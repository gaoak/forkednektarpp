///////////////////////////////////////////////////////////////////////////////
//
// File: TestIMEX.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestIMEX

#include "TestTimeOps.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <optional>

namespace tt = boost::test_tools;

/**
 * @brief Solves of the test problem, with alpha = 1 and beta = -10, that
 * record the start-up context the time integration operator reports in each
 * implicit solve.
 */
struct StartupContextRecorder
{
    TimeOp<double> *op = nullptr;
    Math::MathHelper math;
    // Solution at the start of the step, and the result of the previous
    // implicit solve
    Field<double, FieldState::Phys> *stepStart      = nullptr;
    Field<double, FieldState::Phys> *previousResult = nullptr;
    Field<double, FieldState::Phys> *wsp            = nullptr;
    // Implicit solves so far in the current step
    unsigned int stageInStep = 0;

    std::vector<std::optional<double>> targets;
    std::vector<double> stageCoefficients;
    std::vector<double> initialSolutionErrors;

    void DoExplicitRhs(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out,
                       [[maybe_unused]] const double &time,
                       const double &factor)
    {
        math.mul(factor, in, out);
    }

    void DoImplicit(Field<double, FieldState::Phys> &in,
                    Field<double, FieldState::Phys> &out,
                    [[maybe_unused]] const double &time, const double &lambda)
    {
        if (const auto startupStage = op->GetStartupStage())
        {
            targets.push_back(startupStage->targetImplicitCoefficient);
            stageCoefficients.push_back(startupStage->stageImplicitCoefficient);
            // The first stage starts from the solution at the start of the
            // step, a later one from the result of the previous stage.
            auto &expected = stageInStep == 0 ? *stepStart : *previousResult;
            math.sub(*startupStage->initialSolution, expected, *wsp);
            initialSolutionErrors.push_back(math.l2norm(*wsp));
        }
        else
        {
            targets.push_back(std::nullopt);
        }
        math.mul(1.0 / (1.0 + 10.0 * lambda), in, out);
        previousResult->template Copy<NektarSpaces::HostSpace>(out);
        ++stageInStep;
    }
};

/**
 * @brief Integrate nsteps steps of scheme and check the start-up context of
 * every implicit solve.
 *
 * The first nstartup implicit solves belong to start-up steps: they report
 * the implicit coefficient target of the multistep scheme, their own
 * coefficient stageCoefficient, and the solution they start from.
 * The later solves report no start-up stage.
 */
void CheckStartupContext(const SessionReaderSharedPtr &session,
                         const ExpListSharedPtr &expList,
                         Field<double, FieldState::Phys> &initial,
                         const std::string &scheme, const unsigned int order,
                         const unsigned int nsteps, const unsigned int nstartup,
                         const std::optional<double> target,
                         const double stageCoefficient)
{
    double timeStep = 0.01;
    session->SetParameter("TimeStep", timeStep);

    const unsigned int ncomp = session->GetVariables().size();
    auto blockAttr           = GetBlockAttributes<double, FieldState::Phys>(
        expList, Operator<double>::GetDefaultInterleaveWidth(session));
    Field<double, FieldState::Phys> solution("solution", blockAttr, ncomp, 1);
    Field<double, FieldState::Phys> stepStart("step start", blockAttr, ncomp,
                                              1);
    Field<double, FieldState::Phys> previousResult("previous result", blockAttr,
                                                   ncomp, 1);
    Field<double, FieldState::Phys> wsp("wsp", blockAttr, ncomp, 1);
    solution.template Copy<NektarSpaces::HostSpace>(initial);

    auto op =
        TimeOp<double>::Create(expList, session->GetVariables(), scheme, order);

    StartupContextRecorder recorder;
    recorder.op   = op.get();
    recorder.math = Math::MathHelper(Operator<double>::GetOpExecSpace(session));
    recorder.stepStart      = &stepStart;
    recorder.previousResult = &previousResult;
    recorder.wsp            = &wsp;
    op->DefineExplicitRhs(&StartupContextRecorder::DoExplicitRhs, &recorder);
    op->DefineImplicit(&StartupContextRecorder::DoImplicit, &recorder);
    op->EnableExplicitContributionExtrapolation();

    while (op->GetStep() < nsteps)
    {
        stepStart.template Copy<NektarSpaces::HostSpace>(solution);
        recorder.stageInStep = 0;
        op->Apply(solution);
    }

    BOOST_TEST_REQUIRE(recorder.targets.size() > nstartup);
    for (unsigned int i = 0; i < recorder.targets.size(); ++i)
    {
        if (i < nstartup)
        {
            BOOST_TEST_REQUIRE(target.has_value());
            BOOST_TEST_REQUIRE(recorder.targets[i].has_value());
            BOOST_TEST(*recorder.targets[i] == *target, tt::tolerance(1e-14));
            BOOST_TEST(recorder.stageCoefficients[i] == stageCoefficient,
                       tt::tolerance(1e-14));
        }
        else
        {
            BOOST_TEST(!recorder.targets[i].has_value());
        }
    }

    BOOST_TEST(recorder.initialSolutionErrors.size() == nstartup);
    for (const double error : recorder.initialSolutionErrors)
    {
        BOOST_TEST(error == 0.0);
    }
}

#define TEST_SCHEME(test_name, test, order)                                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase(1.0, -10.0);                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(CheckOrderOfAccuracy("IMEX", "", order));               \
        }                                                                      \
    }

#define TEST_STARTUP_CONTEXT(test_name, test, scheme, order, nsteps, nstartup, \
                             target, stageCoefficient)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase(1.0, -10.0);                                               \
        CheckStartupContext(m_session, m_expList, m_in, scheme, order, nsteps, \
                            nstartup, target, stageCoefficient);               \
    }

BOOST_AUTO_TEST_SUITE(TestSuiteIMEX)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_SCHEME(imex_order_1, segment, 1)

TEST_SCHEME(imex_order_2, segment, 2)

TEST_SCHEME(imex_order_3, segment, 3)

TEST_SCHEME(imex_order_4, segment, 4)

// The start-up steps of IMEX order J are IMEXdirk 2_3_2 steps for J = 2 and
// IMEXdirk 3_4_3 steps for J = 3 and 4, with two and three implicit stages.
TEST_STARTUP_CONTEXT(imex_order_2_startup_context, segment, "IMEX", 2, 3, 2,
                     2.0 / 3.0, (2.0 - 1.414213562373095) / 2.0)

TEST_STARTUP_CONTEXT(imex_order_3_startup_context, segment, "IMEX", 3, 4, 6,
                     6.0 / 11.0, 0.4358665215)

TEST_STARTUP_CONTEXT(imex_order_4_startup_context, segment, "IMEX", 4, 5, 9,
                     12.0 / 25.0, 0.4358665215)

// A one-step scheme has no start-up.
TEST_STARTUP_CONTEXT(imexdirk_startup_context, segment, "IMEXdirk23", 2, 2, 0,
                     std::nullopt, 0.0)
#endif

BOOST_AUTO_TEST_SUITE_END()
