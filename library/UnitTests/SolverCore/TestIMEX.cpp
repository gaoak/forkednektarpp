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

#include <SolverCore/TimeOps/IMEX/IMEXOp.hpp>

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <numeric>
#include <optional>

namespace tt = boost::test_tools;

/**
 * @brief Solves of the test problem, with alpha = 1 and beta = -10, that
 * record what a solver can observe of the start-up of a multistep scheme in
 * each implicit solve.
 */
struct StartupRecorder
{
    TimeOp<double> *op = nullptr;
    Math::MathHelper math;
    // Solution at the start of the step, the result of the previous
    // implicit solve, and the input of the latest explicit RHS evaluation
    Field<double, FieldState::Phys> *stepStart      = nullptr;
    Field<double, FieldState::Phys> *previousResult = nullptr;
    Field<double, FieldState::Phys> *explicitInput  = nullptr;
    Field<double, FieldState::Phys> *wsp            = nullptr;
    // Implicit solves so far in the current step
    unsigned int stageInStep = 0;

    std::vector<unsigned int> steps;
    std::vector<double> implicitCoefficients;
    std::vector<double> explicitInputErrors;
    std::vector<IMEXBaseOp<double>::ExplicitWeights> recordedWeights;

    void DoExplicitRhs(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out,
                       [[maybe_unused]] const double &time,
                       const double &factor)
    {
        explicitInput->template Copy<NektarSpaces::HostSpace>(in);
        math.mul(factor, in, out);
    }

    void DoImplicit(Field<double, FieldState::Phys> &in,
                    Field<double, FieldState::Phys> &out,
                    [[maybe_unused]] const double &time, const double &lambda,
                    const IMEXBaseOp<double>::ExplicitWeights &weights)
    {
        steps.push_back(op->GetStep());
        recordedWeights.push_back(weights);
        implicitCoefficients.push_back(lambda / op->GetTimeStep());
        // The first stage starts from the solution at the start of the step,
        // a later one from the result of the previous stage.
        auto &expected = stageInStep == 0 ? *stepStart : *previousResult;
        math.sub(*explicitInput, expected, *wsp);
        explicitInputErrors.push_back(math.l2norm(*wsp));

        math.mul(1.0 / (1.0 + 10.0 * lambda), in, out);
        previousResult->template Copy<NektarSpaces::HostSpace>(out);
        ++stageInStep;
    }
};

/**
 * @brief Integrate nsteps steps of scheme and check the start-up of every
 * implicit solve.
 *
 * The first nstartup implicit solves belong to start-up steps: GetStep() is
 * below order - 1 in them, their implicit coefficient is stageCoefficient,
 * and the latest explicit RHS was evaluated at the solution they start from.
 * The later solves are steps of the multistep scheme, whose implicit
 * coefficient target is the sum of its explicit contribution coefficients.
 * Every solve receives the coefficients of the explicit contributions it
 * combines; a start-up solve with the history id of its start-up scheme.
 */
void CheckStartup(const SessionReaderSharedPtr &session,
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
    Field<double, FieldState::Phys> explicitInput("explicit input", blockAttr,
                                                  ncomp, 1);
    Field<double, FieldState::Phys> wsp("wsp", blockAttr, ncomp, 1);
    solution.template Copy<NektarSpaces::HostSpace>(initial);

    auto op =
        TimeOp<double>::Create(expList, session->GetVariables(), scheme, order);

    StartupRecorder recorder;
    recorder.op   = op.get();
    recorder.math = Math::MathHelper(Operator<double>::GetOpExecSpace(session));
    recorder.stepStart      = &stepStart;
    recorder.previousResult = &previousResult;
    recorder.explicitInput  = &explicitInput;
    recorder.wsp            = &wsp;
    auto imexOp             = std::dynamic_pointer_cast<IMEXBaseOp<double>>(op);
    BOOST_TEST_REQUIRE(imexOp);
    imexOp->DefineExplicitRhs(&StartupRecorder::DoExplicitRhs, &recorder);
    imexOp->DefineImplicitWithWeights(&StartupRecorder::DoImplicit, &recorder);

    if (target)
    {
        const auto coeffs = imexOp->GetExplicitCoefficients();
        BOOST_TEST(std::accumulate(coeffs.begin(), coeffs.end(), 0.0) ==
                       *target,
                   tt::tolerance(1e-14));
    }

    while (op->GetStep() < nsteps)
    {
        stepStart.template Copy<NektarSpaces::HostSpace>(solution);
        recorder.stageInStep = 0;
        op->Apply(solution);
    }

    const unsigned int startupSteps = target ? order - 1 : 0;
    BOOST_TEST_REQUIRE(recorder.steps.size() > nstartup);
    const unsigned int mainHistoryId =
        recorder.recordedWeights.back().historyId;
    for (unsigned int i = 0; i < recorder.steps.size(); ++i)
    {
        const auto &weights = recorder.recordedWeights[i];
        const auto &coeffs  = weights.coefficients;
        BOOST_TEST(!coeffs.empty());
        BOOST_TEST(coeffs.size() <= weights.historySize);
        if (i < nstartup)
        {
            BOOST_TEST(recorder.steps[i] < startupSteps);
            BOOST_TEST(recorder.implicitCoefficients[i] == stageCoefficient,
                       tt::tolerance(1e-14));
            BOOST_TEST(recorder.explicitInputErrors[i] == 0.0);
            BOOST_TEST(weights.historyId != mainHistoryId);
        }
        else
        {
            BOOST_TEST(recorder.steps[i] >= startupSteps);
            BOOST_TEST(weights.historyId == mainHistoryId);
            if (target)
            {
                BOOST_TEST(recorder.implicitCoefficients[i] == *target,
                           tt::tolerance(1e-14));
                BOOST_TEST(weights.historySize == order);
                BOOST_TEST(std::accumulate(coeffs.begin(), coeffs.end(), 0.0) ==
                               *target,
                           tt::tolerance(1e-14));
            }
        }
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
        CheckStartup(m_session, m_expList, m_in, scheme, order, nsteps,        \
                     nstartup, target, stageCoefficient);                      \
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
