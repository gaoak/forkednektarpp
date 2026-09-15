///////////////////////////////////////////////////////////////////////////////
//
// File: TestTimeOps.hpp
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
// Description: Test fixture for the time-integration schemes, driven through
// the SolverCore TimeOp interface.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/TimeOps/TimeOp.hpp>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <UnitTests/TestBoostSetup.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

struct GlobalConfiguration
{
    // The execution space string, captured once up front: MPI_Init below is
    // handed the live argv and may rewrite it, and reading argv[1] per test
    // without a bounds check is undefined when the argument is missing. The
    // bounds check in the constructor turns that into a usage message
    // instead.
    static std::string &ExecStr()
    {
        static std::string s;
        return s;
    }

    GlobalConfiguration()
    {
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        if (argc > 1)
        {
            ExecStr() = argv[1];
        }

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif

        // The execution space must be present: exe -- ExecName. Anything
        // less gets the usage message.
        if (argc < 2)
        {
            int rank = 0;

#ifdef NEKTAR_USE_MPI
            MPI_Comm comm = MPI_COMM_WORLD;
            MPI_Comm_rank(comm, &rank);
#endif
            if (rank == 0)
            {
                std::string execname(argv[0]);

                std::cerr << "Usage: " << execname << " -- ExecName"
                          << std::endl;
#if defined(NEKTAR_ENABLE_DEVICE) && defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_DEVICE)
                std::cerr << "\t ExecName = Serial, Device" << std::endl;
#elif defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX" << std::endl;
#else
                std::cerr << "\t ExecName = Serial" << std::endl;
#endif
            }
#ifdef NEKTAR_USE_MPI
            MPI_Finalize();
#endif
            exit(1);
        }
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

/**
 * @class TestTimeOps
 *
 * A test fixture that integrates the scalar test problem
 * du/dt = (alpha + beta) u in time, alpha being the non-stiff factor carried
 * by the explicit part of a scheme and beta the stiff factor carried by the
 * implicit part. The exact solution u = e^((alpha + beta) t) with u(t=0) = 1
 * gives the reference the observed order of accuracy is measured against.
 *
 * The fixture constructor (destructor) is called before (after) each call to
 * the BOOST_FIXTURE_TEST_CASE(<test name>, <fixture>) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */
template <typename TData> class TestTimeOps
{
public:
    TestTimeOps()  = default;
    ~TestTimeOps() = default;

    /**
     * @brief Read the session, build the expansion list and allocate the
     * fields the test cases work on.
     */
    void Configure()
    {
        SetSession();
        SetExpList();
        SetFixture();
    }

    /**
     * @brief Set the initial condition and the analytic solution it is
     * expected to reach at the final time.
     *
     * @param alpha  non-stiff factor, integrated by the explicit part
     * @param beta   stiff factor, integrated by the implicit part
     */
    void SetTestCase(const TData alpha, const TData beta)
    {
        // Initialise math kernel
        std::string execName = Operator<TData>::GetOpExecSpace(m_session);
        m_math               = Math::MathHelper(execName);

        // Set initial value
        for (unsigned int blk = 0; blk < m_in.GetBlocks().size(); ++blk)
        {
            auto &block = m_in.GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < m_in.GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = 1.0 + phys;
                    }
                }
                inptr += block.CompSize();
            }
        }

        // Parameters for analytic solution
        m_alpha = alpha; // non-stiff factor
        m_beta  = beta;  // stiff factor

        const double final_time = 0.5;

        // Compute expected solution
        ExpectedSolution(final_time);
    }

    /**
     * @brief Integrate the test problem for @p numsteps steps of the scheme
     * given, leaving the result in the output field.
     */
    void RunTestCase(const std::string scheme, const std::string variant,
                     const unsigned int order,
                     const std::vector<TData> freeParams,
                     const unsigned int numsteps)
    {
        // Copy m_in to m_out since operator uses apply with inout type
        m_out.template Copy<NektarSpaces::HostSpace>(m_in);

        // Initialise Time-stepping operator
        auto op = TimeOp<TData>::Create(m_expList, m_session->GetVariables(),
                                        scheme, order, variant, freeParams);
        op->DefineExplicitRhs(&TestTimeOps::DoExplicitRHS, this);
        op->DefineImplicitRhs(&TestTimeOps::DoImplicitRHS, this);
        op->DefineImplicit(&TestTimeOps::DoLHS, this);
        op->DefineProjection(&TestTimeOps::DoProjection, this);

        // Initialise timestepping operator
        // Loop all steps
        while (op->GetStep() < numsteps)
        {
            // Evolve PDE for one timestep
            op->Apply(m_out);
        }
    }

    /**
     * @brief Fill the expected field with the analytic solution at
     * @p final_time.
     */
    void ExpectedSolution(const double final_time)
    {
        // We solve the analytic problem du/dt = \alpha u + \beta u,
        // where \alpha is mild parameter leading to the explicit part
        // and \beta is a stiff parameter leading to the implicit part.
        // The solution is u = e^((\alpha + \beta)*t) and u(t=0) = 1.0
        m_expected.template Copy<NektarSpaces::HostSpace>(m_in);
        m_math.mul(exp((m_alpha + m_beta) * final_time), m_expected,
                   m_expected);
    }

    /**
     * @brief Integrate the test problem over a sequence of timesteps and
     * check the order the error converges at against the order claimed by
     * the scheme.
     */
    bool CheckOrderOfAccuracy(const std::string scheme,
                              const std::string variant,
                              const unsigned int order,
                              const std::vector<TData> freeParams = {})
    {
        // The timestep ladder is test configuration, not field data: keeping
        // it double makes final_time/dt land exactly on an integer step count
        // whatever TData is. At float, 0.5f/0.001f is 499.99997, which
        // truncates to 499 and stops the integration a step short.
        const double final_time = 0.5;

        std::vector<double> timesteps;

        if (order <= 2)
        {
            timesteps = {0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005};
        }
        else if (order <= 3)
        {
            timesteps = {0.1, 0.05, 0.025, 0.01, 0.005, 0.001};
        }
        else if (order == 4)
        {
            timesteps = {0.1, 0.05, 0.02, 0.01, 0.005};
        }
        else if (order == 5)
        {
            if (scheme == "ImplicitSDC" || scheme == "IMEXSDC" ||
                scheme == "ImplicitGEM" || scheme == "IMEXGEM")
            {
                timesteps = {0.1, 0.05, 0.02, 0.01, 0.005};
            }
            else
            {
                timesteps = {0.2, 0.1, 0.05, 0.025};
            }
        }
        else
        {
            if (scheme == "IMEXSDC")
            {
                timesteps = {0.1, 0.05, 0.02, 0.01, 0.005};
            }
            else
            {
                timesteps = {0.4, 0.2, 0.1};
            }
        }

        std::vector<TData> errors;

        for (double dt : timesteps)
        {
            // Change timestep in m_session to propagate to TimeOp at runtime
            m_session->SetParameter("TimeStep", dt);

            // Run simulation
            RunTestCase(scheme, variant, order, freeParams, final_time / dt);

            // Compute error at final time (L2 norm; adapt as needed)
            m_math.sub(m_out, m_expected, m_out);
            TData error = std::sqrt(m_math.l2norm(m_out) / m_math.l2norm(m_in));
            errors.push_back(error);
        }

        // Compute observed order of accuracy from (dt, error) pairs
        std::vector<TData> orders;
        for (size_t i = 0; i < timesteps.size() - 1; ++i)
        {
            TData order = log(errors[i] / errors[i + 1]) /
                          log(timesteps[i] / timesteps[i + 1]);
            orders.push_back(order);
        }

        // Use last observed order
        TData observedOrder = orders.back();

        // Check if observed order is close to expected order (within tolerance)
        if (std::isnan(observedOrder) || std::isinf(observedOrder) ||
            observedOrder < 0.9 * order) // Choose tolerance
        {
            std::cerr << "Order of accuracy test failed! Observed: "
                      << observedOrder << ", Expected: " << order << std::endl;
            return false;
        }
        else
        {
            std::cout << "Order of accuracy verified: " << observedOrder
                      << std::endl;
            return true;
        }
    }

protected:
    std::string m_meshName = "";
    TData m_alpha          = 0;
    TData m_beta           = 0;

    Math::MathHelper m_math;

    Field<TData, FieldState::Phys> m_in;
    Field<TData, FieldState::Phys> m_out;
    Field<TData, FieldState::Phys> m_expected;
    ExpListSharedPtr m_expList;
    SessionReaderSharedPtr m_session;

    /**
     * @brief Create the session from the mesh named by the derived fixture and
     * the execution space given on the command line.
     */
    void SetSession()
    {
        std::string execStr(GlobalConfiguration::ExecStr());

        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 3;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = strdup(m_meshName.data());
        argv[2]     = strdup(("--opExecSpace=" + execStr).c_str());

        m_session = SessionReader::CreateInstance(argc, argv);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;
    }

    void SetExpList()
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(m_session);
        m_expList  = MemoryManager<ExpList>::AllocateSharedPtr(
            m_session, graph, true, "u", Collections::eNoCollection);
        m_expList->SetDataWarehouse();
    }

    void SetFixture()
    {
        const unsigned int ncomp = m_session->GetVariables().size();
        auto blockAttr = GetBlockAttributes<TData, FieldState::Phys>(m_expList);

        m_in  = Field<TData, FieldState::Phys>("f_in", blockAttr, ncomp, 1);
        m_out = Field<TData, FieldState::Phys>("f_out", blockAttr, ncomp, 1);
        m_expected =
            Field<TData, FieldState::Phys>("f_expected", blockAttr, ncomp, 1);
    }

    /**
     * @brief Implicit solve of the stiff part of the test problem.
     */
    void DoLHS(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out,
               [[maybe_unused]] const TData &time, const TData &lambda)
    {
        // Factor for implicit/stiff part of analytic test problem
        auto factor = 1.0 / (1.0 - lambda * m_beta);

        // Multiply extrapolated rhs
        m_math.mul(factor, in, out);
    }

    /**
     * @brief Right-hand side of the non-stiff part of the test problem.
     */
    void DoExplicitRHS(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Phys> &out,
                       [[maybe_unused]] const TData &time, const TData &factor)
    {
        // Multiply solution by factor
        m_math.mul(m_alpha * factor, in, out);
    }

    /**
     * @brief Right-hand side of the stiff part of the test problem.
     */
    void DoImplicitRHS(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Phys> &out,
                       [[maybe_unused]] const TData &time, const TData &factor)
    {
        // Multiply solution by factor
        m_math.mul(m_beta * factor, in, out);
    }

    /**
     * @brief Projection between timesteps, the identity for this test problem.
     */
    void DoProjection(Field<TData, FieldState::Phys> &in,
                      Field<TData, FieldState::Phys> &out,
                      [[maybe_unused]] const TData &time)
    {
        // Multiply solution by factor
        m_math.mul(1.0, in, out);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestTimeOps<float>                              \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public TestTimeOps<double>                                    \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(segment, "run/segment.xml")

#include <UnitTests/TestBoostTeardown.hpp>
