///////////////////////////////////////////////////////////////////////////////
//
// File: TestPreconOps.hpp
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
// Description: Fixture shared by the DiagPrecon and NullPrecon unit tests.
// Self contained: it owns its session, continuous expansion list and fields,
// and takes the expected result from the legacy MultiRegions preconditioner
// as an independent oracle.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(NEKTAR_USE_MAGMA)
#include "magma_v2.h"
#endif

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"
#include <SolverCore/PreconOps/PreconOp.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/GlobalLinSys.h>
#include <MultiRegions/Preconditioner.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <UnitTests/TestBoostSetup.hpp>

#include <cmath>
#include <cstdio>
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
    // The execution space and implementation strings, captured once up
    // front. Two reasons: MPI_Init below is handed the live argv and may
    // rewrite it, and reading argv[1]/argv[2] per test without a bounds check
    // was undefined when the arguments were missing - which happens more
    // easily than it sounds, e.g. zsh passes an unquoted $var holding
    // "Serial SumFac" as a single argument. The bounds check in the
    // constructor now turns that into a usage message instead.
    static std::string &ExecStr()
    {
        static std::string s;
        return s;
    }
    static std::string &ImplStr()
    {
        static std::string s;
        return s;
    }

    GlobalConfiguration()
    {
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        if (argc > 2)
        {
            ExecStr() = argv[1];
            ImplStr() = argv[2];
        }

#ifdef NEKTAR_USE_MAGMA
        magma_init();
#endif
#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif

        // The execution space and the implementation must both be present:
        // exe -- ExecName ImplName. These tests drive the Helmholtz operator,
        // whose implementation is selected by the second argument, so unlike
        // the other SolverCore fixtures one argument is not enough.
        if (argc < 3)
        {
            int rank = 0;

#ifdef NEKTAR_USE_MPI
            MPI_Comm comm = MPI_COMM_WORLD;
            MPI_Comm_rank(comm, &rank);
#endif
            if (rank == 0)
            {
                std::string execname(argv[0]);

                std::cerr << "Usage: " << execname << " -- ExecName ImplName"
                          << std::endl;
#if defined(NEKTAR_ENABLE_DEVICE) && defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX, Device" << std::endl;
                std::cerr << "\t ImplName = StdMat, SumFac, SumFacTOP"
#elif defined(NEKTAR_ENABLE_DEVICE)
                std::cerr << "\t ExecName = Serial, Device" << std::endl;
                std::cerr << "\t ImplName = StdMat, SumFac, SumFacTOP"
#elif defined(NEKTAR_ENABLE_SIMD)
                std::cerr << "\t ExecName = Serial, AVX" << std::endl;
                std::cerr << "\t ImplName = StdMat, SumFac"
#else
                std::cerr << "\t ExecName = Serial" << std::endl;
                std::cerr << "\t ImplName = StdMat, SumFac"
#endif
                          << std::endl;
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
#ifdef NEKTAR_USE_MAGMA
        magma_finalize();
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
 * @class TestPreconOps
 *
 * @brief Compares a preconditioner operator against its legacy counterpart.
 *
 * The operator under test is built through PreconOp::Create for the requested
 * method and configured against a Helmholtz operator; the expected result
 * comes from the MultiRegions preconditioner of the same name, built on a
 * legacy GlobalLinSys for the same problem. The two reach the same numbers by
 * different routes, so the comparison checks the operator, its configuration
 * from the element operator, and its handling of the assembled storage.
 *
 * The fixture constructor (destructor) is called before (after) each call to
 * the BOOST_FIXTURE_TEST_CASE(<test name>, <fixture>) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */
template <typename TData> class TestPreconOps
{
public:
    TestPreconOps()  = default;
    ~TestPreconOps() = default;

    /**
     * @brief Read the session, build the continuous expansion list and
     * allocate the fields the test cases work on.
     */
    void Configure()
    {
        SetSession();
        SetExpList();
        SetFixture();
    }

    /**
     * @brief Set the input coefficients and the problem parameters, and
     * record the result the preconditioner is expected to produce.
     */
    void SetTestCase(const std::string &method)
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < m_in.GetBlocks().size(); ++blk)
        {
            auto &block = m_in.GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
            {
                for (unsigned int coeff = 0; coeff < block.GetNumData();
                     ++coeff, ++cnt)
                {
                    inptr[cnt] = 1.0;
                }
            }
        }

        // Get lambda from m_session or default to 10.0
        m_lambda = m_session->DefinesParameter("Lambda")
                       ? m_session->GetParameter("Lambda")
                       : 10.0;

        // Set up diffusion coefficient.
        const auto coordDim      = m_expList->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        if (coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (coordDim == 2)
        {
            m_diffCoeff[0] = 1.0; // D00
            m_diffCoeff[2] = 1.0; // D11
        }
        else
        {
            m_diffCoeff[0] = 1.0; // D00
            m_diffCoeff[2] = 1.0; // D11
            m_diffCoeff[5] = 1.0; // D22
        }

        // Compute expected solution.
        ExpectedSolution(method);
    }

    /**
     * @brief Build the preconditioner for @p method, configure it against the
     * Helmholtz operator and apply it to the assembled input.
     */
    void RunTestCase(const std::string &method)
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(m_session);

        auto assmb = AssmbScatrZeroDirOp<TData>::Create(
            m_expList, m_session->GetVariables());
        auto op =
            HelmholtzOp<TData>::Create(m_expList, m_session->GetVariables());
        auto precon = PreconOp<TData>::Create(
            m_expList, m_session->GetVariables(), method);
        op->SetLambda(m_lambda);
        op->SetDiffCoeff(m_diffCoeff);
        precon->Configure(op);
        assmb->Apply(m_in, m_out);

        // reshape m_out
        if (execStr == "AVX")
        {
            m_out.ReshapeStorage(NektarSpaces::GetVectorWidth<TData>(execStr),
                                 execStr);
        }

        precon->Apply(m_out, m_out);
    }

    /**
     * @brief The oracle: the legacy MultiRegions preconditioner of the same
     * name, built on a GlobalLinSys for the same Helmholtz problem.
     */
    void ExpectedSolution(const std::string &method)
    {
        auto nin    = m_session->GetVariables().size();
        auto ncoeff = m_expList->GetNcoeffs();

        // Calculate expected result from Nektar++
        Array<OneD, TData> incoeffs = m_in.ToArray();
        Array<OneD, TData> outcoeffs(nin * ncoeff, 0.0);

        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;

        auto map = std::dynamic_pointer_cast<ContField>(m_expList)
                       ->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map, factors);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, m_expList, map);
        auto precond =
            GetPreconFactory().CreateInstance(method, globalSys, map);
        precond->BuildPreconditioner();
        for (unsigned int n = 0; n < nin; n++)
        {
            Array<OneD, TData> tmp;
            precond->DoPreconditioner(incoeffs + n * ncoeff,
                                      tmp = outcoeffs + n * ncoeff, true);
        }
        m_expected.template CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

    /**
     * @brief Compare the computed field to the expected one, with absolute
     * tolerance tol. The two must have the same storage shape and the same
     * components.
     *
     * @return bool
     */
    bool Compare(TData tol)
    {
        auto rank = m_session->GetComm()->GetRank();

        if (m_expected.GetNumComponents() != m_out.GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (m_expected.GetNumHomoModes() != m_out.GetNumHomoModes())
        {
            std::cout << "Mismatch of number of homogeneous modes."
                      << std::endl;
            return false;
        }

        if (m_expected.GetBlocks().size() != m_out.GetBlocks().size())
        {
            std::cout << "Mismatch of block size." << std::endl;
            return false;
        }

        ReshapeToScalar(m_out);
        ReshapeToScalar(m_expected);

        bool isMatch = true;

        if (rank == 0)
        {
            printf("#elm #pts output               expected            "
                   "difference\n");
        }
        for (unsigned int blk = 0; blk < m_out.GetBlocks().size(); ++blk)
        {
            const TData *outptr =
                m_out.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const TData *expptr =
                m_expected.GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            if ((m_out.GetBlocks()[blk].GetNumElements() !=
                 m_expected.GetBlocks()[blk].GetNumElements()) ||
                (m_out.GetBlocks()[blk].GetNumData() !=
                 m_expected.GetBlocks()[blk].GetNumData()))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            for (unsigned int n = 0;
                 n < m_out.GetNumComponents() * m_out.GetNumHomoModes(); ++n)
            {
                size_t MisMatchcnt = 0, total = 0;

                for (size_t el = 0, cnt = 0;
                     el < m_out.GetBlocks()[blk].GetNumElements(); ++el)
                {
                    for (unsigned int pts = 0;
                         pts < m_out.GetBlocks()[blk].GetNumData();
                         ++pts, ++cnt)
                    {
                        if (std::isnan(outptr[cnt]) ||
                            std::isinf(outptr[cnt]) ||
                            std::abs(outptr[cnt] - expptr[cnt]) > tol)
                        {
                            printf("%04lu %04u %20.16f %20.16f %20.16f\n", el,
                                   pts, outptr[cnt], expptr[cnt],
                                   std::abs(outptr[cnt] - expptr[cnt]));
                            MisMatchcnt++;
                        }
                        total++;
                    }
                }

                outptr += m_out.GetBlocks()[blk].CompSize();
                expptr += m_expected.GetBlocks()[blk].CompSize();

                if (MisMatchcnt)
                {
                    std::cout << "Number of mismatches in component " << n
                              << " on block " << blk << " is " << MisMatchcnt
                              << " out of " << total << " on rank: " << rank
                              << std::endl;
                    isMatch = false;
                }
            }
        }

        return isMatch;
    }

protected:
    std::string m_meshName = "";
    SessionReaderSharedPtr m_session;
    ExpListSharedPtr m_expList;
    Field<TData, FieldState::Coeff> m_in;
    Field<TData, FieldState::Coeff> m_out;
    Field<TData, FieldState::Coeff> m_expected;

    TData m_lambda;
    std::vector<TData> m_diffCoeff;

    void SetSession()
    {
        std::string execStr(GlobalConfiguration::ExecStr());
        std::string implStr(GlobalConfiguration::ImplStr());

        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 4;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = strdup(m_meshName.data());
        argv[2]     = strdup(("--opExecSpace=" + execStr).c_str());
        argv[3]     = strdup(("--opImpl=" + implStr).c_str());

        m_session = SessionReader::CreateInstance(argc, argv);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;
    }

    /**
     * @brief A continuous field: the preconditioners act on the assembled,
     * globally continuous coefficients, and the oracle needs the local to
     * global map that only ContField carries.
     */
    void SetExpList()
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(m_session);
        m_expList  = MemoryManager<ContField>::AllocateSharedPtr(
            m_session, graph, "u", true, false, Collections::eNoCollection);
        m_expList->SetDataWarehouse();
    }

    void SetFixture()
    {
        const unsigned int ncomp = m_session->GetVariables().size();
        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(m_expList);

        m_in  = Field<TData, FieldState::Coeff>("f_in", blockAttr, ncomp, 1);
        m_out = Field<TData, FieldState::Coeff>("f_out", blockAttr, ncomp, 1);
        m_expected =
            Field<TData, FieldState::Coeff>("f_expected", blockAttr, ncomp, 1);
    }

    void ReshapeToScalar(Field<TData, FieldState::Coeff> &in)
    {
        in.ReshapeStorage(1, "Serial");
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestPreconOps<float>                            \
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
    class type : public TestPreconOps<double>                                  \
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

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz1D_3C, "run/Helmholtz1D_3C.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz2D_3C, "run/Helmholtz2D_3C.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")

#include <UnitTests/TestBoostTeardown.hpp>
