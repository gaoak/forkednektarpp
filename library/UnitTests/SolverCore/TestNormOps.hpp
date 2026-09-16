///////////////////////////////////////////////////////////////////////////////
//
// File: TestNormOps.hpp
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
// Description: Fixture for the NormL2 and NormLinf operators. Self contained:
// it owns its session, expansion list and input field, and takes the expected
// norms from the expansion list's own L2/Linf as an independent oracle.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/NormOps/NormL2/NormL2Op.hpp>
#include <SolverCore/NormOps/NormLinf/NormLinfOp.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::Exec);

/**
 * @class TestNormOps
 *
 * A test fixture that fills a physical-space field with an analytic
 * expression and checks NormL2Op and NormLinfOp against the norms the
 * expansion list computes itself, both for the non-interleaved storage and
 * for the interleaved storage the vectorised and device paths use.
 *
 * The fixture constructor (destructor) is called before (after) each call to
 * the BOOST_FIXTURE_TEST_CASE(<test name>, <fixture>) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */
template <typename TData> class TestNormOps
{
public:
    TestNormOps()  = default;
    ~TestNormOps() = default;

    /**
     * @brief Read the session, build the expansion list and allocate the
     * field the test cases work on.
     */
    void Configure()
    {
        SetSession();
        SetExpList();
        SetFixture();
    }

    /**
     * @brief Fill the input field with an analytic expression and record the
     * norms it is expected to have.
     */
    void SetTestCase()
    {
        const auto nphys = m_expList->GetTotPoints();

        // Get coordinates
        Array<OneD, double> x(nphys);
        Array<OneD, double> y(nphys, 0.0);
        Array<OneD, double> z(nphys, 0.0);
        m_expList->GetCoords(x, y, z);

        // Set initial conditions.
        for (unsigned int blk = 0; blk < m_in.GetBlocks().size(); ++blk)
        {
            auto &inblock = m_in.GetBlocks()[blk];
            auto inptr =
                inblock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0;
                 n < m_in.GetNumComponents() * m_in.GetNumHomoModes(); ++n)
            {
                const auto scale = static_cast<TData>(n + 1);
                for (size_t el = 0, cnt = 0; el < inblock.GetNumElements();
                     ++el)
                {
                    for (unsigned int phys = 0; phys < inblock.GetNumData();
                         ++phys, ++cnt)
                    {
                        const auto xc = static_cast<TData>(x[cnt]);
                        const auto yc = static_cast<TData>(y[cnt]);
                        const auto zc = static_cast<TData>(z[cnt]);

                        const auto base =
                            static_cast<TData>(0.3) * scale +
                            static_cast<TData>(0.2) * scale * xc +
                            static_cast<TData>(0.1) * (n + 2) * yc * yc -
                            static_cast<TData>(0.05) * scale * zc;

                        const auto base_plus_error =
                            static_cast<TData>(0.02) * scale *
                                (static_cast<TData>(1.0) + xc * xc +
                                 static_cast<TData>(0.5) * yc +
                                 static_cast<TData>(0.25) * zc * zc) +
                            static_cast<TData>(0.003) * (n + 1) *
                                ((cnt % 5) + 1);
                        inptr[cnt] = base - base_plus_error;
                    }
                }
                inptr += inblock.CompSize();
            }
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    /**
     * @brief Apply both norm operators, once on the non-interleaved storage
     * and once on the interleaved storage.
     */
    void RunTestCase()
    {
        auto l2Op =
            NormL2Op<TData>::Create(m_expList, m_session->GetVariables());
        auto linfOp =
            NormLinfOp<TData>::Create(m_expList, m_session->GetVariables());

        // Non-interleaved case.
        l2Op->Apply(m_in);
        m_l2Test = l2Op->GetNorms();

        linfOp->Apply(m_in);
        m_linfTest = linfOp->GetNorms();

        // Interleaved case.
        std::string execStr =
            Operators::Operator<TData>::GetOpExecSpace(m_session);
        m_in.ReshapeStorage(NektarSpaces::GetVectorWidth<TData>(execStr),
                            execStr);

        l2Op->Apply(m_in);
        m_l2Test2 = l2Op->GetNorms();

        linfOp->Apply(m_in);
        m_linfTest2 = linfOp->GetNorms();
    }

    /**
     * @brief Compare both sets of norms against the expected ones, with
     * absolute tolerance tol.
     *
     * @return bool
     */
    bool Compare(const TData tol)
    {
        const unsigned int numComp =
            m_in.GetNumComponents() * m_in.GetNumHomoModes();

        if (m_l2Test.size() != m_l2Expected.size() ||
            m_linfTest.size() != m_linfExpected.size())
        {
            std::cout << "Mismatch of norm vector sizes." << std::endl;
            return false;
        }

        bool match = true;
        std::cout << " - Non-interleaved cases: " << std::endl;
        for (unsigned int comp = 0; comp < numComp; ++comp)
        {
            const auto l2Diff = std::abs(m_l2Test[comp] - m_l2Expected[comp]);
            const auto linfDiff =
                std::abs(m_linfTest[comp] - m_linfExpected[comp]);

            if (l2Diff > tol)
            {
                std::cout << "Component " << comp
                          << " mismatch: " << "L2(actual=" << m_l2Test[comp]
                          << ", expected=" << m_l2Expected[comp]
                          << ", diff=" << l2Diff << ")" << std::endl;
                match = false;
            }

            if (linfDiff > tol)
            {
                std::cout << "Component " << comp
                          << " mismatch: " << "Linf(actual=" << m_linfTest[comp]
                          << ", expected=" << m_linfExpected[comp]
                          << ", diff=" << linfDiff << ")" << std::endl;
                match = false;
            }
        }

        std::cout << " - Interleaved cases: " << std::endl;
        for (unsigned int comp = 0; comp < numComp; ++comp)
        {
            const auto l2Diff = std::abs(m_l2Test2[comp] - m_l2Expected[comp]);
            const auto linfDiff =
                std::abs(m_linfTest2[comp] - m_linfExpected[comp]);

            if (l2Diff > tol)
            {
                std::cout << "Component " << comp
                          << " mismatch: " << "L2(actual=" << m_l2Test2[comp]
                          << ", expected=" << m_l2Expected[comp]
                          << ", diff=" << l2Diff << ")" << std::endl;
                match = false;
            }

            if (linfDiff > tol)
            {
                std::cout << "Component " << comp << " mismatch: "
                          << "Linf(actual=" << m_linfTest2[comp]
                          << ", expected=" << m_linfExpected[comp]
                          << ", diff=" << linfDiff << ")" << std::endl;
                match = false;
            }
        }

        return match;
    }

protected:
    std::string m_meshName = "";
    SessionReaderSharedPtr m_session;
    ExpListSharedPtr m_expList;
    Field<TData, FieldState::Phys> m_in;

    std::vector<TData> m_l2Test;
    std::vector<TData> m_linfTest;
    std::vector<TData> m_l2Test2;
    std::vector<TData> m_linfTest2;
    std::vector<TData> m_l2Expected;
    std::vector<TData> m_linfExpected;

    void SetSession()
    {
        std::string execStr(GlobalConfiguration::ExecStr());

        BOOST_TEST_MESSAGE("Creating input field");

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

        m_in = Field<TData, FieldState::Phys>("f_in", blockAttr, ncomp, 1);
    }

    /**
     * @brief The oracle: the norms the expansion list computes from the same
     * data, independently of the operators under test.
     */
    void ExpectedSolution()
    {
        const unsigned int numComp =
            m_in.GetNumComponents() * m_in.GetNumHomoModes();
        const auto nphys = m_expList->GetTotPoints();

        Array<OneD, TData> lhs = m_in.template ToArray<TData>();

        m_l2Expected.resize(numComp);
        m_linfExpected.resize(numComp);

        for (unsigned int comp = 0; comp < numComp; ++comp)
        {
            m_l2Expected[comp]   = m_expList->L2(lhs + comp * nphys);
            m_linfExpected[comp] = m_expList->Linf(lhs + comp * nphys);
        }
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestNormOps<float>                              \
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
    class type : public TestNormOps<double>                                    \
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

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Seg3D, "run/segment_3D.xml")

TEST(Quad, "run/square.xml")

TEST(Quad3D, "run/square_3D.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(Tri3D, "run/tri_3D.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(PrismNodal, "run/prism_nodal.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(TetNodal, "run/tet_nodal.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")

#include <UnitTests/TestBoostTeardown.hpp>
