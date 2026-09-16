///////////////////////////////////////////////////////////////////////////////
//
// File: TestFwdTrans.hpp
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
// Description: Fixture for the FwdTrans operator. Builds on the shared
// TestLinearSolver base in TestLinearSolver.hpp.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <UnitTests/SolverCore/TestLinearSolver.hpp>

#include <SolverCore/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp>
#include <SolverCore/PreconOps/PreconOp.hpp>

using namespace Nektar::SolverCore;

template <typename TData, typename TExpList>
class TestFwdTransBase : public TestLinearSolver<TData, TExpList>
{
public:
    TestFwdTransBase() = default;

    void SetTestCase()
    {
        // Set initial conditions.
        const unsigned int numComp =
            this->m_in.GetNumComponents() * this->m_in.GetNumHomoModes();
        const size_t nphys = this->m_expList->GetTotPoints();
        m_coordDim         = this->m_expList->GetCoordim(0);

        Array<OneD, TData> x(nphys);
        Array<OneD, TData> y(nphys);
        Array<OneD, TData> z(nphys);
        Array<OneD, TData> fce(numComp * nphys, 0.0);
        this->m_expList->GetCoords(x, y, z);

        const bool useForcing = !std::is_same_v<TExpList, DisContField> &&
                                this->m_session->DefinesFunction("Forcing");

        if (useForcing)
        {
            for (unsigned int n = 0; n < numComp; ++n)
            {
                auto func = this->m_session->GetFunction(
                    "Forcing", n % this->m_session->GetVariables().size());
                Array<OneD, TData> fceVar = fce + n * nphys;
                func->Evaluate(x, y, z, fceVar);
            }
        }

        for (unsigned int blk = 0; blk < this->m_in.GetBlocks().size(); ++blk)
        {
            auto &block = this->m_in.GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < numComp; ++n)
            {
                auto xptr   = x.data();
                auto yptr   = y.data();
                auto zptr   = z.data();
                auto fceptr = fce.data() + n * nphys;

                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        if (useForcing)
                        {
                            inptr[cnt] = *(fceptr++);
                        }
                        else
                        {
                            inptr[cnt] = 1.0 + n;
                            if (m_coordDim == 1)
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] +=
                                        (n + 1) * p * std::pow(*xptr, p);
                                }
                                xptr++;
                            }
                            else if (m_coordDim == 2)
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] += (n + 1) * p *
                                                  std::pow(*xptr, p) *
                                                  std::pow(*yptr, p);
                                }
                                xptr++;
                                yptr++;
                            }
                            else
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] +=
                                        (n + 1) * p * std::pow(*xptr, p) *
                                        std::pow(*yptr, p) * std::pow(*zptr, p);
                                }
                                xptr++;
                                yptr++;
                                zptr++;
                            }
                        }
                    }
                }

                inptr += block.CompSize();
            }
        }

        this->m_out.template Initialize<NektarSpaces::HostSpace>(0.0);

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase(const std::string &method)
    {
        std::string execStr(GlobalConfiguration::ExecStr());
        std::string implStr(GlobalConfiguration::ImplStr());

        // reshape this->m_in
        if (implStr == "SumFac" || execStr == "AVX")
        {
            this->m_in.ReshapeStorage(
                NektarSpaces::GetVectorWidth<TData>(execStr), execStr);
            this->m_out.SetInterleaveWidth(
                NektarSpaces::GetVectorWidth<TData>(execStr));
        }

        auto op = FwdTransOp<TData>::Create(this->m_expList,
                                            this->m_session->GetVariables());

        if constexpr (!std::is_same_v<TExpList, DisContField>)
        {
            auto precon = PreconOp<TData>::Create(
                this->m_expList, this->m_session->GetVariables(), "Diagonal");
            auto linsolve = LinearSolverOp<TData>::Create(
                this->m_expList, this->m_session->GetVariables(), method);
            op->SetLinearSolver(linsolve);
            op->SetPrecon(precon);
            op->UpdatePrecon();
        }
        op->Apply(this->m_in, this->m_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp =
            this->m_in.GetNumComponents() * this->m_in.GetNumHomoModes();
        const size_t nphys    = this->m_expList->GetTotPoints();
        const size_t ncoeffs  = this->m_expList->GetNcoeffs();
        const auto &variables = this->m_session->GetVariables();

        // Setup input/output arrays
        Array<OneD, TData> inphys = this->m_in.ToArray();
        Array<OneD, TData> outcoeffs(numComp * ncoeffs, 0.0);

        // Solve each component separately
        auto graph = SpatialDomains::MeshGraphIO::Read(this->m_session);
        for (unsigned int n = 0; n < numComp; ++n)
        {
            auto expListVar = MemoryManager<TExpList>::AllocateSharedPtr(
                this->m_session, graph, variables[n % variables.size()], true,
                std::is_same_v<TExpList, DisContField>,
                Collections::eNoCollection);
            Array<OneD, TData> inphysVar    = inphys + n * nphys;
            Array<OneD, TData> outcoeffsVar = outcoeffs + n * ncoeffs;
            expListVar->FwdTrans(inphysVar, outcoeffsVar);
        }

        // Copy solution back
        this->m_expected.template CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

protected:
    unsigned int m_coordDim;
};

template <typename TData>
class TestFwdTransCG : public TestFwdTransBase<TData, ContField>
{
};

template <typename TData>
class TestFwdTransDG : public TestFwdTransBase<TData, DisContField>
{
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestFwdTransCG<float>                           \
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
    class type : public TestFwdTransCG<double>                                 \
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

#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTDGFLOAT(type, filename)                                            \
    class type##float : public TestFwdTransDG<float>                           \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTDGFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDGDOUBLE(type, filename)                                           \
    class type : public TestFwdTransDG<double>                                 \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            m_meshName = filename;                                             \
        }                                                                      \
    };
#else
#define TESTDGDOUBLE(type, filename)
#endif
#define TESTDG(type, filename)                                                 \
    TESTDGFLOAT(type, filename)                                                \
    TESTDGDOUBLE(type, filename)
// clang-format on

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")
TEST(Helmholtz2D_Tri_Quad_3C, "run/Helmholtz2D_3C.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")
TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")
TEST(Helmholtz3D_Prism_3C, "run/Helmholtz3D_Prism_3C.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
TEST(Helmholtz3D_Tet_3C, "run/Helmholtz3D_Tet_3C.xml")

TESTDG(DGSeg, "run/segment.xml")
TESTDG(DGSegSEM, "run/line_sem.xml")
TESTDG(DGSeg3D, "run/segment_3D.xml")

TESTDG(DGQuad, "run/square.xml")
TESTDG(DGQuad3D, "run/square_3D.xml")
TESTDG(DGQuadVarP, "run/square_varp.xml")
TESTDG(DGQuadSEM, "run/square_sem.xml")

TESTDG(DGTri, "run/tri.xml")
TESTDG(DGTri3D, "run/tri_3D.xml")
TESTDG(DGTriVarP, "run/tri_varp.xml")
TESTDG(DGTriNodal, "run/tri_nodal.xml")

TESTDG(DGSquareAllElements, "run/square_all_elements.xml")

TESTDG(DGHex, "run/hex.xml")
TESTDG(DGHexVarP, "run/hex_varp.xml")
TESTDG(DGHexSEM, "run/hex_sem.xml")

TESTDG(DGPrism, "run/prism.xml")
TESTDG(DGPrismVarP, "run/prism_varp.xml")
TESTDG(DGPrismNodal, "run/prism_nodal.xml")

TESTDG(DGPyr, "run/pyr.xml")
TESTDG(DGPyrVarP, "run/pyr_varp.xml")

TESTDG(DGTet, "run/tet.xml")
TESTDG(DGTetVarP, "run/tet_varp.xml")
TESTDG(DGTetNodal, "run/tet_nodal.xml")

TESTDG(DGCubePrismHex, "run/cube_prismhex.xml")

TESTDG(DGCubeAllElements, "run/cube_all_elements.xml")
