///////////////////////////////////////////////////////////////////////////////
//
// File: init_fwdtransfields.hpp
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

#include "init_fields.hpp"

#include "Operators/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp"
#include "Operators/PreconOps/PreconOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData, typename TExpList>
class FwdTransFieldBase
    : public InitFields<TData, FieldState::Phys, FieldState::Coeff, TExpList>
{
public:
    FwdTransFieldBase()
        : InitFields<TData, FieldState::Phys, FieldState::Coeff, TExpList>()
    {
    }

    void SetTestCase()
    {
        // Set initial conditions.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t nphys = this->fixt_explist->GetTotPoints();
        m_coordDim         = this->fixt_explist->GetCoordim(0);

        Array<OneD, TData> x(nphys);
        Array<OneD, TData> y(nphys);
        Array<OneD, TData> z(nphys);
        Array<OneD, TData> fce(numComp * nphys, 0.0);
        this->fixt_explist->GetCoords(x, y, z);

        const bool useForcing =
            !std::is_same_v<TExpList, MultiRegions::DisContField> &&
            this->session->DefinesFunction("Forcing");

        if (useForcing)
        {
            for (unsigned int n = 0; n < numComp; ++n)
            {
                auto func = this->session->GetFunction(
                    "Forcing", n % this->session->GetVariables().size());
                Array<OneD, TData> fceVar = fce + n * nphys;
                func->Evaluate(x, y, z, fceVar);
            }
        }

        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
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

        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase(const std::string &method)
    {
        std::string execStr(
            boost::unit_test::framework::master_test_suite().argv[1]);
        std::string implStr(
            boost::unit_test::framework::master_test_suite().argv[2]);

        // reshape this->fixt_in
        if (implStr == "SumFac" || execStr == "AVX")
        {
            this->fixt_in->ReshapeStorage(
                NektarSpaces::GetVectorWidth<TData>(execStr), execStr);
            this->fixt_out->SetInterleaveWidth(
                NektarSpaces::GetVectorWidth<TData>(execStr));
        }

        auto op = FwdTransOp<TData>::Create(this->fixt_explist,
                                            this->session->GetVariables());

        if constexpr (!std::is_same_v<TExpList, MultiRegions::DisContField>)
        {
            auto precon = PreconOp<TData>::Create(
                this->fixt_explist, this->session->GetVariables(), "Diagonal");
            auto linsolve = LinearSolverOp<TData>::Create(
                this->fixt_explist, this->session->GetVariables(), method);
            op->SetLinearSolver(linsolve);
            op->SetPrecon(precon);
            op->UpdatePrecon();
        }
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t nphys    = this->fixt_explist->GetTotPoints();
        const size_t ncoeffs  = this->fixt_explist->GetNcoeffs();
        const auto &variables = this->session->GetVariables();

        // Setup input/output arrays
        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(numComp * ncoeffs, 0.0);

        // Solve each component separately
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        for (unsigned int n = 0; n < numComp; ++n)
        {
            auto expListVar = MemoryManager<TExpList>::AllocateSharedPtr(
                this->session, graph, variables[n % variables.size()], true,
                std::is_same_v<TExpList, MultiRegions::DisContField>,
                Collections::eNoCollection);
            Array<OneD, TData> inphysVar    = inphys + n * nphys;
            Array<OneD, TData> outcoeffsVar = outcoeffs + n * ncoeffs;
            expListVar->FwdTrans(inphysVar, outcoeffsVar);
        }

        // Copy solution back
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

protected:
    unsigned int m_coordDim;
};

template <typename TData>
class FwdTransField : public FwdTransFieldBase<TData, MultiRegions::ContField>
{
};

template <typename TData>
class FwdTransDGField
    : public FwdTransFieldBase<TData, MultiRegions::DisContField>
{
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public FwdTransField<float>                            \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public FwdTransField<double>                                  \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
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
    class type##float : public FwdTransDGField<float>                          \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTDGFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDGDOUBLE(type, filename)                                           \
    class type : public FwdTransDGField<double>                                \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
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
