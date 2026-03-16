///////////////////////////////////////////////////////////////////////////////
//
// File: init_poissonsolve_fields.hpp
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
#include "Operators/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp"
#include "Operators/PreconOps/PreconOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData>
class PoissonSolveField
    : public InitFields<TData, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    PoissonSolveField()
        : InitFields<TData, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::ContField>()
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

        if (this->session->DefinesFunction("Forcing"))
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
                        if (this->session->DefinesFunction("Forcing"))
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

        // Set up diffusion coefficient.
        const auto diffCoeffSize = m_coordDim * (m_coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        // Set up (isotropic) diffusion coefficient.
        if (m_coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (m_coordDim == 2)
        {
            m_diffCoeff[0] = 2.0; // D00
            m_diffCoeff[2] = 3.0; // D11
        }
        else
        {
            m_diffCoeff[0] = 2.0; // D00
            m_diffCoeff[2] = 3.0; // D11
            m_diffCoeff[5] = 4.0; // D22
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase(const std::string &method)
    {
        auto op     = PoissonSolveOp<TData>::Create(this->fixt_explist,
                                                    this->session->GetVariables());
        auto precon = PreconOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), "Diagonal");
        auto linsolve = LinearSolverOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), method);
        op->SetDiffCoeff(m_diffCoeff);
        op->SetLinearSolver(linsolve);
        op->SetPrecon(precon);
        op->UpdatePrecon();
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

        // Set lambda as constant coefficient
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = 0.0;

        // Set up diffusion coefficient.
        if (m_coordDim == 2)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else if (m_coordDim == 3)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        // Solve each component separately
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        for (unsigned int n = 0; n < numComp; ++n)
        {
            auto expListVar =
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    this->session, graph, variables[n % variables.size()], true,
                    false, Collections::eNoCollection);
            Array<OneD, TData> inphysVar    = inphys + n * nphys;
            Array<OneD, TData> outcoeffsVar = outcoeffs + n * ncoeffs;
            expListVar->HelmSolve(inphysVar, outcoeffsVar, factors);
        }

        // Copy solution back
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

protected:
    unsigned int m_coordDim;
    std::vector<TData> m_diffCoeff;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public PoissonSolveField<float>                        \
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
    class type : public PoissonSolveField<double>                              \
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
// clang-format on

TEST(Poisson1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Poisson2D_Tri_Quad, "run/Helmholtz2D_varP.xml")
TEST(Poisson2D_Tri_Quad_3C, "run/Helmholtz2D_3C.xml")

TEST(Poisson2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Poisson3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")
TEST(Poisson3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Poisson3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Poisson3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Poisson3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
