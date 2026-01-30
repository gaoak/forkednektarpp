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
        Array<OneD, TData> x(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> y(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> z(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> fce(this->fixt_explist->GetTotPoints());
        this->fixt_explist->GetCoords(x, y, z);

        if (this->session->DefinesFunction("Forcing"))
        {
            auto func = this->session->GetFunction("Forcing", 0);
            func->Evaluate(x, y, z, fce);
        }

        auto xptr = x.data(), yptr = y.data(), zptr = z.data(),
             fceptr = fce.data();
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
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
                        inptr[cnt] = 1.0;
                        if (this->fixt_explist->GetCoordim(0) == 1)
                        {
                            for (unsigned int n = 1; n < 4; n++)
                            {
                                inptr[cnt] += n * std::pow(*xptr, n);
                            }
                            xptr++;
                        }
                        else if (this->fixt_explist->GetCoordim(0) == 2)
                        {
                            for (unsigned int n = 1; n < 4; n++)
                            {
                                inptr[cnt] +=
                                    n * std::pow(*xptr, n) * std::pow(*yptr, n);
                            }
                            xptr++;
                            yptr++;
                        }
                        else
                        {
                            for (unsigned int n = 1; n < 4; n++)
                            {
                                inptr[cnt] += n * std::pow(*xptr, n) *
                                              std::pow(*yptr, n) *
                                              std::pow(*zptr, n);
                            }
                            xptr++;
                            yptr++;
                            zptr++;
                        }
                    }
                }
            }
        }

        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Set up diffusion coefficient.
        if (this->fixt_explist->GetCoordim(0) == 1)
        {
            m_diffCoeff.resize(1);
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (this->fixt_explist->GetCoordim(0) == 2)
        {
            m_diffCoeff.resize(4);
            m_diffCoeff[0] = 2.0; // D00
            m_diffCoeff[2] = 3.0; // D11
        }
        else
        {
            m_diffCoeff.resize(6);
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
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(this->fixt_explist->GetNcoeffs(), 0.0);

        // Set up diffusion coefficient.
        StdRegions::FactorMap factors;
        if (this->fixt_explist->GetCoordim(0) == 1)
        {
            factors[StdRegions::eFactorLambda] = 0.0;
        }
        else if (this->fixt_explist->GetCoordim(0) == 2)
        {
            factors[StdRegions::eFactorLambda]   = 0.0;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else
        {
            factors[StdRegions::eFactorLambda]   = 0.0;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        this->fixt_explist->HelmSolve(inphys, outcoeffs, factors);
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

protected:
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

TEST(Poisson1D_Seg, "run/Poisson1D_P8.xml")

TEST(Poisson2D_Tri_Quad, "run/Poisson2D_varP.xml")

TEST(Poisson2D_AllBCs, "run/Poisson2D_P7_AllBCs.xml")

TEST(Poisson3D_Hex, "run/Poisson3D_Hex_Heterogeneous.xml")

TEST(Poisson3D_Prism, "run/Poisson3D_Prism_VarP.xml")

TEST(Poisson3D_Pyr, "run/Poisson3D_Pyr_VarP.xml")

TEST(Poisson3D_Tet, "run/Poisson3D_Tet_VarP.xml")
