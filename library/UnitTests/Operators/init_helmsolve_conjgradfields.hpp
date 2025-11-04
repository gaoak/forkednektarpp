///////////////////////////////////////////////////////////////////////////////
//
// File: init_helmsolve_conjgradfields.hpp
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

#include "Operators/GlobalLinSysOps/ConjGrad/ConjGradOp.hpp"
#include "Operators/GlobalLinSysOps/HelmSolve/HelmSolveOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class HelmSolveField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    HelmSolveField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase()
    {
        // Set initial conditions.
        Array<OneD, double> x(fixt_explist[0]->GetTotPoints());
        Array<OneD, double> y(fixt_explist[0]->GetTotPoints());
        Array<OneD, double> z(fixt_explist[0]->GetTotPoints());
        Array<OneD, double> fce(fixt_explist[0]->GetTotPoints());
        fixt_explist[0]->GetCoords(x, y, z);

        if (session->DefinesFunction("Forcing"))
        {
            auto func = session->GetFunction("Forcing", 0);
            func->Evaluate(x, y, z, fce);
        }

        auto xptr = x.data(), yptr = y.data(), zptr = z.data(),
             fceptr = fce.data();
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
            {
                for (unsigned int phys = 0; phys < block.GetNumData();
                     ++phys, ++cnt)
                {
                    if (session->DefinesFunction("Forcing"))
                    {
                        inptr[cnt] = *(fceptr++);
                    }
                    else
                    {
                        inptr[cnt] = 1.0;
                        if (fixt_explist[0]->GetCoordim(0) == 1)
                        {
                            for (unsigned int n = 1; n < 4; n++)
                            {
                                inptr[cnt] += n * std::pow(*xptr, n);
                            }
                            xptr++;
                        }
                        else if (fixt_explist[0]->GetCoordim(0) == 2)
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

        fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Get lambda from session or default to 10.0
        m_lambda = session->DefinesParameter("Lambda")
                       ? session->GetParameter("Lambda")
                       : 10.0;

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op       = HelmSolveOp<double>::Create(fixt_explist[0]);
        auto precon   = DiagPreconOp<double>::Create(fixt_explist[0]);
        auto linsolve = ConjGradOp<double>::Create(fixt_explist[0]);
        op->SetLambda(m_lambda);
        op->SetLinearSolver(linsolve);
        op->SetPrecon(precon);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(fixt_explist[0]->GetNcoeffs(), 0.0);
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;
        fixt_explist[0]->HelmSolve(inphys, outcoeffs, factors);
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

protected:
    double m_lambda;
};

#define TEST(type, filename)                                                   \
    class type : public HelmSolveField                                         \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
