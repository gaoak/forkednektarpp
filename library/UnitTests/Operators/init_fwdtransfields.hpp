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

#include "Operators/GlobalLinSysOps/OperatorFwdTrans.hpp"

#include <MultiRegions/ContField.h>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class FwdTransField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    FwdTransField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase()
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        Array<OneD, double> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);

        if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
        {
            auto func = fixt_explist->GetSession()->GetFunction("Forcing", 0);
            func->Evaluate(x, y, z, fce);
        }

        double *xptr = x.get(), *yptr = y.get(), *zptr = z.get(),
               *fceptr = fce.get();
        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_in->GetBlocks())
        {
            for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                {
                    if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
                    {
                        inptr[cnt] = *(fceptr++);
                    }
                    else
                    {
                        inptr[cnt] = 1.0;
                        if (fixt_explist->GetCoordim(0) == 1)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                inptr[cnt] += n * std::pow(*xptr, n);
                            }
                            xptr++;
                        }
                        else if (fixt_explist->GetCoordim(0) == 2)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                inptr[cnt] +=
                                    n * std::pow(*xptr, n) * std::pow(*yptr, n);
                            }
                            xptr++;
                            yptr++;
                        }
                        else
                        {
                            for (size_t n = 1; n < 4; n++)
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
            inptr += block.block_size;
        }
        ExpectedSolution();
    }

    template <typename ExecSpace, typename Impl> void RunTestCase()
    {
        auto FwdTransOp =
            FwdTrans<>::template create<ExecSpace, Impl>(fixt_explist);
        auto DiagPreconOp =
            DiagPrecon<>::template create<ExecSpace, Impl>(fixt_explist);
        FwdTransOp->setPrecon(DiagPreconOp);
        FwdTransOp->apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->toArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);
        fixt_explist->FwdTrans(inphys, outcoeffs);
        fixt_expected->copyArray<NektarSpaces::HostSpace>(outcoeffs);
    }
};

#define TEST(type, filename)                                                   \
    class type : public FwdTransField                                          \
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
