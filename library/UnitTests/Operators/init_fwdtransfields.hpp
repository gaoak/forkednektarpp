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
#include <MultiRegions/ContField.h>

using namespace std;
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

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        Array<OneD, NekDouble> x(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> y(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> z(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);

        if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
        {
            auto func = fixt_explist->GetSession()->GetFunction("Forcing", 0);
            func->Evaluate(x, y, z, fce);
        }

        double *xptr = x.get(), *yptr = y.get(), *zptr = z.get(),
               *fceptr = fce.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    if (fixt_explist->GetSession()->DefinesFunction("Forcing"))
                    {
                        *(inptr++) = *(fceptr++);
                    }
                    else
                    {
                        *inptr = 1.0;
                        if (fixt_explist->GetCoordim(0) == 1)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr += n * std::pow(*xptr, n);
                            }
                            inptr++;
                            xptr++;
                        }
                        else if (fixt_explist->GetCoordim(0) == 2)
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr +=
                                    n * std::pow(*xptr, n) * std::pow(*yptr, n);
                            }
                            inptr++;
                            xptr++;
                            yptr++;
                        }
                        else
                        {
                            for (size_t n = 1; n < 4; n++)
                            {
                                *inptr += n * std::pow(*xptr, n) *
                                          std::pow(*yptr, n) *
                                          std::pow(*zptr, n);
                            }
                            inptr++;
                            xptr++;
                            yptr++;
                            zptr++;
                        }
                    }
                }
            }
            if (padding)
            {
                for (size_t el = 0; el < block.num_padding_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        inptr++;
                    }
                }
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->FwdTrans(inphys, outcoeffs);

        // Copy expected result from Array to pointer
        double *coeffptr = outcoeffs.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    (*inptr++) = (*coeffptr++);
                }
            }
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    inptr++;
                }
            }
        }
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

// TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_P7_AllBCs.xml")
TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
