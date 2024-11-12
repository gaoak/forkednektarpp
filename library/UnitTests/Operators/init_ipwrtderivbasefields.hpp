///////////////////////////////////////////////////////////////////////////////
//
// File: init_ipwrtderivbasefields.hpp
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

#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class IProductWRTDerivBaseField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    IProductWRTDerivBaseField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            double *inptr =
                fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>(
                    blk);
            auto &block = fixt_in->GetBlocks()[blk];
            for (unsigned int k = 0; k < fixt_explist->GetCoordim(0); k++)
            {
                for (unsigned int el = 0, cnt = 0; el < block.GetNumElements();
                     ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys + k;
                    }
                }
                inptr += block.size();
            }
        }
        ExpectedSolution();
    }

    template <typename ExecSpace, typename Impl> void RunTestCase()
    {
        IProductWRTDerivBase<>::template Create<ExecSpace, Impl>(fixt_explist)
            ->apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);
        Array<OneD, Array<OneD, double>> inphysarray(
            fixt_explist->GetCoordim(0));
        if (fixt_explist->GetCoordim(0) > 0)
        {
            inphysarray[0] = inphys;
        }
        if (fixt_explist->GetCoordim(0) > 1)
        {
            inphysarray[1] = inphysarray[0] + fixt_explist->GetTotPoints();
        }
        if (fixt_explist->GetCoordim(0) > 2)
        {
            inphysarray[2] = inphysarray[1] + fixt_explist->GetTotPoints();
        }
        fixt_explist->IProductWRTDerivBase(inphysarray, outcoeffs);
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }
};

#define TEST(type, filename)                                                   \
    class type : public IProductWRTDerivBaseField                              \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
