///////////////////////////////////////////////////////////////////////////////
//
// File: init_multiplybyelmtinvmassfields.hpp
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

#include "Operators/ElmtOps/OperatorMultiplyByElmtInvMass.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class MultiplyByElmtInvMassField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff>
{
public:
    MultiplyByElmtInvMassField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (auto const &block : fixt_in->GetBlocks())
        {
            for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    inptr[cnt] = coeff;
                }
            }
            inptr += block.block_size;
        }
        ExpectedSolution();
    }

    template <typename ExecSpace, typename Impl> void RunTestCase()
    {
        MultiplyByElmtInvMass<>::template create<ExecSpace, Impl>(fixt_explist)
            ->apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> incoeffs = fixt_in->toArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs());
        fixt_explist->MultiplyByElmtInvMass(incoeffs, outcoeffs);
        fixt_expected->copyArray<NektarSpaces::HostSpace>(outcoeffs);
    }
};

#define TEST(type, filename)                                                   \
    class type : public MultiplyByElmtInvMassField                             \
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
