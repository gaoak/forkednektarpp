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

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *outptr,
                     bool padding = true)
    {
        for (auto const &block : blocks)
        {
            for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
            {
                for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                    {
                        outptr[cnt] = phys + k;
                    }
                }
                outptr += (padding) ? block.block_size
                                    : block.num_elements * block.num_pts;
            }
        }
    }

    void SetTestCaseNektar(const std::vector<BlockAttributes> &blocks,
                           double *outptr, bool padding = true)
    {
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
            for (auto const &block : blocks)
            {
                for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                    {
                        outptr[cnt] = phys + k;
                    }
                }
                outptr += (padding) ? block.block_size
                                    : block.num_elements * block.num_pts;
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *outptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetCoordim(0) *
                                      fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCaseNektar(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        Array<OneD, Array<OneD, NekDouble>> inphysarray(
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

        // Copy expected result from Array to pointer
        double *ptr = outcoeffs.get();
        for (auto const &block : blocks)
        {
            for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    outptr[cnt] = (*ptr++);
                }
            }
            outptr += block.block_size;
        }
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
