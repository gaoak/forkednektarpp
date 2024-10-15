///////////////////////////////////////////////////////////////////////////////
//
// File: init_assmbscatrfields.hpp
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

#include "Operators/OperatorAssmbScatr.hpp"

#include <LibUtilities/LinearAlgebra/NekLinSysIter.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/GlobalLinSysIterativeFull.h>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

class AssmbScatrField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>
{
public:
    AssmbScatrField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *outptr,
                     bool padding = true)
    {
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    outptr[cnt] = coeff;
                }
            }
            outptr += (padding) ? block.block_size : cnt;
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *outptr)
    {
        Array<OneD, NekDouble> incoeff(fixt_explist->GetNcoeffs());
        Array<OneD, NekDouble> outcoeff(fixt_explist->GetNcoeffs());

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), incoeff.get(), false);

        // Calculate expected result from Nektar++
        auto map = fixt_explist->GetLocalToGlobalMap();
        map->Assemble(incoeff, outcoeff);
        // Vmath::Zero(map->GetNumGlobalDirBndCoeffs(), outcoeff, 1);
        map->GlobalToLocal(outcoeff, outcoeff);

        // Copy expected result from Array to pointer
        double *ptr = outcoeff.get();
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
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
    class type : public AssmbScatrField                                        \
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
