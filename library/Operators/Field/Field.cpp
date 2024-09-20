///////////////////////////////////////////////////////////////////////////////
//
// File: Field.cpp
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

#include "Operators/Field/Field.hpp"

#include <MultiRegions/ExpList.h>

using namespace Nektar;
using namespace LibUtilities;

std::string FieldStateString(FieldState state)
{
    if (state == FieldState::Phys)
    {
        return std::string("Phys");
    }
    else if (state == FieldState::Coeff)
    {
        return std::string("Coeff");
    }
    else
    {
        return std::string("");
    }
}

std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const MultiRegions::ExpListSharedPtr explist,
    size_t VectorWidth)
{
    std::vector<BlockAttributes> blockAttr;

    // initialize the basisKeys
    std::vector<BasisKey> prevbasisKeys(3, NullBasisKey);
    std::vector<BasisKey> thisbasisKeys(3, NullBasisKey);
    int prevIsDeformed = -1, thisIsDeformed = -1;

    // initialize the first block using the first element
    auto expPtr = explist->GetExp(0);
    for (int d = 0; d < expPtr->GetNumBases(); d++)
    {
        prevbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
    }
    prevIsDeformed = expPtr->GetMetricInfo()->GetGtype();
    size_t num_pts = state == FieldState::Phys ? expPtr->GetTotPoints()
                                               : expPtr->GetNcoeffs();
    blockAttr.push_back({1, num_pts});

    // loop over elements
    for (int i = 1; i < explist->GetNumElmts(); i++)
    {
        expPtr = explist->GetExp(i);

        // fetch basiskeys of current element
        for (int d = 0; d < expPtr->GetNumBases(); d++)
        {
            thisbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        thisIsDeformed = expPtr->GetMetricInfo()->GetGtype();

        // if the basis is the same as the previous one,
        // increment the number of elements
        if (thisbasisKeys == prevbasisKeys && thisIsDeformed == prevIsDeformed)
        {
            blockAttr.back().num_elements++;
        }
        else // if not, create a new block with the number of elements = 1
        {
            blockAttr.back().width = VectorWidth;
            blockAttr.back().num_elmt_groups =
                (blockAttr.back().num_elements + blockAttr.back().width - 1) / blockAttr.back().width;
            blockAttr.back().block_size = blockAttr.back().num_elmt_groups *
                                          blockAttr.back().width *
                                          blockAttr.back().num_pts;

            // update num_pts for a new block
            num_pts = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                : expPtr->GetNcoeffs();
            blockAttr.push_back({1, num_pts});
            prevbasisKeys  = thisbasisKeys;
            prevIsDeformed = thisIsDeformed;
        }
    }

    // update the padding elements for the last block
    // compute the padding elements
    blockAttr.back().width = VectorWidth;
    blockAttr.back().num_elmt_groups =
        (blockAttr.back().num_elements + blockAttr.back().width - 1) / blockAttr.back().width;
    blockAttr.back().block_size = blockAttr.back().num_elmt_groups *
                                  blockAttr.back().width * blockAttr.back().num_pts;

    return blockAttr;
}
