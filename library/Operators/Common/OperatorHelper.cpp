///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorHelper.hpp
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

#include "Operators/Common/OperatorHelper.hpp"

namespace Nektar::Operators
{

size_t GetGeometricFactorSize(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::vector<BlockAttributes> &blocks)
{
    size_t gfSize = 0;
    size_t exp_id = 0;

    for (size_t blk = 0; blk < blocks.size(); ++blk)
    {
        const auto expPtr = expansionList->GetExp(exp_id);

        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            gfSize +=
                expPtr->GetTotPoints() *
                (blocks[blk].num_elements + blocks[blk].num_padding_elements);
        }
        else
        {
            gfSize +=
                blocks[blk].num_elements + blocks[blk].num_padding_elements;
        }

        exp_id += blocks[blk].num_elements;
    }

    return gfSize;
}

} // namespace Nektar::Operators
