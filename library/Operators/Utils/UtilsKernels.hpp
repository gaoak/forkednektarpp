///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsKernels.hpp
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

#pragma once

#include "Operators/Utils/UtilsDeviceKernels.hpp"
#include "Operators/Utils/UtilsSerialAVXKernels.hpp"

namespace Nektar::Operators
{

/**
 * @brief Reshapes the storage to a prescribed vector width.
 *
 * This routine reorders the elemental data to interleave elements to a
 * prescribed vector width VW. This therefore puts the same DOF for groups
 * of VW elements contiguously in memory, enabling the efficient use of
 * vectorised instructions. At the moment this routine reshapes from VW0 to
 * VW1 by first reshaping from VW0 to a vector width of 1, and then to VW1.
 *
 * @tparam  targInterleaveWidth     Target vector width.
 * @tparam  currInterleaveWidth     Current vector width.
 */
template <typename ExecSpace, typename TData>
void ReshapeStorage(const unsigned int targInterleaveWidth,
                    const unsigned int currInterleaveWidth, const size_t nelmt,
                    const unsigned int ndata, TData *inoutptr)
{
    if (currInterleaveWidth != targInterleaveWidth)
    {
        // Reshape to scalar shape, if necessary
        if (currInterleaveWidth != 1)
        {
            deInterleave<ExecSpace>(currInterleaveWidth,
                                    nelmt / currInterleaveWidth, ndata,
                                    inoutptr);
        }

        // Reshape to required shape, if necessary
        if (targInterleaveWidth != 1)
        {
            ASSERTL0(
                nelmt % targInterleaveWidth == 0,
                "Number of elements is not divisible by interleave width.");

            interleave<ExecSpace>(targInterleaveWidth,
                                  nelmt / targInterleaveWidth, ndata, inoutptr);
        }
    }
}

} // namespace Nektar::Operators
