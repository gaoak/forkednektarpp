///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondKernels.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

using namespace Nektar;

namespace Nektar::MultiRegions::detail
{

// Every target in mapPtr is unique -- BuildCGBndCondCoeffMaps() has already
// pulled out any local coefficient written by more than one boundary trace
// piece into the group kernel below -- so a plain, non-atomic add is safe:
// no two loop iterations ever touch the same outptr address.
template <typename ExecSpace, typename TData>
void NeuBndCondKernel(const size_t bndExpSize, const size_t *mapPtr,
                      const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, bndExpSize,
        NEKTAR_LAMBDA(const size_t i) { outptr[mapPtr[i]] += inptr[i]; });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void NeuBndCondKernel(const size_t bndExpSize, const TData *signPtr,
                      const size_t *mapPtr, const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, bndExpSize, NEKTAR_LAMBDA(const size_t i) {
            outptr[mapPtr[i]] += signPtr[i] * inptr[i];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

// Companion to NeuBndCondKernel() above for the (rare) local coefficient
// targets written by more than one boundary trace piece of the same element
// -- e.g. a domain corner where two Neumann edges meet. groupOffsetPtr is a
// CSR row pointer: group g's contributions are inptr[groupOffsetPtr[g] ..
// groupOffsetPtr[g + 1]). Each thread sums its (small, fixed-topology) group
// locally and performs a single non-atomic add, so this never contends with
// NeuBndCondKernel() or with any other group -- every group targets a
// distinct address, and groups are disjoint from the unique-target set above
// by construction.
template <typename ExecSpace, typename TData>
void NeuBndCondGroupKernel(const size_t nGroups, const size_t *groupOffsetPtr,
                           const size_t *groupTargetPtr, const TData *inptr,
                           TData *outptr, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nGroups, NEKTAR_LAMBDA(const size_t g) {
            TData acc = static_cast<TData>(0);
            for (size_t j = groupOffsetPtr[g]; j < groupOffsetPtr[g + 1]; ++j)
            {
                acc += inptr[j];
            }
            outptr[groupTargetPtr[g]] += acc;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void NeuBndCondGroupKernel(const size_t nGroups, const size_t *groupOffsetPtr,
                           const size_t *groupTargetPtr, const TData *signPtr,
                           const TData *inptr, TData *outptr,
                           const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nGroups, NEKTAR_LAMBDA(const size_t g) {
            TData acc = static_cast<TData>(0);
            for (size_t j = groupOffsetPtr[g]; j < groupOffsetPtr[g + 1]; ++j)
            {
                acc += signPtr[j] * inptr[j];
            }
            outptr[groupTargetPtr[g]] += acc;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::MultiRegions::detail
