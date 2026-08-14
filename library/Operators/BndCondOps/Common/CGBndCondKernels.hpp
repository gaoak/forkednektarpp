///////////////////////////////////////////////////////////////////////////////
//
// File: CGBndCondKernels.hpp
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
// Description: Small device-portable kernels shared by the CG boundary
// condition operators (Dirichlet lifting, Neumann weak forcing) to keep
// UpdateBndCoeffs() entirely on-device, avoiding host round-trips.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>
#include <type_traits>

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

// dstPtr[i] = srcPtr[idxPtr[i]], for i in [0, nsize). Used to gather the
// compact per-component boundary coefficient array into the per-block
// layout expected by the scatter/atomic-add kernels in v_Apply().
template <typename ExecSpace, typename TData>
void CGBndCondGatherKernel(const size_t nsize, const size_t *idxPtr,
                           const TData *srcPtr, TData *dstPtr,
                           const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize,
        NEKTAR_LAMBDA(const size_t i) { dstPtr[i] = srcPtr[idxPtr[i]]; });

    Nektar::LoopExecutionSetStreamID(0);
}

// Scatter-with-absolute-maximum-combine:
//
//     dstPtr[idxPtr[i]] = absmax(dstPtr[idxPtr[i]], srcPtr[i])
//
// where absmax(a, b) keeps whichever of the two has the larger magnitude,
// preserving its sign (ties keep a). This is the device-side equivalent of
// GSLib's Gs::gs_amax combine, used to fold received cross-rank Dirichlet
// boundary values back into the local coefficient array, and to resolve
// duplicated local copies of the same universal Dirichlet dof.
//
// idxPtr may repeat the same index more than once -- a dof shared by three
// or more ranks contributes one entry per neighbouring rank -- so the
// combine has to be a genuine reduction, not a last-write-wins scatter.
#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
void CGBndCondUnpackAbsMaxKernel(const size_t nsize, const size_t *idxPtr,
                                 const TData *srcPtr, TData *dstPtr,
                                 const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            Nektar::atomic_absmax<NektarSpaces::GlobalScope>(dstPtr + idxPtr[i],
                                                             srcPtr[i]);
        });

    Nektar::LoopExecutionSetStreamID(0);
}
#endif

// Host (Serial/AVX) counterpart. The working set here is only ever the set
// of Dirichlet dofs straddling a partition boundary (plus their duplicated
// local copies), which is small, so a plain scalar loop is used rather than
// a parallel_for with atomics.
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<!std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
void CGBndCondUnpackAbsMaxKernel(const size_t nsize, const size_t *idxPtr,
                                 const TData *srcPtr, TData *dstPtr,
                                 [[maybe_unused]] const unsigned int streamID)
{
    for (size_t i = 0; i < nsize; ++i)
    {
        TData *const dst = dstPtr + idxPtr[i];
        if (std::abs(srcPtr[i]) > std::abs(*dst))
        {
            *dst = srcPtr[i];
        }
    }
}

} // namespace Nektar::Operators::detail
