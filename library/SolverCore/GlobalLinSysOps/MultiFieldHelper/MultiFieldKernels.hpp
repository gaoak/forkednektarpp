///////////////////////////////////////////////////////////////////////////////
//
// File: MultiFieldKernels.hpp
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
// Description: Kernels of the MultiField operations.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

#include <cstdint>
#include <type_traits>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Copy @p in into @p out with its padding zeroed.
 *
 * @p mask holds the padding mask of one component of @p compSize values; the
 * block holds @p ncomp components one after the other.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyWithoutPadding(const size_t compSize,
                                                const unsigned int ncomp,
                                                const std::uint8_t *mask,
                                                const TData *in, TData *out,
                                                const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        // One launch over all components, rather than one per component.
        Nektar::parallel_for<ExecSpace>(
            0, ncomp * compSize, NEKTAR_LAMBDA(const size_t i) {
                out[i] = mask[i % compSize] * in[i];
            });
    }
    else
    {
        // One loop per component, so that the mask is read contiguously.
        for (unsigned int n = 0; n < ncomp; ++n)
        {
            const size_t offset = n * compSize;
            Nektar::parallel_for<ExecSpace>(
                0, compSize, NEKTAR_LAMBDA(const size_t i) {
                    out[offset + i] = mask[i] * in[offset + i];
                });
        }
    }

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Copy the @p size entries of @p in into @p out, those @p mask
 * excludes zeroed.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyMasked(const size_t size,
                                        const std::uint8_t *mask,
                                        const TData *in, TData *out,
                                        const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, size, NEKTAR_LAMBDA(const size_t i) { out[i] = mask[i] * in[i]; });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Sum the per-block dot products into @p out, on stream 0.
 *
 * @p blockDots holds the @p ncols dot products of each of @p nblocks blocks,
 * @p stride apart.
 *
 * The launch is tiny (@p ncols threads, a few loads each), so it leaves the
 * device mostly idle, but it costs about one kernel launch, small next to the
 * matrix products before it and the reduction across ranks after it. It keeps
 * @p out in device memory for that reduction and for MultiAxpy, without a
 * device-to-host copy, and lets each block run its product on its own stream.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void SumBlockDots(const unsigned int nblocks,
                                          const unsigned int stride,
                                          const unsigned int ncols,
                                          const TData *blockDots, TData *out)
{
    Nektar::LoopExecutionSetStreamID(0);

    Nektar::parallel_for<ExecSpace>(
        0, ncols, NEKTAR_LAMBDA(const size_t j) {
            TData sum = 0.0;
            for (unsigned int b = 0; b < nblocks; ++b)
            {
                sum += blockDots[b * stride + j];
            }
            out[j] = sum;
        });
}

} // namespace Nektar::SolverCore::detail
