///////////////////////////////////////////////////////////////////////////////
//
// File: HOPressureBndCondKernels.hpp
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
// Description: Kernels of the higher-order pressure boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Copy the boundary-adjacent elements of @p nComp components of a
 * volume field into the field on those elements:
 * dst[c * dstCompSize + dstIdx[i]] = src[c * srcCompSize + srcIdx[i]]. Both
 * index maps are built on the host, so no index is divided here.
 *
 * @details
 * src is a block of the volume field and dst a block of the field on the
 * boundary-adjacent elements, which has fewer elements in general, so the
 * two component sizes differ. The threads are numbered by destination point,
 * the element lane varying fastest, so that the writes are coalesced.
 */
template <typename ExecSpace, typename TData>
void HOPressureVolumeGatherKernel(const size_t nsize, const unsigned int nComp,
                                  const size_t srcCompSize,
                                  const size_t dstCompSize,
                                  const size_t *srcIdx, const size_t *dstIdx,
                                  const TData *src, TData *dst,
                                  const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            const size_t srcPoint = srcIdx[i];
            const size_t dstPoint = dstIdx[i];
            for (unsigned int c = 0; c < nComp; ++c)
            {
                dst[c * dstCompSize + dstPoint] =
                    src[c * srcCompSize + srcPoint];
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief flux = scale * (flux - kinvis * curlcurl), in place, for the
 * NDIM components of a vector field on the boundary-adjacent elements.
 *
 * @details
 * Extracting onto the boundary and projecting onto the normal are linear,
 * so combining the two terms here, before extraction, needs one extraction
 * per boundary region instead of two.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOPressureVolumeCombineKernel(const size_t nsize, const size_t compSize,
                                   const TData kinvis, const TData scale,
                                   const TData *curlcurl, TData *flux,
                                   const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * compSize + i;
                flux[idx]        = scale * (flux[idx] - kinvis * curlcurl[idx]);
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief out = flux . normals at the points of a boundary region.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOPressureBoundaryCombineKernel(const size_t nsize, const size_t compSize,
                                     const TData *flux, const TData *normals,
                                     TData *out, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            TData normalFlux = TData(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * compSize + i;
                normalFlux += normals[idx] * flux[idx];
            }
            out[i] = normalFlux;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief normalFlux -= scale * velocity . normals at the points of a
 * boundary region.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOPressureSubtractNormalVelocityKernel(
    const size_t nsize, const size_t compSize, const TData scale,
    const TData *velocity, const TData *normals, TData *normalFlux,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            TData value = TData(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * compSize + i;
                value += velocity[idx] * normals[idx];
            }
            normalFlux[i] -= scale * value;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
