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
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Copy the boundary-adjacent elements of the NDIM components of a
 * volume field into the field on those elements:
 * dst[c * dstCompSize + dstIdx[i]] = src[c * srcCompSize + srcIdx[i]], or
 * subtract them from it with SUBTRACT. Both index maps are built on the
 * host, so no index is divided here.
 *
 * @details
 * src is a block of the volume field and dst a block of the field on the
 * boundary-adjacent elements, which has fewer elements in general, so the
 * two component sizes differ. The threads are numbered by destination point,
 * the element lane varying fastest, so that the writes are coalesced.
 */
template <typename ExecSpace, unsigned int NDIM, bool SUBTRACT, typename TData>
void HOPressureVolumeGatherKernel(const size_t nsize, const size_t srcCompSize,
                                  const size_t dstCompSize,
                                  const size_t *srcIdx, const size_t *dstIdx,
                                  const TData *src, TData *dst,
                                  const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            const auto srcPtr = src + srcIdx[i];
            auto dstPtr       = dst + dstIdx[i];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t offset = d * dstCompSize;
                const TData value   = srcPtr[d * srcCompSize];
                if constexpr (SUBTRACT)
                {
                    dstPtr[offset] -= value;
                }
                else
                {
                    dstPtr[offset] = value;
                }
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief curlcurl = scale * (forcing - kinvis * curlcurl), in place, for the
 * NDIM components of a vector field on the boundary-adjacent elements, or
 * curlcurl = -scale * kinvis * curlcurl without FORCING.
 *
 * @details
 * Extracting onto the boundary and projecting onto the normal are linear,
 * so combining the two terms here, before extraction, needs one extraction
 * per boundary region instead of two. Without FORCING, the flux of the weak
 * pressure form, forcing is not read.
 */
template <typename ExecSpace, unsigned int NDIM, bool FORCING, typename TData>
void HOPressureVolumeCombineKernel(const size_t nsize, const size_t compSize,
                                   const TData kinvis, const TData scale,
                                   const TData *forcing, TData *curlcurl,
                                   const unsigned int streamID)
{
    // Explicit vectorisation for AVX backend, vec_t =
    // tinysimd::simd<TData> for AVX, vec_t = TData otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    ASSERTL1(nsize % vec_width == 0,
             "The block size is not a multiple of the vector width.");
    const size_t nVecs       = nsize / vec_width;
    const size_t vecCompSize = compSize / vec_width;
    const auto forcingVec    = reinterpret_cast<const vec_t *>(forcing);
    auto curlcurlVec         = reinterpret_cast<vec_t *>(curlcurl);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nVecs, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * vecCompSize + i;
                const vec_t forcingValue =
                    FORCING ? forcingVec[idx] : vec_t(0.0);
                curlcurlVec[idx] =
                    vec_t(scale) *
                    (forcingValue - vec_t(kinvis) * curlcurlVec[idx]);
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief out = flux . normals at the points of a boundary region.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOPressureBoundaryCombineKernel(const size_t bndSize,
                                     const size_t compSize, const TData *flux,
                                     const TData *normals, TData *out,
                                     const unsigned int streamID)
{
    // Explicit vectorisation for AVX backend, vec_t =
    // tinysimd::simd<TData> for AVX, vec_t = TData otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    ASSERTL1(bndSize % vec_width == 0,
             "The block size is not a multiple of the vector width.");
    const size_t nVecs       = bndSize / vec_width;
    const size_t vecCompSize = compSize / vec_width;
    const auto fluxVec       = reinterpret_cast<const vec_t *>(flux);
    const auto normalsVec    = reinterpret_cast<const vec_t *>(normals);
    auto outVec              = reinterpret_cast<vec_t *>(out);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nVecs, NEKTAR_LAMBDA(const size_t i) {
            vec_t normalFlux = vec_t(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * vecCompSize + i;
                normalFlux += normalsVec[idx] * fluxVec[idx];
            }
            outVec[i] = normalFlux;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief normalFlux -= scale * velocity . normals at the points of a
 * boundary region.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOPressureSubtractNormalVelocityKernel(
    const size_t bndSize, const size_t compSize, const TData scale,
    const TData *velocity, const TData *normals, TData *normalFlux,
    const unsigned int streamID)
{
    // Explicit vectorisation for AVX backend, vec_t =
    // tinysimd::simd<TData> for AVX, vec_t = TData otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    ASSERTL1(bndSize % vec_width == 0,
             "The block size is not a multiple of the vector width.");
    const size_t nVecs       = bndSize / vec_width;
    const size_t vecCompSize = compSize / vec_width;
    const auto velocityVec   = reinterpret_cast<const vec_t *>(velocity);
    const auto normalsVec    = reinterpret_cast<const vec_t *>(normals);
    auto normalFluxVec       = reinterpret_cast<vec_t *>(normalFlux);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nVecs, NEKTAR_LAMBDA(const size_t i) {
            vec_t value = vec_t(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * vecCompSize + i;
                value += velocityVec[idx] * normalsVec[idx];
            }
            normalFluxVec[i] -= vec_t(scale) * value;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
