///////////////////////////////////////////////////////////////////////////////
//
// File: HOutflowBndCondKernels.hpp
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
// Description: Kernels of Dong's high-order outflow boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Copy the boundary-adjacent elements of the NDIM components of a
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
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOutflowVolumeGatherKernel(const size_t nsize, const size_t srcCompSize,
                                const size_t dstCompSize, const size_t *srcIdx,
                                const size_t *dstIdx, const TData *src,
                                TData *dst, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            const size_t srcPoint = srcIdx[i];
            const size_t dstPoint = dstIdx[i];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                dst[d * dstCompSize + dstPoint] =
                    src[d * srcCompSize + srcPoint];
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief tanh of every lane of @p x, which tinysimd does not provide.
 */
template <typename vec_t> NEK_HOSTDEVICE_INLINE vec_t vecTanh(const vec_t x)
{
    if constexpr (std::is_floating_point_v<vec_t>)
    {
#if defined(__SYCL_DEVICE_ONLY__)
        return sycl::tanh(x);
#else
        return std::tanh(x);
#endif
    }
    else
    {
        alignas(vec_t::alignment) typename vec_t::scalarArray tmp;
        x.store(tmp, tinysimd::is_aligned);
        for (unsigned int l = 0; l < vec_t::width; ++l)
        {
            tmp[l] = std::tanh(tmp[l]);
        }
        vec_t y;
        y.load(tmp, tinysimd::is_aligned);
        return y;
    }
}

/**
 * @brief Evaluate Dong's pressure condition at the points of a boundary
 * region, keeping E(n, u*) - f_b and div(u*) for the velocity condition.
 *
 * @details
 * @p velocity holds the NDIM velocity components and @p gradient the
 * NDIM * NDIM components of its gradient, du_d/dx_j at d * NDIM + j, on the
 * points of the region, at the component size @p compSize of the normals.
 * The velocity and its gradient are multiplied by @p velocityScale.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOutflowPressureKernel(const size_t bndSize, const size_t compSize,
                            const TData *velocity, const TData *gradient,
                            const TData *normals, const TData *fb,
                            const TData kinvis, const TData velocityScale,
                            const TData theta, const TData alpha1,
                            const TData alpha2, const TData U0,
                            const TData delta, TData *pressure,
                            TData *energyMinusFb, TData *divergence,
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
    const auto gradientVec   = reinterpret_cast<const vec_t *>(gradient);
    const auto normalsVec    = reinterpret_cast<const vec_t *>(normals);
    const auto fbVec         = reinterpret_cast<const vec_t *>(fb);
    auto pressureVec         = reinterpret_cast<vec_t *>(pressure);
    auto energyMinusFbVec    = reinterpret_cast<vec_t *>(energyMinusFb);
    auto divergenceVec       = reinterpret_cast<vec_t *>(divergence);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nVecs, NEKTAR_LAMBDA(const size_t i) {
            vec_t u[NDIM];
            vec_t grad[NDIM * NDIM];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u[d] = vec_t(velocityScale) * velocityVec[d * vecCompSize + i];
                for (unsigned int j = 0; j < NDIM; ++j)
                {
                    grad[d * NDIM + j] =
                        vec_t(velocityScale) *
                        gradientVec[(d * NDIM + j) * vecCompSize + i];
                }
            }

            vec_t un = vec_t(0.0), u2 = vec_t(0.0), div = vec_t(0.0);
            vec_t nGradUn = vec_t(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const vec_t nd = normalsVec[d * vecCompSize + i];
                un += nd * u[d];
                u2 += u[d] * u[d];
                div += grad[d * NDIM + d];
                for (unsigned int j = 0; j < NDIM; ++j)
                {
                    nGradUn += nd * grad[d * NDIM + j] *
                               normalsVec[j * vecCompSize + i];
                }
            }

            const vec_t S0 =
                vec_t(0.5) * (vec_t(1.0) - vecTanh(un / vec_t(U0 * delta)));
            const vec_t k1 = vec_t(TData(0.5) * (theta + alpha2));
            const vec_t k2 = vec_t(TData(0.5) * (TData(1) - theta + alpha1));
            vec_t energyNormal = vec_t(0.0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * vecCompSize + i;
                const vec_t nd   = normalsVec[idx];
                const vec_t value =
                    (k1 * u2 * nd + k2 * un * u[d]) * S0 - fbVec[idx];
                energyMinusFbVec[idx] = value;
                energyNormal += nd * value;
            }
            divergenceVec[i] = div;
            pressureVec[i]   = vec_t(kinvis) * nGradUn - energyNormal;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Evaluate Dong's velocity flux at the points of a boundary region,
 * from the solved pressure and the data kept by HOutflowPressureKernel.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOutflowVelocityKernel(const size_t bndSize, const size_t compSize,
                            const TData *solvedPressure, const TData *normals,
                            const TData *energyMinusFb, const TData *divergence,
                            const TData kinvis, TData *velocityFlux,
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
    const auto solvedPressureVec =
        reinterpret_cast<const vec_t *>(solvedPressure);
    const auto normalsVec = reinterpret_cast<const vec_t *>(normals);
    const auto energyMinusFbVec =
        reinterpret_cast<const vec_t *>(energyMinusFb);
    const auto divergenceVec = reinterpret_cast<const vec_t *>(divergence);
    auto velocityFluxVec     = reinterpret_cast<vec_t *>(velocityFlux);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nVecs, NEKTAR_LAMBDA(const size_t i) {
            const vec_t p = solvedPressureVec[i];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * vecCompSize + i;
                velocityFluxVec[idx] =
                    ((p - vec_t(kinvis) * divergenceVec[i]) * normalsVec[idx] +
                     energyMinusFbVec[idx]) /
                    vec_t(kinvis);
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
