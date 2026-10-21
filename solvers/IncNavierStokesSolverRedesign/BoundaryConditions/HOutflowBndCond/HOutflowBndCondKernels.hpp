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
void HOutflowVolumeGatherKernel(const size_t nsize, const unsigned int nComp,
                                const size_t srcCompSize,
                                const size_t dstCompSize, const size_t *srcIdx,
                                const size_t *dstIdx, const TData *src,
                                TData *dst, const unsigned int streamID)
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
void HOutflowPressureKernel(const size_t nsize, const size_t compSize,
                            const TData *velocity, const TData *gradient,
                            const TData *normals, const TData *fb,
                            const TData kinvis, const TData velocityScale,
                            const TData theta, const TData alpha1,
                            const TData alpha2, const TData U0,
                            const TData delta, TData *pressure,
                            TData *energyMinusFb, TData *divergence,
                            const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            TData u[NDIM];
            TData grad[NDIM * NDIM];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u[d] = velocityScale * velocity[d * compSize + i];
                for (unsigned int j = 0; j < NDIM; ++j)
                {
                    grad[d * NDIM + j] =
                        velocityScale * gradient[(d * NDIM + j) * compSize + i];
                }
            }

            TData un = TData(0), u2 = TData(0), div = TData(0);
            TData nGradUn = TData(0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const TData nd = normals[d * compSize + i];
                un += nd * u[d];
                u2 += u[d] * u[d];
                div += grad[d * NDIM + d];
                for (unsigned int j = 0; j < NDIM; ++j)
                {
                    nGradUn +=
                        nd * grad[d * NDIM + j] * normals[j * compSize + i];
                }
            }

            const TData S0 = TData(0.5) * (TData(1) - tanh(un / (U0 * delta)));
            const TData k1 = TData(0.5) * (theta + alpha2);
            const TData k2 = TData(0.5) * (TData(1) - theta + alpha1);
            TData energyNormal = TData(0);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * compSize + i;
                const TData nd   = normals[idx];
                const TData value =
                    (k1 * u2 * nd + k2 * un * u[d]) * S0 - fb[idx];
                energyMinusFb[idx] = value;
                energyNormal += nd * value;
            }
            divergence[i] = div;
            pressure[i]   = kinvis * nGradUn - energyNormal;
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Evaluate Dong's velocity flux at the points of a boundary region,
 * from the solved pressure and the data kept by HOutflowPressureKernel.
 */
template <typename ExecSpace, unsigned int NDIM, typename TData>
void HOutflowVelocityKernel(const size_t nsize, const size_t compSize,
                            const TData *solvedPressure, const TData *normals,
                            const TData *energyMinusFb, const TData *divergence,
                            const TData kinvis, TData *velocityFlux,
                            const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            const TData p = solvedPressure[i];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const size_t idx = d * compSize + i;
                velocityFlux[idx] =
                    ((p - kinvis * divergence[i]) * normals[idx] +
                     energyMinusFb[idx]) /
                    kinvis;
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
