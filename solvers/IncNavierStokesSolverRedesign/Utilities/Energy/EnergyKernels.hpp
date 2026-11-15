///////////////////////////////////////////////////////////////////////////////
//
// File: EnergyKernels.hpp
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
// Description: Device-portable kinetic-energy and enstrophy kernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Write the squared velocity and vorticity magnitudes at storage
 * index @p idx, and the volume it stands for, each times @p weight, to
 * vals[0], vals[1] and vals[2].
 *
 * The offsets are those of the velocity components and of the gradient
 * components forming the curl; in 2D the w and z-derivative offsets are not
 * read.
 */
template <unsigned int DIMENSION, typename TData>
NEK_HOSTDEVICE_INLINE void EnergyKernel(
    const size_t idx, const TData weight, const TData *NEK_RESTRICT velocity,
    const TData *NEK_RESTRICT gradient, const size_t uOffset,
    const size_t vOffset, [[maybe_unused]] const size_t wOffset,
    const size_t dudyOffset, [[maybe_unused]] const size_t dudzOffset,
    const size_t dvdxOffset, [[maybe_unused]] const size_t dvdzOffset,
    [[maybe_unused]] const size_t dwdxOffset,
    [[maybe_unused]] const size_t dwdyOffset, TData *vals)
{
    if constexpr (DIMENSION == 2)
    {
        const TData velocitySquared =
            velocity[uOffset + idx] * velocity[uOffset + idx] +
            velocity[vOffset + idx] * velocity[vOffset + idx];
        const TData curl =
            gradient[dvdxOffset + idx] - gradient[dudyOffset + idx];

        vals[0] = weight * velocitySquared;
        vals[1] = weight * (curl * curl);
    }
    else if constexpr (DIMENSION == 3)
    {
        const TData velocitySquared =
            velocity[uOffset + idx] * velocity[uOffset + idx] +
            velocity[vOffset + idx] * velocity[vOffset + idx] +
            velocity[wOffset + idx] * velocity[wOffset + idx];
        const TData curlX =
            gradient[dwdyOffset + idx] - gradient[dvdzOffset + idx];
        const TData curlY =
            gradient[dudzOffset + idx] - gradient[dwdxOffset + idx];
        const TData curlZ =
            gradient[dvdxOffset + idx] - gradient[dudyOffset + idx];

        vals[0] = weight * velocitySquared;
        vals[1] = weight * (curlX * curlX + curlY * curlY + curlZ * curlZ);
    }
    vals[2] = weight;
}

/**
 * @brief Add the integrals of the squared velocity and vorticity magnitudes
 * and the volume of one block to result[0], result[1] and result[2].
 *
 * The velocity, its gradient and the quadrature metric share the block's
 * storage layout, the caller passing the metric at the interleave width of
 * the velocity. One storage index then addresses the same point in all three,
 * whatever that width, and every entry of a component is one term of a
 * parallel sum: the kernel walks the storage as it is laid out, neighbouring
 * threads reading neighbouring entries, with no reshape at either width. The
 * padding of the metric is zero, so the padded entries add nothing. The
 * gradient holds the derivative of velocity component @c c in direction @c d
 * as component <tt>c * DIMENSION + d</tt>.
 *
 * Each component holds @p nPlanes homogeneous planes of @p planeSize entries,
 * one after the other, each weighted by its entry of @p planeWeights, which
 * the kernel finds from the plane of an entry, its only index arithmetic. A
 * single plane is that of a field without a homogeneous direction, of unit
 * weight, and is summed without it.
 */
template <unsigned int DIMENSION, typename ExecSpace, typename TData>
void EnergyKernelLauncher(const size_t planeSize, const unsigned int nPlanes,
                          const unsigned int velocityComponent0,
                          const unsigned int velocityComponent1,
                          const unsigned int velocityComponent2,
                          const TData *metric, const TData *planeWeights,
                          const TData *velocity, const TData *gradient,
                          TData *result, const unsigned int streamID)
{
    static_assert(DIMENSION == 2 || DIMENSION == 3,
                  "EnergyOp supports only dimensions 2 and 3.");

    // Entries of a component, over all its planes.
    const size_t compSize = nPlanes * planeSize;

    Nektar::LoopExecutionSetStreamID(streamID);
    if (nPlanes == 1)
    {
        const size_t uOffset = velocityComponent0 * compSize;
        const size_t vOffset = velocityComponent1 * compSize;
        const size_t wOffset = velocityComponent2 * compSize;
        const size_t dudyOffset =
            (velocityComponent0 * DIMENSION + 1) * compSize;
        const size_t dudzOffset =
            (velocityComponent0 * DIMENSION + 2) * compSize;
        const size_t dvdxOffset =
            (velocityComponent1 * DIMENSION + 0) * compSize;
        const size_t dvdzOffset =
            (velocityComponent1 * DIMENSION + 2) * compSize;
        const size_t dwdxOffset =
            (velocityComponent2 * DIMENSION + 0) * compSize;
        const size_t dwdyOffset =
            (velocityComponent2 * DIMENSION + 1) * compSize;

        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 3>(
            0, compSize,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                EnergyKernel<DIMENSION>(idx, metric[idx], velocity, gradient,
                                        uOffset, vOffset, wOffset, dudyOffset,
                                        dudzOffset, dvdxOffset, dvdzOffset,
                                        dwdxOffset, dwdyOffset, vals);
            },
            result);
    }
    else
    {
        // Offsets of the velocity components and of the gradient components
        // forming the curl; those of w and of the z derivatives are not read in
        // 2D.
        const size_t uOffset = velocityComponent0 * compSize;
        const size_t vOffset = velocityComponent1 * compSize;
        const size_t wOffset = velocityComponent2 * compSize;
        const size_t dudyOffset =
            (velocityComponent0 * DIMENSION + 1) * compSize;
        const size_t dudzOffset =
            (velocityComponent0 * DIMENSION + 2) * compSize;
        const size_t dvdxOffset =
            (velocityComponent1 * DIMENSION + 0) * compSize;
        const size_t dvdzOffset =
            (velocityComponent1 * DIMENSION + 2) * compSize;
        const size_t dwdxOffset =
            (velocityComponent2 * DIMENSION + 0) * compSize;
        const size_t dwdyOffset =
            (velocityComponent2 * DIMENSION + 1) * compSize;

        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 3>(
            0, compSize,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                const size_t plane = idx / planeSize;
                const TData weight =
                    metric[idx - plane * planeSize] * planeWeights[plane];
                EnergyKernel<DIMENSION>(idx, weight, velocity, gradient,
                                        uOffset, vOffset, wOffset, dudyOffset,
                                        dudzOffset, dvdxOffset, dvdzOffset,
                                        dwdxOffset, dwdyOffset, vals);
            },
            result);
    }
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
