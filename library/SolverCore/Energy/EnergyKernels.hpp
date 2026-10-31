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

namespace Nektar::SolverCore::detail
{

/**
 * @brief Add the integrals of the squared velocity and vorticity magnitudes
 * and the volume of one block to result[0], result[1] and result[2].
 *
 * The velocity, its gradient and the quadrature metric share the block's
 * storage layout, the caller passing the metric at the interleave width of
 * the velocity. One storage index then addresses the same point in all three,
 * whatever that width, and every entry of a component is one term of a
 * parallel sum: the kernel walks the storage as it is laid out, neighbouring
 * threads reading neighbouring entries, with no index arithmetic and no
 * reshape at either width. The padding of the metric is zero, so the padded
 * entries add nothing. The gradient holds the derivative of velocity
 * component @c c in direction @c d as component <tt>c * DIMENSION + d</tt>.
 */
template <unsigned int DIMENSION, typename ExecSpace, typename TData>
void EnergyKernel(const size_t compSize, const unsigned int velocityComponent0,
                  const unsigned int velocityComponent1,
                  const unsigned int velocityComponent2, const TData *metric,
                  const TData *velocity, const TData *gradient, TData *result,
                  const unsigned int streamID)
{
    static_assert(DIMENSION == 2 || DIMENSION == 3,
                  "Energy supports only dimensions 2 and 3.");

    // Offsets of the velocity components and of the gradient components
    // forming the curl. The dimension is dispatched outside the lambdas,
    // which nvcc does not let capture a variable first in an if constexpr.
    const size_t uOffset    = velocityComponent0 * compSize;
    const size_t vOffset    = velocityComponent1 * compSize;
    const size_t dudyOffset = (velocityComponent0 * DIMENSION + 1) * compSize;
    const size_t dvdxOffset = (velocityComponent1 * DIMENSION + 0) * compSize;

    Nektar::LoopExecutionSetStreamID(streamID);
    if constexpr (DIMENSION == 2)
    {
        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 3>(
            0, compSize,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                const TData weight = metric[idx];
                const TData velocitySquared =
                    velocity[uOffset + idx] * velocity[uOffset + idx] +
                    velocity[vOffset + idx] * velocity[vOffset + idx];
                const TData curl =
                    gradient[dvdxOffset + idx] - gradient[dudyOffset + idx];

                vals[0] = weight * velocitySquared;
                vals[1] = weight * (curl * curl);
                vals[2] = weight;
            },
            result);
    }
    else
    {
        const size_t wOffset = velocityComponent2 * compSize;
        const size_t dudzOffset =
            (velocityComponent0 * DIMENSION + 2) * compSize;
        const size_t dvdzOffset =
            (velocityComponent1 * DIMENSION + 2) * compSize;
        const size_t dwdxOffset =
            (velocityComponent2 * DIMENSION + 0) * compSize;
        const size_t dwdyOffset =
            (velocityComponent2 * DIMENSION + 1) * compSize;

        Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 3>(
            0, compSize,
            NEKTAR_LAMBDA(const size_t idx, TData *vals) {
                const TData weight = metric[idx];
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
                vals[1] =
                    weight * (curlX * curlX + curlY * curlY + curlZ * curlZ);
                vals[2] = weight;
            },
            result);
    }
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
