///////////////////////////////////////////////////////////////////////////////
//
// File: AeroForcesKernels.hpp
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
// Description: Kernels of AeroForcesOp: boundary traction and its surface
// integral.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Write the pressure and viscous tractions of one boundary quadrature
 * point, and their moments about the pivot, to @p vals.
 *
 * @p vals receives the pressure force, the viscous force, the pressure moment
 * and the viscous moment as 3-vectors starting at entries 0, 3, 6 and 9; in
 * 2D the z forces and the x and y moments are zero, and the z moments hold
 * the in-plane moments.
 *
 * The pressure and the velocity gradient at the point are restricted from
 * the volume by one row each of a sparse restriction: row @p point has the
 * entries from pressureRowOffset[point] up to pressureRowOffset[point + 1],
 * and likewise for the gradient, each with the storage offset of its source
 * within a component and its weight. A row which is a gather has one entry of
 * weight one. The offsets are computed on the host for the interleave width
 * the pressure and the gradient are laid out at, so that both are read where
 * they are stored, at either width. The velocity gradient holds the
 * derivative of velocity component @c c in direction @c d as component
 * <tt>c * DIMENSION + d</tt>. The normals are weighted by the quadrature
 * metric of the boundary, and the moment arms are positions relative to the
 * pivot; both are stored point-minor, component @c d of point @p point at
 * <tt>d * nPoints + point</tt>, so that neighbouring threads read
 * neighbouring entries.
 */
template <unsigned int DIMENSION, typename TData>
NEK_HOSTDEVICE_INLINE void AeroForcesPoint(
    const size_t point, const size_t nPoints,
    const size_t *NEK_RESTRICT pressureRowOffset,
    const size_t *NEK_RESTRICT pressureSource,
    const TData *NEK_RESTRICT pressureWeight,
    const size_t *NEK_RESTRICT gradientRowOffset,
    const size_t *NEK_RESTRICT gradientSource,
    const TData *NEK_RESTRICT gradientWeight,
    const TData *NEK_RESTRICT weightedNormals,
    const TData *NEK_RESTRICT momentArms, const TData *NEK_RESTRICT pressure,
    const TData *NEK_RESTRICT gradient, const size_t gradientCompSize,
    const unsigned int velocityComponent0,
    const unsigned int velocityComponent1,
    const unsigned int velocityComponent2, const TData rho, const TData mu,
    TData *vals)
{
    TData p = TData(0);
    for (size_t entry = pressureRowOffset[point];
         entry < pressureRowOffset[point + 1]; ++entry)
    {
        p += pressureWeight[entry] * pressure[pressureSource[entry]];
    }

    TData grad[DIMENSION * DIMENSION] = {};
    for (size_t entry = gradientRowOffset[point];
         entry < gradientRowOffset[point + 1]; ++entry)
    {
        const TData weight  = gradientWeight[entry];
        const size_t source = gradientSource[entry];
        for (unsigned int c = 0; c < DIMENSION; ++c)
        {
            const unsigned int component = c == 0   ? velocityComponent0
                                           : c == 1 ? velocityComponent1
                                                    : velocityComponent2;
            for (unsigned int d = 0; d < DIMENSION; ++d)
            {
                grad[c * DIMENSION + d] +=
                    weight *
                    gradient[(component * DIMENSION + d) * gradientCompSize +
                             source];
            }
        }
    }

    TData normal[DIMENSION];
    for (unsigned int j = 0; j < DIMENSION; ++j)
    {
        normal[j] = weightedNormals[j * nPoints + point];
    }
    TData fp[3] = {};
    TData fv[3] = {};
    for (unsigned int j = 0; j < DIMENSION; ++j)
    {
        fp[j] = rho * p * normal[j];
        for (unsigned int k = 0; k < DIMENSION; ++k)
        {
            fv[j] -= mu * (grad[k * DIMENSION + j] + grad[j * DIMENSION + k]) *
                     normal[k];
        }
    }
    for (unsigned int j = 0; j < 3; ++j)
    {
        vals[0 + j] = fp[j];
        vals[3 + j] = fv[j];
    }

    const TData r[3] = {momentArms[point], momentArms[nPoints + point],
                        momentArms[2 * nPoints + point]};
    if constexpr (DIMENSION == 2)
    {
        vals[6]  = TData(0);
        vals[7]  = TData(0);
        vals[8]  = r[0] * fp[1] - r[1] * fp[0];
        vals[9]  = TData(0);
        vals[10] = TData(0);
        vals[11] = r[0] * fv[1] - r[1] * fv[0];
    }
    else
    {
        vals[6]  = r[1] * fp[2] - r[2] * fp[1];
        vals[7]  = r[2] * fp[0] - r[0] * fp[2];
        vals[8]  = r[0] * fp[1] - r[1] * fp[0];
        vals[9]  = r[1] * fv[2] - r[2] * fv[1];
        vals[10] = r[2] * fv[0] - r[0] * fv[2];
        vals[11] = r[0] * fv[1] - r[1] * fv[0];
    }
}

/**
 * @brief Add the forces and moments on the boundary quadrature points of one
 * block to the 3-vectors at result[0], result[3], result[6] and result[9], as
 * AeroForcesPoint() lays them out.
 *
 * The twelve sums are reduced over the points, one point per thread, by one
 * parallel_reduce of twelve values on stream @p streamID.
 */
template <unsigned int DIMENSION, typename ExecSpace, typename TData>
void AeroForcesKernel(const size_t nPoints, const size_t *pressureRowOffset,
                      const size_t *pressureSource, const TData *pressureWeight,
                      const size_t *gradientRowOffset,
                      const size_t *gradientSource, const TData *gradientWeight,
                      const TData *weightedNormals, const TData *momentArms,
                      const TData *pressure, const TData *gradient,
                      const size_t gradientCompSize,
                      const unsigned int velocityComponent0,
                      const unsigned int velocityComponent1,
                      const unsigned int velocityComponent2, const TData rho,
                      const TData mu, TData *result,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, 12>(
        0, nPoints,
        NEKTAR_LAMBDA(const size_t point, TData *vals) {
            AeroForcesPoint<DIMENSION>(
                point, nPoints, pressureRowOffset, pressureSource,
                pressureWeight, gradientRowOffset, gradientSource,
                gradientWeight, weightedNormals, momentArms, pressure, gradient,
                gradientCompSize, velocityComponent0, velocityComponent1,
                velocityComponent2, rho, mu, vals);
        },
        result);
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
