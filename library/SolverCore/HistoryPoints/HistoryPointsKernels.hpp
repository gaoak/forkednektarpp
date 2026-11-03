///////////////////////////////////////////////////////////////////////////////
//
// File: HistoryPointsKernels.hpp
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
// Description: Kernel interpolating physical fields at points from per-point
// weights.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Interpolate every component of the field of one block at the history
 * points in its elements, one point per thread.
 *
 * The block is laid out at interleave width @p INTERLEAVE_WIDTH, which is one
 * or the width of the execution space. Quadrature point @c q of the element
 * holding point @c p is at <tt>elementOffset[p] + q * INTERLEAVE_WIDTH</tt>
 * within a component, the offset of the element being computed on the host. The
 * points are sorted by element, so that at the execution space's width
 * neighbouring threads read neighbouring lanes of an element group.
 *
 * The interpolation weights of each direction are stored point-minor, the
 * weight of quadrature point @c i of point @c p at <tt>i * nPoints + p</tt>,
 * so that neighbouring threads read neighbouring weights. The value of
 * component @c c at point @c p is written to
 * <tt>result[pointIds[p] * nComponents + c]</tt>.
 */
template <unsigned int DIMENSION, unsigned int INTERLEAVE_WIDTH,
          typename ExecSpace, typename TData>
void HistoryPointsKernel(const size_t nPoints, const unsigned int nComponents,
                         const unsigned int nq0, const unsigned int nq1,
                         const unsigned int nq2, const size_t compSize,
                         const size_t *elementOffset, const size_t *pointIds,
                         const TData *weights0, const TData *weights1,
                         const TData *weights2, const TData *field,
                         TData *result, const unsigned int streamID)
{
    static_assert(DIMENSION >= 1 && DIMENSION <= 3,
                  "HistoryPoints supports dimensions 1, 2 and 3.");

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, nPoints, NEKTAR_LAMBDA(const size_t point) {
            // Read what the higher directions use through locals, as nvcc
            // does not let a lambda capture a variable first inside an if
            // constexpr.
            [[maybe_unused]] const TData *direction1Weights = weights1;
            [[maybe_unused]] const TData *direction2Weights = weights2;
            const size_t weightStride                       = nPoints;
            const size_t offset       = elementOffset[point];
            const size_t resultOffset = pointIds[point] * nComponents;

            for (unsigned int c = 0; c < nComponents; ++c)
            {
                const TData *values = field + c * compSize + offset;
                TData value         = TData(0);
                unsigned int q      = 0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    TData wk = TData(1);
                    if constexpr (DIMENSION == 3)
                    {
                        wk = direction2Weights[k * weightStride + point];
                    }
                    for (unsigned int j = 0; j < nq1; ++j)
                    {
                        TData wjk = wk;
                        if constexpr (DIMENSION >= 2)
                        {
                            wjk *= direction1Weights[j * weightStride + point];
                        }
                        TData line = TData(0);
                        for (unsigned int i = 0; i < nq0; ++i, ++q)
                        {
                            line += weights0[i * weightStride + point] *
                                    values[q * INTERLEAVE_WIDTH];
                        }
                        value += wjk * line;
                    }
                }
                result[resultOffset + c] = value;
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
