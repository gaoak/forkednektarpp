///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceKernels.hpp
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
// Description: Device kernel the implementations of the lift-and-project
// element operator share.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceKernels.hpp
 * @brief The Device kernel of the Pointwise form at width one, shared by the
 * SumFacTOP and StdMat implementations of the lift-and-project operator:
 * each thread block striding over the elements and its threads over the
 * element's points, so that consecutive threads touch consecutive entries.
 * It is restricted to Implementation SumFacTOP or StdMat and has the name
 * and arguments of the SumFac kernel
 * (PhysLiftProjectBlockOpDeviceSumFacKernels.hpp), so that each caller
 * launches `PhysLiftProjectPointwiseKernel<Implementation, ...>`.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

#include <type_traits>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief The Pointwise form at width one: `out (+)= scale / (w_q J) * in` at
 * every point, zero on the padding elements. Grid dimension 1 is the
 * component.
 *
 * @param   weights  The quadrature weights at every point (WeightsKey at
 *                   width one).
 * @param   jac      The Jacobians (JacobianKey at width one), one per
 *                   element, or one per point when @p DEFORMED.
 */
template <typename Implementation, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> ||
                               std::is_same_v<Implementation, StdMat>,
                           bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectPointwiseKernel(
    const size_t nelmt, const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getBlockIdx<0>(threadBlock); e < nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const size_t offset = (nelmt * c + e) * nqTot;
        for (unsigned int q = getLocalIdx<0>(threadBlock); q < nqTot;
             q += getLocalRange<0>(threadBlock))
        {
            TData jacobian;
            if constexpr (DEFORMED)
            {
                jacobian = jac[e * nqTot + q];
            }
            else
            {
                jacobian = jac[e];
            }
            const TData factor = SafeDivide(scale, weights[q] * jacobian);
            if constexpr (APPEND)
            {
                out[offset + q] += factor * in[offset + q];
            }
            else
            {
                out[offset + q] = factor * in[offset + q];
            }
        }
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
