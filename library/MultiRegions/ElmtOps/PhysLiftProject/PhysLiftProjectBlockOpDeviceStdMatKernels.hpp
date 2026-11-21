///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceStdMatKernels.hpp
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
// Description: Device StdMat kernels of the lift-and-project element
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceStdMatKernels.hpp
 * @brief Device StdMat kernels of the lift-and-project operator: the
 * Pointwise form at width one.
 *
 * @details
 * The threads stride over the points of all the elements, one point
 * each, so that consecutive threads touch consecutive entries. The kernel
 * has the name and arguments of its SumFac and SumFacTOP counterparts
 * (PhysLiftProjectBlockOpDeviceSumFacKernels.hpp,
 * PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp), restricted to
 * Implementation StdMat, so that the caller launches
 * `PhysLiftProjectPointwiseKernel<Implementation, ...>`.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
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
          std::enable_if_t<std::is_same_v<Implementation, StdMat>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectPointwiseKernel(
    const size_t nelmt, const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const size_t nsize              = nqTot * nelmt;
    const unsigned int c            = getBlockIdx<1>(threadBlock);
    const TData *NEK_RESTRICT inptr = in + nsize * c;
    TData *NEK_RESTRICT outptr      = out + nsize * c;
    for (size_t idx = getGlobalIdx<0>(threadBlock); idx < nsize;
         idx += getGlobalRange<0>(threadBlock))
    {
        const unsigned int q = idx % nqTot;
        TData jacobian;
        if constexpr (DEFORMED)
        {
            jacobian = jac[idx];
        }
        else
        {
            jacobian = jac[idx / nqTot];
        }
        const TData factor = SafeDivide(scale, weights[q] * jacobian);
        if constexpr (APPEND)
        {
            outptr[idx] += factor * inptr[idx];
        }
        else
        {
            outptr[idx] = factor * inptr[idx];
        }
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
