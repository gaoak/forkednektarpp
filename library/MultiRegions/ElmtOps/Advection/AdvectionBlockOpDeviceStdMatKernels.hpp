///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionBlockOpDeviceStdMatKernels.hpp
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
// Description: Pointwise derivative-factor and velocity kernel of the Device
// StdMat advection block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionBlockOpDeviceStdMatKernels.hpp
 * @brief Pointwise derivative-factor/velocity kernel of the Device StdMat
 * advection block operator (see AdvectionBlockOpDeviceStdMat.hpp).
 *
 * @details
 * The kernel runs over all quadrature points of the block, launched by the
 * block operator on its device stream; each variable and homogeneous plane
 * of the block rides the second grid dimension of that one launch.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief From reference-space derivatives, form the Cartesian gradient and
 * contract it with the advection velocity:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d, evaluated at every
 * quadrature point of every element of one variable and plane.
 *
 * The derivative-factor entry k * dimension + d maps the
 * reference-direction-d derivative into the Cartesian-k gradient
 * component. The factors are read in the layout produced by
 * DerivFactorKey with the transpose flag set: for regular geometry one set
 * of ncoord * dimension values per element, for deformed geometry one such
 * set per element in which each factor component is contiguous over the
 * element's nqTot points.
 *
 * Work items run over the quadrature points of every element on the first
 * grid dimension; the second grid dimension picks the variable and plane,
 * @p nelmt * @p nqTot values apart in @p in and @p out. The velocity is
 * taken from the plane, that grid index modulo @p nhomo, and the same
 * geometric factors serve every plane.
 *
 * @tparam APPEND       Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian gradient/velocity components.
 * @param   dimension   Reference directions covered by @p in.
 * @param   nelmt       Elements in the block, including padding.
 * @param   inoffset    Stride between the reference-direction planes of
 *                      @p in.
 * @param   dfptr       Derivative factors of the block, laid out as
 *                      described above.
 * @param   advVel      Velocity data; component k starts at
 *                      k * advelsize.
 * @param   advelsize   Stride between velocity components.
 * @param   nhomo       Homogeneous planes per variable.
 * @param   in          Reference-space derivatives, one plane of
 *                      @p inoffset values per direction.
 * @param   out         Output values, one per work item.
 * @param   scale       Scalar factor applied to the result.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByDerivFactorAndAdvecVelKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *advVel, const size_t advelsize,
    const unsigned int nhomo, const TData *in, TData *out, const TData scale,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx             = getGlobalIdx<0>(threadBlock);
    const unsigned int c   = getBlockIdx<1>(threadBlock);
    const TData *inptr     = in + nsize * c;
    TData *outptr          = out + nsize * c;
    const TData *advVelPtr = advVel + nsize * (c % nhomo);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3], tmp0 = 0;
        if constexpr (DEFORMED)
        {
            const size_t i = idx - e * nqTot;
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[ndf * nqTot * e + (k * dimension) * nqTot + i] *
                         inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[ndf * nqTot * e +
                                    (k * dimension + d) * nqTot + i] *
                              inptr[idx + d * inoffset];
                }
                tmp0 += advVelPtr[k * advelsize + idx] * tmp[k];
            }
        }
        else
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[(ndf * e + k * dimension)] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + d * inoffset];
                }
                tmp0 += advVelPtr[k * advelsize + idx] * tmp[k];
            }
        }

        if constexpr (APPEND)
        {
            outptr[idx] += scale * tmp0;
        }
        else
        {
            outptr[idx] = scale * tmp0;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
