///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseBlockOpDeviceStdMatKernels.hpp
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
// Description: Device StdMat kernels of the inner product with the
// gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseBlockOpDeviceStdMatKernels.hpp
 * @brief Device StdMat kernels of the inner product with the gradient of
 * the basis: the geometric stage that turns the input vector into one
 * physical-space field per reference direction.
 *
 * @details
 * Both kernels below evaluate, for every quadrature point of every element,
 * homogeneous plane and output component of the block,
 * \f[ g_d = J \sum_k \frac{\partial \xi_d}{\partial x_k} f_k , \f]
 * optionally weighted by the quadrature weights, in a single launch on the
 * caller's stream: one work item per (element, point) pair on the first
 * grid dimension, the homogeneous planes and the components on the second
 * and third (see IProductWRTDerivBaseBlockOpDeviceStdMat.hpp for the two
 * stages). They do not apply the operator's scale factor, which is folded into
 * the GEMM alpha instead.
 *
 * The geometric factors are shared between planes and components. They are
 * read from the transposed DerivFactorKey layout, in which the nqTot values
 * of one factor of one element are contiguous: factor c of element e at
 * point i sits at ndf * nqTot * e + nqTot * c + i, written as
 * (ndf - 1) * nqTot * e + nqTot * c + idx because the flat index idx
 * already carries one nqTot * e. On a regular block there is instead one
 * set of factors and one Jacobian per element.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Multiply the input vector by the derivative factors and the
 * Jacobian, without the quadrature weights (Coeff output path).
 *
 * The second grid dimension picks the homogeneous plane m and the third the
 * output component c. Each component holds inDim input slots of one plane
 * per homogeneous plane (inDim is 3 in 3DH1, coordDim otherwise) and one
 * output slot per plane, so the pair selects slot inDim * nhomo * c + m of
 * @p in and slot nhomo * c + m of @p out, nqTot * nelmt values apart.
 *
 * @tparam DEFORMED     True when the geometric factors vary within an
 *                      element.
 * @tparam TthreadBlock Back-end thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   dimension   Reference dimension: number of output fields
 *                      written.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   outoffset   Stride between the output fields of successive
 *                      reference directions.
 * @param   jacptr      Jacobians of the block.
 * @param   dfptr       Derivative factors of the block, transposed layout.
 * @param   in          Input vector \f${\bf f}\f$; coordinate component k
 *                      of a slot starts k * inoffset after it.
 * @param   out         Receives the @p dimension fields \f$g_d\f$, field d
 *                      of a slot starting d * outoffset after it.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void JacobianDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *jacptr, const TData *dfptr,
    const TData *in, TData *out, const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
    const TData *inptr       = in + nsize * (inDim * nmode * c + m);
    TData *outptr            = out + nsize * (nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] =
                    dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + k * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + k * inoffset];
                }
            }
        }

        const auto wj = DEFORMED ? jacptr[idx] : jacptr[e];
        for (unsigned int d = 0; d < dimension; d++)
        {
            outptr[d * outoffset + idx] = tmp[d] * wj;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Multiply the input vector by the derivative factors, the Jacobian
 * and the quadrature weights (Phys output path).
 *
 * Identical to JacobianDerivFactorKernel except that each value is
 * additionally weighted by @p weights, the element's quadrature weight at
 * that point. The Phys path needs this because its standard-region matrix
 * (ePhysDerivStdMatTranspose) carries no weights.
 *
 * @tparam DEFORMED     True when the geometric factors vary within an
 *                      element.
 * @tparam TthreadBlock Back-end thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   dimension   Reference dimension: number of output fields
 *                      written.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   outoffset   Stride between the output fields of successive
 *                      reference directions.
 * @param   jacptr      Jacobians of the block.
 * @param   dfptr       Derivative factors of the block, transposed layout.
 * @param   weights     Quadrature weight per point of the element.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   out         Receives the @p dimension weighted fields.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void JacobianDerivFactorWeightsKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *jacptr, const TData *dfptr,
    const TData *weights, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
    const TData *inptr       = in + nsize * (inDim * nmode * c + m);
    TData *outptr            = out + nsize * (nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] =
                    dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + k * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + k * inoffset];
                }
            }
        }

        const size_t i = idx - e * nqTot;
        const auto wj  = (DEFORMED ? jacptr[idx] : jacptr[e]) * weights[i];
        for (unsigned int d = 0; d < dimension; d++)
        {
            outptr[d * outoffset + idx] = tmp[d] * wj;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
