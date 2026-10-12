///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXStdMatKernels.hpp
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
// Description: Chain-rule kernels of the Serial/AVX StdMat physical
// derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivSerialAVXStdMatKernels.hpp
 * @brief Chain-rule kernels of the Serial/AVX StdMat physical
 * derivative: contract standard-region derivatives with the element's
 * geometric derivative factors.
 *
 * @details
 * The StdMat implementations produce, for each element, the `dimension`
 * derivatives \f$\partial u / \partial \xi_d\f$ with respect to the
 * standard-element coordinates. These kernels perform the remaining
 * chain rule
 * \f[ \frac{\partial u}{\partial x_k}
 *     = \sum_{d} \frac{\partial \xi_d}{\partial x_k}
 *       \frac{\partial u}{\partial \xi_d} \f]
 * against the derivative factors of the block, which are laid out
 * point-major -- `coordDim * dimension` consecutive values per
 * quadrature point for a deformed element, one such set per element for
 * a regular one (DerivFactorKey with `transpose = false`, see
 * GeometricDataWarehouse.hpp) -- and are addressed as
 * `k * dimension + d`. The DEFORMED template flag selects between the
 * two indexings; nothing else about the two branches differs.
 *
 * MultiplyByDerivFactorKernel produces all `coordDim` components at
 * once and is used by PhysDeriv itself (and by the AdvectionDealias
 * StdMat implementation); MultiplyByDerivDirFactorKernel produces one
 * chosen component and is used by the Divergence StdMat implementation,
 * which accumulates the diagonal terms of
 * \f$\nabla \cdot \mathbf{u}\f$ into a single output.
 *
 * These kernels are instantiated with @p TData bound to the SIMD vector
 * type of the execution space (tinysimd::scalarT on Serial), so each
 * element of the interleaved lane group is handled in its own lane.
 */

#pragma once

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Contract the standard-region derivatives of a lane group into
 * all @p ncoord Cartesian derivatives.
 *
 * Safe to run in place (@p outptr equal to @p inptr, as PhysDeriv does):
 * all @p ncoord results for one point are formed in registers before
 * any of them is stored.
 *
 * @tparam ExecSpace  Execution space; unused.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      SIMD vector type of the interleaved lane group.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian components to produce (coordDim); at
 *                      most 3, the size of the per-point register
 *                      buffer.
 * @param   dimension   Standard directions to contract over.
 * @param   nelmt       Element groups covered by this call; the StdMat
 *                      block operator passes 1 and advances the
 *                      pointers itself.
 * @param   inoffset    Stride, in SIMD vectors, between the
 *                      standard-direction planes of @p inptr.
 * @param   outoffset   Stride, in SIMD vectors, between the Cartesian
 *                      planes of @p outptr.
 * @param   dfptr       Derivative factors of the group, from the first
 *                      point (deformed) or the first element (regular).
 * @param   inptr       Standard-direction derivatives, plane @em d at
 *                      `d * inoffset`.
 * @param   outptr      Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *dfptr, const TData *inptr,
    TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * inoffset + idx]);
                }
            }
            for (unsigned int k = 0; k < ncoord; k++)
            {
                outptr[k * outoffset + idx] = tmp[k];
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] =
                        dfptr[ndf * e + k * dimension] * inptr[nqTot * e + i];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + d * inoffset]);
                    }
                }
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    outptr[k * outoffset + nqTot * e + i] = tmp[k];
                }
            }
        }
    }
}

/**
 * @brief Contract the standard-region derivatives of a lane group into
 * the single Cartesian derivative @p dir, overwriting or accumulating.
 *
 * The runtime @p dir argument plays the role that the DIR template
 * parameter plays in the sum-factorised kernels: it selects the
 * derivative-factor rows `dir * dimension + d`.
 *
 * @tparam ExecSpace  Execution space; unused.
 * @tparam APPEND     Accumulate onto @p outptr instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      SIMD vector type of the interleaved lane group.
 *
 * @param   dir         Cartesian component to produce.
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian components the factor array holds
 *                      (coordDim); sets the per-point factor stride.
 * @param   dimension   Standard directions to contract over.
 * @param   nelmt       Element groups covered by this call.
 * @param   inoffset    Stride, in SIMD vectors, between the
 *                      standard-direction planes of @p inptr.
 * @param   dfptr       Derivative factors of the group.
 * @param   inptr       Standard-direction derivatives, plane @em d at
 *                      `d * inoffset`.
 * @param   outptr      Single Cartesian derivative plane.
 */
template <typename ExecSpace, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivDirFactorKernel(
    const unsigned dir, const unsigned nqTot, const unsigned ncoord,
    const unsigned dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *inptr, TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp;
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            tmp = dfptr[ndf * idx + dir * dimension] * inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp.fma(dfptr[ndf * idx + dir * dimension + d],
                        inptr[d * inoffset + idx]);
            }
            if constexpr (APPEND)
            {
                outptr[idx] += tmp;
            }
            else
            {
                outptr[idx] = tmp;
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                tmp = dfptr[ndf * e + dir * dimension] * inptr[nqTot * e + i];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp.fma(dfptr[ndf * e + dir * dimension + d],
                            inptr[nqTot * e + i + d * inoffset]);
                }
                if constexpr (APPEND)
                {
                    outptr[nqTot * e + i] += tmp;
                }
                else
                {
                    outptr[nqTot * e + i] = tmp;
                }
            }
        }
    }
}

} // namespace Nektar::MultiRegions::detail
