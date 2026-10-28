///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceBlockOpDeviceStdMatKernels.hpp
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
// Description: Chain-rule kernel of the device StdMat divergence.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file DivergenceBlockOpDeviceStdMatKernels.hpp
 * @brief Chain-rule kernel of the device StdMat divergence: contract the
 * standard-region derivatives of every component with the element's
 * derivative factors and sum them, one work item per point of the block.
 *
 * @details
 * The derivative factors are laid out transposed (DerivFactorKey with
 * `transpose = true`, see GeometricDataWarehouse.hpp): per element the
 * `ndf = coordDim * dimension` factors form contiguous planes, of one value
 * each for a regular element and of nqTot values each for a deformed one.
 * The deformed branch reaches factor `c * dimension + d` of element @em e at
 * `ndf * nqTot * e + nqTot * (c * dimension + d) + i`, written as
 * `(ndf - 1) * nqTot * e + ...` because the within-plane flat index added
 * last, `idx`, already carries one `nqTot * e`.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Form the divergence of a whole block from the standard-region
 * derivatives of its components, writing every point once.
 *
 * Each work item sums
 * \f$\sum_c \sum_d \partial\xi_d/\partial x_c\,
 * \partial u_c/\partial\xi_d\f$ at its point in a register and stores it.
 * The second grid dimension picks the homogeneous plane, `nqTot * nelmt`
 * values apart in both @p in and @p out; the same geometry serves every
 * plane.
 *
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Back-end thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian components the factor array holds
 *                      (coordDim); sets the per-element factor stride.
 * @param   dimension   Components of the vector field, and standard
 *                      directions to contract over.
 * @param   nelmt       Elements in the block, including padding.
 * @param   compoffset  Stride between the components of @p in.
 * @param   diroffset   Stride between the standard directions of @p in.
 * @param   dfptr       Derivative factors of the block, transposed layout.
 * @param   in          Standard-direction derivatives, direction @em d of
 *                      component @em c at `c * compoffset + d * diroffset`.
 * @param   out         Divergence of the block.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void DivergenceDerivFactorKernel(
    const unsigned nqTot, const unsigned ncoord, const unsigned dimension,
    const size_t nelmt, const size_t compoffset, const size_t diroffset,
    const TData *dfptr, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx           = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    const TData *inptr   = in + nsize * p;
    TData *outptr        = out + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData div = 0.0;
        for (unsigned int c = 0; c < dimension; c++)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                const TData df = DEFORMED
                                     ? dfptr[(ndf - 1) * nqTot * e +
                                             nqTot * (c * dimension + d) + idx]
                                     : dfptr[ndf * e + c * dimension + d];
                div += df * inptr[c * compoffset + d * diroffset + idx];
            }
        }

        outptr[idx] = div;

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
