///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpSerialAVXKernels.hpp
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
// Description: Serial/AVX kernels the implementations of the lift-and-project
// element operator share.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpSerialAVXKernels.hpp
 * @brief Serial/AVX kernels of the Pointwise form of the lift-and-project
 * operator, shared by its SumFac and StdMat implementations: one SIMD vector
 * of elements per call.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief The Pointwise form's factor of one element group,
 * `pointScale[q] = scale / (w_q J)`, zero on the padding elements, whose
 * Jacobian is zero. @p weights holds the quadrature weights at every point
 * (WeightsKey), @p jac the group's Jacobian, one per element or, when
 * @p DEFORMED, one per point.
 */
template <bool DEFORMED, typename simd_type, typename TData>
NEK_FORCE_INLINE static void PhysLiftProjectPointScaleKernel(
    const unsigned int nqTot, const TData scale, const simd_type *weights,
    const simd_type *jac, simd_type *pointScale)
{
    for (unsigned int q = 0; q < nqTot; ++q)
    {
        const simd_type wJ =
            DEFORMED ? weights[q] * jac[q] : weights[q] * jac[0];
        pointScale[q] = SafeDivide(simd_type(scale), wJ);
    }
}

/// The Pointwise form of one element group: `out (+)= pointScale * vol`.
template <bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void PhysLiftProjectPointwiseKernel(
    const unsigned int nqTot, const simd_type *pointScale, const simd_type *vol,
    simd_type *out)
{
    for (unsigned int q = 0; q < nqTot; ++q)
    {
        if constexpr (APPEND)
        {
            out[q].fma(pointScale[q], vol[q]);
        }
        else
        {
            out[q] = pointScale[q] * vol[q];
        }
    }
}

} // namespace Nektar::MultiRegions::detail
