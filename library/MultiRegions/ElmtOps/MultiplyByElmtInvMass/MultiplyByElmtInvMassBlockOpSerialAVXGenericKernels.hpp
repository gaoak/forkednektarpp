///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassBlockOpSerialAVXGenericKernels.hpp
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
// Description: Serial/AVX kernels of the inverse mass of an affine element.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief The standard inverse mass of one element group where it is
 * diagonal (StdRegions::StdExpansion::StdInvMassIsDiagonal()), scaled: `out[m]
 * = scale * diag[m] * in[m]`, with @p diag the nmTot values of the warehouse's
 * StdRegions::eInvMassStdMatDiagonal. On an affine element the inverse mass
 * is the standard one over the Jacobian, which @p scale may carry, one value
 * per lane. Where it is not diagonal the callers take the libxsmm product
 * instead.
 *
 * @tparam simd_type  SIMD vector type, one element per lane.
 */
template <typename simd_type, typename TData>
NEK_FORCE_INLINE static void StdInvMassDiagonalScaleKernel(
    const unsigned int nmTot, const TData *diag, const simd_type scale,
    const simd_type *in, simd_type *out)
{
    for (unsigned int m = 0; m < nmTot; ++m)
    {
        out[m] = in[m] * (scale * simd_type(diag[m]));
    }
}

} // namespace Nektar::MultiRegions::detail
