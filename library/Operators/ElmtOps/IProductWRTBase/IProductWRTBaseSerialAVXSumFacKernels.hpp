///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialAVXSumFacKernels.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <LibUtilities/BasicUtils/NekInline.hpp>

#include "StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct1DWorkspace(
    [[maybe_unused]] const size_t nm0, [[maybe_unused]] const size_t nq0)

{
    // Check preconditions
    // None
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct2DWorkspace(
    [[maybe_unused]] const size_t nm0, [[maybe_unused]] const size_t nm1,
    [[maybe_unused]] const size_t nq0, const size_t nq1, size_t &wsp0Size)
{
    wsp0Size = std::max(wsp0Size, nq1);
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct3DWorkspace(
    [[maybe_unused]] const size_t nm0, const size_t nm1,
    [[maybe_unused]] const size_t nm2, [[maybe_unused]] const size_t nq0,
    const size_t nq1, const size_t nq2, size_t &wsp0Size, size_t &wsp1Size,
    size_t &wsp2Size)
{
    wsp0Size = std::max(wsp0Size, nq1 * nq2);
    wsp1Size = std::max(wsp1Size, nq2);
    wsp2Size = std::max(wsp0Size, nm1);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct1DKernel(
    const size_t nm0, const size_t nq0,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *w0, const simd_type *jac,
    typename simd_type::scalarType *out,
    typename simd_type::scalarType scale = 1.0)
{
    IProductSegKernel<SCALE, APPEND, DEFORMED>(nm0, nq0, in, basis0, w0, jac,
                                               out, scale);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct2DKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    [[maybe_unused]] const bool correct,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *w0, const simd_type *w1,
    const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &wsp0,
    typename simd_type::scalarType *out,
    typename simd_type::scalarType scale = 1.0)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        IProductTriKernel<SCALE, APPEND, DEFORMED>(nm0, nm1, nq0, nq1, correct,
                                                   in, basis0, basis1, w0, w1,
                                                   jac, wsp0, out, scale);
    }
    else
    {
        IProductQuadKernel<SCALE, APPEND, DEFORMED>(nm0, nm1, nq0, nq1, in,
                                                    basis0, basis1, w0, w1, jac,
                                                    wsp0, out, scale);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct3DKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, [[maybe_unused]] const bool correct,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &wsp0,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &wsp1,
    [[maybe_unused]] std::vector<simd_type, tinysimd::allocator<simd_type>>
        &wsp2,
    typename simd_type::scalarType *out,
    typename simd_type::scalarType scale = 1.0)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        IProductHexKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, in, basis0, basis1, basis2, w0, w1,
            w2, jac, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        IProductTetKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, correct, in, basis0, basis1, basis2,
            w0, w1, w2, jac, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        IProductPrismKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, correct, in, basis0, basis1, basis2,
            w0, w1, w2, jac, wsp0, wsp1, wsp2, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        IProductPyrKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, correct, in, basis0, basis1, basis2,
            w0, w1, w2, jac, wsp0, wsp1, out, scale);
    }
}
