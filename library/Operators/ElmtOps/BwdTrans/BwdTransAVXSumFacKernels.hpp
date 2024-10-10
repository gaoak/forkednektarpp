///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransAVXSumFacKernels.hpp
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

#include "StdRegions/Operators/BwdTransSumFacStdKernels.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.
template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans1DWorkspace(
    [[maybe_unused]] const size_t nm0, [[maybe_unused]] const size_t nq0)

{
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans2DWorkspace(
    [[maybe_unused]] const size_t nm0, [[maybe_unused]] const size_t nm1,
    [[maybe_unused]] const size_t nq0, [[maybe_unused]] const size_t nq1,
    size_t &wsp0Size)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        wsp0Size = std::max(wsp0Size, nm0);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm1 * nq0);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans3DWorkspace(
    [[maybe_unused]] const size_t nm0, [[maybe_unused]] const size_t nm1,
    [[maybe_unused]] const size_t nm2, [[maybe_unused]] const size_t nq0,
    [[maybe_unused]] const size_t nq1, [[maybe_unused]] const size_t nq2,
    size_t &wsp0Size, size_t &wsp1Size)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        wsp0Size = std::max(wsp0Size, nq0 * nm1 * nm2);
        wsp1Size = std::max(wsp1Size, nq0 * nq1 * nm2);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm0 * nm1);
        wsp1Size = std::max(wsp1Size, nm0);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans1DKernel(const size_t nm0,
                                              const size_t nq0,
                                              const vec_t *basis0,
                                              const vec_t::vectorType *in,
                                              vec_t::scalarType *out)
{
    BwdTransSegKernel(nm0, nq0, basis0, in, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans2DKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    [[maybe_unused]] const bool correct, const vec_t *basis0,
    const vec_t *basis1, std::vector<vec_t, tinysimd::allocator<vec_t>> &wsp0,
    const vec_t::vectorType *in, vec_t::scalarType *out)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        BwdTransTriKernel(nm0, nm1, nq0, nq1, correct, basis0, basis1, wsp0, in,
                          out);
    }
    else
    {
        BwdTransQuadKernel(nm0, nm1, nq0, nq1, basis0, basis1, wsp0, in, out);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void BwdTrans3DKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, [[maybe_unused]] const bool correct,
    const vec_t *basis0, const vec_t *basis1, const vec_t *basis2,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &wsp0,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &wsp1,
    const vec_t::vectorType *in, vec_t::scalarType *out)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        BwdTransHexKernel(nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                          wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        BwdTransTetKernel(nm0, nm1, nm2, nq0, nq1, nq2, correct, basis0, basis1,
                          basis2, wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        BwdTransPrismKernel(nm0, nm1, nm2, nq0, nq1, nq2, correct, basis0,
                            basis1, basis2, wsp0, wsp1, in, out);
    }
    else
    {
        BwdTransPyrKernel(nm0, nm1, nm2, nq0, nq1, nq2, correct, basis0, basis1,
                          basis2, wsp0, wsp1, in, out);
    }
}
