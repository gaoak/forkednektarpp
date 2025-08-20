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

namespace Nektar::Operators::detail
{

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct1DWorkspace(
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nq0)

{
    // Check preconditions
    // None
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct2DWorkspace(
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0, const unsigned int nq1,
    unsigned int &wsp0Size)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        wsp0Size = std::max(wsp0Size, nq1 + nm0 * (nm0 + 1) / 2);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nq1);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void IProduct3DWorkspace(
    [[maybe_unused]] const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2,
    [[maybe_unused]] const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, unsigned int &wsp0Size, unsigned int &wsp1Size,
    unsigned int &wsp2Size)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        wsp0Size =
            std::max(wsp0Size, nq1 * nq2 + nm0 * (nm0 + 1) * (nm0 + 2) / 6);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        wsp0Size = std::max(wsp0Size, nq1 * nq2 + nm0 * nm0 * (nm0 + 1) / 2);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nq1 * nq2);
    }
    wsp1Size = std::max(wsp1Size, nq2);
    wsp2Size = std::max(wsp2Size, nm1);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct1DKernel(
    const unsigned int nm0, const unsigned int nq0, const simd_type *in,
    const simd_type *B0, const simd_type *w0, const simd_type *jac,
    simd_type *out, typename simd_type::scalarType scale = 1.0)
{
    IProductSegKernel<SCALE, APPEND, DEFORMED>(nm0, nq0, in, B0, w0, jac, out,
                                               scale);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, [[maybe_unused]] const bool isModified,
    const simd_type *in, const simd_type *B0, const simd_type *B1,
    const simd_type *w0, const simd_type *w1,
    [[maybe_unused]] const simd_type *NtoMTrans, const simd_type *jac,
    simd_type *wsp0, simd_type *out, typename simd_type::scalarType scale = 1.0)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        const auto nmTot  = nm0 * (nm0 + 1) / 2;
        simd_type *outtmp = wsp0 + nq1;

        IProductTriKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nq0, nq1, isModified, in, B0, B1, w0, w1, jac, wsp0,
            outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        IProductTriKernel<SCALE, APPEND, DEFORMED>(nm0, nm1, nq0, nq1,
                                                   isModified, in, B0, B1, w0,
                                                   w1, jac, wsp0, out, scale);
    }
    else
    {
        IProductQuadKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nq0, nq1, in, B0, B1, w0, w1, jac, wsp0, out, scale);
    }
}

// inner product without quadrature metric wJ
template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          typename simd_type>
NEK_FORCE_INLINE static void IProduct2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, [[maybe_unused]] const bool isModified,
    const simd_type *in, const simd_type *B0, const simd_type *B1,
    [[maybe_unused]] const simd_type *NtoMTrans, simd_type *wsp0,
    simd_type *out, typename simd_type::scalarType scale = 1.0)

{
    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        const auto nmTot  = nm0 * (nm0 + 1) / 2;
        simd_type *outtmp = wsp0 + nq1;

        IProductTriKernel<SCALE, APPEND, simd_type>(
            nm0, nm1, nq0, nq1, isModified, in, B0, B1, wsp0, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        IProductTriKernel<SCALE, APPEND, simd_type>(
            nm0, nm1, nq0, nq1, isModified, in, B0, B1, wsp0, out, scale);
    }
    else
    {
        IProductQuadKernel<SCALE, APPEND, simd_type>(nm0, nm1, nq0, nq1, in, B0,
                                                     B1, wsp0, out, scale);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProduct3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const bool isModified, const simd_type *in,
    const simd_type *B0, const simd_type *B1, const simd_type *B2,
    const simd_type *w0, const simd_type *w1, const simd_type *w2,
    [[maybe_unused]] const simd_type *NtoMTrans, const simd_type *jac,
    simd_type *wsp0, simd_type *wsp1, [[maybe_unused]] simd_type *wsp2,
    simd_type *out, typename simd_type::scalarType scale = 1.0)

{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        IProductHexKernel<SCALE, APPEND, DEFORMED>(nm0, nm1, nm2, nq0, nq1, nq2,
                                                   in, B0, B1, B2, w0, w1, w2,
                                                   jac, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        IProductTetKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, w0, w1,
            w2, jac, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        const auto nmTot  = nm0 * (nm0 + 1) * (nm0 + 2) / 6;
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductTetKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, w0, w1,
            w2, jac, wsp0, wsp1, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        IProductPrismKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, w0, w1,
            w2, jac, wsp0, wsp1, wsp2, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        const auto nmTot  = nm0 * nm0 * (nm0 + 1) / 2;
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductPrismKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, w0, w1,
            w2, jac, wsp0, wsp1, wsp2, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        IProductPyrKernel<SCALE, APPEND, DEFORMED>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, w0, w1,
            w2, jac, wsp0, wsp1, out, scale);
    }
}

// inner product without quadrature metric wJ
template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          typename simd_type>
NEK_FORCE_INLINE static void IProduct3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const bool isModified, const simd_type *in,
    const simd_type *B0, const simd_type *B1, const simd_type *B2,
    [[maybe_unused]] const simd_type *NtoMTrans, simd_type *wsp0,
    simd_type *wsp1, [[maybe_unused]] simd_type *wsp2, simd_type *out,
    typename simd_type::scalarType scale = 1.0)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        IProductHexKernel<SCALE, APPEND, simd_type>(nm0, nm1, nm2, nq0, nq1,
                                                    nq2, in, B0, B1, B2, wsp0,
                                                    wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        IProductTetKernel<SCALE, APPEND, simd_type>(nm0, nm1, nm2, nq0, nq1,
                                                    nq2, isModified, in, B0, B1,
                                                    B2, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        const auto nmTot  = nm0 * (nm0 + 1) * (nm0 + 2) / 6;
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductTetKernel<SCALE, APPEND, simd_type>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, wsp0,
            wsp1, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        IProductPrismKernel<SCALE, APPEND, simd_type>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, wsp0,
            wsp1, wsp2, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        const auto nmTot  = nm0 * nm0 * (nm0 + 1) / 2;
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductPrismKernel<SCALE, APPEND, simd_type>(
            nm0, nm1, nm2, nq0, nq1, nq2, isModified, in, B0, B1, B2, wsp0,
            wsp1, wsp2, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        IProductPyrKernel<SCALE, APPEND, simd_type>(nm0, nm1, nm2, nq0, nq1,
                                                    nq2, isModified, in, B0, B1,
                                                    B2, wsp0, wsp1, out, scale);
    }
}

} // namespace Nektar::Operators::detail
