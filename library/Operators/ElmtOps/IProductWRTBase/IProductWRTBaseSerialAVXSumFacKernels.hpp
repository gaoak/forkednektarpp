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

#include "StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.

NEK_FORCE_INLINE static void IProduct1DWorkspace(
    [[maybe_unused]] LibUtilities::ShapeType SHAPE_TYPE,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nq0)

{
    // Check preconditions
    // None
}

NEK_FORCE_INLINE static void IProduct2DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0, const unsigned int nq1,
    unsigned int &wsp0Size)
{
    if (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        wsp0Size = std::max(wsp0Size, nq1 + nm0 * (nm0 + 1) / 2);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nq1);
    }
}

NEK_FORCE_INLINE static void IProduct3DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    const unsigned int nm1, [[maybe_unused]] const unsigned int nm2,
    [[maybe_unused]] const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, unsigned int &wsp0Size, unsigned int &wsp1Size,
    unsigned int &wsp2Size)
{
    if (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        wsp0Size =
            std::max(wsp0Size, nq1 * nq2 + nm0 * (nm0 + 1) * (nm0 + 2) / 6);
    }
    else if (SHAPE_TYPE == LibUtilities::eNodalPrism)
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
          bool DEFORMED, typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter1D sizeParam1D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    const typename simd_type::scalarType *w0,
    [[maybe_unused]] const simd_type *NtoMTrans, const simd_type *jac,
    simd_type *out, typename simd_type::scalarType scale = 1.0)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    IProductSegKernel<SCALE, APPEND, DEFORMED>(nm0, nq0, in, B0, w0, jac, out,
                                               scale);
}

// inner product without quadrature metric wJ
template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter1D sizeParam1D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    [[maybe_unused]] const simd_type *NtoMTrans, simd_type *out,
    typename simd_type::scalarType scale = 1.0)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    IProductSegKernel<SCALE, APPEND>(nm0, nq0, in, B0, out, scale);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter2D sizeParam2D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    const typename simd_type::scalarType *B1,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    [[maybe_unused]] const simd_type *NtoMTrans, const simd_type *jac,
    simd_type *wsp0, simd_type *out, typename simd_type::scalarType scale = 1.0)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nm0                    = sizeParam2D.nm0();
    const unsigned int nm1                    = sizeParam2D.nm1();
    [[maybe_unused]] const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0                    = sizeParam2D.nq0();
    const unsigned int nq1                    = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
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
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter2D sizeParam2D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    const typename simd_type::scalarType *B1,
    [[maybe_unused]] const simd_type *NtoMTrans, simd_type *wsp0,
    simd_type *out, typename simd_type::scalarType scale = 1.0)

{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nm0                    = sizeParam2D.nm0();
    const unsigned int nm1                    = sizeParam2D.nm1();
    [[maybe_unused]] const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0                    = sizeParam2D.nq0();
    const unsigned int nq1                    = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
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
          bool DEFORMED, typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter3D sizeParam3D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    const typename simd_type::scalarType *B1,
    const typename simd_type::scalarType *B2,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2,
    [[maybe_unused]] const simd_type *NtoMTrans, const simd_type *jac,
    simd_type *wsp0, simd_type *wsp1, [[maybe_unused]] simd_type *wsp2,
    simd_type *out, typename simd_type::scalarType scale = 1.0)

{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    const unsigned int nm0                    = sizeParam3D.nm0();
    const unsigned int nm1                    = sizeParam3D.nm1();
    const unsigned int nm2                    = sizeParam3D.nm2();
    [[maybe_unused]] const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0                    = sizeParam3D.nq0();
    const unsigned int nq1                    = sizeParam3D.nq1();
    const unsigned int nq2                    = sizeParam3D.nq2();
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
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTBaseKernelLauncher(
    const TSizeParameter3D sizeParam3D, [[maybe_unused]] const bool isModified,
    const simd_type *in, const typename simd_type::scalarType *B0,
    const typename simd_type::scalarType *B1,
    const typename simd_type::scalarType *B2,
    [[maybe_unused]] const simd_type *NtoMTrans, simd_type *wsp0,
    simd_type *wsp1, [[maybe_unused]] simd_type *wsp2, simd_type *out,
    typename simd_type::scalarType scale = 1.0)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    const unsigned int nm0                    = sizeParam3D.nm0();
    const unsigned int nm1                    = sizeParam3D.nm1();
    const unsigned int nm2                    = sizeParam3D.nm2();
    [[maybe_unused]] const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0                    = sizeParam3D.nq0();
    const unsigned int nq1                    = sizeParam3D.nq1();
    const unsigned int nq2                    = sizeParam3D.nq2();
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
