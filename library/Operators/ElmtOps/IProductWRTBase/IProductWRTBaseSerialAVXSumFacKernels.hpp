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
// Description: Kernel launchers and workspace sizing of the
// Serial/AVX sum-factorised inner product with the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseSerialAVXSumFacKernels.hpp
 * @brief Kernel launchers and workspace sizing of the Serial/AVX
 * sum-factorised inner product with the basis, one element per SIMD lane.
 *
 * @details
 * The IProductWRTBaseKernelLauncher overloads select the shape-specific
 * kernel -- defined in
 * StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp -- at compile
 * time from their SHAPE_TYPE template parameter, appending the
 * modal-to-nodal conversion for the nodal shapes; the type of the size
 * parameter picks the one-, two- or three-dimensional overload. Each comes
 * in two forms, distinguished by their argument list rather than by a
 * template parameter: the first takes the quadrature weights and the
 * Jacobians and evaluates the full inner product; the second omits both
 * and applies the basis transpose alone, for input that already carries
 * the quadrature metric (see IProductWRTBaseOp::SetIntegration). The
 * IProduct{1,2,3}DWorkspace functions supply the workspace sizes the
 * kernels require; the block operator allocates the workspaces once at
 * construction (see IProductWRTBaseSerialAVXSumFac.hpp). Workspace sizes
 * are counted in SIMD vectors: one vector holds the same value for each
 * element of an interleaved element group.
 *
 * As the note below explains, the kernels are force-inlined rather than
 * duplicated per element size, so that the size-templated OperatorND()
 * instantiations see compile-time loop bounds and unroll. The launchers
 * are shared with the other element operators whose sum-factorised paths
 * end in a basis contraction, which is why they carry an APPEND parameter
 * that IProductWRTBase itself never sets true.
 */

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

/// @brief 1D workspace query: the segment kernel contracts its only
/// direction straight into the output and needs no workspace, so this is a
/// no-op kept for uniformity with the 2D and 3D forms.
NEK_FORCE_INLINE static void IProduct1DWorkspace(
    [[maybe_unused]] LibUtilities::ShapeType SHAPE_TYPE,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nq0)

{
    // Check preconditions
    // None
}

/**
 * @brief Grow @p wsp0Size to the workspace the 2D launcher needs for the
 * given shape and sizes.
 *
 * Updates a running maximum so one workspace can be sized across several
 * queries. Sizes are in SIMD vectors: every 2D shape stages nq1
 * direction-0 sums, one per direction-1 quadrature point, and nodal
 * triangles additionally need nmTot = nm0 * (nm0 + 1) / 2 vectors behind
 * those to hold the modal-basis result before it is mapped to the nodal
 * coefficients (see the 2D IProductWRTBaseKernelLauncher).
 *
 * @param        SHAPE_TYPE  Shape of the block's elements.
 * @param        nm0,nm1     Modes per direction.
 * @param        nq0,nq1     Quadrature points per direction.
 * @param[in,out] wsp0Size   Raised to the required size if smaller.
 */
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

/**
 * @brief Grow @p wsp0Size, @p wsp1Size and @p wsp2Size to the workspaces
 * the 3D launcher needs for the given shape and sizes.
 *
 * As IProduct2DWorkspace, in SIMD vectors and as running maxima. wsp0
 * holds the nq1 * nq2 direction-0 sums and wsp1 the nq2 sums of the second
 * stage; for the nodal shapes wsp0 is enlarged by the shape's mode count
 * to hold the modal-basis result before the nodal mapping. wsp2 (nm1
 * vectors) is only read by the prism kernels, which accumulate their
 * collapsed-edge correction there, one entry per direction-1 mode.
 *
 * @param        SHAPE_TYPE    Shape of the block's elements.
 * @param        nm0,nm1,nm2   Modes per direction.
 * @param        nq0,nq1,nq2   Quadrature points per direction.
 * @param[in,out] wsp0Size     Raised to the required size if smaller.
 * @param[in,out] wsp1Size     Raised to the required size if smaller.
 * @param[in,out] wsp2Size     Raised to the required size if smaller.
 */
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

/**
 * @brief 1D launcher: forward one element group's inner product to the
 * segment kernel, quadrature metric included.
 *
 * @tparam SHAPE_TYPE       Shape tag (segments only in 1D).
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED         @p jac holds one vector per quadrature point
 *                          rather than one per element group.
 * @tparam TSizeParameter1D 1D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type; each vector holds one value of
 *                          every element of the interleaved group.
 *
 * @param   sizeParam1D Element sizes; nm0 and nq0 are read.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   in          Interleaved physical values of the element group.
 * @param   B0          One-dimensional basis table, B0[p * nq0 + i] being
 *                      mode p at point i.
 * @param   w0          Direction-0 quadrature weights.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix; unused in 1D.
 * @param   jac         Interleaved Jacobians of the element group.
 * @param   out         Interleaved coefficients of the element group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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

/**
 * @brief 1D launcher without the quadrature metric: applies the basis
 * transpose alone to one element group; selected by the absence of the
 * weight and Jacobian arguments.
 *
 * @param   sizeParam1D Element sizes; nm0 and nq0 are read.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   in          Interleaved physical values of the element group,
 *                      assumed to carry the quadrature metric already.
 * @param   B0          One-dimensional basis table.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix; unused in 1D.
 * @param   out         Interleaved coefficients of the element group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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

/**
 * @brief 2D launcher: select the quadrilateral or (nodal) triangular kernel
 * for one element group at compile time, quadrature metric included.
 *
 * For NodalTri the triangular kernel writes the modal-basis result to the
 * tail of @p wsp0 (beyond its first nq1 vectors) and MatVecKernel then maps
 * those nmTot values to the nodal coefficients with @p NtoMTrans. For the
 * triangles, @p isModified triggers the extra collapsed-vertex contribution
 * inside IProductTriKernel.
 *
 * @note On NodalTri the @p APPEND flag does not reach @p out. It is handed
 * to IProductTriKernel, which then accumulates into the workspace slice
 * holding the pre-nodal coefficients, a slice nothing zeroes beforehand,
 * and MatVecKernel (LibUtilities/BasicUtils/Utils/UtilsSerialAVXKernels.hpp)
 * stores its result to @p out unconditionally. Quad and Tri apply the flag
 * to @p out. IProductWRTBase itself instantiates the launcher with APPEND
 * false only.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED         @p jac holds one vector per quadrature point.
 * @tparam TSizeParameter2D 2D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam2D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   in          Interleaved physical values of the group.
 * @param   B0          Direction-0 basis table (nm0 rows of nq0).
 * @param   B1          Direction-1 basis table; for the triangles indexed
 *                      by the combined (p,q) mode, one row per mode.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix (nodal shapes
 *                      only).
 * @param   jac         Interleaved Jacobians of the group.
 * @param   wsp0        Group workspace, sized by IProduct2DWorkspace.
 * @param   out         Interleaved coefficients of the group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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

/**
 * @brief 2D launcher without the quadrature metric: applies the basis
 * transpose alone, otherwise identical to the metric overload, including
 * the NodalTri behaviour of @p APPEND noted there.
 *
 * @param   sizeParam2D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   in          Interleaved physical values of the group, assumed
 *                      to carry the quadrature metric already.
 * @param   B0,B1       Basis tables per direction.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix (nodal shapes
 *                      only).
 * @param   wsp0        Group workspace, sized by IProduct2DWorkspace.
 * @param   out         Interleaved coefficients of the group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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

        IProductTriKernel<SCALE, APPEND>(nm0, nm1, nq0, nq1, isModified, in, B0,
                                         B1, wsp0, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        IProductTriKernel<SCALE, APPEND>(nm0, nm1, nq0, nq1, isModified, in, B0,
                                         B1, wsp0, out, scale);
    }
    else
    {
        IProductQuadKernel<SCALE, APPEND>(nm0, nm1, nq0, nq1, in, B0, B1, wsp0,
                                          out, scale);
    }
}

/**
 * @brief 3D launcher: select the hexahedral, (nodal) tetrahedral, (nodal)
 * prismatic or pyramidal kernel for one element group at compile time,
 * quadrature metric included.
 *
 * For the nodal shapes the shape kernel writes the modal-basis result to
 * the tail of @p wsp0 (beyond its first nq1 * nq2 vectors) and
 * MatVecKernel then maps those nmTot values to the nodal coefficients with
 * @p NtoMTrans. @p isModified triggers the collapsed vertex and edge
 * corrections inside the tetrahedral, prismatic and pyramidal kernels.
 *
 * @note On NodalTet and NodalPrism the @p APPEND flag does not reach
 * @p out, for the reason given in the 2D overload: the shape kernel
 * accumulates into the pre-nodal workspace slice and MatVecKernel stores
 * to @p out unconditionally. The other shapes apply the flag to @p out.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED         @p jac holds one vector per quadrature point.
 * @tparam TSizeParameter3D 3D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam3D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   in          Interleaved physical values of the group.
 * @param   B0,B1,B2    Basis tables per direction; along a collapsed
 *                      direction indexed by the shape's combined mode.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix (nodal shapes
 *                      only).
 * @param   jac         Interleaved Jacobians of the group.
 * @param   wsp0,wsp1   Group workspaces of the two contraction stages,
 *                      sized by IProduct3DWorkspace.
 * @param   wsp2        Correction workspace; prisms only.
 * @param   out         Interleaved coefficients of the group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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

/**
 * @brief 3D launcher without the quadrature metric: applies the basis
 * transpose alone, otherwise identical to the metric overload, including
 * the nodal-shape behaviour of @p APPEND noted there.
 *
 * @param   sizeParam3D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   in          Interleaved physical values of the group, assumed
 *                      to carry the quadrature metric already.
 * @param   B0,B1,B2    Basis tables per direction.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix (nodal shapes
 *                      only).
 * @param   wsp0,wsp1   Group workspaces of the two contraction stages.
 * @param   wsp2        Correction workspace; prisms only.
 * @param   out         Interleaved coefficients of the group.
 * @param   scale       Factor applied when @p SCALE is set.
 */
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
        IProductHexKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, in, B0,
                                         B1, B2, wsp0, wsp1, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        IProductTetKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2,
                                         isModified, in, B0, B1, B2, wsp0, wsp1,
                                         out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductTetKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2,
                                         isModified, in, B0, B1, B2, wsp0, wsp1,
                                         outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        IProductPrismKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, in, B0, B1, B2, wsp0,
                                           wsp1, wsp2, out, scale);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        simd_type *outtmp = wsp0 + nq1 * nq2;

        IProductPrismKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, in, B0, B1, B2, wsp0,
                                           wsp1, wsp2, outtmp, scale);
        MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        IProductPyrKernel<SCALE, APPEND>(nm0, nm1, nm2, nq0, nq1, nq2,
                                         isModified, in, B0, B1, B2, wsp0, wsp1,
                                         out, scale);
    }
}

} // namespace Nektar::Operators::detail
