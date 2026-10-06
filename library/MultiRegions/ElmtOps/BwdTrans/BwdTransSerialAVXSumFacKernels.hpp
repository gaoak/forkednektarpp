///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXSumFacKernels.hpp
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
// Description: Dimension-level dispatch and workspace sizing for the
// Serial/AVX sum-factorised backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransSerialAVXSumFacKernels.hpp
 * @brief Dimension-level dispatch kernels and workspace sizing for the
 * Serial/AVX sum-factorised backward transform.
 *
 * The BwdTransKernelLauncher overloads select the shape-specific
 * kernel -- defined in
 * StdRegions/Operators/BwdTransSumFacStdKernels.hpp -- at compile time
 * from their SHAPE_TYPE template parameter, prepending the
 * nodal-to-modal conversion for the nodal shapes. The
 * BwdTrans{1,2,3}DWorkspace functions supply the workspace sizes those
 * kernels require; the block operator allocates the workspaces once at
 * construction (see BwdTransSerialAVXSumFac.hpp). The other
 * sum-factorised operator families include this header for the same
 * kernels, so their signatures are shared. Workspace sizes are
 * counted in SIMD vectors: one vector holds the same value for each
 * element of an interleaved element group.
 *
 * As the note below explains, the kernels are force-inlined rather than
 * duplicated per element size, so that a call reaching them through a
 * compile-time size parameter sees compile-time loop bounds and unrolls.
 *
 * @see BwdTransDeviceSumFacKernels.hpp for the device form of the same
 * decomposition, expressed over warp lanes instead of SIMD vectors.
 */

#pragma once

#include "StdRegions/Operators/BwdTransSumFacStdKernels.hpp"

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.
/// @brief 1D workspace query: the segment kernel needs no workspace, so
/// this is a no-op kept for uniformity with the 2D and 3D forms.
NEK_FORCE_INLINE static void BwdTrans1DWorkspace(
    [[maybe_unused]] LibUtilities::ShapeType SHAPE_TYPE,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nq0)

{
}

/**
 * @brief Grow @p wsp0Size to the workspace the 2D kernels need for the
 * given shape and sizes.
 *
 * Updates a running maximum so one workspace can be sized across several
 * queries. Sizes are in SIMD vectors: quadrilaterals stage nm1 * nq0
 * direction-0 sums; triangles only nm0 per direction-1 point, the
 * collapsed basis making the first stage per-point; nodal triangles
 * additionally need room behind that for the modal-converted
 * coefficients the launcher writes at offset nm0.
 *
 * @param   SHAPE_TYPE  Shape of the block's elements.
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param[in,out] wsp0Size  Raised to the required size if smaller.
 */
NEK_FORCE_INLINE static void BwdTrans2DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1, unsigned int &wsp0Size)
{
    if (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        wsp0Size = std::max(wsp0Size, nm0 + nm0 * (nm0 + 1) / 2);
    }
    else if (SHAPE_TYPE == LibUtilities::Tri)
    {
        wsp0Size = std::max(wsp0Size, nm0);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm1 * nq0);
    }
}

/**
 * @brief Grow @p wsp0Size and @p wsp1Size to the workspaces the 3D
 * kernels need for the given shape and sizes.
 *
 * As BwdTrans2DWorkspace, in SIMD vectors and as running maxima. The two
 * workspaces hold the intermediates of the first and second
 * sum-factorisation stages respectively; for the nodal shapes wsp0 is
 * enlarged to carry the modal-converted coefficients behind the first
 * stage's own area.
 *
 * @param   SHAPE_TYPE    Shape of the block's elements.
 * @param   nm0,nm1,nm2   Modes per direction.
 * @param   nq0,nq1,nq2   Quadrature points per direction.
 * @param[in,out] wsp0Size    Raised to the required size if smaller.
 * @param[in,out] wsp1Size    Raised to the required size if smaller.
 */
NEK_FORCE_INLINE static void BwdTrans3DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2, unsigned int &wsp0Size,
    unsigned int &wsp1Size)
{
    if (SHAPE_TYPE == LibUtilities::Hex)
    {
        wsp0Size = std::max(wsp0Size, nq0 * nm1 * nm2);
        wsp1Size = std::max(wsp1Size, nq0 * nq1 * nm2);
    }
    else if (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        wsp0Size =
            std::max(wsp0Size, nm0 * nm1 + nm0 * (nm0 + 1) * (nm0 + 2) / 6);
        wsp1Size = std::max(wsp1Size, nm0);
    }
    else if (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        wsp0Size = std::max(wsp0Size, nm0 * nm1 + nm0 * nm0 * (nm0 + 1) / 2);
        wsp1Size = std::max(wsp1Size, nm0);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm0 * nm1);
        wsp1Size = std::max(wsp1Size, nm0);
    }
}

/**
 * @brief 1D dispatch: forward one element group's transform to the
 * segment kernel.
 *
 * @tparam SHAPE_TYPE       Shape tag (segments only in 1D).
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter1D 1D size parameter, in the runtime or the
 *                          compile-time form; with the latter the sizes
 *                          below are compile-time constants and the
 *                          kernel's loops unroll.
 * @tparam simd_type        SIMD vector type; each vector holds one value
 *                          of every element of the interleaved group.
 *
 * @param   sizeParam1D Modal and quadrature sizes of the expansion.
 * @param   isModified  First-direction basis is eModified_A; the segment
 *                      kernel needs no correction, so unused.
 * @param   basis0      1D basis table, `basis0[p * nq0 + i]` = mode p at
 *                      point i.
 * @param   NtoM        Nodal-to-modal matrix; no 1D nodal shape, so
 *                      unused.
 * @param   in          Interleaved coefficients of the element group.
 * @param   out         Interleaved physical values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void BwdTransKernelLauncher(
    const TSizeParameter1D sizeParam1D, [[maybe_unused]] const bool isModified,
    const typename simd_type::scalarType *basis0,
    [[maybe_unused]] const simd_type *NtoM, const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    BwdTransSegKernel<APPEND>(nm0, nq0, basis0, in, out);
}

/**
 * @brief 2D dispatch: select the quadrilateral or (nodal) triangular
 * kernel for one element group at compile time.
 *
 * For NodalTri the coefficients are first mapped to the modified modal
 * basis by MatVecKernel with @p NtoM, using the tail of @p wsp0 beyond
 * its first nm0 vectors as destination; the triangular kernel then runs
 * on the converted coefficients. For Tri, @p isModified triggers the
 * extra collapsed vertex-mode contribution inside BwdTransTriKernel.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter2D 2D size parameter, runtime or compile-time
 *                          form.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam2D Modal and quadrature sizes per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table; for triangles indexed by
 *                      the combined (p,q) mode, one row per mode.
 * @param   NtoM        Nodal-to-modal matrix (nodal shapes only).
 * @param   wsp0        Workspace sized by BwdTrans2DWorkspace.
 * @param   in          Interleaved coefficients of the element group.
 * @param   out         Interleaved physical values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void BwdTransKernelLauncher(
    const TSizeParameter2D sizeParam2D, [[maybe_unused]] const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    [[maybe_unused]] const simd_type *NtoM, simd_type *wsp0,
    const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nm0                    = sizeParam2D.nm0();
    const unsigned int nm1                    = sizeParam2D.nm1();
    [[maybe_unused]] const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0                    = sizeParam2D.nq0();
    const unsigned int nq1                    = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        simd_type *in1 = wsp0 + nm0;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransTriKernel<APPEND>(nm0, nm1, nq0, nq1, isModified, basis0,
                                  basis1, wsp0, in1, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        BwdTransTriKernel<APPEND>(nm0, nm1, nq0, nq1, isModified, basis0,
                                  basis1, wsp0, in, out);
    }
    else
    {
        BwdTransQuadKernel<APPEND>(nm0, nm1, nq0, nq1, basis0, basis1, wsp0, in,
                                   out);
    }
}

/**
 * @brief 3D dispatch: select the hexahedral, (nodal) tetrahedral,
 * (nodal) prismatic or pyramidal kernel for one element group at
 * compile time.
 *
 * As in 2D, the nodal shapes are first mapped to the modified modal
 * basis by MatVecKernel with @p NtoM, the destination sitting behind the
 * shape's own area inside @p wsp0, and @p isModified triggers the
 * collapsed-mode corrections inside the shape kernels.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter3D 3D size parameter, runtime or compile-time
 *                          form.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam3D Modal and quadrature sizes per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table; combined-mode indexed
 *                      for the tetrahedron.
 * @param   basis2      Direction-2 basis table; combined-mode indexed
 *                      for the collapsed shapes.
 * @param   NtoM        Nodal-to-modal matrix (nodal shapes only).
 * @param   wsp0,wsp1   Workspaces sized by BwdTrans3DWorkspace.
 * @param   in          Interleaved coefficients of the element group.
 * @param   out         Interleaved physical values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void BwdTransKernelLauncher(
    const TSizeParameter3D sizeParam3D, [[maybe_unused]] const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    [[maybe_unused]] const simd_type *NtoM, simd_type *wsp0, simd_type *wsp1,
    const simd_type *in, simd_type *out)
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

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        BwdTransHexKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1,
                                  basis2, wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        BwdTransTetKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        simd_type *in1 = wsp0 + nm0 * nm1;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransTetKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in1, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        BwdTransPrismKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                    basis0, basis1, basis2, wsp0, wsp1, in,
                                    out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        simd_type *in1 = wsp0 + nm0 * nm1;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransPrismKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                    basis0, basis1, basis2, wsp0, wsp1, in1,
                                    out);
    }
    else
    {
        BwdTransPyrKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in, out);
    }
}

} // namespace Nektar::MultiRegions::detail
