///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialAVXSumFacKernels.hpp
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
// Description: Kernel launchers of the Serial/AVX sum-factorised mass
// operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassSerialAVXSumFacKernels.hpp
 * @brief Kernel launchers of the Serial/AVX sum-factorised mass operator:
 * the two composed families' launchers strung together, one element per
 * SIMD lane.
 *
 * @details
 * The mass operator has no sum-factorised shape kernels of its own on this
 * execution space. The MassKernelLauncher overloads below -- one per
 * dimension, selected by the type of the size parameter -- run, for one
 * element group, BwdTransKernelLauncher (BwdTransSerialAVXSumFacKernels.hpp)
 * into the physical-value buffer and then IProductWRTBaseKernelLauncher
 * (IProductWRTBaseSerialAVXSumFacKernels.hpp) from that buffer into the
 * output, in its form with the quadrature weights and the Jacobians. The
 * transform is instantiated with APPEND false and the inner product with
 * SCALE and APPEND false, so the output is overwritten and the nodal-shape
 * APPEND caveat noted on the IProductWRTBaseKernelLauncher overloads does
 * not arise here. Both launchers stage their directional intermediate sums
 * in the same workspaces, which the block implementation
 * (MassSerialAVXSumFac.hpp) sizes to the larger of the two families'
 * workspace queries, pulled in through the two kernel headers.
 *
 * @see MassDeviceSumFacKernels.hpp and MassDeviceSumFacTOPKernels.hpp for
 * the device counterparts, which compose the same two families' shape
 * kernels inside fused device kernels.
 */

#pragma once

#include "Operators/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

/**
 * @brief 1D launcher: backward transform then inner product with the basis
 * for one element group of segments.
 *
 * @tparam SHAPE_TYPE       Shape tag (segments only in 1D).
 * @tparam DEFORMED         @p jac holds one vector per quadrature point
 *                          rather than one per element group.
 * @tparam TSizeParameter1D 1D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type; each vector holds one value of
 *                          every element of the interleaved group.
 *
 * @param   sizeParam1D Element sizes; nm0 and nq0 are read.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      One-dimensional basis table, basis0[p * nq0 + i]
 *                      being mode p at point i.
 * @param   w0          Direction-0 quadrature weights.
 * @param   NtoM        Nodal-to-modal matrix; unused in 1D.
 * @param   NtoMTrans   Transposed nodal-to-modal matrix; unused in 1D.
 * @param   jac         Interleaved Jacobians of the element group.
 * @param   bwd         Interleaved physical values of the group, written
 *                      by the transform and read by the inner product.
 * @param   in          Interleaved coefficients of the element group.
 * @param   out         Interleaved coefficients of the element group;
 *                      overwritten.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter1D sizeParam1D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *w0, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *bwd,
    const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam1D, isModified, basis0,
                                              NtoM, in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam1D, isModified, bwd, basis0, w0, NtoMTrans, jac, out);
}

/**
 * @brief 2D launcher: backward transform then inner product with the basis
 * for one element group of quadrilaterals or (nodal) triangles.
 *
 * For NodalTri the transform maps the coefficients to the modal basis with
 * @p NtoM before its tensor-product stages and the inner product maps its
 * modal result back with @p NtoMTrans afterwards. For the triangles,
 * @p isModified triggers the extra collapsed-vertex contribution inside
 * the shape kernels.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam DEFORMED         @p jac holds one vector per quadrature point.
 * @tparam TSizeParameter2D 2D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam2D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table; for the triangles indexed
 *                      by the combined (p,q) mode, one row per mode.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   NtoM        Nodal-to-modal matrix (nodal shapes only).
 * @param   NtoMTrans   Its transpose (nodal shapes only).
 * @param   jac         Interleaved Jacobians of the group.
 * @param   wsp0        Group workspace shared by the two stages.
 * @param   bwd         Interleaved physical values of the group, written
 *                      by the transform and read by the inner product.
 * @param   in          Interleaved coefficients of the group.
 * @param   out         Interleaved coefficients of the group; overwritten.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter2D sizeParam2D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *wsp0,
    simd_type *bwd, const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam2D, isModified, basis0,
                                              basis1, NtoM, wsp0, in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam2D, isModified, bwd, basis0, basis1, w0, w1, NtoMTrans, jac,
        wsp0, out);
}

/**
 * @brief 3D launcher: backward transform then inner product with the basis
 * for one element group of hexahedra, (nodal) tetrahedra, (nodal) prisms
 * or pyramids.
 *
 * The nodal mappings and the @p isModified corrections apply as in the 2D
 * overload. The transform uses @p wsp0 and @p wsp1 and the inner product
 * all three workspaces; the block implementation sizes each to the larger
 * of the two stages' demands.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED         @p jac holds one vector per quadrature point.
 * @tparam TSizeParameter3D 3D size parameter, templated or not.
 * @tparam simd_type        SIMD vector type of the interleaved group.
 *
 * @param   sizeParam3D Element sizes: modes and points per direction and
 *                      the total mode count.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction; along a
 *                      collapsed direction indexed by the shape's
 *                      combined mode.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   NtoM        Nodal-to-modal matrix (nodal shapes only).
 * @param   NtoMTrans   Its transpose (nodal shapes only).
 * @param   jac         Interleaved Jacobians of the group.
 * @param   wsp0,wsp1   Group workspaces of the two contraction stages,
 *                      shared by transform and inner product.
 * @param   wsp2        Correction workspace of the inner product; prisms
 *                      only.
 * @param   bwd         Interleaved physical values of the group, written
 *                      by the transform and read by the inner product.
 * @param   in          Interleaved coefficients of the group.
 * @param   out         Interleaved coefficients of the group; overwritten.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter3D sizeParam3D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *wsp0,
    simd_type *wsp1, simd_type *wsp2, simd_type *bwd, const simd_type *in,
    simd_type *out)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam3D, isModified, basis0,
                                              basis1, basis2, NtoM, wsp0, wsp1,
                                              in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam3D, isModified, bwd, basis0, basis1, basis2, w0, w1, w2,
        NtoMTrans, jac, wsp0, wsp1, wsp2, out);
}

} // namespace Nektar::Operators::detail
