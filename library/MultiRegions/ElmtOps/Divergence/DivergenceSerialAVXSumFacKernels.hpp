///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceSerialAVXSumFacKernels.hpp
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

/**
 * @file DivergenceSerialAVXSumFacKernels.hpp
 * @brief Kernel launchers of the Serial/AVX sum-factorised divergence:
 * per element group, tensor-product differentiation of each vector
 * component followed by the contraction with its row of derivative
 * factors, one element per SIMD lane.
 *
 * @details
 * The divergence is assembled entirely from the PhysDeriv kernels: for
 * each vector component, PhysDerivTensor{1,2,3}DKernel
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp) produces the
 * reference-direction derivatives, and
 * PhysDerivDir{1,2,3}DKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>
 * (PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp, included here)
 * corrects them for a collapsed coordinate where the shape requires
 * it and contracts them with row DIR of the derivative factors, either
 * overwriting or -- with APPEND set -- accumulating onto the output.
 * DivergenceKernelLauncher is the entry point the block operator
 * (DivergenceSerialAVXSumFac.hpp) calls for one element group; its
 * one-, two- and three-dimensional overloads are selected by the type
 * of the size parameter, and each chains those two calls per
 * component, DIR following the component index, so that the output
 * ends up holding \f$\sum_c \partial u_c/\partial x_c\f$.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Differentiate one element group of a 1D block:
 * out = du/dx, with u the single component at @p in0.
 *
 * Unpacks the size parameter -- runtime or compile-time, both
 * satisfying IsPhysSizeParameter1D_v as the static_assert enforces --
 * forms du/dxi in @p wsp0 with PhysDerivTensor1DKernel and scales it
 * with PhysDerivDir1DKernel by the group's derivative factor for
 * direction 0, the first of the ncoord factors each point (deformed)
 * or element (regular) carries, overwriting @p out.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam1D Element sizes: coordinate dimension and
 *                      quadrature points.
 * @param   D0          1D derivative matrix of the direction.
 * @param   df_ptr      Derivative factors of the group.
 * @param   wsp0        Workspace for the reference derivative.
 * @param   in0         Input values of the group.
 * @param   out         Output values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *D0, const simd_type *df_ptr,
    simd_type *wsp0, const simd_type *in0, simd_type *out)
{
    static_assert(IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter1D or "
                  "TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    // du/dx
    PhysDerivTensor1DKernel(nq0, in0, D0, wsp0);
    PhysDerivDir1DKernel<SHAPE_TYPE, false, DEFORMED, 0>(nq0, ncoord, df_ptr,
                                                         wsp0, out);
}

/**
 * @brief Differentiate one element group of a 2D block:
 * out = du/dx + dv/dy, with u and v the components at @p in0 and
 * @p in1.
 *
 * Unpacks the size parameter -- runtime or compile-time, both
 * satisfying IsPhysSizeParameter2D_v as the static_assert enforces --
 * then, for u, forms the two reference-direction derivatives in
 * @p wsp0 and @p wsp1 with PhysDerivTensor2DKernel and hands them to
 * PhysDerivDir2DKernel with DIR = 0, which applies the
 * collapsed-coordinate correction for triangles (using @p f0 and
 * @p f1), contracts with row 0 of the derivative factors and
 * overwrites @p out; the same pair of calls with v, DIR = 1 and APPEND
 * then adds dv/dy.
 *
 * @tparam SHAPE_TYPE           Element shape; selects the
 *                              collapsed-coordinate correction.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam2D Element sizes: coordinate dimension and
 *                      quadrature points per direction.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df_ptr      Derivative factors of the group, 2 * ncoord per
 *                      element group, or per point when deformed.
 * @param   wsp0        Workspace for the direction-0 derivative.
 * @param   wsp1        Workspace for the direction-1 derivative.
 * @param   in0         First component of the group.
 * @param   in1         Second component of the group.
 * @param   out         Output values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, simd_type *wsp0,
    simd_type *wsp1, const simd_type *in0, const simd_type *in1, simd_type *out)
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    // du/dx
    PhysDerivTensor2DKernel(nq0, nq1, in0, D0, D1, wsp0, wsp1);
    PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
        nq0, nq1, ncoord, f0, f1, df_ptr, wsp0, wsp1, out);

    // dv/dy
    PhysDerivTensor2DKernel(nq0, nq1, in1, D0, D1, wsp0, wsp1);
    PhysDerivDir2DKernel<SHAPE_TYPE, true, DEFORMED, 1>(
        nq0, nq1, ncoord, f0, f1, df_ptr, wsp0, wsp1, out);
}

/**
 * @brief Differentiate one element group of a 3D block:
 * out = du/dx + dv/dy + dw/dz, with u, v and w the components at
 * @p in0, @p in1 and @p in2.
 *
 * As in two dimensions, each component in turn is differentiated in
 * the three reference directions by PhysDerivTensor3DKernel into
 * @p wsp0 to @p wsp2 and contracted with its row of derivative factors
 * by PhysDerivDir3DKernel, which also applies the collapsed-coordinate
 * correction of the shape (using @p f0, @p f1, @p f1m and @p f2); the
 * DIR = 0 call overwrites @p out and the DIR = 1, 2 calls set APPEND
 * and accumulate. The element is assumed to have coordinate dimension
 * three, so it carries nine derivative factors.
 *
 * @tparam SHAPE_TYPE           Element shape; selects the
 *                              collapsed-coordinate correction.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam3D Element sizes: quadrature points per direction.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   D2          1D derivative matrix of direction 2.
 * @param   f0          \f$\tfrac{1}{2}(1+\eta_0)\f$ at the direction-0
 *                      points.
 * @param   f1          \f$\tfrac{1}{2}(1+\eta_1)\f$ at the direction-1
 *                      points.
 * @param   f1m         \f$2/(1-\eta_1)\f$ at the direction-1 points.
 * @param   f2          \f$2/(1-\eta_2)\f$ at the direction-2 points.
 *                      All four are unused for hexahedra.
 * @param   df_ptr      Derivative factors of the group, nine per
 *                      element group, or per point when deformed.
 * @param   wsp0        Workspace for the direction-0 derivative.
 * @param   wsp1        Workspace for the direction-1 derivative.
 * @param   wsp2        Workspace for the direction-2 derivative.
 * @param   in0         First component of the group.
 * @param   in1         Second component of the group.
 * @param   in2         Third component of the group.
 * @param   out         Output values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *df_ptr, simd_type *wsp0, simd_type *wsp1, simd_type *wsp2,
    const simd_type *in0, const simd_type *in1, const simd_type *in2,
    simd_type *out)
{
    static_assert(IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter3D or "
                  "TemplatedPhysSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // du/dx
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in0, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);

    // dv/dy
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in1, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, true, DEFORMED, 1>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);

    // dw/dz
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in2, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, true, DEFORMED, 2>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);
}

} // namespace Nektar::MultiRegions::detail
