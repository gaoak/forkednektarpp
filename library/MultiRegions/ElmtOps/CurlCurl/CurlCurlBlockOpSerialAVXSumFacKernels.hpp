///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlBlockOpSerialAVXSumFacKernels.hpp
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
 * @file CurlCurlBlockOpSerialAVXSumFacKernels.hpp
 * @brief Kernels of the Serial/AVX sum-factorised curl-curl operator:
 * tensor-product differentiation, collapsed-coordinate correction,
 * derivative factors and curl, one element per SIMD lane.
 *
 * @details
 * CurlCurlKernelLauncher is the entry point the block operator
 * (CurlCurlBlockOpSerialAVXSumFac.hpp) calls for one element group; its one-,
 * two- and three-dimensional overloads are selected by the type of the
 * size parameter. The 2D and 3D overloads differentiate the group's
 * components along the reference tensor directions with the
 * PhysDerivTensor kernels
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp), hand the result to
 * a curl kernel that forms the vorticity omega = curl u, differentiate
 * omega the same way and hand that to a curl kernel that writes
 * curl(omega). The curl kernels apply, in one sweep over the quadrature
 * points, the collapsed-to-standard coordinate correction for simplex-type
 * shapes -- the tensor directions being the collapsed coordinates there,
 * the derivatives are first chain-ruled back to the standard coordinates
 * using the pointwise factors f0/f1/f1m/f2, tabulations of (1 + eta)/2 and
 * 2/(1 - eta) at the quadrature points of the respective direction (see
 * CurlCurlBlockOpSerialAVXSumFac.hpp for how they are fetched) -- then the
 * geometric derivative factors, then the curl. Curl3DH1KernelLauncher is
 * the entry point for one plane of a multi-plane 3DH1 block: it forms the
 * plane part of a single curl of three components with the 2D curl
 * kernels.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpSerialAVXSumFacKernels.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Scalar curl of a two-dimensional vector field,
 * omega = d(u1)/dx0 - d(u0)/dx1, from the tensor-direction derivatives of
 * both components, per quadrature point with one element per SIMD lane.
 *
 * For triangles the tensor directions are the collapsed coordinates: each
 * direction-0 derivative is multiplied by f1[j] = 2/(1 - eta_1) and the
 * direction-1 derivative augmented by it times f0[i] = (1 + eta_0)/2,
 * recovering the standard-coordinate derivatives before the derivative
 * factors are applied. The derivative-factor entry k * 2 + d then maps the
 * direction-d derivative into the Cartesian-k gradient component, and the
 * two off-diagonal Cartesian derivatives are combined into omega. All of
 * this happens in a single sweep rather than in separate passes.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   ncoord      Coordinate dimension; sets the stride 2 * ncoord
 *                      between the factor sets of consecutive points.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df_ptr      Derivative factors (2 * ncoord per element group,
 *                      or per point when deformed); the first four are
 *                      read.
 * @param   tderiv0_0   Tensor-direction-0 derivative of component u0.
 * @param   tderiv1_0   Tensor-direction-1 derivative of component u0.
 * @param   tderiv0_1   Tensor-direction-0 derivative of component u1.
 * @param   tderiv1_1   Tensor-direction-1 derivative of component u1.
 * @param   out         Output plane of nq0 * nq1 vectors receiving omega.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void Curl2DScalarSumFacKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int ncoord,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0_0,
    const simd_type *tderiv1_0, const simd_type *tderiv0_1,
    const simd_type *tderiv1_1, simd_type *out)
{
    const unsigned int ndf = 2 * ncoord;
    simd_type df_tmp[4];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
    }

    for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        simd_type xfrm1;
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            xfrm1 = f1[j]; // Load 1x
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            simd_type d0u = tderiv0_0[cnt_ji]; // Load 1x
            simd_type d1u = tderiv1_0[cnt_ji]; // Load 1x
            simd_type d0v = tderiv0_1[cnt_ji]; // Load 1x
            simd_type d1v = tderiv1_1[cnt_ji]; // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                // Moving from collapsed to standard coordinates
                simd_type xfrm0 = f0[i]; // Load 1x
                d0u *= xfrm1;
                d1u.fma(d0u, xfrm0);
                d0v *= xfrm1;
                d1v.fma(d0v, xfrm0);
            }

            // Multiply by derivative factors
            if constexpr (DEFORMED)
            {
                df_tmp[0] = df_ptr[cnt_ji * ndf + 0];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_ptr[cnt_ji * ndf + 3];
            }

            simd_type dvdx = d0v * df_tmp[0];
            dvdx.fma(d1v, df_tmp[1]);
            simd_type dudy = d0u * df_tmp[2];
            dudy.fma(d1u, df_tmp[3]);

            out[cnt_ji] = dvdx - dudy; // Store 1x
        }
    }
}

/**
 * @brief Vector curl of a two-dimensional scalar field,
 * out = ( d(omega)/dx1, -d(omega)/dx0 ), from the tensor-direction
 * derivatives of omega, per quadrature point with one element per SIMD
 * lane.
 *
 * The collapsed-coordinate correction and the derivative factors are
 * applied as in Curl2DScalarSumFacKernel; both output components come out
 * of the same sweep, so the sign flip needs no pass of its own.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   ncoord      Coordinate dimension; sets the stride 2 * ncoord
 *                      between the factor sets of consecutive points.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df_ptr      Derivative factors (2 * ncoord per element group,
 *                      or per point when deformed); the first four are
 *                      read.
 * @param   tderiv0     Tensor-direction-0 derivative of omega.
 * @param   tderiv1     Tensor-direction-1 derivative of omega.
 * @param   out0        First output component, nq0 * nq1 vectors.
 * @param   out1        Second output component, nq0 * nq1 vectors.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void Curl2DVectorSumFacKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int ncoord,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0, const simd_type *tderiv1,
    simd_type *out0, simd_type *out1)
{
    const unsigned int ndf = 2 * ncoord;
    simd_type df_tmp[4];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
    }

    for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        simd_type xfrm1;
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            xfrm1 = f1[j]; // Load 1x
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            simd_type d0 = tderiv0[cnt_ji]; // Load 1x
            simd_type d1 = tderiv1[cnt_ji]; // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                // Moving from collapsed to standard coordinates
                simd_type xfrm0 = f0[i]; // Load 1x
                d0 *= xfrm1;
                d1.fma(d0, xfrm0);
            }

            // Multiply by derivative factors
            if constexpr (DEFORMED)
            {
                df_tmp[0] = df_ptr[cnt_ji * ndf + 0];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_ptr[cnt_ji * ndf + 3];
            }

            simd_type dwdx = d0 * df_tmp[0];
            dwdx.fma(d1, df_tmp[1]);
            simd_type dwdy = d0 * df_tmp[2];
            dwdy.fma(d1, df_tmp[3]);

            out0[cnt_ji] = dwdy;  // Store 1x
            out1[cnt_ji] = -dwdx; // Store 1x
        }
    }
}

/**
 * @brief Curl of a three-dimensional vector field from the
 * tensor-direction derivatives of its three components, per quadrature
 * point with one element per SIMD lane; used for both passes of the
 * curl-curl operator, omega = curl u and out = curl omega.
 *
 * For the simplex-type shapes the tensor directions are the collapsed
 * coordinates and each component's derivatives are first chain-ruled back
 * to the standard coordinates: a prism scales the direction-0 derivative
 * by f2[k] and adds f0[i] times it to direction 2; a pyramid scales
 * directions 0 and 1 by f2[k] and adds f0[i] and f1[j] times them to
 * direction 2; a tetrahedron scales direction 0 by f1m[j] * f2[k],
 * rebuilds direction 1 from f0[i] times that and f2[k] times the old
 * direction-1 derivative, and adds both contributions -- the second
 * weighted by f1[j] -- to direction 2. Hexahedra need no correction. The
 * derivative-factor entry k * 3 + d then maps the direction-d derivative
 * into the Cartesian-k gradient component; only the six off-diagonal
 * Cartesian derivatives are formed, the diagonal ones cancelling out of
 * the curl:
 *     out0 = d(u2)/dx1 - d(u1)/dx2,
 *     out1 = d(u0)/dx2 - d(u2)/dx0,
 *     out2 = d(u1)/dx0 - d(u0)/dx1.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   nq2         Quadrature points in direction 2.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (prisms, pyramids and
 *                      tetrahedra).
 * @param   f1          Collapsed-coordinate factor (1 + eta_1)/2 at the
 *                      direction-1 points (pyramids and tetrahedra).
 * @param   f1m         Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (tetrahedra only).
 * @param   f2          Collapsed-coordinate factor 2/(1 - eta_2) at the
 *                      direction-2 points (prisms, pyramids and
 *                      tetrahedra).
 * @param   df_ptr      Derivative factors (nine per element group, or per
 *                      point when deformed).
 * @param   tderiv0_0   Tensor-direction-0 derivative of component 0.
 * @param   tderiv1_0   Tensor-direction-1 derivative of component 0.
 * @param   tderiv2_0   Tensor-direction-2 derivative of component 0.
 * @param   tderiv0_1   Tensor-direction-0 derivative of component 1.
 * @param   tderiv1_1   Tensor-direction-1 derivative of component 1.
 * @param   tderiv2_1   Tensor-direction-2 derivative of component 1.
 * @param   tderiv0_2   Tensor-direction-0 derivative of component 2.
 * @param   tderiv1_2   Tensor-direction-1 derivative of component 2.
 * @param   tderiv2_2   Tensor-direction-2 derivative of component 2.
 * @param   out0        First output component, nq0 * nq1 * nq2 vectors.
 * @param   out1        Second output component, nq0 * nq1 * nq2 vectors.
 * @param   out2        Third output component, nq0 * nq1 * nq2 vectors.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void Curl3DSumFacKernel(
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    [[maybe_unused]] const simd_type *f1m, [[maybe_unused]] const simd_type *f2,
    const simd_type *df_ptr, const simd_type *tderiv0_0,
    const simd_type *tderiv1_0, const simd_type *tderiv2_0,
    const simd_type *tderiv0_1, const simd_type *tderiv1_1,
    const simd_type *tderiv2_1, const simd_type *tderiv0_2,
    const simd_type *tderiv1_2, const simd_type *tderiv2_2, simd_type *out0,
    simd_type *out1, simd_type *out2)
{
    constexpr unsigned int ndf = 9u;
    simd_type df_tmp[9];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
        df_tmp[4] = df_ptr[4];
        df_tmp[5] = df_ptr[5];
        df_tmp[6] = df_ptr[6];
        df_tmp[7] = df_ptr[7];
        df_tmp[8] = df_ptr[8];
    }

    for (unsigned int k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        simd_type xfrm_eta2;
        if constexpr (SHAPE_TYPE != LibUtilities::Hex)
        {
            xfrm_eta2 = f2[k];
        }

        for (unsigned int j = 0; j < nq1; ++j)
        {
            simd_type xfrm_eta1, xfrm;
            if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                xfrm_eta1 = f1[j];
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                               SHAPE_TYPE == LibUtilities::NodalTet)
            {
                xfrm_eta1 = f1[j];
                xfrm      = f1m[j] * xfrm_eta2;
            }

            for (unsigned int i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                simd_type d0[3], d1[3], d2[3];
                d0[0] = tderiv0_0[cnt_ijk]; // Load 1x
                d1[0] = tderiv1_0[cnt_ijk]; // Load 1x
                d2[0] = tderiv2_0[cnt_ijk]; // Load 1x
                d0[1] = tderiv0_1[cnt_ijk]; // Load 1x
                d1[1] = tderiv1_1[cnt_ijk]; // Load 1x
                d2[1] = tderiv2_1[cnt_ijk]; // Load 1x
                d0[2] = tderiv0_2[cnt_ijk]; // Load 1x
                d1[2] = tderiv1_2[cnt_ijk]; // Load 1x
                d2[2] = tderiv2_2[cnt_ijk]; // Load 1x

                // Chain-rule to construct cartesian derivatives for non Hex
                // shapes. Applied to each component in turn.
                if constexpr (SHAPE_TYPE != LibUtilities::Hex)
                {
                    const simd_type xfrm_eta0 = f0[i]; // Load 1x

                    auto collapse = [&](simd_type &a0, simd_type &a1,
                                        simd_type &a2) {
                        if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                                      (SHAPE_TYPE == LibUtilities::NodalPrism))
                        {
                            a0 *= xfrm_eta2;
                            a2.fma(xfrm_eta0, a0);
                        }
                        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                        {
                            a0 *= xfrm_eta2;
                            a1 *= xfrm_eta2;
                            a2.fma(xfrm_eta0, a0);
                            a2.fma(xfrm_eta1, a1);
                        }
                        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                                           SHAPE_TYPE == LibUtilities::NodalTet)
                        {
                            a0 *= xfrm;
                            simd_type tmp0 = xfrm_eta0 * a0;
                            simd_type tmp1 = a1 * xfrm_eta2;
                            a1             = tmp0 + tmp1;
                            tmp0.fma(tmp1, xfrm_eta1);
                            a2 += tmp0;
                        }
                    };

                    collapse(d0[0], d1[0], d2[0]);
                    collapse(d0[1], d1[1], d2[1]);
                    collapse(d0[2], d1[2], d2[2]);
                }

                // Multiply by derivative factors
                if constexpr (DEFORMED)
                {
                    df_tmp[0] = df_ptr[cnt_ijk * ndf + 0];
                    df_tmp[1] = df_ptr[cnt_ijk * ndf + 1];
                    df_tmp[2] = df_ptr[cnt_ijk * ndf + 2];
                    df_tmp[3] = df_ptr[cnt_ijk * ndf + 3];
                    df_tmp[4] = df_ptr[cnt_ijk * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ijk * ndf + 5];
                    df_tmp[6] = df_ptr[cnt_ijk * ndf + 6];
                    df_tmp[7] = df_ptr[cnt_ijk * ndf + 7];
                    df_tmp[8] = df_ptr[cnt_ijk * ndf + 8];
                }

                // Physical derivative of component c in direction m. Only
                // the six off-diagonal entries are needed, the diagonal ones
                // cancel out.
                simd_type dudy = d0[0] * df_tmp[3];
                dudy.fma(d1[0], df_tmp[4]);
                dudy.fma(d2[0], df_tmp[5]);

                simd_type dudz = d0[0] * df_tmp[6];
                dudz.fma(d1[0], df_tmp[7]);
                dudz.fma(d2[0], df_tmp[8]);

                simd_type dvdx = d0[1] * df_tmp[0];
                dvdx.fma(d1[1], df_tmp[1]);
                dvdx.fma(d2[1], df_tmp[2]);

                simd_type dvdz = d0[1] * df_tmp[6];
                dvdz.fma(d1[1], df_tmp[7]);
                dvdz.fma(d2[1], df_tmp[8]);

                simd_type dwdx = d0[2] * df_tmp[0];
                dwdx.fma(d1[2], df_tmp[1]);
                dwdx.fma(d2[2], df_tmp[2]);

                simd_type dwdy = d0[2] * df_tmp[3];
                dwdy.fma(d1[2], df_tmp[4]);
                dwdy.fma(d2[2], df_tmp[5]);

                out0[cnt_ijk] = dwdy - dvdz; // Store 1x
                out1[cnt_ijk] = dudz - dwdx; // Store 1x
                out2[cnt_ijk] = dvdx - dudy; // Store 1x
            }
        }
    }
}

/**
 * @brief 1D entry point: the curl-curl is undefined on a segment, so the
 * group's single component is differentiated once -- PhysDerivTensor1DKernel
 * followed by PhysDerivDir1DKernel with the direction-0 derivative factor
 * -- and the derivative written to the output.
 *
 * Exists so that the generated segment ShapeBlock() compiles; the block
 * operator's constructor asserts that the block is two- or
 * three-dimensional (see CurlCurlBlockOpSerialAVXSumFac.hpp).
 *
 * @tparam SHAPE_TYPE           Element shape (segments only).
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter, runtime
 *                              or compile-time (checked by static_assert).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam1D Element sizes: coordinate dimension and quadrature
 *                      points.
 * @param   D0          1D derivative matrix of the direction.
 * @param   df_ptr      Derivative factors of the group.
 * @param   wsp         Workspace of at least nq0 vectors for the
 *                      tensor-direction derivative.
 * @param   in0         Input values of the group.
 * @param   out0        Output values of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void CurlCurlKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *D0, const simd_type *df_ptr,
    simd_type *wsp, const simd_type *in0, simd_type *out0)
{
    static_assert(IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter1D or "
                  "TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    // The curl of a scalar field on a segment reduces to its derivative.
    PhysDerivTensor1DKernel(nq0, in0, D0, wsp);
    PhysDerivDir1DKernel<SHAPE_TYPE, false, DEFORMED, 0>(nq0, ncoord, df_ptr,
                                                         wsp, out0);
}

/**
 * @brief 2D entry point: evaluate curl(curl u) for one element group.
 *
 * Lays the workspace out as the two tensor-direction derivatives of u0,
 * the two of u1 and one plane for omega -- five planes of nqTot vectors,
 * the count NumWorkspace() in the block operator provides. Differentiates
 * both components with PhysDerivTensor2DKernel, forms omega with
 * Curl2DScalarSumFacKernel, differentiates omega into the planes of u0 and
 * writes the two output components with Curl2DVectorSumFacKernel. The
 * coordinate dimension passed to the curl kernels is two, so the factor
 * sets are read at a stride of four.
 *
 * @tparam SHAPE_TYPE           Element shape; selects the
 *                              collapsed-coordinate correction.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter, runtime
 *                              or compile-time (checked by static_assert).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam2D Element sizes: coordinate dimension and
 *                      per-direction quadrature points.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor of direction 0.
 * @param   f1          Collapsed-coordinate factor of direction 1.
 * @param   df_ptr      Derivative factors of the group.
 * @param   wsp         Workspace of at least 5 * nqTot vectors.
 * @param   in0         Input component u0 of the group.
 * @param   in1         Input component u1 of the group.
 * @param   out0        First output component of the group.
 * @param   out1        Second output component of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void CurlCurlKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, simd_type *wsp,
    const simd_type *in0, const simd_type *in1, simd_type *out0,
    simd_type *out1)
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    // Tensorial derivatives of both components.
    simd_type *tderiv_u = wsp;
    simd_type *tderiv_v = tderiv_u + 2u * nqTot;
    simd_type *omega    = tderiv_v + 2u * nqTot;

    PhysDerivTensor2DKernel(nq0, nq1, in0, D0, D1, tderiv_u, tderiv_u + nqTot);
    PhysDerivTensor2DKernel(nq0, nq1, in1, D0, D1, tderiv_v, tderiv_v + nqTot);

    // omega_z = dv/dx - du/dy
    Curl2DScalarSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, 2, f0, f1, df_ptr, tderiv_u, tderiv_u + nqTot, tderiv_v,
        tderiv_v + nqTot, omega);

    // q = {d(omega_z)/dy, -d(omega_z)/dx}
    PhysDerivTensor2DKernel(nq0, nq1, omega, D0, D1, tderiv_u,
                            tderiv_u + nqTot);
    Curl2DVectorSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, 2, f0, f1, df_ptr, tderiv_u, tderiv_u + nqTot, out0, out1);
}

// One curl's plane part on a 3DH1 element: three components in, three out.
//
// The curl of a 3DH1 field splits into a part that differentiates in the
// plane, (df_z/dy, -df_z/dx, df_y/dx - df_x/dy), and a part that
// differentiates along z. This writes the first; the operator's z-op adds the
// second, and CurlCurlOpImpl takes the pair twice, the output carrying omega
// in between. Both halves of
// the plane part are already in the 2D kernels: the scalar one is the third
// component, taken from the derivatives of the first two, and the vector one
// is the first two components, taken from the derivatives of the third.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void Curl3DH1KernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, simd_type *wsp,
    const simd_type *in0, const simd_type *in1, const simd_type *in2,
    simd_type *out0, simd_type *out1, simd_type *out2)
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    // Tensorial derivatives of the first two components, then of the third
    // once the first two have been consumed.
    simd_type *tderiv_u = wsp;
    simd_type *tderiv_v = tderiv_u + 2u * nqTot;

    PhysDerivTensor2DKernel(nq0, nq1, in0, D0, D1, tderiv_u, tderiv_u + nqTot);
    PhysDerivTensor2DKernel(nq0, nq1, in1, D0, D1, tderiv_v, tderiv_v + nqTot);

    // The third component, df_y/dx - df_x/dy.
    Curl2DScalarSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, 2, f0, f1, df_ptr, tderiv_u, tderiv_u + nqTot, tderiv_v,
        tderiv_v + nqTot, out2);

    // The first two components, {df_z/dy, -df_z/dx}.
    PhysDerivTensor2DKernel(nq0, nq1, in2, D0, D1, tderiv_u, tderiv_u + nqTot);
    Curl2DVectorSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, 2, f0, f1, df_ptr, tderiv_u, tderiv_u + nqTot, out0, out1);
}

/**
 * @brief 3D entry point: evaluate curl(curl u) for one element group.
 *
 * Lays the workspace out as the nine tensor-direction derivatives -- three
 * per component -- followed by the three planes of omega: twelve planes of
 * nqTot vectors, the count NumWorkspace() provides. Differentiates the
 * three components with PhysDerivTensor3DKernel, forms omega with
 * Curl3DSumFacKernel, differentiates the components of omega into the same
 * nine planes and writes the three output components with a second
 * Curl3DSumFacKernel call. Each tensor-direction derivative is thus formed
 * once and reused for every Cartesian derivative that needs it.
 *
 * @tparam SHAPE_TYPE           Element shape; selects the
 *                              collapsed-coordinate correction.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter, runtime
 *                              or compile-time (checked by static_assert).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam3D Element sizes: per-direction quadrature points.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   D2          1D derivative matrix of direction 2.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 of
 *                      direction 0.
 * @param   f1          Collapsed-coordinate factor (1 + eta_1)/2 of
 *                      direction 1.
 * @param   f1m         Collapsed-coordinate factor 2/(1 - eta_1) of
 *                      direction 1.
 * @param   f2          Collapsed-coordinate factor 2/(1 - eta_2) of
 *                      direction 2.
 * @param   df_ptr      Derivative factors of the group.
 * @param   wsp         Workspace of at least 12 * nqTot vectors.
 * @param   in0         Input component u0 of the group.
 * @param   in1         Input component u1 of the group.
 * @param   in2         Input component u2 of the group.
 * @param   out0        First output component of the group.
 * @param   out1        Second output component of the group.
 * @param   out2        Third output component of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void CurlCurlKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *df_ptr, simd_type *wsp, const simd_type *in0,
    const simd_type *in1, const simd_type *in2, simd_type *out0,
    simd_type *out1, simd_type *out2)
{
    static_assert(IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter3D or "
                  "TemplatedPhysSizeParameter3D.");

    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = sizeParam3D.nqTot();

    // Tensorial derivatives of the three components. Each one is evaluated
    // once and reused for every physical direction that needs it.
    simd_type *tderiv = wsp;
    simd_type *omega  = tderiv + 9u * nqTot;

    PhysDerivTensor3DKernel(nq0, nq1, nq2, in0, D0, D1, D2, tderiv,
                            tderiv + nqTot, tderiv + 2u * nqTot);
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in1, D0, D1, D2, tderiv + 3u * nqTot,
                            tderiv + 4u * nqTot, tderiv + 5u * nqTot);
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in2, D0, D1, D2, tderiv + 6u * nqTot,
                            tderiv + 7u * nqTot, tderiv + 8u * nqTot);

    // omega = curl(u)
    Curl3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, tderiv, tderiv + nqTot,
        tderiv + 2u * nqTot, tderiv + 3u * nqTot, tderiv + 4u * nqTot,
        tderiv + 5u * nqTot, tderiv + 6u * nqTot, tderiv + 7u * nqTot,
        tderiv + 8u * nqTot, omega, omega + nqTot, omega + 2u * nqTot);

    // Tensorial derivatives of omega.
    PhysDerivTensor3DKernel(nq0, nq1, nq2, omega, D0, D1, D2, tderiv,
                            tderiv + nqTot, tderiv + 2u * nqTot);
    PhysDerivTensor3DKernel(nq0, nq1, nq2, omega + nqTot, D0, D1, D2,
                            tderiv + 3u * nqTot, tderiv + 4u * nqTot,
                            tderiv + 5u * nqTot);
    PhysDerivTensor3DKernel(nq0, nq1, nq2, omega + 2u * nqTot, D0, D1, D2,
                            tderiv + 6u * nqTot, tderiv + 7u * nqTot,
                            tderiv + 8u * nqTot);

    // out = curl(omega)
    Curl3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, tderiv, tderiv + nqTot,
        tderiv + 2u * nqTot, tderiv + 3u * nqTot, tderiv + 4u * nqTot,
        tderiv + 5u * nqTot, tderiv + 6u * nqTot, tderiv + 7u * nqTot,
        tderiv + 8u * nqTot, out0, out1, out2);
}

} // namespace Nektar::MultiRegions::detail
