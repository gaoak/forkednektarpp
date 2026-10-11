///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionSerialAVXSumFacKernels.hpp
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
// Description: Serial/AVX sum-factorised advection kernels, one element per
// SIMD lane.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionSerialAVXSumFacKernels.hpp
 * @brief Kernels of the Serial/AVX sum-factorised advection operator:
 * tensor-product differentiation, collapsed-coordinate correction,
 * derivative factors and velocity dot product, one element per SIMD lane.
 *
 * @details
 * AdvectionKernelLauncher is the entry point the block operator
 * (AdvectionSerialAVXSumFac.hpp) calls for one element group; its one-,
 * two- and three-dimensional overloads are selected by the type of the
 * size parameter. Each unpacks the sizes, differentiates the group along
 * the reference tensor directions with the PhysDerivTensor kernels
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp) and hands the
 * result to the matching Advection*DKernel, which writes
 * scale * (V . grad u) per quadrature point. For simplex-type shapes the
 * tensor directions are the collapsed coordinates, and the kernels first
 * chain-rule the derivatives back to the standard coordinates using the
 * pointwise factors f0/f1/f1m/f2 -- tabulations of (1 + eta)/2 and
 * 2/(1 - eta) at the quadrature points of the respective direction (see
 * AdvectionSerialAVXSumFac.hpp for how they are fetched).
 */

#pragma once

#include "StdRegions/Operators/PhysDerivSumFacStdKernels.hpp"

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief 1D advection finish: out = scale * du/dxi * sum_k df(k) V_k per
 * quadrature point.
 *
 * A segment has a single reference direction, so the velocity contraction
 * reduces to scaling the one reference derivative by the k-th derivative
 * factor and velocity component for each of the up to three coordinate
 * directions of a possibly embedded segment.
 *
 * @tparam SHAPE_TYPE Element shape (segments only; unused).
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points per element.
 * @param   ncoord      Coordinate dimension, equal to the number of
 *                      derivative factors per point.
 * @param   df_ptr      Derivative factors (ncoord per element group, or
 *                      per point when deformed).
 * @param   advVel0_ptr Velocity component 0 at the quadrature points.
 * @param   advVel1_ptr Velocity component 1; read when ncoord >= 2.
 * @param   advVel2_ptr Velocity component 2; read when ncoord == 3.
 * @param   deriv0      Reference-direction derivative du/dxi.
 * @param   out         Output plane of nq0 vectors.
 * @param   scale       Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename simd_type>
NEK_FORCE_INLINE static void Advection1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const simd_type *df_ptr,
    const simd_type *advVel0_ptr, [[maybe_unused]] const simd_type *advVel1_ptr,
    [[maybe_unused]] const simd_type *advVel2_ptr, simd_type *deriv0,
    simd_type *out, const typename simd_type::scalarType scale)
{
    simd_type df_tmp[3];

    if constexpr (!DEFORMED)
    {
        // unroll very small loops
        df_tmp[0] = df_ptr[0];
        if (ncoord >= 2)
        {
            df_tmp[1] = df_ptr[1];
        }
        if (ncoord == 3)
        {
            df_tmp[2] = df_ptr[2];
        }
    }

    for (unsigned int j = 0; j < nq0; ++j)
    {
        if constexpr (DEFORMED)
        {
            df_tmp[0] = df_ptr[j * ncoord]; // load 1x
            if (ncoord >= 2)
            {
                df_tmp[1] = df_ptr[j * ncoord + 1]; // load 1x
            }
            if (ncoord == 3)
            {
                df_tmp[2] = df_ptr[j * ncoord + 2]; // load 1x
            }
        }

        // Multiply by derivative factors
        simd_type in, tmp;
        in = deriv0[j]; // Load 1x
        // Multiply by derivative factor, and Vx
        tmp = df_tmp[0] * advVel0_ptr[j];
        if (ncoord >= 2)
        {
            // Multiply by derivative factor, and Vy
            tmp.fma(df_tmp[1], advVel1_ptr[j]);
        }
        if (ncoord == 3)
        {
            // Multiply by derivative factor, and Vz
            tmp.fma(df_tmp[2], advVel2_ptr[j]);
        }

        if constexpr (APPEND)
        {
            out[j] += scale * in * tmp;
        }
        else
        {
            out[j] = scale * in * tmp;
        }
    }
}

/**
 * @brief 2D advection finish: chain-rule the tensor-direction derivatives
 * into Cartesian ones and contract with the velocity,
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d per point.
 *
 * For triangles the tensor directions are the collapsed coordinates: d0 is
 * multiplied by f1[j] = 2/(1 - eta_1) and d1 augmented by d0 * f0[i] with
 * f0[i] = (1 + eta_0)/2, recovering the standard-coordinate derivatives
 * before the derivative factors are applied. With @p ncoord == 3 the
 * element is embedded in three-dimensional space and a third
 * factor/velocity pair contributes.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   ncoord      Coordinate dimension, i.e. number of
 *                      gradient/velocity components (2 or 3).
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df_ptr      Derivative factors (2 * ncoord per element group,
 *                      or per point when deformed).
 * @param   advVel0_ptr Velocity component 0 at the quadrature points.
 * @param   advVel1_ptr Velocity component 1 at the quadrature points.
 * @param   advVel2_ptr Velocity component 2; read when ncoord == 3.
 * @param   deriv0      Tensor-direction-0 derivative of the group.
 * @param   deriv1      Tensor-direction-1 derivative of the group.
 * @param   out         Output plane of nq0 * nq1 vectors.
 * @param   scale       Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename simd_type>
NEK_FORCE_INLINE static void Advection2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, [[maybe_unused]] const simd_type *advVel2_ptr,
    simd_type *deriv0, simd_type *deriv1, simd_type *out,
    const typename simd_type::scalarType scale)
{
    auto ndf = 2 * ncoord;
    simd_type vx, vy, df_tmp[6];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];

        if (ncoord == 3)
        {
            df_tmp[4] = df_ptr[4];
            df_tmp[5] = df_ptr[5];
        }
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
            simd_type d0, d1;
            d0 = deriv0[cnt_ji]; // Load 1x
            d1 = deriv1[cnt_ji]; // Load 1x

            // Get advection velocity.
            vx = advVel0_ptr[cnt_ji];
            vy = advVel1_ptr[cnt_ji];

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                // Moving from collapsed to standard coordinates
                simd_type xfrm0 = f0[i]; // Load 1x
                d0 *= xfrm1;
                d1.fma(d0, xfrm0);
            }

            if constexpr (DEFORMED)
            {
                df_tmp[0] = df_ptr[cnt_ji * ndf];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_ptr[cnt_ji * ndf + 3];

                if (ncoord == 3)
                {
                    df_tmp[4] = df_ptr[cnt_ji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ji * ndf + 5];
                }
            }

            // Multiply by derivative factors
            simd_type tmp, tmp0;
            tmp = d0 * df_tmp[0]; // d0 * df0 + d1 * df1
            tmp.fma(d1, df_tmp[1]);
            // Multiply advection vel, Vx
            tmp0 = tmp * vx;

            tmp = d0 * df_tmp[2]; // d0 * df2 + d1 * df3
            tmp.fma(d1, df_tmp[3]);
            // Multiply advection vel, Vy
            tmp0.fma(tmp, vy);

            if (ncoord == 3)
            {
                simd_type vz = advVel2_ptr[cnt_ji];
                tmp          = d0 * df_tmp[4]; // d0 * df4 + d1 * df5
                tmp.fma(d1, df_tmp[5]);
                // Multiply advection vel, Vz
                tmp0.fma(tmp, vz);
            }

            if constexpr (APPEND)
            {
                out[cnt_ji].fma(tmp0, scale);
            }
            else
            {
                out[cnt_ji] = tmp0 * scale;
            }
        }
    }
}

/**
 * @brief 3D advection finish: chain-rule the tensor-direction derivatives
 * into Cartesian ones and contract with the velocity,
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d per point.
 *
 * Hexahedra need no correction and ignore the factor tables. For the other
 * shapes the tensor directions are the collapsed coordinates and are
 * chain-ruled back to standard coordinates before the nine derivative
 * factors are applied:
 * - prisms scale d0 by f2[k] and add f0[i] * d0 to d2;
 * - pyramids scale both d0 and d1 by f2[k] and add
 *   f0[i] * d0 + f1[j] * d1 to d2;
 * - tetrahedra scale d0 by f1m[j] * f2[k] and d1 by f2[k], then add
 *   f0[i] * d0 to d1 and f0[i] * d0 + f1[j] * d1 to d2, where the d1 in
 *   the d2 term is the scaled value taken before that addition.
 *
 * The factor arguments arrive in the push order of the block operator's
 * table vector (m_f[0] .. m_f[3]), which pairs them with the parameter
 * names as follows: @p f0 and @p f1 are the (1 + eta)/2 tabulations of
 * directions 0 and 1, while @p f1m and @p f2 are the 2/(1 - eta)
 * tabulations of directions 1 and 2. Note that @p f1 and @p f1m are
 * therefore two different factors of the same direction, and that the
 * parameter order is not the direction order.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 * @tparam simd_type  SIMD vector type; one element per lane.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   nq2         Quadrature points in direction 2.
 * @param   f0          (1 + eta_0)/2 at the direction-0 points.
 * @param   f1          (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m         2/(1 - eta_1) at the direction-1 points.
 * @param   f2          2/(1 - eta_2) at the direction-2 points.
 * @param   df_ptr      Derivative factors (nine per element group, or per
 *                      point when deformed).
 * @param   advVel0_ptr Velocity component 0 at the quadrature points.
 * @param   advVel1_ptr Velocity component 1 at the quadrature points.
 * @param   advVel2_ptr Velocity component 2 at the quadrature points.
 * @param   deriv0      Tensor-direction-0 derivative of the group.
 * @param   deriv1      Tensor-direction-1 derivative of the group.
 * @param   deriv2      Tensor-direction-2 derivative of the group.
 * @param   out         Output plane of nq0 * nq1 * nq2 vectors.
 * @param   scale       Scalar factor applied to the result.
 *
 * @see Advection3DSumFacKernel in AdvectionDeviceSumFacKernels.hpp for the
 * Device counterpart of the same arithmetic.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename simd_type>
NEK_FORCE_INLINE static void Advection3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    [[maybe_unused]] const simd_type *f1m, [[maybe_unused]] const simd_type *f2,
    const simd_type *df_ptr, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    simd_type *deriv0, simd_type *deriv1, simd_type *deriv2, simd_type *out,
    const typename simd_type::scalarType scale)
{
    constexpr auto ndf = 9;
    simd_type vx, vy, vz, df_tmp[ndf];

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
                simd_type d0, d1, d2;
                d0 = deriv0[cnt_ijk];
                d1 = deriv1[cnt_ijk];
                d2 = deriv2[cnt_ijk];

                // Get advection velocity.
                vx = advVel0_ptr[cnt_ijk];
                vy = advVel1_ptr[cnt_ijk];
                vz = advVel2_ptr[cnt_ijk];

                simd_type xfrm_eta0, tmp;

                // Chain-rule  to construct cartesian  derivatives for non
                // Hex shapes
                if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                              (SHAPE_TYPE == LibUtilities::NodalPrism))
                {
                    d0 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= xfrm_eta2;
                    d1 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                    d2.fma(xfrm_eta1, d1);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                                   SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    d0 *= xfrm;
                    xfrm_eta0 = f0[i] * d0;
                    tmp       = d1 * xfrm_eta2;
                    d1        = xfrm_eta0 + tmp;
                    xfrm_eta0.fma(tmp, xfrm_eta1);
                    d2 += xfrm_eta0;
                }

                if constexpr (DEFORMED)
                {
                    df_tmp[0] = df_ptr[cnt_ijk * ndf];
                    df_tmp[1] = df_ptr[cnt_ijk * ndf + 1];
                    df_tmp[2] = df_ptr[cnt_ijk * ndf + 2];
                    df_tmp[3] = df_ptr[cnt_ijk * ndf + 3];
                    df_tmp[4] = df_ptr[cnt_ijk * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ijk * ndf + 5];
                    df_tmp[6] = df_ptr[cnt_ijk * ndf + 6];
                    df_tmp[7] = df_ptr[cnt_ijk * ndf + 7];
                    df_tmp[8] = df_ptr[cnt_ijk * ndf + 8];
                }

                // Metric for eta_0, xi_1, eta_2
                simd_type tmp0;
                tmp = d0 * df_tmp[0];
                tmp.fma(d1, df_tmp[1]);
                tmp.fma(d2, df_tmp[2]);
                // Multiply advection velocity, Vx
                tmp0 = tmp * vx;

                tmp = d0 * df_tmp[3];
                tmp.fma(d1, df_tmp[4]);
                tmp.fma(d2, df_tmp[5]);
                // Multiply advection velocity, Vy
                tmp0.fma(tmp, vy);

                tmp = d0 * df_tmp[6];
                tmp.fma(d1, df_tmp[7]);
                tmp.fma(d2, df_tmp[8]);
                // Multiply advection velocity, Vz
                tmp0.fma(tmp, vz);

                if constexpr (APPEND)
                {
                    out[cnt_ijk].fma(scale, tmp0);
                }
                else
                {
                    out[cnt_ijk] = scale * tmp0;
                }
            }
        }
    }
}

/**
 * @brief Advect one element group of a 1D block: tensor-product
 * differentiation followed by Advection1DKernel.
 *
 * Unpacks the size parameter -- runtime or compile-time, both satisfying
 * IsPhysSizeParameter1D_v as the static_assert enforces -- forms du/dxi in
 * @p deriv0 with PhysDerivTensor1DKernel and hands it to Advection1DKernel
 * together with the group's derivative factors and velocity.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam1D Element sizes: coordinate dimension and quadrature
 *                      points.
 * @param   D0          1D derivative matrix of the direction.
 * @param   df_ptr      Derivative factors of the group.
 * @param   advVel0_ptr Velocity component 0 of the group.
 * @param   advVel1_ptr Velocity component 1; read for embedded segments.
 * @param   advVel2_ptr Velocity component 2; read for embedded segments.
 * @param   deriv0      Workspace for the reference derivative.
 * @param   in          Input values of the group.
 * @param   out         Output values of the group.
 * @param   scale       Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TPhysSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void AdvectionKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *D0, const simd_type *df_ptr,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, simd_type *deriv0, const simd_type *in,
    simd_type *out, const typename simd_type::scalarType scale)
{
    static_assert(IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter1D or "
                  "TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    // Get the basic derivative.
    PhysDerivTensor1DKernel(nq0, in, D0, deriv0);

    // Calculate physical derivative.
    Advection1DKernel<SHAPE_TYPE, APPEND, DEFORMED>(
        nq0, ncoord, df_ptr, advVel0_ptr, advVel1_ptr, advVel2_ptr, deriv0, out,
        scale);
}

/**
 * @brief Advect one element group of a 2D block: tensor-product
 * differentiation followed by Advection2DKernel.
 *
 * As in 1D, with PhysDerivTensor2DKernel forming the two tensor-direction
 * derivatives in @p deriv0 and @p deriv1 and Advection2DKernel applying
 * the triangle collapsed-coordinate correction (through @p f0 and @p f1),
 * the derivative factors and the velocity contraction.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam2D Element sizes: coordinate dimension, which is 3 for
 *                      elements embedded in three-dimensional space, and
 *                      per-direction quadrature points.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor of direction 0.
 * @param   f1          Collapsed-coordinate factor of direction 1.
 * @param   df_ptr      Derivative factors of the group.
 * @param   advVel0_ptr Velocity component 0 of the group.
 * @param   advVel1_ptr Velocity component 1 of the group.
 * @param   advVel2_ptr Velocity component 2; read for embedded elements.
 * @param   deriv0      Workspace for the direction-0 derivative.
 * @param   deriv1      Workspace for the direction-1 derivative.
 * @param   in          Input values of the group.
 * @param   out         Output values of the group.
 * @param   scale       Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void AdvectionKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    simd_type *deriv0, simd_type *deriv1, const simd_type *in, simd_type *out,
    const typename simd_type::scalarType scale)
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    // Get the basic derivative.
    PhysDerivTensor2DKernel(nq0, nq1, in, D0, D1, deriv0, deriv1);

    // Calculate physical derivative.
    Advection2DKernel<SHAPE_TYPE, APPEND, DEFORMED>(
        nq0, nq1, ncoord, f0, f1, df_ptr, advVel0_ptr, advVel1_ptr, advVel2_ptr,
        deriv0, deriv1, out, scale);
}

/**
 * @brief Advect one element group of a 3D block: tensor-product
 * differentiation followed by Advection3DKernel.
 *
 * Counterpart of the 2D overload for three-dimensional shapes; the size
 * parameter carries no coordinate dimension, since a 3D element always
 * has three. PhysDerivTensor3DKernel forms the three tensor-direction
 * derivatives in @p deriv0 .. @p deriv2 and Advection3DKernel applies the
 * shape-specific collapsed-coordinate correction (through @p f0 .. @p f2),
 * the nine derivative factors and the velocity contraction.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam3D Element sizes: per-direction quadrature points.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   D2          1D derivative matrix of direction 2.
 * @param   f0          (1 + eta_0)/2 at the direction-0 points.
 * @param   f1          (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m         2/(1 - eta_1) at the direction-1 points.
 * @param   f2          2/(1 - eta_2) at the direction-2 points.
 * @param   df_ptr      Derivative factors of the group.
 * @param   advVel0_ptr Velocity component 0 of the group.
 * @param   advVel1_ptr Velocity component 1 of the group.
 * @param   advVel2_ptr Velocity component 2 of the group.
 * @param   deriv0      Workspace for the direction-0 derivative.
 * @param   deriv1      Workspace for the direction-1 derivative.
 * @param   deriv2      Workspace for the direction-2 derivative.
 * @param   in          Input values of the group.
 * @param   out         Output values of the group.
 * @param   scale       Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TPhysSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void AdvectionKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *df_ptr, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    simd_type *deriv0, simd_type *deriv1, simd_type *deriv2,
    const simd_type *in, simd_type *out,
    const typename simd_type::scalarType scale)
{
    static_assert(IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter3D or "
                  "TemplatedPhysSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Get the basic derivative.
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in, D0, D1, D2, deriv0, deriv1,
                            deriv2);

    // Calculate physical derivative.
    Advection3DKernel<SHAPE_TYPE, APPEND, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, advVel0_ptr, advVel1_ptr,
        advVel2_ptr, deriv0, deriv1, deriv2, out, scale);
}

} // namespace Nektar::MultiRegions::detail
