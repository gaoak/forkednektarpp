///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXSumFacKernels.hpp
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
// Description: Chain-rule kernels and kernel launchers of the
// Serial/AVX sum-factorised physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivSerialAVXSumFacKernels.hpp
 * @brief Chain-rule kernels and kernel launchers of the Serial/AVX
 * sum-factorised physical derivative: collapsed-coordinate correction
 * followed by the geometric chain rule, one element per SIMD lane.
 *
 * @details
 * The sum-factorised block operator first applies the tensorial
 * derivative kernels of StdRegions/Operators/PhysDerivSumFacStdKernels.hpp
 * (PhysDerivTensor{1,2,3}DKernel), which differentiate along each of
 * the element's collapsed reference directions \f$\eta_d\f$. The
 * kernels in this file complete the operation for one interleaved
 * element group:
 * -# for the simplex-type shapes they convert those collapsed
 *    derivatives into derivatives with respect to the standard-element
 *    coordinates \f$\xi_d\f$, using the per-point factors
 *    \f$\tfrac{1}{2}(1+\eta)\f$ and \f$2/(1-\eta)\f$ tabulated in the
 *    data warehouse (BasisDataKey with eHalfMultOnePlusZero and
 *    eTwoOverOneMinusZero); hexahedra, quadrilaterals and segments need
 *    no such correction;
 * -# they contract the result with the element's derivative factors
 *    \f$\partial \xi_d / \partial x_k\f$ to give the Cartesian
 *    derivatives.
 *
 * The derivative factors are laid out point-major -- `ndf = coordDim *
 * dimension` consecutive values per quadrature point for a deformed
 * element, one such set per element for a regular one -- and are
 * addressed as `k * dimension + d`; the DEFORMED template flag selects
 * between the two, hoisting the loads out of the point loop in the
 * regular case.
 *
 * Two forms of each kernel are provided. PhysDeriv{1,2,3}DKernel
 * overwrite all `coordDim` Cartesian components in place in the arrays
 * that carry the standard derivatives on entry; the
 * PhysDerivKernelLauncher overloads at the end of the file chain them
 * behind the tensorial derivative for one element group and are what
 * the block operator (PhysDerivSerialAVXSumFac.hpp) calls.
 * PhysDerivDir{1,2,3}DKernel produce one chosen component DIR from
 * separate input arrays and can accumulate; PhysDeriv itself never
 * calls them, the Divergence kernels (and, in one dimension, the
 * CurlCurl kernels) do.
 *
 * @see PhysDerivDeviceSumFacKernels.hpp for the device counterparts,
 * which apply the same corrections in the same order.
 */

#pragma once

#include "StdRegions/Operators/PhysDerivSumFacStdKernels.hpp"

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief One dimension: scale a segment's standard derivative by the
 * derivative factors, in place, for every Cartesian component.
 *
 * With one standard direction there is nothing to contract and no
 * collapsed-coordinate correction: `out[k][j] = out[0][j] * df[k]`,
 * evaluated from k = ncoord-1 downwards so that `out[0]` -- which is
 * both the input and an output -- is overwritten last.
 *
 * @tparam SHAPE_TYPE Shape tag (segments only in 1D); unused.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam simd_type  SIMD vector type; each vector holds one value of
 *                    every element of the interleaved group.
 *
 * @param   nq0     Quadrature points of the element.
 * @param   ncoord  Coordinate dimension (1, 2 or 3), equal to the
 *                  derivative factors per point.
 * @param   df_ptr  Derivative factors of the element group.
 * @param   out     The `ncoord` Cartesian planes; on entry `out[0]`
 *                  holds the standard derivative, on exit all `ncoord`
 *                  planes hold Cartesian derivatives.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int nq0,
                                               const unsigned int ncoord,
                                               const simd_type *df_ptr,
                                               simd_type *out[3])
{
    simd_type df_tmp[3] = {};

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
        in = out[0][j]; // Load 1x
        if (ncoord == 3)
        {
            tmp       = in * df_tmp[2];
            out[2][j] = tmp;
        }
        if (ncoord >= 2)
        {
            tmp       = in * df_tmp[1];
            out[1][j] = tmp;
        }
        tmp       = in * df_tmp[0];
        out[0][j] = tmp;
    }
}

/**
 * @brief One dimension, single component: scale a segment's standard
 * derivative by the derivative factor of direction DIR.
 *
 * @tparam SHAPE_TYPE Shape tag (segments only in 1D); unused.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce.
 * @tparam simd_type  SIMD vector type of the interleaved group.
 *
 * @param   nq0     Quadrature points of the element.
 * @param   ncoord  Coordinate dimension, equal to the derivative
 *                  factors per point; sets the per-point factor stride.
 * @param   df_ptr  Derivative factors of the element group.
 * @param   in      Standard derivative of the element group.
 * @param   out     Cartesian component DIR.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivDir1DKernel(const unsigned int nq0,
                                                  const unsigned int ncoord,
                                                  const simd_type *df_ptr,
                                                  const simd_type *in,
                                                  simd_type *out)
{
    simd_type df_tmp;

    if constexpr (!DEFORMED)
    {
        // unroll very small loops
        df_tmp = df_ptr[DIR];
    }

    for (unsigned int j = 0; j < nq0; ++j)
    {
        if constexpr (DEFORMED)
        {
            df_tmp = df_ptr[j * ncoord + DIR]; // load 1x
        }

        // Multiply by derivative factors
        if constexpr (APPEND)
        {
            out[j].fma(in[j], df_tmp);
        }
        else
        {
            out[j] = in[j] * df_tmp;
        }
    }
}

/**
 * @brief Two dimensions: correct the collapsed derivatives of a
 * triangle and contract with the derivative factors, in place, for
 * every Cartesian component.
 *
 * For a (nodal) triangle the collapsed-to-standard correction
 * \f[ \frac{\partial u}{\partial \xi_0} =
 *     \frac{2}{1-\eta_1}\frac{\partial u}{\partial \eta_0}, \qquad
 *     \frac{\partial u}{\partial \xi_1} =
 *     \frac{\partial u}{\partial \eta_1}
 *     + \frac{1+\eta_0}{2}\frac{\partial u}{\partial \xi_0} \f]
 * is applied first; quadrilaterals skip it, @p f0 and @p f1 then being
 * unused. Each output component is then
 * `out[k] = d0 * df[2k] + d1 * df[2k+1]`, the third being produced only
 * for an element embedded in three dimensions.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam simd_type  SIMD vector type of the interleaved group.
 *
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   ncoord      Cartesian components to produce (coordDim, 2 or
 *                      3); the derivative factors come in
 *                      `ndf = 2 * ncoord` per point.
 * @param   f0          Table of \f$\tfrac{1}{2}(1+\eta_0)\f$ at the
 *                      direction-0 points (triangles only).
 * @param   f1          Table of \f$2/(1-\eta_1)\f$ at the direction-1
 *                      points (triangles only).
 * @param   df_ptr      Derivative factors of the element group.
 * @param   out         The three Cartesian planes; on entry `out[0]`
 *                      and `out[1]` hold the collapsed derivatives, on
 *                      exit `out[0..ncoord-1]` hold Cartesian ones.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, simd_type *out[3])
{
    const unsigned int ndf = 2 * ncoord;
    simd_type df_tmp[6]    = {};

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
            d0 = out[0][cnt_ji]; // Load 1x
            d1 = out[1][cnt_ji]; // Load 1x

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
            simd_type tmp;
            tmp = d0 * df_tmp[0]; // d0 * df0 + d1 * df1
            tmp.fma(d1, df_tmp[1]);
            out[0][cnt_ji] = tmp;

            tmp = d0 * df_tmp[2]; // d0 * df2 + d1 * df3
            tmp.fma(d1, df_tmp[3]);
            out[1][cnt_ji] = tmp;

            if (ncoord == 3)
            {
                tmp = d0 * df_tmp[4]; // d0 * df4 + d1 * df5
                tmp.fma(d1, df_tmp[5]);
                out[2][cnt_ji] = tmp;
            }
        }
    }
}

/**
 * @brief Two dimensions, single component: as PhysDeriv2DKernel, but
 * reading the collapsed derivatives from separate arrays and producing
 * only the Cartesian component DIR.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce; selects the
 *                    derivative-factor rows `2 * DIR` and `2 * DIR + 1`.
 * @tparam simd_type  SIMD vector type of the interleaved group.
 *
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   ncoord      Cartesian components the factor array holds
 *                      (coordDim); sets the per-point factor stride
 *                      `ndf = 2 * ncoord`.
 * @param   f0,f1       Collapsed-coordinate factor tables, as in
 *                      PhysDeriv2DKernel (triangles only).
 * @param   df_ptr      Derivative factors of the element group.
 * @param   tderiv0     Collapsed derivative along direction 0.
 * @param   tderiv1     Collapsed derivative along direction 1.
 * @param   out         Cartesian component DIR.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivDir2DKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int ncoord,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0, const simd_type *tderiv1,
    simd_type *out)
{
    const unsigned int ndf = 2 * ncoord;
    simd_type df_tmp[2];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[2 * DIR];
        df_tmp[1] = df_ptr[2 * DIR + 1];
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
            d0 = tderiv0[cnt_ji]; // Load 1x
            d1 = tderiv1[cnt_ji]; // Load 1x

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
                df_tmp[0] = df_ptr[cnt_ji * ndf + 2 * DIR];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 2 * DIR + 1];
            }

            simd_type tmp;
            tmp = d0 * df_tmp[0];
            tmp.fma(d1, df_tmp[1]);
            if constexpr (APPEND)
            {
                out[cnt_ji] += tmp;
            }
            else
            {
                out[cnt_ji] = tmp;
            }
        }
    }
}

/**
 * @brief Three dimensions: correct the collapsed derivatives of a
 * non-hexahedral shape and contract with the derivative factors, in
 * place, for all three Cartesian components.
 *
 * The collapsed-to-standard correction is shape specific and uses the
 * four factor tables: a prism scales direction 0 by @p f2 and adds its
 * @p f0 multiple onto direction 2; a pyramid scales directions 0 and 1
 * by @p f2 and adds their @p f0 and @p f1 multiples onto direction 2; a
 * (nodal) tetrahedron scales direction 0 by @p f1m and @p f2 and
 * direction 1 by @p f2, then folds both into directions 1 and 2.
 * Hexahedra need no correction and leave @p f0 -- @p f2 unused. Each
 * output component is then `out[k] = sum_d d_d * df[3k+d]`, the nine
 * derivative factors of a three-dimensional element.
 *
 * @tparam SHAPE_TYPE Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam simd_type  SIMD vector type of the interleaved group.
 *
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   f0          \f$\tfrac{1}{2}(1+\eta_0)\f$ at the direction-0
 *                      points.
 * @param   f1          \f$\tfrac{1}{2}(1+\eta_1)\f$ at the direction-1
 *                      points.
 * @param   f1m         \f$2/(1-\eta_1)\f$ at the direction-1 points.
 * @param   f2          \f$2/(1-\eta_2)\f$ at the direction-2 points.
 * @param   df_ptr      Derivative factors of the element group, nine
 *                      per point (deformed) or per element (regular).
 * @param   out_d0      Collapsed derivative along direction 0 on entry,
 *                      \f$\partial u/\partial x_0\f$ on exit.
 * @param   out_d1      As @p out_d0 for direction 1 and \f$x_1\f$.
 * @param   out_d2      As @p out_d0 for direction 2 and \f$x_2\f$.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    [[maybe_unused]] const simd_type *f1m, [[maybe_unused]] const simd_type *f2,
    const simd_type *df_ptr, simd_type *out_d0, simd_type *out_d1,
    simd_type *out_d2)
{
    constexpr unsigned int ndf = 9;
    simd_type df_tmp[ndf];

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
                d0 = out_d0[cnt_ijk];
                d1 = out_d1[cnt_ijk];
                d2 = out_d2[cnt_ijk];

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
                tmp = d0 * df_tmp[0];
                tmp.fma(d1, df_tmp[1]);
                tmp.fma(d2, df_tmp[2]);
                out_d0[cnt_ijk] = tmp;

                tmp = d0 * df_tmp[3];
                tmp.fma(d1, df_tmp[4]);
                tmp.fma(d2, df_tmp[5]);
                out_d1[cnt_ijk] = tmp;

                tmp = d0 * df_tmp[6];
                tmp.fma(d1, df_tmp[7]);
                tmp.fma(d2, df_tmp[8]);
                out_d2[cnt_ijk] = tmp;
            }
        }
    }
}

/**
 * @brief Three dimensions, single component: as PhysDeriv3DKernel, but
 * reading the collapsed derivatives from separate arrays and producing
 * only the Cartesian component DIR.
 *
 * @tparam SHAPE_TYPE Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce; selects the
 *                    derivative-factor rows `3 * DIR + 0,1,2`.
 * @tparam simd_type  SIMD vector type of the interleaved group.
 *
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables, as in
 *                          PhysDeriv3DKernel.
 * @param   df_ptr          Derivative factors of the element group.
 * @param   tderiv0         Collapsed derivative along direction 0.
 * @param   tderiv1         Collapsed derivative along direction 1.
 * @param   tderiv2         Collapsed derivative along direction 2.
 * @param   out             Cartesian component DIR.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivDir3DKernel(
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    [[maybe_unused]] const simd_type *f1m, [[maybe_unused]] const simd_type *f2,
    const simd_type *df_ptr, const simd_type *tderiv0, const simd_type *tderiv1,
    const simd_type *tderiv2, simd_type *out)
{
    constexpr unsigned int ndf = 9u;
    simd_type df_tmp[3];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[3 * DIR];
        df_tmp[1] = df_ptr[3 * DIR + 1];
        df_tmp[2] = df_ptr[3 * DIR + 2];
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
                d0 = tderiv0[cnt_ijk];
                d1 = tderiv1[cnt_ijk];
                d2 = tderiv2[cnt_ijk];

                simd_type xfrm_eta0, tmp;

                // Chain-rule to construct cartesian  derivatives for non
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
                    df_tmp[0] = df_ptr[cnt_ijk * ndf + 3 * DIR];
                    df_tmp[1] = df_ptr[cnt_ijk * ndf + 3 * DIR + 1];
                    df_tmp[2] = df_ptr[cnt_ijk * ndf + 3 * DIR + 2];
                }

                tmp = d0 * df_tmp[0];
                tmp.fma(d1, df_tmp[1]);
                tmp.fma(d2, df_tmp[2]);
                if constexpr (APPEND)
                {
                    out[cnt_ijk] += tmp;
                }
                else
                {
                    out[cnt_ijk] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Differentiate one element group of a 1D block: the standard
 * derivative into `out[0]`, then all `ncoord` Cartesian derivatives in
 * place.
 *
 * Unpacks the size parameter -- runtime or compile-time, both
 * satisfying IsPhysSizeParameter1D_v as the static_assert enforces --
 * forms du/dxi in `out[0]` with PhysDerivTensor1DKernel and hands the
 * planes to PhysDeriv1DKernel, which scales that derivative by the
 * group's `ncoord` derivative factors into `out[0 .. ncoord-1]`.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam1D Element sizes: coordinate dimension and
 *                      quadrature points.
 * @param   D0          1D differentiation matrix of the direction.
 * @param   df_ptr      Derivative factors of the group.
 * @param   in          Input values of the group.
 * @param   out         The Cartesian output planes of the group;
 *                      `out[0 .. ncoord-1]` are written.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *D0, const simd_type *df_ptr,
    const simd_type *in, simd_type *out[3])
{
    static_assert(IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter1D or "
                  "TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    // Get the basic derivative.
    PhysDerivTensor1DKernel(nq0, in, D0, out[0]);

    // Calculate physical derivative.
    PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, ncoord, df_ptr, out);
}

/**
 * @brief Differentiate one element group of a 2D block: the two
 * collapsed derivatives into `out[0]` and `out[1]`, then all `ncoord`
 * Cartesian derivatives in place.
 *
 * Unpacks the size parameter -- runtime or compile-time, both
 * satisfying IsPhysSizeParameter2D_v as the static_assert enforces --
 * forms the two reference-direction derivatives with
 * PhysDerivTensor2DKernel and hands the planes to PhysDeriv2DKernel,
 * which applies the collapsed-coordinate correction for triangles
 * (using @p f0 and @p f1) and contracts with the derivative factors
 * into `out[0 .. ncoord-1]`.
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
 * @param   D0          1D differentiation matrix of direction 0.
 * @param   D1          1D differentiation matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df_ptr      Derivative factors of the group, `2 * ncoord`
 *                      per element group, or per point when deformed.
 * @param   in          Input values of the group.
 * @param   out         The Cartesian output planes of the group;
 *                      `out[0 .. ncoord-1]` are written.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, const simd_type *in,
    simd_type *out[3])
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    // Get the basic derivative.
    PhysDerivTensor2DKernel(nq0, nq1, in, D0, D1, out[0], out[1]);

    // Calculate physical derivative.
    PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, ncoord, f0, f1, df_ptr,
                                            out);
}

/**
 * @brief Differentiate one element group of a 3D block: the three
 * collapsed derivatives into `out[0 .. 2]`, then the three Cartesian
 * derivatives in place.
 *
 * Unpacks the size parameter -- runtime or compile-time, both
 * satisfying IsPhysSizeParameter3D_v as the static_assert enforces --
 * forms the three reference-direction derivatives with
 * PhysDerivTensor3DKernel and hands the planes to PhysDeriv3DKernel,
 * which applies the collapsed-coordinate correction of the shape
 * (using @p f0, @p f1, @p f1m and @p f2) and contracts with the nine
 * derivative factors. No coordinate dimension is carried: a
 * three-dimensional element always has three of both.
 *
 * @tparam SHAPE_TYPE           Element shape; selects the
 *                              collapsed-coordinate correction.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter (see
 *                              ElmtHelper.hpp).
 * @tparam simd_type            SIMD vector type; one element per lane.
 *
 * @param   sizeParam3D Element sizes: quadrature points per direction.
 * @param   D0          1D differentiation matrix of direction 0.
 * @param   D1          1D differentiation matrix of direction 1.
 * @param   D2          1D differentiation matrix of direction 2.
 * @param   f0          Collapsed-coordinate factor table of
 *                      direction 0.
 * @param   f1          Collapsed-coordinate factor table of
 *                      direction 1.
 * @param   f1m         Second factor table of direction 1, used by the
 *                      tetrahedral correction.
 * @param   f2          Collapsed-coordinate factor table of
 *                      direction 2. All four are unused for hexahedra.
 * @param   df_ptr      Derivative factors of the group, nine per
 *                      element group, or per point when deformed.
 * @param   in          Input values of the group.
 * @param   out         The three Cartesian output planes of the group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *df_ptr, const simd_type *in, simd_type *out[3])
{
    static_assert(IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter3D or "
                  "TemplatedPhysSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Get the basic derivative.
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in, D0, D1, D2, out[0], out[1],
                            out[2]);

    // Calculate physical derivative.
    PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nq2, f0, f1, f1m, f2,
                                            df_ptr, out[0], out[1], out[2]);
}

} // namespace Nektar::MultiRegions::detail
