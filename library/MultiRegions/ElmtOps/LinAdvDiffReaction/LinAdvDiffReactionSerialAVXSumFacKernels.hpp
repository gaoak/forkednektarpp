///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXSumFacKernels.hpp
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
// Description: Serial/AVX sum-factorised kernels of the linearised
// advection-diffusion-reaction operator, one element per SIMD lane.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionSerialAVXSumFacKernels.hpp
 * @brief Kernels of the Serial/AVX sum-factorised linearised
 * advection-diffusion-reaction operator: the advection kernels and the
 * launchers that string the stages together, one element per SIMD lane.
 *
 * @details
 * Only the advection term is specific to this family, so only its
 * kernels live here; the header pulls in the backward-transform,
 * derivative, inner-product and diffusion-metric kernels of the
 * BwdTrans, PhysDeriv, IProductWRTBase and Helmholtz families, which the
 * LinAdvDiffReactionKernelLauncher overloads drive around them. The
 * launcher is the entry point the block operator
 * (LinAdvDiffReactionSerialAVXSumFac.hpp) calls for one element group;
 * its one-, two- and three-dimensional overloads are selected by the
 * number of arguments the block operator expands.
 *
 * Each AddAdvection kernel handles one interleaved element group: it
 * receives the reference-space derivatives that the tensor-derivative
 * stage produced, completes the transformation to Cartesian derivatives
 * -- the collapsed-coordinate factors first, for the shapes that have
 * them, then the element's derivative factors -- dots the result with
 * the advection velocity, and folds the reaction term in by leaving
 *
 *     out = V . grad u + scale * out
 *
 * in the buffer that arrived holding \f$u\f$. The quadrature weights and
 * the Jacobian are not applied here; the Helmholtz diffusion-metric
 * stage that follows takes this buffer as its physical-values argument
 * with a unit coefficient, the reaction term having been folded in
 * already.
 *
 * The velocity components are passed as three separate pointers, of
 * which only the first @p ncoord are dereferenced. The caller always
 * forms all three as component offsets into the block's velocity
 * storage, so when the field holds fewer than three components the
 * trailing pointers are formed past the end of that storage; the
 * kernels never dereference them.
 */

#pragma once

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp>

#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Segment advection kernel: add the advection term to the scaled
 * physical values of one element group.
 *
 * A segment has a single reference direction, so the Cartesian
 * derivative in coordinate direction \f$k\f$ is
 * \f$(\partial\xi_0/\partial x_k)\,\partial u/\partial\xi_0\f$, and no
 * collapsed-coordinate factors are involved.
 *
 * @tparam DEFORMED    Geometry is deformed, so the derivative factors
 *                     vary from point to point.
 * @tparam simd_type   SIMD vector type of the interleaved group.
 *
 * @param   ncoord      Coordinate dimension: how many velocity
 *                      components and derivative factors are used.
 * @param   nq0         Quadrature points of the element.
 * @param   advVel0_ptr Velocity component 0 at the quadrature points.
 * @param   advVel1_ptr Velocity component 1; read only when
 *                      @p ncoord > 1.
 * @param   advVel2_ptr Velocity component 2; read only when
 *                      @p ncoord > 2.
 * @param   df_ptr      Derivative factors
 *                      \f$\partial\xi_0/\partial x_k\f$. When the
 *                      geometry is regular the kernel reads @p ncoord
 *                      values for the whole element, `df_ptr[k]`; when
 *                      it is deformed, `df_ptr[i * ncoord + k]` at
 *                      point i.
 * @param   deriv0      Reference derivatives
 *                      \f$\partial u/\partial\xi_0\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out:
 *                      \f$\mathbf{V}\cdot\nabla u + \mathrm{scale}\,u\f$.
 * @param   scale       Reaction coefficient \f$\lambda\f$.
 */
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvection1DKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, const simd_type *df_ptr, simd_type *deriv0,
    simd_type *out, const typename simd_type::scalarType scale)
{
    simd_type vx, vy, vz, df0, df1, df2, d0;

    // Precompute constant derivative factors.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        if (ncoord > 1)
        {
            df1 = df_ptr[1];
        }
        if (ncoord > 2)
        {
            df2 = df_ptr[2];
        }
    }

    // Apply metrics on all quad points.
    for (unsigned int i = 0; i < nq0; ++i)
    {
        // Set deformed derivative factors.
        if constexpr (DEFORMED)
        {
            df0 = df_ptr[i * ncoord];
            if (ncoord > 1)
            {
                df1 = df_ptr[i * ncoord + 1];
            }
            if (ncoord > 2)
            {
                df2 = df_ptr[i * ncoord + 2];
            }
        }

        // Get advection velocity.
        vx = advVel0_ptr[i];
        if (ncoord > 1)
        {
            vy = advVel1_ptr[i];
        }
        if (ncoord > 2)
        {
            vz = advVel2_ptr[i];
        }

        // Get derivatives.
        d0 = deriv0[i];

        // x-advection * scale.
        simd_type adv = df0 * d0;
        adv *= vx;

        if (ncoord > 1)
        {
            simd_type tmp = df1 * d0;
            adv.fma(vy, tmp);
        }
        if (ncoord > 2)
        {
            simd_type tmp = df2 * d0;
            adv.fma(vz, tmp);
        }

        adv.fma(out[i], simd_type(scale));
        out[i] = adv;
    }
}

/**
 * @brief Two-dimensional advection kernel: add the advection term to
 * the scaled physical values of one element group.
 *
 * For triangles the incoming derivatives are those of the collapsed
 * coordinate system, so the kernel first applies the standard
 * correction with the two factor tables -- the direction-0 derivative
 * is scaled by \f$2/(1-\eta_1)\f$ and folded into the direction-1
 * derivative with weight \f$(1+\eta_0)/2\f$ -- before the derivative
 * factors take it to Cartesian space. Quadrilaterals skip that step.
 * The derivative-factor block holds `2 * ncoord` values per point (or
 * per element, when the geometry is regular), ordered so that the entry
 * for coordinate \f$k\f$ and reference direction \f$d\f$ is at
 * `2 * k + d`.
 *
 * @tparam SHAPE_TYPE  Quad, Tri or NodalTri.
 * @tparam DEFORMED    Geometry is deformed.
 * @tparam simd_type   SIMD vector type of the interleaved group.
 *
 * @param   ncoord      Coordinate dimension (2, or 3 for a surface
 *                      element embedded in three dimensions).
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   hfac0       Collapsed-coordinate factor \f$(1+\eta_0)/2\f$,
 *                      one value per direction-0 point; triangles only.
 * @param   hfac1       Collapsed-coordinate factor
 *                      \f$2/(1-\eta_1)\f$, one value per direction-1
 *                      point; triangles only.
 * @param   advVel0_ptr,advVel1_ptr     Velocity components 0 and 1.
 * @param   advVel2_ptr Velocity component 2; read only when
 *                      @p ncoord == 3.
 * @param   df_ptr      Derivative factors, laid out as described above.
 * @param   deriv0,deriv1   Reference derivatives; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out:
 *                      \f$\mathbf{V}\cdot\nabla u + \mathrm{scale}\,u\f$.
 * @param   scale       Reaction coefficient \f$\lambda\f$.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvection2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    [[maybe_unused]] const simd_type *hfac0,
    [[maybe_unused]] const simd_type *hfac1, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    const simd_type *df_ptr, simd_type *deriv0, simd_type *deriv1,
    simd_type *out, const typename simd_type::scalarType scale)
{
    const auto ndf = 2 * ncoord;
    simd_type vx, vy, vz, df0, df1, df2, df3, df4, df5, d0, d1, h0, h1;

    // Precompute constant derivative factors.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        if (ncoord == 3)
        {
            df4 = df_ptr[4];
            df5 = df_ptr[5];
        }
    }

    // Apply metrics on all quad points.
    for (unsigned int q = 0; q < nq1; ++q)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            h1 = hfac1[q];
        }
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt = q * nq0 + p;

            // Set deformed derivative factors.
            if constexpr (DEFORMED)
            {
                df0 = df_ptr[cnt * ndf];
                df1 = df_ptr[cnt * ndf + 1];
                df2 = df_ptr[cnt * ndf + 2];
                df3 = df_ptr[cnt * ndf + 3];
                if (ncoord == 3)
                {
                    df4 = df_ptr[cnt * ndf + 4];
                    df5 = df_ptr[cnt * ndf + 5];
                }
            }

            // Get advection velocity.
            vx = advVel0_ptr[cnt];
            vy = advVel1_ptr[cnt];
            if (ncoord == 3)
            {
                vz = advVel2_ptr[cnt];
            }

            // Get derivatives.
            d0 = deriv0[cnt];
            d1 = deriv1[cnt];

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                h0 = hfac0[p];
                d0 *= h1;
                d1.fma(d0, h0);
            }

            // x-advection * scale.
            simd_type adv = df0 * d0;
            adv.fma(df1, d1);
            adv *= vx;

            simd_type dy = df2 * d0;
            dy.fma(df3, d1);

            adv.fma(vy, dy);

            if (ncoord == 3)
            {
                simd_type dz = df4 * d0;
                dz.fma(df5, d1);

                adv.fma(vz, dz);
            }

            adv.fma(out[cnt], simd_type(scale));
            out[cnt] = adv;
        }
    }
}

/**
 * @brief Three-dimensional advection kernel: add the advection term to
 * the scaled physical values of one element group.
 *
 * As in two dimensions, the collapsed-coordinate correction comes
 * first, and which of the four factor tables enter depends on the
 * shape: the hexahedron needs none; the prism uses @p hfac0 and
 * @p hfac3; the pyramid adds @p hfac1; the tetrahedron uses all four.
 * The derivative factors then take the corrected derivatives to
 * Cartesian space. There is no coordinate-dimension argument -- a
 * three-dimensional element has coordinate dimension three, so the
 * derivative-factor block is always nine values per point (or per
 * element, when the geometry is regular), with the entry for coordinate
 * \f$k\f$ and reference direction \f$d\f$ at `3 * k + d`.
 *
 * @tparam SHAPE_TYPE  Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED    Geometry is deformed.
 * @tparam simd_type   SIMD vector type of the interleaved group.
 *
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   hfac0       \f$(1+\eta_0)/2\f$, per direction-0 point.
 * @param   hfac1       \f$(1+\eta_1)/2\f$, per direction-1 point.
 * @param   hfac2       \f$2/(1-\eta_1)\f$, per direction-1 point.
 * @param   hfac3       \f$2/(1-\eta_2)\f$, per direction-2 point.
 * @param   advVel0_ptr,advVel1_ptr,advVel2_ptr  Velocity components.
 * @param   df_ptr      Derivative factors, laid out as described above.
 * @param   deriv0,deriv1,deriv2    Reference derivatives; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out:
 *                      \f$\mathbf{V}\cdot\nabla u + \mathrm{scale}\,u\f$.
 * @param   scale       Reaction coefficient \f$\lambda\f$.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvection3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const simd_type *hfac0,
    [[maybe_unused]] const simd_type *hfac1,
    [[maybe_unused]] const simd_type *hfac2,
    [[maybe_unused]] const simd_type *hfac3, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    const simd_type *df_ptr, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2, simd_type *out,
    const typename simd_type::scalarType scale)
{
    constexpr auto ndf = 9;
    simd_type vx, vy, vz;
    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type d0, d1, d2, h0, h1, h2, h3;

    // Precompute constant derivative factors.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        df4 = df_ptr[4];
        df5 = df_ptr[5];
        df6 = df_ptr[6];
        df7 = df_ptr[7];
        df8 = df_ptr[8];
    }

    // Apply metrics on all quad points.
    for (unsigned int r = 0; r < nq2; ++r)
    {
        if constexpr (SHAPE_TYPE != LibUtilities::Hex)
        {
            h3 = hfac3[r];
        }

        for (unsigned int q = 0; q < nq1; ++q)
        {
            if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                h1 = hfac1[q];
            }
            else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                               (SHAPE_TYPE == LibUtilities::NodalTet))
            {
                h1 = hfac1[q];
                h2 = hfac2[q];
            }

            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = r * nq0 * nq1 + q * nq0 + p;

                // Set deformed derivative factors.
                if constexpr (DEFORMED)
                {
                    df0 = df_ptr[cnt * ndf];
                    df1 = df_ptr[cnt * ndf + 1];
                    df2 = df_ptr[cnt * ndf + 2];
                    df3 = df_ptr[cnt * ndf + 3];
                    df4 = df_ptr[cnt * ndf + 4];
                    df5 = df_ptr[cnt * ndf + 5];
                    df6 = df_ptr[cnt * ndf + 6];
                    df7 = df_ptr[cnt * ndf + 7];
                    df8 = df_ptr[cnt * ndf + 8];
                }

                // Get advection velocity.
                vx = advVel0_ptr[cnt];
                vy = advVel1_ptr[cnt];
                vz = advVel2_ptr[cnt];

                // Get derivatives.
                d0 = deriv0[cnt];
                d1 = deriv1[cnt];
                d2 = deriv2[cnt];

                // Chain-rule local to cartesian.
                if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                              (SHAPE_TYPE == LibUtilities::NodalPrism))
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d2.fma(h0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d1 *= h3;
                    d2.fma(h0, d0);
                    d2.fma(h1, d1);
                }
                else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                                   (SHAPE_TYPE == LibUtilities::NodalTet))
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d0 *= h2;
                    d1 *= h3;
                    d2.fma(h0, d0);
                    d2.fma(h1, d1);
                    d1.fma(h0, d0);
                }

                // Advection.
                // u.dx
                simd_type adv = df0 * d0;
                adv.fma(df1, d1);
                adv.fma(df2, d2);
                adv *= vx;

                // v.dy
                simd_type dy = df3 * d0;
                dy.fma(df4, d1);
                dy.fma(df5, d2);
                adv.fma(vy, dy);

                // w.dz
                simd_type dz = df6 * d0;
                dz.fma(df7, d1);
                dz.fma(df8, d2);
                adv.fma(vz, dz);

                adv.fma(out[cnt], simd_type(scale));
                out[cnt] = adv;
            }
        }
    }
}

/**
 * @brief Apply the whole operator to one element group of a 1D block.
 *
 * Unpacks the size parameter -- runtime or compile-time, both satisfying
 * IsSizeParameter1D_v as the static_assert enforces -- and runs the six
 * stages on the group:
 * -# BwdTransKernelLauncher: coefficients to \f$u\f$ at the quadrature
 *    points, in @p bwd;
 * -# PhysDerivTensor1DKernel: the reference derivative
 *    \f$\partial u/\partial\xi_0\f$, in @p deriv0;
 * -# AddAdvection1DKernel: overwrites @p bwd with
 *    \f$\mathbf{V}\cdot\nabla u + \lambda u\f$, applying the chain rule
 *    to the reference derivative as it goes;
 * -# DiffusionCoeffwithWJ1DKernel of the Helmholtz family: replaces
 *    @p deriv0 by the diffusion flux pulled back to the reference
 *    direction and applies the quadrature weights and the Jacobian to
 *    it and to @p bwd, with a unit coefficient since the reaction term
 *    is already in @p bwd; the varying-diffusivity arguments receive
 *    @p nullVec;
 * -# SumDerivTensor1DKernel: contracts the flux against the transposed
 *    derivative matrix and adds it onto @p bwd, which then holds the
 *    whole integrand;
 * -# IProductWRTBaseKernelLauncher without weights or Jacobian (already
 *    applied), writing the output coefficients.
 *
 * @tparam SHAPE_TYPE          Element shape (segments only).
 * @tparam DEFORMED            Jacobian and derivative factors vary point
 *                             by point.
 * @tparam TSizeParameter1D    1D size parameter (see ElmtHelper.hpp).
 * @tparam simd_type           SIMD vector type; one element per lane.
 *
 * @param   sizeParam1D Element sizes: modes and quadrature points.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   ncoord      Coordinate dimension.
 * @param   diffCoeff   Packed diffusivity tensor (see
 *                      LinAdvDiffReactionOp::SetDiffCoeff).
 * @param   nullVec     Empty vector standing in for the
 *                      varying-diffusivity tables.
 * @param   basis0      1D basis table.
 * @param   D0          1D collocation derivative matrix.
 * @param   w0          1D quadrature weights.
 * @param   NtoM        Nodal-to-modal matrix; unused for segments.
 * @param   NtoMTrans   Its transpose; unused for segments.
 * @param   jac         Jacobians of the group.
 * @param   df          Derivative factors of the group.
 * @param   advVel0_ptr Velocity component 0 of the group.
 * @param   advVel1_ptr Velocity component 1; read for embedded segments.
 * @param   advVel2_ptr Velocity component 2; read for embedded segments.
 * @param   deriv0      Quadrature-space buffer for the derivative and
 *                      then the flux.
 * @param   bwd         Quadrature-space buffer for \f$u\f$ and then the
 *                      integrand.
 * @param   in          Coefficients of the group.
 * @param   out         Output coefficients of the group, overwritten.
 * @param   lambda      Reaction coefficient.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter1D sizeParam1D, const bool isModified,
    const unsigned int ncoord, const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *w0, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, const simd_type *df,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, simd_type *deriv0, simd_type *bwd,
    const simd_type *in, simd_type *out,
    const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nq0 = sizeParam1D.nq0();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam1D, isModified, basis0,
                                              NtoM, in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor1DKernel(nq0, bwd, D0, deriv0);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvection1DKernel<DEFORMED>(ncoord, nq0, advVel0_ptr, advVel1_ptr,
                                   advVel2_ptr, df, deriv0, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ1DKernel<SHAPE_TYPE, true, DEFORMED>(
        ncoord, nq0, true, diffCoeff, false, nullVec, nullVec, nullVec, nullVec,
        nullVec, nullVec, jac, w0, df, deriv0, bwd, 1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor1DKernel<true>(nq0, deriv0, D0, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam1D, isModified, bwd, basis0, NtoMTrans, out);
}

/**
 * @brief Apply the whole operator to one element group of a 2D block
 * (quadrilaterals and modal or nodal triangles).
 *
 * As the 1D overload, with the two-dimensional stages: the backward
 * transform and the inner product use the workspace @p wsp0 and, for
 * NodalTri, convert from and back to the nodal coefficients through
 * @p NtoM and @p NtoMTrans; PhysDerivTensor2DKernel leaves the two
 * reference derivatives -- still in collapsed coordinates for triangles
 * -- in @p deriv0 and @p deriv1; AddAdvection2DKernel and
 * DiffusionCoeffwithWJ2DKernel each complete the collapsed-coordinate
 * chain rule with @p f0 and @p f1 before applying the derivative
 * factors; SumDerivTensor2DKernel sums both flux components onto
 * @p bwd.
 *
 * @tparam SHAPE_TYPE          Quad, Tri or NodalTri.
 * @tparam DEFORMED            Jacobian and derivative factors vary point
 *                             by point.
 * @tparam TSizeParameter2D    2D size parameter (see ElmtHelper.hpp).
 * @tparam simd_type           SIMD vector type; one element per lane.
 *
 * @param   sizeParam2D Element sizes: modes and quadrature points per
 *                      direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   ncoord      Coordinate dimension (2, or 3 for a surface
 *                      element embedded in three dimensions).
 * @param   diffCoeff   Packed diffusivity tensor.
 * @param   nullVec     Empty vector standing in for the
 *                      varying-diffusivity tables.
 * @param   basis0,basis1   1D basis tables.
 * @param   D0,D1       1D collocation derivative matrices.
 * @param   w0,w1       1D quadrature weights.
 * @param   f0          Collapsed-coordinate factor \f$(1+\eta_0)/2\f$,
 *                      per direction-0 point; triangles only.
 * @param   f1          Collapsed-coordinate factor \f$2/(1-\eta_1)\f$,
 *                      per direction-1 point; triangles only.
 * @param   NtoM        Nodal-to-modal matrix; NodalTri only.
 * @param   NtoMTrans   Its transpose; NodalTri only.
 * @param   jac         Jacobians of the group.
 * @param   df          Derivative factors of the group.
 * @param   advVel0_ptr,advVel1_ptr     Velocity components 0 and 1.
 * @param   advVel2_ptr Velocity component 2; read when @p ncoord == 3.
 * @param   wsp0        Shape-kernel workspace.
 * @param   deriv0,deriv1   Quadrature-space buffers for the derivatives
 *                          and then the flux.
 * @param   bwd         Quadrature-space buffer for \f$u\f$ and then the
 *                      integrand.
 * @param   in          Coefficients of the group.
 * @param   out         Output coefficients of the group, overwritten.
 * @param   lambda      Reaction coefficient.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter2D sizeParam2D, const bool isModified,
    const unsigned int ncoord, const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1, const simd_type *f0,
    const simd_type *f1, const simd_type *NtoM, const simd_type *NtoMTrans,
    const simd_type *jac, const simd_type *df, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr, simd_type *wsp0,
    simd_type *deriv0, simd_type *deriv1, simd_type *bwd, const simd_type *in,
    simd_type *out, const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam2D, isModified, basis0,
                                              basis1, NtoM, wsp0, in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor2DKernel(nq0, nq1, bwd, D0, D1, deriv0, deriv1);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, f0, f1, advVel0_ptr, advVel1_ptr, advVel2_ptr, df,
        deriv0, deriv1, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ2DKernel<SHAPE_TYPE, true, DEFORMED>(
        ncoord, nq0, nq1, true, diffCoeff, false, nullVec, nullVec, nullVec,
        nullVec, nullVec, nullVec, jac, w0, w1, df, f0, f1, deriv0, deriv1, bwd,
        1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor2DKernel<true>(nq0, nq1, deriv0, deriv1, D0, D1, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam2D, isModified, bwd, basis0, basis1, NtoMTrans, wsp0, out);
}

/**
 * @brief Apply the whole operator to one element group of a 3D block
 * (hexahedra, modal or nodal tetrahedra and prisms, and pyramids).
 *
 * The three-dimensional counterpart of the 2D overload: three
 * workspaces, three derivative buffers and the four collapsed-coordinate
 * factor tables, which AddAdvection3DKernel and
 * DiffusionCoeffwithWJ3DKernel use according to the shape. The
 * coordinate dimension is accepted but not used -- a three-dimensional
 * element has coordinate dimension three.
 *
 * @tparam SHAPE_TYPE          Hex, Tet, NodalTet, Prism, NodalPrism or
 *                             Pyr.
 * @tparam DEFORMED            Jacobian and derivative factors vary point
 *                             by point.
 * @tparam TSizeParameter3D    3D size parameter (see ElmtHelper.hpp).
 * @tparam simd_type           SIMD vector type; one element per lane.
 *
 * @param   sizeParam3D Element sizes: modes and quadrature points per
 *                      direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   ncoord      Coordinate dimension; unused.
 * @param   diffCoeff   Packed diffusivity tensor.
 * @param   nullVec     Empty vector standing in for the
 *                      varying-diffusivity tables.
 * @param   basis0,basis1,basis2    1D basis tables.
 * @param   D0,D1,D2    1D collocation derivative matrices.
 * @param   w0,w1,w2    1D quadrature weights.
 * @param   f0          \f$(1+\eta_0)/2\f$, per direction-0 point.
 * @param   f1          \f$(1+\eta_1)/2\f$, per direction-1 point.
 * @param   f1m         \f$2/(1-\eta_1)\f$, per direction-1 point.
 * @param   f2          \f$2/(1-\eta_2)\f$, per direction-2 point.
 * @param   NtoM        Nodal-to-modal matrix; nodal shapes only.
 * @param   NtoMTrans   Its transpose; nodal shapes only.
 * @param   jac         Jacobians of the group.
 * @param   df          Derivative factors of the group.
 * @param   advVel0_ptr,advVel1_ptr,advVel2_ptr  Velocity components.
 * @param   wsp0,wsp1,wsp2  Shape-kernel workspaces.
 * @param   deriv0,deriv1,deriv2    Quadrature-space buffers for the
 *                                  derivatives and then the flux.
 * @param   bwd         Quadrature-space buffer for \f$u\f$ and then the
 *                      integrand.
 * @param   in          Coefficients of the group.
 * @param   out         Output coefficients of the group, overwritten.
 * @param   lambda      Reaction coefficient.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter3D sizeParam3D, const bool isModified,
    [[maybe_unused]] const unsigned int ncoord,
    const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *NtoM, const simd_type *NtoMTrans, const simd_type *jac,
    const simd_type *df, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr, simd_type *wsp0,
    simd_type *wsp1, simd_type *wsp2, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2, simd_type *bwd, const simd_type *in, simd_type *out,
    const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam3D, isModified, basis0,
                                              basis1, basis2, NtoM, wsp0, wsp1,
                                              in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor3DKernel(nq0, nq1, nq2, bwd, D0, D1, D2, deriv0, deriv1,
                            deriv2);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, advVel0_ptr, advVel1_ptr, advVel2_ptr,
        df, deriv0, deriv1, deriv2, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ3DKernel<SHAPE_TYPE, true, DEFORMED>(
        nq0, nq1, nq2, true, diffCoeff, false, nullVec, nullVec, nullVec,
        nullVec, nullVec, nullVec, jac, w0, w1, w2, df, f0, f1, f1m, f2, deriv0,
        deriv1, deriv2, bwd, 1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor3DKernel<true>(nq0, nq1, nq2, deriv0, deriv1, deriv2, D0, D1,
                                 D2, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam3D, isModified, bwd, basis0, basis1, basis2, NtoMTrans, wsp0,
        wsp1, wsp2, out);
}

} // namespace Nektar::MultiRegions::detail
