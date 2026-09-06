///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlSerialAVXSumFacKernels.hpp
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

// Scalar curl of a two-dimensional vector field, omega = dv/dx - du/dy.
//
// The tensorial derivatives of both components are consumed in a single
// sweep, so the collapsed coordinate correction, the chain rule and the curl
// are applied together rather than in three separate passes.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void Curl2DScalarKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int outdim,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0_0,
    const simd_type *tderiv1_0, const simd_type *tderiv0_1,
    const simd_type *tderiv1_1, simd_type *out)
{
    const unsigned int ndf = 2 * outdim;
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
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                      SHAPE_TYPE == LibUtilities::eNodalTri)
        {
            xfrm1 = f1[j]; // Load 1x
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            simd_type d0u = tderiv0_0[cnt_ji]; // Load 1x
            simd_type d1u = tderiv1_0[cnt_ji]; // Load 1x
            simd_type d0v = tderiv0_1[cnt_ji]; // Load 1x
            simd_type d1v = tderiv1_1[cnt_ji]; // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                          SHAPE_TYPE == LibUtilities::eNodalTri)
            {
                // Moving from standard to collapsed coordinates
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

// Vector curl of a two-dimensional scalar field,
// out = {d(omega)/dy, -d(omega)/dx}.
//
// Both output components come out of the same sweep, so the sign flip does
// not need a pass of its own.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void Curl2DVectorKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int outdim,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0, const simd_type *tderiv1,
    simd_type *out0, simd_type *out1)
{
    const unsigned int ndf = 2 * outdim;
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
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                      SHAPE_TYPE == LibUtilities::eNodalTri)
        {
            xfrm1 = f1[j]; // Load 1x
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            simd_type d0 = tderiv0[cnt_ji]; // Load 1x
            simd_type d1 = tderiv1[cnt_ji]; // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                          SHAPE_TYPE == LibUtilities::eNodalTri)
            {
                // Moving from standard to collapsed coordinates
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

// Curl of a three-dimensional vector field. Used for both passes of the
// curl-curl operator, i.e. omega = curl(u) and out = curl(omega).
//
// The tensorial derivatives of the three components are consumed in a single
// sweep. Only the six off-diagonal physical derivatives are formed, the
// diagonal ones cancel out.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void Curl3DKernel(
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
        if constexpr (SHAPE_TYPE != LibUtilities::eHexahedron)
        {
            xfrm_eta2 = f2[k];
        }

        for (unsigned int j = 0; j < nq1; ++j)
        {
            simd_type xfrm_eta1, xfrm;
            if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
            {
                xfrm_eta1 = f1[j];
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                               SHAPE_TYPE == LibUtilities::eNodalTet)
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
                if constexpr (SHAPE_TYPE != LibUtilities::eHexahedron)
                {
                    const simd_type xfrm_eta0 = f0[i]; // Load 1x

                    auto collapse = [&](simd_type &a0, simd_type &a1,
                                        simd_type &a2) {
                        if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                                      (SHAPE_TYPE == LibUtilities::eNodalPrism))
                        {
                            a0 *= xfrm_eta2;
                            a2.fma(xfrm_eta0, a0);
                        }
                        else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                        {
                            a0 *= xfrm_eta2;
                            a1 *= xfrm_eta2;
                            a2.fma(xfrm_eta0, a0);
                            a2.fma(xfrm_eta1, a1);
                        }
                        else if constexpr (SHAPE_TYPE ==
                                               LibUtilities::eTetrahedron ||
                                           SHAPE_TYPE ==
                                               LibUtilities::eNodalTet)
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
    Curl2DScalarKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, 2, f0, f1, df_ptr,
                                             tderiv_u, tderiv_u + nqTot,
                                             tderiv_v, tderiv_v + nqTot, omega);

    // q = {d(omega_z)/dy, -d(omega_z)/dx}
    PhysDerivTensor2DKernel(nq0, nq1, omega, D0, D1, tderiv_u,
                            tderiv_u + nqTot);
    Curl2DVectorKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, 2, f0, f1, df_ptr, tderiv_u, tderiv_u + nqTot, out0, out1);
}

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
    Curl3DKernel<SHAPE_TYPE, DEFORMED>(
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
    Curl3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, tderiv, tderiv + nqTot,
        tderiv + 2u * nqTot, tderiv + 3u * nqTot, tderiv + 4u * nqTot,
        tderiv + 5u * nqTot, tderiv + 6u * nqTot, tderiv + 7u * nqTot,
        tderiv + 8u * nqTot, out0, out1, out2);
}

} // namespace Nektar::Operators::detail
