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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "StdRegions/Operators/PhysDerivSumFacStdKernels.hpp"

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

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
                // Moving from standard to collapsed coordinates
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
