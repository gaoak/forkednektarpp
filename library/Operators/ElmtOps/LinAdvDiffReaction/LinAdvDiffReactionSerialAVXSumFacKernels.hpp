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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"
#include "ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

#include "ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvectionSegKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, const simd_type *df_ptr, simd_type *deriv0,
    simd_type *out, const typename simd_type::scalarType scale)
{
    simd_type vx, vy, vz, df0, df1, df2, d0;

    // Precompute Laplacian metrics.
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
            df0 = df_ptr[i];
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

    // Precompute Laplacian metrics.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        if (ncoord)
        {
            df4 = df_ptr[4];
            df5 = df_ptr[5];
        }
    }

    // Apply metrics on all quad points.
    for (unsigned int q = 0; q < nq1; ++q)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                      SHAPE_TYPE == LibUtilities::eNodalTri)
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

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                          SHAPE_TYPE == LibUtilities::eNodalTri)
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

    // Precompute Laplacian metrics.
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
        if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                      (SHAPE_TYPE == LibUtilities::NodalPrism))
        {
            h3 = hfac3[r];
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            h3 = hfac3[r];
        }
        else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                           (SHAPE_TYPE == LibUtilities::NodalTet))
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

} // namespace Nektar::Operators::detail
