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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

#include "StdRegions/Operators/PhysDerivSumFacStdKernels.hpp"

namespace Nektar::Operators::detail
{

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void PhysDeriv1DKernel(const unsigned int nq0,
                                        const unsigned int ndf,
                                        const simd_type *df_ptr,
                                        simd_type *out[3])
{
    simd_type df_tmp[3];

    if constexpr (!DEFORMED)
    {
        // unroll very small loops
        df_tmp[0] = df_ptr[0];
        if (ndf >= 2)
        {
            df_tmp[1] = df_ptr[1];
        }
        if (ndf == 3)
        {
            df_tmp[2] = df_ptr[2];
        }
    }

    for (unsigned int j = 0; j < nq0; ++j)
    {
        if constexpr (DEFORMED)
        {
            df_tmp[0] = df_ptr[j * ndf]; // load 1x
            if (ndf >= 2)
            {
                df_tmp[1] = df_ptr[j * ndf + 1]; // load 1x
            }
            if (ndf == 3)
            {
                df_tmp[2] = df_ptr[j * ndf + 2]; // load 1x
            }
        }

        // Multiply by derivative factors
        simd_type in, tmp;
        in = out[0][j]; // Load 1x
        if (ndf == 3)
        {
            tmp       = in * df_tmp[2]; // Store 1x
            out[2][j] = tmp;
        }
        if (ndf >= 2)
        {
            tmp       = in * df_tmp[1]; // Store 1x
            out[1][j] = tmp;
        }
        tmp       = in * df_tmp[0]; // Store 1x
        out[0][j] = tmp;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int outdim,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, simd_type *out[3])
{
    auto ndf = 2 * outdim;
    simd_type df_tmp[6];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];

        if (outdim == 3)
        {
            df_tmp[4] = df_ptr[4];
            df_tmp[5] = df_ptr[5];
        }
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
            simd_type d0, d1;
            d0 = out[0][cnt_ji]; // Load 1x
            d1 = out[1][cnt_ji]; // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                          SHAPE_TYPE == LibUtilities::eNodalTri)
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

                if (outdim == 3)
                {
                    df_tmp[4] = df_ptr[cnt_ji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ji * ndf + 5];
                }
            }

            // Multiply by derivative factors
            simd_type tmp;
            tmp = d0 * df_tmp[0]; // d0 * df0 + d1 * df1
            tmp.fma(d1, df_tmp[1]);
            out[0][cnt_ji] = tmp; // Store 1x

            tmp = d0 * df_tmp[2]; // d0 * df2 + d1 * df3
            tmp.fma(d1, df_tmp[3]);
            out[1][cnt_ji] = tmp; // Store 1x

            if (outdim == 3)
            {
                tmp = d0 * df_tmp[4]; // d0 * df4 + d1 * df5
                tmp.fma(d1, df_tmp[5]);
                out[2][cnt_ji] = tmp; // Store 1x
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    [[maybe_unused]] const simd_type *f1m, [[maybe_unused]] const simd_type *f2,
    const simd_type *df_ptr,
    [[maybe_unused]] simd_type *wsp0, // Tets only
    [[maybe_unused]] simd_type *wsp1, // Tets only
    simd_type *out_d0, simd_type *out_d1, simd_type *out_d2)
{
    constexpr auto ndf = 9;
    simd_type df_tmp[ndf];

    if constexpr ((SHAPE_TYPE == LibUtilities::eTetrahedron) ||
                  (SHAPE_TYPE == LibUtilities::eNodalTet))
    {
        for (unsigned int k = 0, eta0 = 0; k < nq2; ++k)
        {
            simd_type xfrm_eta2 = f2[k]; // Load 1x

            for (unsigned int j = 0; j < nq1; ++j)
            {
                simd_type xfrm_eta1 = f1m[j]; // Load 1x
                simd_type xfrm      = xfrm_eta1 * xfrm_eta2;
                for (unsigned int i = 0; i < nq0; ++i, ++eta0)
                {
                    simd_type d0;
                    d0 = out_d0[eta0]; // Load 1x
                    d0 *= xfrm;        // Load 1x
                    out_d0[eta0] = d0; // Store 1x
                    wsp0[eta0]   = d0; // Store 1x partial form for reuse
                }
            }
        }

        for (unsigned int k = 0, eta0 = 0; k < nq2; ++k)
        {
            simd_type xfrm_eta2 = f2[k]; // Load 1x

            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int i = 0; i < nq0; ++i, ++eta0)
                {
                    simd_type xfrm_eta0 = f0[i]; // Load 1x

                    simd_type out0 = xfrm_eta0 * wsp0[eta0]; // Load 1x
                    wsp0[eta0]     = out0; // 2 * (1 + eta_0) / (1 -
                                           // eta_1)(1-eta2) | store 1x
                    simd_type d1;
                    d1 = out_d1[eta0]; // Load 1x
                    d1 *= xfrm_eta2;
                    wsp1[eta0] = d1; // Store 1x partial form for reuse
                    d1 += out0;
                    out_d1[eta0] = d1; // Store 1x
                }
            }
        }

        for (unsigned int k = 0, eta0 = 0; k < nq2; ++k)
        {
            for (unsigned int j = 0; j < nq1; ++j)
            {
                simd_type xfrm_eta1 = f1[j]; // Load 1x

                for (unsigned int i = 0; i < nq0; ++i, ++eta0)
                {
                    simd_type out = wsp0[eta0]; // Load 1x
                    simd_type d1  = wsp1[eta0]; // Load 1x
                    out.fma(d1, xfrm_eta1);
                    d1 = out_d2[eta0]; // Load 1x
                    d1 += out;
                    out_d2[eta0] = d1; // Store 1x
                }
            }
        }
    }

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
        if constexpr (SHAPE_TYPE == LibUtilities::ePrism ||
                      SHAPE_TYPE == LibUtilities::eNodalPrism ||
                      SHAPE_TYPE == LibUtilities::ePyramid)
        {
            xfrm_eta2 = f2[k]; // Load 1x
        }

        for (unsigned int j = 0; j < nq1; ++j)
        {
            simd_type xfrm_eta1;
            if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
            {
                xfrm_eta1 = f1[j]; // Load 1x
            }
            for (unsigned int i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                simd_type d0, d1, d2;
                d0 = out_d0[cnt_ijk]; // Load 1x
                d1 = out_d1[cnt_ijk]; // Load 1x
                d2 = out_d2[cnt_ijk]; // Load 1x

                simd_type xfrm_eta0;
                if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                              (SHAPE_TYPE == LibUtilities::eNodalPrism))
                {
                    // Chain-rule for eta_0 and eta_2
                    d0 *= xfrm_eta2; // Load 1x

                    xfrm_eta0 = f0[i]; // Load 1x
                    d2.fma(xfrm_eta0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                {
                    // Chain-rule for eta_0 and eta_2
                    d0 *= xfrm_eta2; // Load 1x
                    d1 *= xfrm_eta2; // Load 1x

                    xfrm_eta0 = f0[i]; // Load 1x
                    d2.fma(xfrm_eta0, d0);
                    d2.fma(xfrm_eta1, d1);
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
                simd_type tmp;
                tmp = d0 * df_tmp[0];
                tmp.fma(d1, df_tmp[1]);
                tmp.fma(d2, df_tmp[2]);
                out_d0[cnt_ijk] = tmp; // Store 1x

                tmp = d0 * df_tmp[3];
                tmp.fma(d1, df_tmp[4]);
                tmp.fma(d2, df_tmp[5]);
                out_d1[cnt_ijk] = tmp; // Store 1x

                tmp = d0 * df_tmp[6];
                tmp.fma(d1, df_tmp[7]);
                tmp.fma(d2, df_tmp[8]);
                out_d2[cnt_ijk] = tmp; // Store 1x
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void PhysDeriv3DWorkspace(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2,
    [[maybe_unused]] unsigned int &wsp1Size,
    [[maybe_unused]] unsigned int &wsp2Size)
{
    if constexpr ((SHAPE_TYPE == LibUtilities::ShapeType::Tet) ||
                  (SHAPE_TYPE == LibUtilities::eNodalTet))
    {
        wsp1Size = std::max(wsp1Size, nq0 * nq1 * nq2);
        wsp2Size = std::max(wsp2Size, nq0 * nq1 * nq2);
    }
}

} // namespace Nektar::Operators::detail
