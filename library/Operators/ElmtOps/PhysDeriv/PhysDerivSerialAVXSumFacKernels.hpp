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
            tmp       = in * df_tmp[2];
            out[2][j] = tmp;
        }
        if (ndf >= 2)
        {
            tmp       = in * df_tmp[1];
            out[1][j] = tmp;
        }
        tmp       = in * df_tmp[0];
        out[0][j] = tmp;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE void PhysDerivDir1DKernel(const unsigned int nq0,
                                           const unsigned int ndf,
                                           const simd_type *df_ptr,
                                           const simd_type *in, simd_type *out)
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
            df_tmp = df_ptr[j * ndf + DIR]; // load 1x
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

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int outdim,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, simd_type *out[3])
{
    const unsigned int ndf = 2 * outdim;
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
            out[0][cnt_ji] = tmp;

            tmp = d0 * df_tmp[2]; // d0 * df2 + d1 * df3
            tmp.fma(d1, df_tmp[3]);
            out[1][cnt_ji] = tmp;

            if (outdim == 3)
            {
                tmp = d0 * df_tmp[4]; // d0 * df4 + d1 * df5
                tmp.fma(d1, df_tmp[5]);
                out[2][cnt_ji] = tmp;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE void PhysDerivDir2DKernel(
    const unsigned nq0, const unsigned nq1, const unsigned int outdim,
    [[maybe_unused]] const simd_type *f0, [[maybe_unused]] const simd_type *f1,
    const simd_type *df_ptr, const simd_type *tderiv0, const simd_type *tderiv1,
    simd_type *out)
{
    const unsigned int ndf = 2 * outdim;
    simd_type df_tmp[2];

    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[2 * DIR];
        df_tmp[1] = df_ptr[2 * DIR + 1];
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
            d0 = tderiv0[cnt_ji]; // Load 1x
            d1 = tderiv1[cnt_ji]; // Load 1x

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

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE void PhysDeriv3DKernel(
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
            else if (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                     SHAPE_TYPE == LibUtilities::eNodalTet)
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
                if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                              (SHAPE_TYPE == LibUtilities::eNodalPrism))
                {
                    d0 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                {
                    d0 *= xfrm_eta2;
                    d1 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                    d2.fma(xfrm_eta1, d1);
                }
                else if (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                         SHAPE_TYPE == LibUtilities::eNodalTet)
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

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename simd_type>
NEK_FORCE_INLINE void PhysDerivDir3DKernel(
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
            else if (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                     SHAPE_TYPE == LibUtilities::eNodalTet)
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
                if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                              (SHAPE_TYPE == LibUtilities::eNodalPrism))
                {
                    d0 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                {
                    d0 *= xfrm_eta2;
                    d1 *= xfrm_eta2;
                    xfrm_eta0 = f0[i];
                    d2.fma(xfrm_eta0, d0);
                    d2.fma(xfrm_eta1, d1);
                }
                else if (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                         SHAPE_TYPE == LibUtilities::eNodalTet)
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

} // namespace Nektar::Operators::detail
