///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivAVXSumFacKernels.hpp
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

namespace Nektar::Operators::detail
{

using namespace tinysimd;
using vec_t = simd<NekDouble>;

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
NEK_FORCE_INLINE void PhysDeriv1DKernel(const int nq0, const size_t ndf,
                                        const vec_t *df_ptr,
                                        vec_t::scalarType *out[3])
{
    vec_t df_tmp[3];

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

    for (int j = 0; j < nq0; ++j)
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
        vec_t in, tmp;
        in.load(out[0] + j * vec_t::width);
        if (ndf == 3)
        {
            tmp = in * df_tmp[2]; // Store 1x
            tmp.store(out[2] + j * vec_t::width);
        }
        if (ndf >= 2)
        {
            tmp = in * df_tmp[1]; // Store 1x
            tmp.store(out[1] + j * vec_t::width);
        }
        tmp = in * df_tmp[0]; // Store 1x
        tmp.store(out[0] + j * vec_t::width);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
NEK_FORCE_INLINE void PhysDeriv2DKernel(const int nq0, const int nq1,
                                        const size_t outdim,
                                        [[maybe_unused]] const vec_t *Z0,
                                        [[maybe_unused]] const vec_t *Z1,
                                        const vec_t *df_ptr,
                                        vec_t::scalarType *out[3])
{
    auto ndf = 2 * outdim;
    vec_t df_tmp[6];

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

    for (int j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        vec_t xfrm0;
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            xfrm0 = 2.0 / (1.0 - Z1[j]); // Load 1x
        }

        for (int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            vec_t d0, d1;
            d0.load(out[0] + cnt_ji * vec_t::width); // Load 1x
            d1.load(out[1] + cnt_ji * vec_t::width); // Load 1x

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
            {
                // Moving from standard to collapsed coordinates
                vec_t xfrm1 = 0.5 * (1.0 + Z0[i]); // Load 1x
                d0 *= xfrm0;
                d1.fma(d0, xfrm1);
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
            vec_t tmp;
            tmp = d0 * df_tmp[0]; // d0 * df0 + d1 * df1
            tmp.fma(d1, df_tmp[1]);
            tmp.store(out[0] + cnt_ji * vec_t::width); // Store 1x

            tmp = d0 * df_tmp[2]; // d0 * df2 + d1 * df3
            tmp.fma(d1, df_tmp[3]);
            tmp.store(out[1] + cnt_ji * vec_t::width); // Store 1x

            if (outdim == 3)
            {
                tmp = d0 * df_tmp[4]; // d0 * df4 + d1 * df5
                tmp.fma(d1, df_tmp[5]);
                tmp.store(out[2] + cnt_ji * vec_t::width); // Store 1x
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
NEK_FORCE_INLINE void PhysDeriv3DKernel(
    const int nq0, const int nq1, const int nq2,
    [[maybe_unused]] const vec_t *Z0, [[maybe_unused]] const vec_t *Z1,
    [[maybe_unused]] const vec_t *Z2, const vec_t *df_ptr,
    [[maybe_unused]] std::vector<vec_t, allocator<vec_t>> &wsp0, // Tets only
    [[maybe_unused]] std::vector<vec_t, allocator<vec_t>> &wsp1, // Tets only
    vec_t::scalarType *out_d0, vec_t::scalarType *out_d1,
    vec_t::scalarType *out_d2)
{
    constexpr auto ndf = 9;
    vec_t df_tmp[ndf];

    if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        for (int k = 0, eta0 = 0; k < nq2; ++k)
        {
            vec_t xfrm_eta2 = 2.0 / (1.0 - Z2[k]); // Load 1x

            for (int j = 0; j < nq1; ++j)
            {
                vec_t xfrm_eta1 = 2.0 / (1.0 - Z1[j]); // Load 1x
                vec_t xfrm      = xfrm_eta1 * xfrm_eta2;
                for (int i = 0; i < nq0; ++i, ++eta0)
                {
                    vec_t d0;
                    d0.load(out_d0 + eta0 * vec_t::width);  // Load 1x
                    d0 *= xfrm;                             // Load 1x
                    d0.store(out_d0 + eta0 * vec_t::width); // Store 1x
                    wsp0[eta0] = d0; // Store 1x partial form for reuse
                }
            }
        }

        for (int k = 0, eta0 = 0; k < nq2; ++k)
        {
            vec_t xfrm_eta2 = 2.0 / (1.0 - Z2[k]); // Load 1x

            for (int j = 0; j < nq1; ++j)
            {
                for (int i = 0; i < nq0; ++i, ++eta0)
                {
                    vec_t xfrm_eta0 = 0.5 * (1.0 + Z0[i]); // Load 1x

                    vec_t out0 = xfrm_eta0 * wsp0[eta0]; // Load 1x
                    wsp0[eta0] = out0; // 2 * (1 + eta_0) / (1 -
                                       // eta_1)(1-eta2) | store 1x
                    vec_t d1;
                    d1.load(out_d1 + eta0 * vec_t::width); // Load 1x
                    d1 *= xfrm_eta2;
                    wsp1[eta0] = d1; // Store 1x partial form for reuse
                    d1 += out0;
                    d1.store(out_d1 + eta0 * vec_t::width); // Store 1x
                }
            }
        }

        for (int k = 0, eta0 = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j)
            {
                vec_t xfrm_eta1 = 0.5 * (1.0 + Z1[j]); // Load 1x

                for (int i = 0; i < nq0; ++i, ++eta0)
                {
                    // out_d2[eta0] += wsp0[eta0] * xfrm_eta1 + wsp1[eta0]
                    vec_t out = wsp0[eta0]; // Load 1x
                    vec_t d1  = wsp1[eta0]; // Load 1x
                    out.fma(d1, xfrm_eta1);
                    d1.load(out_d2 + eta0 * vec_t::width); // Load 1x
                    d1 += out;
                    d1.store(out_d2 + eta0 * vec_t::width); // Store 1x
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

    for (int k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        vec_t xfrm_eta2;
        if constexpr (SHAPE_TYPE == LibUtilities::ePrism ||
                      SHAPE_TYPE == LibUtilities::ePyramid)
        {
            xfrm_eta2 = 2.0 / (1.0 - Z2[k]); // Load 1x
        }

        for (int j = 0; j < nq1; ++j)
        {
            vec_t xfrm_eta1;
            if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
            {
                xfrm_eta1 = 0.5 * (1.0 + Z1[j]); // Load 1x
            }
            for (int i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                vec_t d0, d1, d2;
                d0.load(out_d0 + cnt_ijk * vec_t::width); // Load 1x
                d1.load(out_d1 + cnt_ijk * vec_t::width); // Load 1x
                d2.load(out_d2 + cnt_ijk * vec_t::width); // Load 1x

                vec_t xfrm_eta0;
                if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
                {
                    // Chain-rule for eta_0 and eta_2
                    d0 *= xfrm_eta2; // Load 1x

                    xfrm_eta0 = 0.5 * (1.0 + Z0[i]); // Load 1x
                    d2.fma(xfrm_eta0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                {
                    // Chain-rule for eta_0 and eta_2
                    d0 *= xfrm_eta2; // Load 1x
                    d1 *= xfrm_eta2; // Load 1x

                    xfrm_eta0 = 0.5 * (1.0 + Z0[i]); // Load 1x
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
                vec_t tmp;
                tmp = d0 * df_tmp[0];
                tmp.fma(d1, df_tmp[1]);
                tmp.fma(d2, df_tmp[2]);
                tmp.store(out_d0 + cnt_ijk * vec_t::width); // Store 1x

                tmp = d0 * df_tmp[3];
                tmp.fma(d1, df_tmp[4]);
                tmp.fma(d2, df_tmp[5]);
                tmp.store(out_d1 + cnt_ijk * vec_t::width); // Store 1x

                tmp = d0 * df_tmp[6];
                tmp.fma(d1, df_tmp[7]);
                tmp.fma(d2, df_tmp[8]);
                tmp.store(out_d2 + cnt_ijk * vec_t::width); // Store 1x
            }
        }
    }
}

NEK_FORCE_INLINE static void PhysDerivTensor1DKernel(
    const size_t nq0, const vec_t::vectorType *in, const vec_t *D0,
    vec_t::scalarType *out_d0)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    for (int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix

        vec_t prod_sum = 0.0;
        for (int k = 0; k < nq0; ++k)
        {                               // Col index of D0, row index of IN
            vec_t v1 = D0[k * nq0 + i]; // Load 1x
            vec_t v2 = vec_t(in[k]);    // Load 1x

            prod_sum.fma(v1, v2);
        }

        prod_sum.store(out_d0 + i * vec_t::width);
    }
}

NEK_FORCE_INLINE static void PhysDerivTensor2DKernel(
    const size_t nq0, const size_t nq1, const vec_t::vectorType *in,
    const vec_t *D0, const vec_t *D1, vec_t::scalarType *out_d0,
    vec_t::scalarType *out_d1)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    for (int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix
        for (int j = 0; j < nq1; ++j)
        { // Col index of IN matrix

            vec_t prod_sum = 0.0;
            for (int k = 0; k < nq0; ++k)
            {                               // Col index of D0, row index of IN
                vec_t v1 = D0[k * nq0 + i]; // Load 1x
                vec_t v2 = vec_t(in[j * nq0 + k]); // Load 1x

                prod_sum.fma(v1, v2);
            }

            prod_sum.store(out_d0 + (j * nq0 + i) * vec_t::width); // Store 1x
        }
    }

    // in * D1^T
    for (int i = 0; i < nq0; ++i)
    { // row index for grid
        for (int j = 0; j < nq1; ++j)
        { // Column index for D1^T (row idx for D1)

            vec_t prod_sum = 0.0;
            for (int k = 0; k < nq1; ++k)
            {
                vec_t v1 = vec_t(in[k * nq0 + i]); // Load 1x
                vec_t v2 = D1[k * nq1 + j];        // Load 1x

                prod_sum.fma(v1, v2);
            }

            prod_sum.store(out_d1 + (j * nq0 + i) * vec_t::width); // Store 1x
        }
    }
}

NEK_FORCE_INLINE static void PhysDerivTensor3DKernel(
    const size_t nq0, const size_t nq1, const size_t nq2,
    const vec_t::vectorType *in, const vec_t *D0, const vec_t *D1,
    const vec_t *D2, vec_t::scalarType *out_d0, vec_t::scalarType *out_d1,
    vec_t::scalarType *out_d2)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // Direction 1
    for (int i = 0; i < nq0; ++i)
    {
        for (int j = 0; j < nq1 * nq2; ++j)
        {
            vec_t prod_sum = 0.0;
            for (int k = 0; k < nq0; ++k)
            {
                vec_t v1 = D0[k * nq0 + i];        // Load 1x
                vec_t v2 = vec_t(in[j * nq0 + k]); // Load 1x

                prod_sum.fma(v1, v2);
            }

            // out_d0[j * nq0 + i] = prod_sum; // Store 1x
            prod_sum.store(out_d0 + (j * nq0 + i) * vec_t::width);
        }
    }

    // Direction 2
    for (int block = 0; block < nq2; ++block)
    {
        int start = block * nq0 * nq1;

        for (int i = 0; i < nq0; ++i)
        {
            for (int j = 0; j < nq1; ++j)
            {
                vec_t prod_sum = 0.0;
                for (int k = 0; k < nq1; ++k)
                {
                    vec_t v1 = vec_t(in[start + k * nq0 + i]); // Load 1x
                    vec_t v2 = D1[k * nq1 + j];                // Load 1x

                    prod_sum.fma(v1, v2);
                }

                // out_d1[start + j * nq0 + i] = prod_sum; // Store 1x
                prod_sum.store(out_d1 + (start + j * nq0 + i) * vec_t::width);
            }
        }
    }

    // Direction 3
    for (int i = 0; i < nq0 * nq1; ++i)
    {
        for (int j = 0; j < nq2; ++j)
        {
            vec_t prod_sum = 0.0;
            for (int k = 0; k < nq2; ++k)
            {
                vec_t v1 = vec_t(in[k * nq0 * nq1 + i]); // Load 1x
                vec_t v2 = D2[k * nq2 + j];              // Load 1x

                prod_sum.fma(v1, v2);
            }

            // out_d2[j * nq0 * nq1 + i] = prod_sum; // Store 1x
            prod_sum.store(out_d2 + (j * nq0 * nq1 + i) * vec_t::width);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_FORCE_INLINE static void PhysDeriv3DWorkspace(
    [[maybe_unused]] const size_t nq0, [[maybe_unused]] const size_t nq1,
    [[maybe_unused]] const size_t nq2, [[maybe_unused]] size_t &wsp1Size,
    [[maybe_unused]] size_t &wsp2Size)
{
    // Check preconditions
    ASSERTL1((SHAPE_TYPE == LibUtilities::ShapeType::Hex && nq0 == nq1 &&
              nq0 == nq2) ||
                 (SHAPE_TYPE == LibUtilities::ShapeType::Tet &&
                  nq0 == nq1 + 1 && nq0 == nq2 + 1) ||
                 (SHAPE_TYPE == LibUtilities::ShapeType::Prism && nq0 == nq1 &&
                  nq0 == nq2 + 1) ||
                 (SHAPE_TYPE == LibUtilities::ShapeType::Pyr && nq0 == nq1 &&
                  nq0 == nq2 + 1),
             "PhysDeriv3DWorkspace: Requires homogenous points.");

    if constexpr (SHAPE_TYPE == LibUtilities::ShapeType::Tet)
    {
        wsp1Size = std::max(wsp1Size, nq0 * nq1 * nq2);
        wsp2Size = std::max(wsp2Size, nq0 * nq1 * nq2);
    }
}

} // namespace Nektar::Operators::detail
