///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseAVXSumFacKernels.hpp
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

#include "StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp"

template <bool DEFORMED>
NEK_FORCE_INLINE static void StdAlignDerivBase1D(
    const size_t nq0, const size_t indim, const vec_t *df_ptr,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &df_tmp, const size_t inSize,
    const vec_t::vectorType *in, vec_t::scalarType *out)

{
    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if (!DEFORMED)
    {
        // unroll very small loops
        df_tmp[0] = df_ptr[0];
        if (indim >= 2)
        {
            df_tmp[1] = df_ptr[1];
        }
        if (indim == 3)
        {
            df_tmp[2] = df_ptr[2];
        }
    }

    for (int i = 0; i < nq0; ++i)
    {
        if (DEFORMED)
        {
            df_tmp[0] = df_ptr[i * indim];
            if (indim >= 2)
            {
                df_tmp[1] = df_ptr[i * indim + 1];
            }
            if (indim == 3)
            {
                df_tmp[2] = df_ptr[i * indim + 2];
            }
        }

        vec_t sum = 0.0;
        for (int d = 0; d < indim; ++d)
        {
            vec_t inval = vec_t(in[d * inSize + i]); // possibly large stride
            sum.fma(inval, df_tmp[d]);
        }
        sum.store(out + i * vec_t::width);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
NEK_FORCE_INLINE static void StdAlignDerivBase2D(
    const size_t nq0, const size_t nq1, const size_t indim, const vec_t *df_Ptr,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &df_tmp, const size_t inSize,
    const vec_t::vectorType *inPtr, vec_t::scalarType *out[2],
    [[maybe_unused]] const vec_t *Fac0, [[maybe_unused]] const vec_t *Fac1)
{
    const auto ndf = 2 * indim;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if (!DEFORMED)
    {
        df_tmp[0] = df_Ptr[0];
        df_tmp[1] = df_Ptr[1];
        df_tmp[2] = df_Ptr[2];
        df_tmp[3] = df_Ptr[3];

        if (indim == 3)
        {
            df_tmp[4] = df_Ptr[4];
            df_tmp[5] = df_Ptr[5];
        }
    }

    vec_t f1;
    size_t cnt_ji   = 0;
    size_t inoffset = inSize / vec_t::width;
    for (size_t j = 0; j < nq1; ++j)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            f1 = vec_t(Fac1[j]);
        }

        for (size_t i = 0; i < nq0; ++i, ++cnt_ji)
        {

            if (DEFORMED)
            {
                df_tmp[0] = df_Ptr[cnt_ji * ndf];
                df_tmp[1] = df_Ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_Ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_Ptr[cnt_ji * ndf + 3];

                if (indim == 3)
                {
                    df_tmp[4] = df_Ptr[cnt_ji * ndf + 4];
                    df_tmp[5] = df_Ptr[cnt_ji * ndf + 5];
                }
            }

            vec_t in0 = vec_t(inPtr[cnt_ji]);
            vec_t in1 = vec_t(inPtr[inoffset + cnt_ji]);

            vec_t out0 = df_tmp[0] * in0;
            out0.fma(df_tmp[2], in1);

            vec_t out1 = df_tmp[1] * in0;
            out1.fma(df_tmp[3], in1);

            if (indim == 3)
            {
                vec_t in2 = vec_t(inPtr[2 * inSize + cnt_ji]);
                out0.fma(df_tmp[4], in2);
                out1.fma(df_tmp[5], in2);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
            {
                // Multiply by geometric factors
                vec_t f0 = vec_t(Fac0[i]);

                // Scale by geometric factor 2/(1-z1)
                out0 *= f1;
                // Scale by geometric factor (1+z0)/(1-z1)
                vec_t c1 = f0 * out1;
                out0.fma(c1, f1);
            }

            // store ouputs
            out0.store(out[0] + cnt_ji * vec_t::width);
            out1.store(out[1] + cnt_ji * vec_t::width);
        }
    }
}

template <bool DEFORMED>
NEK_FORCE_INLINE static void StdAlignDerivBaseHex(
    const size_t nq0, const size_t nq1, const size_t nq2, const vec_t *df_Ptr,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &df_tmp, const size_t inSize,
    const vec_t::vectorType *inPtr, vec_t::scalarType *out[3])
{
    const auto ndf   = 9;
    const auto nqTot = nq0 * nq1 * nq2;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if (!DEFORMED)
    {
        df_tmp[0] = df_Ptr[0];
        df_tmp[1] = df_Ptr[1];
        df_tmp[2] = df_Ptr[2];
        df_tmp[3] = df_Ptr[3];
        df_tmp[4] = df_Ptr[4];
        df_tmp[5] = df_Ptr[5];
        df_tmp[6] = df_Ptr[6];
        df_tmp[7] = df_Ptr[7];
        df_tmp[8] = df_Ptr[8];
    }

    size_t inoffset = inSize / vec_t::width;
    for (int i = 0; i < nqTot; ++i)
    {
        if (DEFORMED)
        {
            df_tmp[0] = df_Ptr[i * ndf];
            df_tmp[1] = df_Ptr[i * ndf + 1];
            df_tmp[2] = df_Ptr[i * ndf + 2];
            df_tmp[3] = df_Ptr[i * ndf + 3];
            df_tmp[4] = df_Ptr[i * ndf + 4];
            df_tmp[5] = df_Ptr[i * ndf + 5];
            df_tmp[6] = df_Ptr[i * ndf + 6];
            df_tmp[7] = df_Ptr[i * ndf + 7];
            df_tmp[8] = df_Ptr[i * ndf + 8];
        }

        vec_t in0 = vec_t(inPtr[i]);
        vec_t in1 = vec_t(inPtr[inoffset + i]);
        vec_t in2 = vec_t(inPtr[2 * inoffset + i]);

        vec_t out0 = df_tmp[0] * in0;
        out0.fma(df_tmp[3], in1);
        out0.fma(df_tmp[6], in2);

        vec_t out1 = df_tmp[1] * in0;
        out1.fma(df_tmp[4], in1);
        out1.fma(df_tmp[7], in2);

        vec_t out2 = df_tmp[2] * in0;
        out2.fma(df_tmp[5], in1);
        out2.fma(df_tmp[8], in2);

        // store ouputs
        out0.store(out[0] + i * vec_t::width);
        out1.store(out[1] + i * vec_t::width);
        out2.store(out[2] + i * vec_t::width);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
NEK_FORCE_INLINE static void StdAlignDerivBase3D(
    const size_t nq0, const size_t nq1, const size_t nq2, const vec_t *df_Ptr,
    std::vector<vec_t, tinysimd::allocator<vec_t>> &df_tmp, const size_t inSize,
    const vec_t *Fac0, const vec_t *Fac1, const vec_t *Fac1a, const vec_t *Fac2,
    const vec_t::vectorType *inPtr, vec_t::scalarType *out[3])
{
    const auto ndf = 9;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if (!DEFORMED)
    {
        df_tmp[0] = df_Ptr[0];
        df_tmp[1] = df_Ptr[1];
        df_tmp[2] = df_Ptr[2];
        df_tmp[3] = df_Ptr[3];
        df_tmp[4] = df_Ptr[4];
        df_tmp[5] = df_Ptr[5];
        df_tmp[6] = df_Ptr[6];
        df_tmp[7] = df_Ptr[7];
        df_tmp[8] = df_Ptr[8];
    }

    size_t cnt_kji = 0;
    vec_t f0, f1, f1a, f2;
    size_t inoffset = inSize / vec_t::width;

    for (size_t k = 0; k < nq2; ++k)
    {
        f2 = vec_t(Fac2[k]);

        for (size_t j = 0; j < nq1; ++j)
        {
            if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
            {
                f1  = vec_t(Fac1[j]);
                f1a = vec_t(Fac1a[j]);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
            {
                f1 = vec_t(Fac1[j]);
            }

            for (size_t i = 0; i < nq0; ++i, ++cnt_kji)
            {

                if (DEFORMED)
                {
                    df_tmp[0] = df_Ptr[cnt_kji * ndf];
                    df_tmp[1] = df_Ptr[cnt_kji * ndf + 1];
                    df_tmp[2] = df_Ptr[cnt_kji * ndf + 2];
                    df_tmp[3] = df_Ptr[cnt_kji * ndf + 3];
                    df_tmp[4] = df_Ptr[cnt_kji * ndf + 4];
                    df_tmp[5] = df_Ptr[cnt_kji * ndf + 5];
                    df_tmp[6] = df_Ptr[cnt_kji * ndf + 6];
                    df_tmp[7] = df_Ptr[cnt_kji * ndf + 7];
                    df_tmp[8] = df_Ptr[cnt_kji * ndf + 8];
                }

                vec_t in0 = vec_t(inPtr[cnt_kji]);
                vec_t in1 = vec_t(inPtr[inoffset + cnt_kji]);
                vec_t in2 = vec_t(inPtr[2 * inoffset + cnt_kji]);

                vec_t out0 = df_tmp[0] * in0;
                out0.fma(df_tmp[3], in1);
                out0.fma(df_tmp[6], in2);

                vec_t out1 = df_tmp[1] * in0;
                out1.fma(df_tmp[4], in1);
                out1.fma(df_tmp[7], in2);

                vec_t out2 = df_tmp[2] * in0;
                out2.fma(df_tmp[5], in1);
                out2.fma(df_tmp[8], in2);

                if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
                {
                    // (out0 + (out1 + out2)*(1+z0)/2) * 2/(1 - z1) * 2/(1 - z2)
                    f0 = vec_t(Fac0[i]);
                    out0.fma(out1 + out2, f0);
                    out0 *= f1a * f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }

                if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    vec_t f0 = vec_t(Fac0[i]);
                    out0.fma(out2, f0);
                    out0 *= f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }

                if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    vec_t f0 = vec_t(Fac0[i]);
                    out0.fma(out2, f0);
                    out0 *= f2;
                }

                // store ouputs
                out0.store(out[0] + cnt_kji * vec_t::width);
                out1.store(out[1] + cnt_kji * vec_t::width);
                out2.store(out[2] + cnt_kji * vec_t::width);
            }
        }
    }
}
