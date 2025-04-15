///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialAVXSumFacKernels.hpp
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

#include "Operators/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

#include <LibUtilities/BasicUtils/NekInline.hpp>

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffSegKernel(
    const unsigned int nq0, const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const simd_type *df_ptr, simd_type *deriv0)
{
    constexpr auto ndf = 1;

    const auto nqTot = nq0;

    simd_type d00 = {1.0};
    simd_type df0;
    simd_type metric00;

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
    }

    // Precompute Laplacian metricsp
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];

        if (!isConstVarDiff && !isVarDiff)
        {
            metric00 = df0 * df0;
        }
        else if (isConstVarDiff)
        {
            metric00 = df0 * df0 * d00;
        }
    }

    // Step 4: Apply Laplacian metrics & inner product
    if (!isVarDiff)
    {
        if constexpr (DEFORMED)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                df0 = df_ptr[i * ndf];

                if (!isConstVarDiff)
                {
                    metric00 = df0 * df0;
                }
                else
                {
                    metric00 = df0 * df0 * d00;
                }

                deriv0[i] = metric00 * deriv0[i];
            }
        }
        else
        {
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                deriv0[i] = metric00 * deriv0[i];
            }
        }
    }
    else
    {
        if constexpr (DEFORMED)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                df0      = df_ptr[i * ndf];
                d00      = varD00[i];
                metric00 = df0 * df0 * d00;

                deriv0[i] = metric00 * deriv0[i];
            }
        }
        else
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                d00      = varD00[i];
                metric00 = df0 * df0 * d00;

                deriv0[i] = metric00 * deriv0[i];
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffTriKernel(
    const unsigned int nq0, const unsigned int nq1, const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *df_ptr, const simd_type *hfac0, const simd_type *hfac1,
    simd_type *diffderiv0, simd_type *diffderiv1, simd_type *deriv0,
    simd_type *deriv1)
{
    constexpr auto ndf = 4;

    simd_type df0, df1, df2, df3;
    simd_type metric00, metric01, metric11;

    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};                // var diffusion terms
    simd_type dtmp0, dtmp1, dtmp2, dtmp3; // metric products

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
    }

    // Precompute Laplacian metricsp
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
    }

    for (unsigned int q = 0; q < nq1; ++q)
    {
        simd_type h1j = hfac1[q];
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt   = q * nq0 + p;
            simd_type prod_sum = 0.0;
            for (unsigned int i = 0; i < nq0; ++i)
            {
                simd_type v1 = D0[i * nq0 + p]; // Load 1x
                simd_type v2 = in[q * nq0 + i]; // Load 1x

                prod_sum.fma(v1, v2);
            }

            simd_type prod_sum1 = 0.0;
            for (unsigned int j = 0; j < nq1; ++j)
            {
                simd_type v1 = in[j * nq0 + p]; // Load 1x
                simd_type v2 = D1[j * nq1 + q]; // Load 1x

                prod_sum1.fma(v1, v2);
            }

            if constexpr (DEFORMED)
            {
                df0 = df_ptr[cnt * ndf];
                df1 = df_ptr[cnt * ndf + 1];
                df2 = df_ptr[cnt * ndf + 2];
                df3 = df_ptr[cnt * ndf + 3];
            }

            simd_type h0i = hfac0[p];

            // M = [M_00, df1; M_10; df3]
            metric00      = h1j * (df0 + h0i * df1); // M_00
            simd_type tmp = h1j * (df2 + h0i * df3); // M_10

            if (!isConstVarDiff && !isVarDiff)
            {
                metric01 = metric00 * df1;
                metric00 = metric00 * metric00;

                metric01.fma(tmp, df3);
                metric00.fma(tmp, tmp);

                metric11 = df1 * df1;
                metric11.fma(df3, df3);
            }
            else
            {
                if (isVarDiff)
                {
                    d00 = varD00[cnt];
                    d01 = varD01[cnt];
                    d11 = varD11[cnt];
                }
                // M = [M_00, df1; M_10; df3]
                dtmp0 = metric00 * d00;
                dtmp0.fma(tmp, d01);
                dtmp1 = metric00 * d01;
                dtmp1.fma(tmp, d11);
                dtmp2 = df1 * d00;
                dtmp2.fma(df3, d01);
                dtmp3 = df1 * d01;
                dtmp3.fma(df3, d11);

                metric00 = metric00 * dtmp0;
                metric00.fma(tmp, dtmp1);
                metric01 = df1 * dtmp0;
                metric01.fma(df3, dtmp1);
                metric11 = df1 * dtmp2;
                metric11.fma(df3, dtmp3);
            }

            if (deriv0)
            {
                deriv0[cnt] = prod_sum;
            }

            tmp = metric00 * prod_sum;
            tmp.fma(metric01, prod_sum1);
            diffderiv0[cnt] = tmp;

            if (deriv1)
            {
                deriv1[cnt] = prod_sum1;
            }

            tmp = metric01 * prod_sum;
            tmp.fma(metric11, prod_sum1);
            diffderiv1[cnt] = tmp;
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffQuadKernel(
    const unsigned int nq0, const unsigned int nq1, const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *df_ptr, simd_type *diffderiv0, simd_type *diffderiv1,
    simd_type *deriv0, simd_type *deriv1)
{
    constexpr auto ndf = 4;

    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};                // var diffusion terms
    simd_type dtmp0, dtmp1, dtmp2, dtmp3; // temp for vardiff
    simd_type df0, df1, df2, df3;
    simd_type metric00, metric01, metric11;

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
    }

    // Precompute Laplacian metricsp
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];

        if (!isConstVarDiff && !isVarDiff)
        {
            metric00 = df0 * df0;
            metric00.fma(df2, df2);
            metric01 = df0 * df1;
            metric01.fma(df2, df3);
            metric11 = df1 * df1;
            metric11.fma(df3, df3);
        }
        else if (isConstVarDiff)
        {
            dtmp0 = df0 * d00;
            dtmp0.fma(df2, d01);
            dtmp1 = df0 * d01;
            dtmp1.fma(df2, d11);
            dtmp2 = df1 * d00;
            dtmp2.fma(df3, d01);
            dtmp3 = df1 * d01;
            dtmp3.fma(df3, d11);

            metric00 = df0 * dtmp0;
            metric00.fma(df2, dtmp1);
            metric01 = df1 * dtmp0;
            metric01.fma(df3, dtmp1);
            metric11 = df1 * dtmp2;
            metric11.fma(df3, dtmp3);
        }
    }

    for (unsigned int q = 0; q < nq1; ++q)
    {
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt   = q * nq0 + p;
            simd_type prod_sum = 0.0;
            for (unsigned int i = 0; i < nq0; ++i)
            {
                simd_type v1 = D0[i * nq0 + p]; // Load 1x
                simd_type v2 = in[q * nq0 + i]; // Load 1x

                prod_sum.fma(v1, v2);
            }

            simd_type prod_sum1 = 0.0;
            for (unsigned int j = 0; j < nq1; ++j)
            {
                simd_type v1 = in[j * nq0 + p]; // Load 1x
                simd_type v2 = D1[j * nq1 + q]; // Load 1x

                prod_sum1.fma(v1, v2);
            }

            if constexpr (DEFORMED)
            {
                df0 = df_ptr[cnt * ndf];
                df1 = df_ptr[cnt * ndf + 1];
                df2 = df_ptr[cnt * ndf + 2];
                df3 = df_ptr[cnt * ndf + 3];
            }

            if (!isConstVarDiff)
            {
                metric00 = df0 * df0;
                metric00.fma(df2, df2);
                metric01 = df0 * df1;
                metric01.fma(df2, df3);
                metric11 = df1 * df1;
                metric11.fma(df3, df3);
            }
            else
            {
                if (isVarDiff)
                {
                    d00 = varD00[cnt];
                    d01 = varD01[cnt];
                    d11 = varD11[cnt];
                }

                dtmp0 = df0 * d00;
                dtmp0.fma(df2, d01);
                dtmp1 = df0 * d01;
                dtmp1.fma(df2, d11);
                dtmp2 = df1 * d00;
                dtmp2.fma(df3, d01);
                dtmp3 = df1 * d01;
                dtmp3.fma(df3, d11);

                metric00 = df0 * dtmp0;
                metric00.fma(df2, dtmp1);
                metric01 = df1 * dtmp0;
                metric01.fma(df3, dtmp1);
                metric11 = df1 * dtmp2;
                metric11.fma(df3, dtmp3);
            }

            if (deriv0)
            {
                deriv0[cnt] = prod_sum;
            }

            simd_type tmp = metric00 * prod_sum;
            tmp.fma(metric01, prod_sum1);
            diffderiv0[cnt] = tmp;

            if (deriv1)
            {
                deriv1[cnt] = prod_sum1;
            }

            simd_type tmp1 = metric01 * prod_sum;
            tmp1.fma(metric11, prod_sum1);
            diffderiv1[cnt] = tmp1;
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffHexKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *D2, const simd_type *df_ptr, simd_type *diffderiv0,
    simd_type *diffderiv1, simd_type *diffderiv2, simd_type *deriv0,
    simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type metric00, metric01, metric02, metric11, metric12, metric22;

    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0}; // var diffusion terms
    simd_type td0, td1, td2, td3, td4, td5, td6, td7,
        td8; // temp terms for vardiff

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        d02 = constVarDiff[3];
        d12 = constVarDiff[4];
        d22 = constVarDiff[5];
    }

    // Precompute Laplacian metricsp
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

        if (!isConstVarDiff && !isConstVarDiff)
        {
            metric00 = df0 * df0;
            metric00.fma(df3, df3);
            metric00.fma(df6, df6);

            metric01 = df0 * df1;
            metric01.fma(df3, df4);
            metric01.fma(df6, df7);

            metric02 = df0 * df2;
            metric02.fma(df3, df5);
            metric02.fma(df6, df8);

            metric11 = df1 * df1;
            metric11.fma(df4, df4);
            metric11.fma(df7, df7);

            metric12 = df1 * df2;
            metric12.fma(df4, df5);
            metric12.fma(df7, df8);

            metric22 = df2 * df2;
            metric22.fma(df5, df5);
            metric22.fma(df8, df8);
        }
        else if (isConstVarDiff)
        { // with vardiff
            td0 = df0 * d00;
            td0.fma(df3, d01);
            td0.fma(df6, d02);

            td1 = df0 * d01;
            td1.fma(df3, d11);
            td1.fma(df6, d12);

            td2 = df0 * d02;
            td2.fma(df3, d12);
            td2.fma(df6, d22);

            td3 = df1 * d00;
            td3.fma(df4, d01);
            td3.fma(df7, d02);

            td4 = df1 * d01;
            td4.fma(df4, d11);
            td4.fma(df7, d12);

            td5 = df1 * d02;
            td5.fma(df4, d12);
            td5.fma(df7, d22);

            td6 = df2 * d00;
            td6.fma(df5, d01);
            td6.fma(df8, d02);

            td7 = df2 * d01;
            td7.fma(df5, d11);
            td7.fma(df8, d12);

            td8 = df2 * d02;
            td8.fma(df5, d12);
            td8.fma(df8, d22);

            metric00 = td0 * df0;
            metric00.fma(td1, df3);
            metric00.fma(td2, df6);

            metric01 = td0 * df1;
            metric01.fma(td1, df4);
            metric01.fma(td2, df7);

            metric02 = td0 * df2;
            metric02.fma(td1, df5);
            metric02.fma(td2, df8);

            metric11 = td3 * df1;
            metric11.fma(td4, df4);
            metric11.fma(td5, df7);

            metric12 = td3 * df2;
            metric12.fma(td4, df5);
            metric12.fma(td5, df8);

            metric22 = td6 * df2;
            metric22.fma(td7, df5);
            metric22.fma(td8, df8);
        }
    }

    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    for (unsigned int r = 0; r < nq2; ++r)
    {
        for (unsigned int q = 0; q < nq1; ++q)
        {
            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;
                // 1. get deriv in std coordinate
                simd_type prod_sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i)
                {
                    simd_type v1 = D0[i * nq0 + p];      // Load 1x
                    simd_type v2 = in[cnt_qr * nq0 + i]; // Load 1x
                    prod_sum.fma(v1, v2);
                }
                // prod_sum.store(deriv0 + cnt * simd_type::width);
                simd_type prod_sum1 = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    unsigned int cnt_jr = r * nq1 + j;
                    simd_type v1        = D1[j * nq1 + q];      // Load 1x
                    simd_type v2        = in[cnt_jr * nq0 + p]; // Load 1x
                    prod_sum1.fma(v1, v2);
                }
                // prod_sum1.store(deriv1 + cnt * simd_type::width);
                simd_type prod_sum2 = 0.0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    unsigned int cnt_qk = (k * nq1 + q);
                    simd_type v2        = D2[k * nq2 + r];      // Load 1x
                    simd_type v1        = in[cnt_qk * nq0 + p]; // Load 1x
                    prod_sum2.fma(v1, v2);
                }
                // prod_sum2.store(deriv2 + cnt * simd_type::width);

                // 2. evaluate diffusion coeff if deformed
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

                if (!isConstVarDiff && !isVarDiff)
                {
                    metric00 = df0 * df0;
                    metric00.fma(df3, df3);
                    metric00.fma(df6, df6);

                    metric01 = df0 * df1;
                    metric01.fma(df3, df4);
                    metric01.fma(df6, df7);

                    metric02 = df0 * df2;
                    metric02.fma(df3, df5);
                    metric02.fma(df6, df8);

                    metric11 = df1 * df1;
                    metric11.fma(df4, df4);
                    metric11.fma(df7, df7);

                    metric12 = df1 * df2;
                    metric12.fma(df4, df5);
                    metric12.fma(df7, df8);

                    metric22 = df2 * df2;
                    metric22.fma(df5, df5);
                    metric22.fma(df8, df8);
                }
                else
                { // with vardiff

                    if (isVarDiff)
                    {
                        d00 = varD00[cnt];
                        d01 = varD01[cnt];
                        d11 = varD11[cnt];
                        d02 = varD02[cnt];
                        d12 = varD12[cnt];
                        d22 = varD22[cnt];
                    }

                    td0 = df0 * d00;
                    td0.fma(df3, d01);
                    td0.fma(df6, d02);

                    td1 = df0 * d01;
                    td1.fma(df3, d11);
                    td1.fma(df6, d12);

                    td2 = df0 * d02;
                    td2.fma(df3, d12);
                    td2.fma(df6, d22);

                    td3 = df1 * d00;
                    td3.fma(df4, d01);
                    td3.fma(df7, d02);

                    td4 = df1 * d01;
                    td4.fma(df4, d11);
                    td4.fma(df7, d12);

                    td5 = df1 * d02;
                    td5.fma(df4, d12);
                    td5.fma(df7, d22);

                    td6 = df2 * d00;
                    td6.fma(df5, d01);
                    td6.fma(df8, d02);

                    td7 = df2 * d01;
                    td7.fma(df5, d11);
                    td7.fma(df8, d12);

                    td8 = df2 * d02;
                    td8.fma(df5, d12);
                    td8.fma(df8, d22);

                    metric00 = td0 * df0;
                    metric00.fma(td1, df3);
                    metric00.fma(td2, df6);

                    metric01 = td0 * df1;
                    metric01.fma(td1, df4);
                    metric01.fma(td2, df7);

                    metric02 = td0 * df2;
                    metric02.fma(td1, df5);
                    metric02.fma(td2, df8);

                    metric11 = td3 * df1;
                    metric11.fma(td4, df4);
                    metric11.fma(td5, df7);

                    metric12 = td3 * df2;
                    metric12.fma(td4, df5);
                    metric12.fma(td5, df8);

                    metric22 = td6 * df2;
                    metric22.fma(td7, df5);
                    metric22.fma(td8, df8);
                }

                // store tensor derivs if required
                if (deriv0)
                {
                    deriv0[cnt] = prod_sum;
                }
                if (deriv1)
                {
                    deriv1[cnt] = prod_sum1;
                }
                if (deriv2)
                {
                    deriv2[cnt] = prod_sum2;
                }

                // 3. apply diffusion coeff to deriv and get output
                simd_type tmp = metric00 * prod_sum;
                tmp.fma(metric01, prod_sum1);
                tmp.fma(metric02, prod_sum2);
                diffderiv0[cnt] = tmp;

                simd_type tmp1 = metric01 * prod_sum;
                tmp1.fma(metric11, prod_sum1);
                tmp1.fma(metric12, prod_sum2);
                diffderiv1[cnt] = tmp1;

                simd_type tmp2 = metric02 * prod_sum;
                tmp2.fma(metric12, prod_sum1);
                tmp2.fma(metric22, prod_sum2);
                diffderiv2[cnt] = tmp2;
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffTetKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    [[maybe_unused]] const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *D2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac1, const simd_type *hfac2, const simd_type *hfac3,
    simd_type *diffderiv0, simd_type *diffderiv1, simd_type *diffderiv2,
    simd_type *deriv0, simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type g0, g1, g2, g3, g4, g5;
    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0}; // var diffusion terms
    simd_type td0, td1, td2, td3, td4, td5, td6, td7,
        td8; // temp terms for vardiff

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        d02 = constVarDiff[3];
        d12 = constVarDiff[4];
        d22 = constVarDiff[5];
    }

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

    for (unsigned int r = 0; r < nq2; ++r)
    {
        simd_type h3 = hfac3[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type h1   = hfac1[q];
            simd_type h2   = hfac2[q];
            simd_type h2h3 = h2 * h3;
            simd_type h1h3 = h1 * h3;

            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;
                // 1. get deriv in std coordinate
                simd_type prod_sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i)
                {
                    simd_type v1 = D0[i * nq0 + p];      // Load 1x
                    simd_type v2 = in[cnt_qr * nq0 + i]; // Load 1x
                    prod_sum.fma(v1, v2);
                }
                // prod_sum.store(deriv0 + cnt * simd_type::width);
                simd_type prod_sum1 = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    unsigned int cnt_jr = r * nq1 + j;
                    simd_type v1        = D1[j * nq1 + q];      // Load 1x
                    simd_type v2        = in[cnt_jr * nq0 + p]; // Load 1x
                    prod_sum1.fma(v1, v2);
                }
                // prod_sum1.store(deriv1 + cnt * simd_type::width);
                simd_type prod_sum2 = 0.0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    unsigned int cnt_qk = (k * nq1 + q);
                    simd_type v2        = D2[k * nq2 + r];      // Load 1x
                    simd_type v1        = in[cnt_qk * nq0 + p]; // Load 1x
                    prod_sum2.fma(v1, v2);
                }
                // prod_sum2.store(deriv2 + cnt * simd_type::width);

                // 3. apply diffusion coeff to deriv and fill output
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

                simd_type h0h2h3 = hfac0[p] * h2h3;

                simd_type tmp1 = h0h2h3 * (df1 + df2);
                tmp1.fma(df0, h2h3);
                simd_type tmp2 = h0h2h3 * (df4 + df5);
                tmp2.fma(df3, h2h3);
                simd_type tmp3 = h0h2h3 * (df7 + df8);
                tmp3.fma(df6, h2h3);

                simd_type tmp4 = df1 * h3;
                tmp4.fma(df2, h1h3);
                simd_type tmp5 = df4 * h3;
                tmp5.fma(df5, h1h3);
                simd_type tmp6 = df7 * h3;
                tmp6.fma(df8, h1h3);

                if (!isConstVarDiff && !isVarDiff)
                {
                    g0 = tmp1 * tmp1;
                    g0.fma(tmp2, tmp2);
                    g0.fma(tmp3, tmp3);

                    g4 = df2 * tmp1;
                    g4.fma(df5, tmp2);
                    g4.fma(df8, tmp3);

                    g3 = tmp1 * tmp4;
                    g3.fma(tmp2, tmp5);
                    g3.fma(tmp3, tmp6);

                    g1 = tmp4 * tmp4;
                    g1.fma(tmp5, tmp5);
                    g1.fma(tmp6, tmp6);

                    g5 = df2 * tmp4;
                    g5.fma(df5, tmp5);
                    g5.fma(df8, tmp6);

                    g2 = df2 * df2;
                    g2.fma(df5, df5);
                    g2.fma(df8, df8);
                }
                else
                {
                    if (isVarDiff)
                    {
                        d00 = varD00[cnt];
                        d01 = varD01[cnt];
                        d02 = varD02[cnt];
                        d12 = varD12[cnt];
                        d22 = varD22[cnt];
                    }

                    td0 = tmp1 * d00;
                    td0.fma(tmp2, d01);
                    td0.fma(tmp3, d02);

                    td1 = tmp1 * d01;
                    td1.fma(tmp2, d11);
                    td1.fma(tmp3, d12);

                    td2 = tmp1 * d02;
                    td2.fma(tmp2, d12);
                    td2.fma(tmp3, d22);

                    td3 = tmp4 * d00;
                    td3.fma(tmp5, d01);
                    td3.fma(tmp6, d02);

                    td4 = tmp4 * d01;
                    td4.fma(tmp5, d11);
                    td4.fma(tmp6, d12);

                    td5 = tmp4 * d02;
                    td5.fma(tmp5, d12);
                    td5.fma(tmp6, d22);

                    td6 = df2 * d00;
                    td6.fma(df5, d01);
                    td6.fma(df8, d02);

                    td7 = df2 * d01;
                    td7.fma(df5, d11);
                    td7.fma(df8, d12);

                    td8 = df2 * d02;
                    td8.fma(df5, d12);
                    td8.fma(df8, d22);

                    g0 = td0 * tmp1;
                    g0.fma(td1, tmp2);
                    g0.fma(td2, tmp3);

                    g3 = td0 * tmp4;
                    g3.fma(td1, tmp5);
                    g3.fma(td2, tmp6);

                    g4 = td0 * df2;
                    g4.fma(td1, df5);
                    g4.fma(td2, df8);

                    g1 = td3 * tmp4;
                    g1.fma(td4, tmp5);
                    g1.fma(td5, tmp6);

                    g5 = td3 * df2;
                    g5.fma(td4, df5);
                    g5.fma(td5, df8);

                    g2 = td6 * df2;
                    g2.fma(td7, df5);
                    g2.fma(td8, df8);
                }

                // store tensor derivs if required
                if (deriv0)
                {
                    deriv0[cnt] = prod_sum;
                }
                if (deriv1)
                {
                    deriv1[cnt] = prod_sum1;
                }
                if (deriv2)
                {
                    deriv2[cnt] = prod_sum2;
                }

                tmp1 = g0 * prod_sum;
                tmp1.fma(g3, prod_sum1);
                tmp1.fma(g4, prod_sum2);
                diffderiv0[cnt] = tmp1;

                tmp2 = g3 * prod_sum;
                tmp2.fma(g1, prod_sum1);
                tmp2.fma(g5, prod_sum2);
                diffderiv1[cnt] = tmp2;

                tmp3 = g4 * prod_sum;
                tmp3.fma(g5, prod_sum1);
                tmp3.fma(g2, prod_sum2);
                diffderiv2[cnt] = tmp3;
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffPrismKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *D2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac1, simd_type *diffderiv0, simd_type *diffderiv1,
    simd_type *diffderiv2, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type g0, g1, g2, g3, g4, g5; // metrics
    [[maybe_unused]] simd_type d00 = {1.0};
    [[maybe_unused]] simd_type d01 = {0.0};
    [[maybe_unused]] simd_type d11 = {1.0};
    [[maybe_unused]] simd_type d02 = {0.0};
    [[maybe_unused]] simd_type d12 = {0.0};
    [[maybe_unused]] simd_type d22 = {1.0}; // var diffusion terms
    simd_type td0, td1, td2, td3, td4, td5, td6, td7,
        td8; // temp terms for vardiff

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        d02 = constVarDiff[3];
        d12 = constVarDiff[4];
        d22 = constVarDiff[5];
    }

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

    for (unsigned int r = 0; r < nq2; ++r)
    {
        simd_type h1 = hfac1[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;
                // 1. get deriv in std coordinate
                simd_type prod_sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i)
                {
                    simd_type v1 = D0[i * nq0 + p];      // Load 1x
                    simd_type v2 = in[cnt_qr * nq0 + i]; // Load 1x
                    prod_sum.fma(v1, v2);
                }
                // prod_sum.store(deriv0 + cnt * simd_type::width);
                simd_type prod_sum1 = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    unsigned int cnt_jr = r * nq1 + j;
                    simd_type v1        = D1[j * nq1 + q];      // Load 1x
                    simd_type v2        = in[cnt_jr * nq0 + p]; // Load 1x
                    prod_sum1.fma(v1, v2);
                }
                // prod_sum1.store(deriv1 + cnt * simd_type::width);
                simd_type prod_sum2 = 0.0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    unsigned int cnt_qk = (k * nq1 + q);
                    simd_type v2        = D2[k * nq2 + r];      // Load 1x
                    simd_type v1        = in[cnt_qk * nq0 + p]; // Load 1x
                    prod_sum2.fma(v1, v2);
                }
                // prod_sum2.store(deriv2 + cnt * simd_type::width);

                // 3. apply diffusion coeff to deriv and fill output
                simd_type h0 = hfac0[p];

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
                simd_type tmp1 = h1 * (h0 * df2 + df0);
                simd_type tmp2 = h1 * (h0 * df5 + df3);
                simd_type tmp3 = h1 * (h0 * df8 + df6);

                if (!isConstVarDiff && !isVarDiff)
                {
                    g0 = tmp1 * tmp1;
                    g0.fma(tmp2, tmp2);
                    g0.fma(tmp3, tmp3);

                    g3 = df1 * tmp1;
                    g3.fma(df4, tmp2);
                    g3.fma(df7, tmp3);

                    g4 = df2 * tmp1;
                    g4.fma(df5, tmp2);
                    g4.fma(df8, tmp3);

                    g1 = df1 * df1;
                    g1.fma(df4, df4);
                    g1.fma(df7, df7);

                    g2 = df2 * df2;
                    g2.fma(df5, df5);
                    g2.fma(df8, df8);

                    g5 = df1 * df2;
                    g5.fma(df4, df5);
                    g5.fma(df7, df8);
                }
                else
                {
                    // vardiff
                    if (isVarDiff)
                    {
                        d00 = varD00[cnt];
                        d01 = varD01[cnt];
                        d11 = varD11[cnt];
                        d02 = varD02[cnt];
                        d12 = varD12[cnt];
                        d22 = varD22[cnt];
                    }

                    td0 = tmp1 * d00;
                    td0.fma(tmp2, d01);
                    td0.fma(tmp3, d02);

                    td1 = tmp1 * d01;
                    td1.fma(tmp2, d11);
                    td1.fma(tmp3, d12);

                    td2 = tmp1 * d02;
                    td2.fma(tmp2, d12);
                    td2.fma(tmp3, d22);

                    td3 = df1 * d00;
                    td3.fma(df4, d01);
                    td3.fma(df7, d02);

                    td4 = df1 * d01;
                    td4.fma(df4, d11);
                    td4.fma(df7, d12);

                    td5 = df1 * d02;
                    td5.fma(df4, d12);
                    td5.fma(df7, d22);

                    td6 = df2 * d00;
                    td6.fma(df5, d01);
                    td6.fma(df8, d02);

                    td7 = df2 * d01;
                    td7.fma(df5, d11);
                    td7.fma(df8, d12);

                    td8 = df2 * d02;
                    td8.fma(df5, d12);
                    td8.fma(df8, d22);

                    g0 = td0 * tmp1;
                    g0.fma(td1, tmp2);
                    g0.fma(td2, tmp3);

                    g3 = td0 * df1;
                    g3.fma(td1, df4);
                    g3.fma(td2, df7);

                    g4 = td0 * df2;
                    g4.fma(td1, df5);
                    g4.fma(td2, df8);

                    g1 = td3 * df1;
                    g1.fma(td4, df4);
                    g1.fma(td5, df7);

                    g5 = td3 * df2;
                    g5.fma(td4, df5);
                    g5.fma(td5, df8);

                    g2 = td6 * df2;
                    g2.fma(td7, df5);
                    g2.fma(td8, df8);
                }

                // store tensor derivs if required
                if (deriv0)
                {
                    deriv0[cnt] = prod_sum;
                }
                if (deriv1)
                {
                    deriv1[cnt] = prod_sum1;
                }
                if (deriv2)
                {
                    deriv2[cnt] = prod_sum2;
                }

                tmp1 = g0 * prod_sum;
                tmp1.fma(g3, prod_sum1);
                tmp1.fma(g4, prod_sum2);
                diffderiv0[cnt] = tmp1;

                tmp2 = g3 * prod_sum;
                tmp2.fma(g1, prod_sum1);
                tmp2.fma(g5, prod_sum2);
                diffderiv1[cnt] = tmp2;

                tmp3 = g4 * prod_sum;
                tmp3.fma(g5, prod_sum1);
                tmp3.fma(g2, prod_sum2);
                diffderiv2[cnt] = tmp3;
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeffPyrKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *D2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac1, const simd_type *hfac2, simd_type *diffderiv0,
    simd_type *diffderiv1, simd_type *diffderiv2, simd_type *deriv0,
    simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type g0, g1, g2, g3, g4, g5; // metrics
    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0}; // var diffusion terms
    simd_type td0, td1, td2, td3, td4, td5, td6, td7,
        td8; // temp terms for vardiff

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        d02 = constVarDiff[3];
        d12 = constVarDiff[4];
        d22 = constVarDiff[5];
    }

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

    for (unsigned int r = 0; r < nq2; ++r)
    {
        simd_type h2 = hfac2[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type h1        = hfac1[q];
            simd_type h1h2      = h1 * h2;
            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;
                // 1. get deriv in std coordinate
                simd_type prod_sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i)
                {
                    simd_type v1 = D0[i * nq0 + p];      // Load 1x
                    simd_type v2 = in[cnt_qr * nq0 + i]; // Load 1x
                    prod_sum.fma(v1, v2);
                }
                // prod_sum.store(deriv0 + cnt * simd_type::width);
                simd_type prod_sum1 = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    unsigned int cnt_jr = r * nq1 + j;
                    simd_type v1        = D1[j * nq1 + q];      // Load 1x
                    simd_type v2        = in[cnt_jr * nq0 + p]; // Load 1x
                    prod_sum1.fma(v1, v2);
                }
                // prod_sum1.store(deriv1 + cnt * simd_type::width);
                simd_type prod_sum2 = 0.0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    unsigned int cnt_qk = (k * nq1 + q);
                    simd_type v2        = D2[k * nq2 + r];      // Load 1x
                    simd_type v1        = in[cnt_qk * nq0 + p]; // Load 1x
                    prod_sum2.fma(v1, v2);
                }
                // prod_sum2.store(deriv2 + cnt * simd_type::width);

                // 3. apply diffusion coeff to deriv and fill output
                simd_type h0   = hfac0[p];
                simd_type h0h2 = h0 * h2;

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
                simd_type tmp0 = h2 * df0;
                tmp0.fma(h0h2, df2);
                simd_type tmp1 = h2 * df3;
                tmp1.fma(h0h2, df5);
                simd_type tmp2 = h2 * df6;
                tmp2.fma(h0h2, df8);

                simd_type tmp3 = h2 * df1;
                tmp3.fma(h1h2, df2);
                simd_type tmp4 = h2 * df4;
                tmp4.fma(h1h2, df5);
                simd_type tmp5 = h2 * df7;
                tmp5.fma(h1h2, df8);

                if (!isConstVarDiff && !isVarDiff)
                {
                    g0 = tmp0 * tmp0;
                    g0.fma(tmp1, tmp1);
                    g0.fma(tmp2, tmp2);

                    g1 = tmp3 * tmp3;
                    g1.fma(tmp4, tmp4);
                    g1.fma(tmp5, tmp5);

                    g2 = df2 * df2;
                    g2.fma(df5, df5);
                    g2.fma(df8, df8);

                    g3 = tmp0 * tmp3;
                    g3.fma(tmp1, tmp4);
                    g3.fma(tmp2, tmp5);

                    g4 = df2 * tmp0;
                    g4.fma(df5, tmp1);
                    g4.fma(df8, tmp2);

                    g5 = df2 * tmp3;
                    g5.fma(df5, tmp4);
                    g5.fma(df8, tmp5);
                }
                else
                {
                    if (isVarDiff)
                    {
                        d00 = varD00[cnt];
                        d01 = varD01[cnt];
                        d11 = varD11[cnt];
                        d02 = varD02[cnt];
                        d12 = varD12[cnt];

                        d22 = varD22[cnt];
                    }

                    td0 = tmp0 * d00;
                    td0.fma(tmp1, d01);
                    td0.fma(tmp2, d02);

                    td1 = tmp0 * d01;
                    td1.fma(tmp1, d11);
                    td1.fma(tmp2, d12);

                    td2 = tmp0 * d02;
                    td2.fma(tmp1, d12);
                    td2.fma(tmp2, d22);

                    td3 = tmp3 * d00;
                    td3.fma(tmp4, d01);
                    td3.fma(tmp5, d02);

                    td4 = tmp3 * d01;
                    td4.fma(tmp4, d11);
                    td4.fma(tmp5, d12);

                    td5 = tmp3 * d02;
                    td5.fma(tmp4, d12);
                    td5.fma(tmp5, d22);

                    td6 = df2 * d00;
                    td6.fma(df5, d01);
                    td6.fma(df8, d02);

                    td7 = df2 * d01;
                    td7.fma(df5, d11);
                    td7.fma(df8, d12);

                    td8 = df2 * d02;
                    td8.fma(df5, d12);
                    td8.fma(df8, d22);

                    g0 = td0 * tmp0;
                    g0.fma(td1, tmp1);
                    g0.fma(td2, tmp2);

                    g3 = td0 * tmp3;
                    g3.fma(td1, tmp4);
                    g3.fma(td2, tmp5);

                    g4 = td0 * df2;
                    g4.fma(td1, df5);
                    g4.fma(td2, df8);

                    g1 = td3 * tmp3;
                    g1.fma(td4, tmp4);
                    g1.fma(td5, tmp5);

                    g5 = td3 * df2;
                    g5.fma(td4, df5);
                    g5.fma(td5, df8);

                    g2 = td6 * df2;
                    g2.fma(td7, df5);
                    g2.fma(td8, df8);
                }

                // store tensor derivs if required
                if (deriv0)
                {
                    deriv0[cnt] = prod_sum;
                }
                if (deriv1)
                {
                    deriv1[cnt] = prod_sum1;
                }
                if (deriv2)
                {
                    deriv2[cnt] = prod_sum2;
                }

                tmp1 = g0 * prod_sum;
                tmp1.fma(g3, prod_sum1);
                tmp1.fma(g4, prod_sum2);
                diffderiv0[cnt] = tmp1;

                tmp2 = g3 * prod_sum;
                tmp2.fma(g1, prod_sum1);
                tmp2.fma(g5, prod_sum2);
                diffderiv1[cnt] = tmp2;

                tmp3 = g4 * prod_sum;
                tmp3.fma(g5, prod_sum1);
                tmp3.fma(g2, prod_sum2);
                diffderiv2[cnt] = tmp3;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeff2DKernel(
    const unsigned int nq0, const unsigned int nq1, const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *df_ptr, [[maybe_unused]] const simd_type *h0,
    [[maybe_unused]] const simd_type *h1, simd_type *diffderiv0,
    simd_type *diffderiv1, simd_type *deriv0 = nullptr,
    simd_type *deriv1 = nullptr)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                  SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        TensorDerivWithDiffuCoeffTriKernel<DEFORMED>(
            nq0, nq1, isConstVarDiff, constVarDiff, isVarDiff, varD00, varD01,
            varD11, in, D0, D1, df_ptr, h0, h1, diffderiv0, diffderiv1, deriv0,
            deriv1);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
    {
        TensorDerivWithDiffuCoeffQuadKernel<DEFORMED>(
            nq0, nq1, isConstVarDiff, constVarDiff, isVarDiff, varD00, varD01,
            varD11, in, D0, D1, df_ptr, diffderiv0, diffderiv1, deriv0, deriv1);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void TensorDerivWithDiffuCoeff3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const std::vector<typename simd_type::scalarType> &constVarDiff,
    const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *in, const simd_type *D0, const simd_type *D1,
    const simd_type *D2, const simd_type *df_ptr,
    [[maybe_unused]] const simd_type *h0, [[maybe_unused]] const simd_type *h1,
    [[maybe_unused]] const simd_type *h2, [[maybe_unused]] const simd_type *h3,
    simd_type *diffderiv0, simd_type *diffderiv1, simd_type *diffderiv2,
    simd_type *deriv0 = nullptr, simd_type *deriv1 = nullptr,
    simd_type *deriv2 = nullptr)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        TensorDerivWithDiffuCoeffHexKernel<DEFORMED, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, in, D0, D1, D2, df_ptr,
            diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1, deriv2);
    }
    else if constexpr ((SHAPE_TYPE == LibUtilities::eTetrahedron) ||
                       (SHAPE_TYPE == LibUtilities::eNodalTet))
    {
        TensorDerivWithDiffuCoeffTetKernel<DEFORMED, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, in, D0, D1, D2, df_ptr, h0,
            h1, h2, h3, diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1,
            deriv2);
    }
    else if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                       (SHAPE_TYPE == LibUtilities::eNodalPrism))
    {
        TensorDerivWithDiffuCoeffPrismKernel<DEFORMED, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, in, D0, D1, D2, df_ptr, h0,
            h3, diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1, deriv2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        TensorDerivWithDiffuCoeffPyrKernel<DEFORMED, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, in, D0, D1, D2, df_ptr, h0,
            h1, h3, diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1, deriv2);
    }
}
