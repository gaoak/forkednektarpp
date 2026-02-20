///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianSerialAVXSumFacKernels.hpp
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

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffSegKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *df_ptr, simd_type *deriv0)
{
    const auto ndf = ncoord;

    const auto nqTot = nq0;

    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0};
    simd_type df0, df1, df2;
    simd_type metric00;

    if (isConstVarDiff)
    {
        if (ncoord == 1)
        {
            d00 = constVarDiff[0];
        }
        else if (ncoord == 2)
        {
            d00 = constVarDiff[0];
            d01 = constVarDiff[1];
            d11 = constVarDiff[2];
        }
        else if (ncoord == 3)
        {
            d00 = constVarDiff[0];
            d01 = constVarDiff[1];
            d11 = constVarDiff[2];
            d02 = constVarDiff[3];
            d12 = constVarDiff[4];
            d22 = constVarDiff[5];
        }
    }

    // Precompute Laplacian metricsp
    if constexpr (!DEFORMED)
    {
        if (!isConstVarDiff && !isVarDiff)
        {
            metric00 = df0 * df0;
            if (ncoord > 1)
            {
                metric00.fma(df1, df1);
            }
            if (ncoord > 2)
            {
                metric00.fma(df2, df2);
            }
        }
        else if (isConstVarDiff)
        {
            if (ncoord == 1)
            {
                df0      = df_ptr[0];
                metric00 = df0 * df0 * d00;
            }
            else if (ncoord == 2)
            {
                df0            = df_ptr[0];
                df1            = df_ptr[1];
                simd_type tmp0 = d00 * df0;
                tmp0.fma(d01, df1);
                simd_type tmp1 = d01 * df0;
                tmp1.fma(d11, df1);
                metric00 = tmp0 * df0;
                metric00.fma(tmp1, df1);
            }
            else if (ncoord == 3)
            {
                df0            = df_ptr[0];
                df1            = df_ptr[1];
                df2            = df_ptr[2];
                simd_type tmp0 = d00 * df0;
                tmp0.fma(d01, df1);
                tmp0.fma(d02, df2);
                simd_type tmp1 = d01 * df0;
                tmp1.fma(d11, df1);
                tmp1.fma(d12, df2);
                simd_type tmp2 = d02 * df0;
                tmp1.fma(d12, df1);
                tmp1.fma(d22, df2);
                metric00 = tmp0 * df0;
                metric00.fma(tmp1, df1);
                metric00.fma(tmp2, df2);
            }
        }
    }

    // Step 4: Apply Laplacian metrics & inner product
    if (!isVarDiff)
    {
        if constexpr (DEFORMED)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                if (!isConstVarDiff)
                {
                    df0      = df_ptr[i * ndf];
                    metric00 = df0 * df0;
                    if (ncoord > 1)
                    {
                        df1 = df_ptr[i * ndf + 1];
                        metric00.fma(df1, df1);
                    }
                    if (ncoord > 2)
                    {
                        df2 = df_ptr[i * ndf + 2];
                        metric00.fma(df2, df2);
                    }
                }
                else
                {
                    metric00 = df0 * df0 * d00;
                    if (ncoord == 1)
                    {
                        df0      = df_ptr[i * ndf];
                        metric00 = df0 * df0 * d00;
                    }
                    else if (ncoord == 2)
                    {
                        df0            = df_ptr[i * ndf];
                        df1            = df_ptr[i * ndf + 1];
                        simd_type tmp0 = d00 * df0;
                        tmp0.fma(d01, df1);
                        simd_type tmp1 = d01 * df0;
                        tmp1.fma(d11, df1);
                        metric00 = tmp0 * df0;
                        metric00.fma(tmp1, df1);
                    }
                    else if (ncoord == 3)
                    {
                        df0            = df_ptr[i * ndf];
                        df1            = df_ptr[i * ndf + 1];
                        df2            = df_ptr[i * ndf + 2];
                        simd_type tmp0 = d00 * df0;
                        tmp0.fma(d01, df1);
                        tmp0.fma(d02, df2);
                        simd_type tmp1 = d01 * df0;
                        tmp1.fma(d11, df1);
                        tmp1.fma(d12, df2);
                        simd_type tmp2 = d02 * df0;
                        tmp1.fma(d12, df1);
                        tmp1.fma(d22, df2);
                        metric00 = tmp0 * df0;
                        metric00.fma(tmp1, df1);
                        metric00.fma(tmp2, df2);
                    }
                }

                deriv0[i] = metric00 * deriv0[i];
            }
        }
        else
        {
            // Precompute Laplacian metricsp
            df0 = df_ptr[0];

            if (isConstVarDiff)
            {
                metric00 = df0 * df0 * d00;
            }
            else if (isConstVarDiff)
            {
                metric00 = df0 * df0;
            }

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
                if (ncoord == 1)
                {
                    df0      = df_ptr[i * ndf];
                    d00      = varD00[i];
                    metric00 = df0 * df0 * d00;
                }
                else if (ncoord == 2)
                {
                    df0            = df_ptr[i * ndf];
                    df1            = df_ptr[i * ndf + 1];
                    d00            = varD00[i];
                    d01            = varD01[i];
                    d11            = varD11[i];
                    simd_type tmp0 = d00 * df0;
                    tmp0.fma(d01, df1);
                    simd_type tmp1 = d01 * df0;
                    tmp1.fma(d11, df1);
                    metric00 = tmp0 * df0;
                    metric00.fma(tmp1, df1);
                }
                else if (ncoord == 3)
                {
                    df0            = df_ptr[i * ndf];
                    df1            = df_ptr[i * ndf + 1];
                    df2            = df_ptr[i * ndf + 2];
                    d00            = varD00[i];
                    d01            = varD01[i];
                    d11            = varD11[i];
                    d02            = varD02[i];
                    d12            = varD12[i];
                    d22            = varD22[i];
                    simd_type tmp0 = d00 * df0;
                    tmp0.fma(d01, df1);
                    tmp0.fma(d02, df2);
                    simd_type tmp1 = d01 * df0;
                    tmp1.fma(d11, df1);
                    tmp1.fma(d12, df2);
                    simd_type tmp2 = d02 * df0;
                    tmp1.fma(d12, df1);
                    tmp1.fma(d22, df2);
                    metric00 = tmp0 * df0;
                    metric00.fma(tmp1, df1);
                    metric00.fma(tmp2, df2);
                }
                deriv0[i] = metric00 * deriv0[i];
            }
        }
        else
        {
            df0 = df_ptr[0];

            for (unsigned int i = 0; i < nq0; ++i)
            {
                metric00 = df0 * df0 * d00;
                if (ncoord == 1)
                {
                    d00      = varD00[i];
                    metric00 = df0 * df0 * d00;
                }
                else if (ncoord == 2)
                {
                    d00            = varD00[i];
                    d01            = varD01[i];
                    d11            = varD11[i];
                    simd_type tmp0 = d00 * df0;
                    tmp0.fma(d01, df1);
                    simd_type tmp1 = d01 * df0;
                    tmp1.fma(d11, df1);
                    metric00 = tmp0 * df0;
                    metric00.fma(tmp1, df1);
                }
                else if (ncoord == 3)
                {
                    d00            = varD00[i];
                    d01            = varD01[i];
                    d11            = varD11[i];
                    d02            = varD02[i];
                    d12            = varD12[i];
                    d22            = varD22[i];
                    simd_type tmp0 = d00 * df0;
                    tmp0.fma(d01, df1);
                    tmp0.fma(d02, df2);
                    simd_type tmp1 = d01 * df0;
                    tmp1.fma(d11, df1);
                    tmp1.fma(d12, df2);
                    simd_type tmp2 = d02 * df0;
                    tmp1.fma(d12, df1);
                    tmp1.fma(d22, df2);
                    metric00 = tmp0 * df0;
                    metric00.fma(tmp1, df1);
                    metric00.fma(tmp2, df2);
                }

                deriv0[i] = metric00 * deriv0[i];
            }
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJTriKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *df_ptr, const simd_type *hfac0, const simd_type *hfac1,
    simd_type *deriv0, simd_type *deriv1)
{
    const auto ndf = 2 * ncoord;

    simd_type jac = {1.0};
    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0};                // var diffusion terms
    simd_type dtmp0, dtmp1, dtmp2, dtmp3; // temp for vardiff
    simd_type dtmp4, dtmp5;
    simd_type df0, df1, df2, df3, df4, df5;
    simd_type metric00, metric01, metric11;

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        if (ncoord == 3)
        {
            d02 = constVarDiff[3];
            d12 = constVarDiff[4];
            d22 = constVarDiff[5];
        }
    }

    // Precompute Laplacian metricsp
    if constexpr (!DEFORMED)
    {
        jac = jac_ptr[0];

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

    for (unsigned int q = 0; q < nq1; ++q)
    {
        simd_type h1  = hfac1[q];
        simd_type w1J = jac * w1[q];
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt = q * nq0 + p;

            simd_type wJ = w1J * w0[p];

            if constexpr (DEFORMED)
            {
                wJ *= jac_ptr[cnt];
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

            simd_type h0 = hfac0[p];

            // M = [M_00, df1; M_10; df3]
            metric00      = h1 * (df0 + h0 * df1); // M_00
            simd_type tmp = h1 * (df2 + h0 * df3); // M_10

            if (!isConstVarDiff && !isVarDiff)
            {
                metric01 = metric00 * df1;
                metric00 = metric00 * metric00;

                metric01.fma(tmp, df3);
                metric00.fma(tmp, tmp);

                metric11 = df1 * df1;
                metric11.fma(df3, df3);

                if (ncoord == 3)
                {
                    metric00.fma(df4, df4);
                    metric01.fma(df4, df5);
                    metric11.fma(df5, df5);
                }
            }
            else
            {
                if (isVarDiff)
                {
                    d00 = varD00[cnt];
                    d01 = varD01[cnt];
                    d11 = varD11[cnt];
                    if (ncoord == 3)
                    {
                        d02 = varD02[cnt];
                        d12 = varD12[cnt];
                        d22 = varD22[cnt];
                    }
                }

                if (ncoord == 2)
                {
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
                else
                {
                    dtmp0 = metric00 * d00;
                    dtmp0.fma(tmp, d01);
                    dtmp0.fma(df4, d02);
                    dtmp1 = metric00 * d01;
                    dtmp1.fma(metric00, d11);
                    dtmp1.fma(df3, d12);
                    dtmp2 = metric00 * d02;
                    dtmp2.fma(tmp, d12);
                    dtmp2.fma(df3, d22);
                    dtmp3 = df1 * d00;
                    dtmp3.fma(df3, d01);
                    dtmp3.fma(df5, d02);
                    dtmp4 = df1 * d01;
                    dtmp4.fma(df3, d11);
                    dtmp4.fma(df5, d12);
                    dtmp5 = df1 * d02;
                    dtmp5.fma(df3, d12);
                    dtmp5.fma(df5, d22);

                    metric00 = metric00 * dtmp0;
                    metric00.fma(tmp, dtmp1);
                    metric00.fma(df4, dtmp2);
                    metric01 = df1 * dtmp0;
                    metric01.fma(df3, dtmp1);
                    metric01.fma(df5, dtmp2);
                    metric11 = df1 * dtmp3;
                    metric11.fma(df3, dtmp4);
                    metric11.fma(df5, dtmp5);
                }
            }

            simd_type d0 = deriv0[cnt];
            simd_type d1 = deriv1[cnt];

            tmp = metric00 * d0;
            tmp.fma(metric01, d1);
            deriv0[cnt] = tmp * wJ;

            tmp = metric01 * d0;
            tmp.fma(metric11, d1);
            deriv1[cnt] = tmp * wJ;
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJQuadKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *df_ptr, simd_type *deriv0, simd_type *deriv1)
{
    const auto ndf = 2 * ncoord;

    simd_type jac = {1.0};
    simd_type d00 = {1.0};
    simd_type d01 = {0.0};
    simd_type d11 = {1.0};
    simd_type d02 = {0.0};
    simd_type d12 = {0.0};
    simd_type d22 = {1.0};                // var diffusion terms
    simd_type dtmp0, dtmp1, dtmp2, dtmp3; // temp for vardiff
    simd_type dtmp4, dtmp5;
    simd_type df0, df1, df2, df3, df4, df5;
    simd_type metric00, metric01, metric11;

    if (isConstVarDiff)
    {
        d00 = constVarDiff[0];
        d01 = constVarDiff[1];
        d11 = constVarDiff[2];
        if (ncoord == 3)
        {
            d02 = constVarDiff[3];
            d12 = constVarDiff[4];
            d22 = constVarDiff[5];
        }
    }

    // Precompute Laplacian metrics
    if constexpr (!DEFORMED)
    {
        jac = jac_ptr[0];

        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        if (ncoord == 3)
        {
            df4 = df_ptr[4];
            df5 = df_ptr[5];
        }

        if (!isConstVarDiff && !isVarDiff)
        {
            metric00 = df0 * df0;
            metric00.fma(df2, df2);
            metric01 = df0 * df1;
            metric01.fma(df2, df3);
            metric11 = df1 * df1;
            metric11.fma(df3, df3);
            if (ncoord == 3)
            {
                metric00.fma(df4, df4);
                metric01.fma(df4, df5);
                metric11.fma(df5, df5);
            }
        }
        else if (isConstVarDiff)
        {
            if (ncoord == 2)
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
            else
            {
                dtmp0 = df0 * d00;
                dtmp0.fma(df2, d01);
                dtmp0.fma(df4, d02);
                dtmp1 = df0 * d01;
                dtmp1.fma(df2, d11);
                dtmp1.fma(df3, d12);
                dtmp2 = df0 * d02;
                dtmp2.fma(df2, d12);
                dtmp2.fma(df3, d22);
                dtmp3 = df1 * d00;
                dtmp3.fma(df3, d01);
                dtmp3.fma(df5, d02);
                dtmp4 = df1 * d01;
                dtmp4.fma(df3, d11);
                dtmp4.fma(df5, d12);
                dtmp5 = df1 * d02;
                dtmp5.fma(df3, d12);
                dtmp5.fma(df5, d22);

                metric00 = df0 * dtmp0;
                metric00.fma(df2, dtmp1);
                metric00.fma(df4, dtmp2);
                metric01 = df1 * dtmp0;
                metric01.fma(df3, dtmp1);
                metric01.fma(df5, dtmp2);
                metric11 = df1 * dtmp3;
                metric11.fma(df3, dtmp4);
                metric11.fma(df5, dtmp5);
            }
        }
    }

    if (DEFORMED || isVarDiff)
    {
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type w1J = jac * w1[q];
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = q * nq0 + p;

                simd_type wJ = w1J * w0[p];

                if constexpr (DEFORMED)
                {
                    wJ *= jac_ptr[cnt];
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

                    if (ncoord == 3)
                    {
                        metric00.fma(df4, df4);
                        metric01.fma(df4, df5);
                        metric11.fma(df5, df5);
                    }
                }
                else
                {
                    if (isVarDiff)
                    {
                        d00 = varD00[cnt];
                        d01 = varD01[cnt];
                        d11 = varD11[cnt];
                        if (ncoord == 3)
                        {
                            d02 = varD02[cnt];
                            d12 = varD12[cnt];
                            d22 = varD22[cnt];
                        }
                    }

                    if (ncoord == 2)
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
                    else
                    {
                        dtmp0 = df0 * d00;
                        dtmp0.fma(df2, d01);
                        dtmp0.fma(df4, d02);
                        dtmp1 = df0 * d01;
                        dtmp1.fma(df2, d11);
                        dtmp1.fma(df3, d12);
                        dtmp2 = df0 * d02;
                        dtmp2.fma(df2, d12);
                        dtmp2.fma(df3, d22);
                        dtmp3 = df1 * d00;
                        dtmp3.fma(df3, d01);
                        dtmp3.fma(df5, d02);
                        dtmp4 = df1 * d01;
                        dtmp4.fma(df3, d11);
                        dtmp4.fma(df5, d12);
                        dtmp5 = df1 * d02;
                        dtmp5.fma(df3, d12);
                        dtmp5.fma(df5, d22);

                        metric00 = df0 * dtmp0;
                        metric00.fma(df2, dtmp1);
                        metric00.fma(df4, dtmp2);
                        metric01 = df1 * dtmp0;
                        metric01.fma(df3, dtmp1);
                        metric01.fma(df5, dtmp2);
                        metric11 = df1 * dtmp3;
                        metric11.fma(df3, dtmp4);
                        metric11.fma(df5, dtmp5);
                    }
                }

                simd_type d0 = deriv0[cnt];
                simd_type d1 = deriv1[cnt];

                simd_type tmp = metric00 * d0;
                tmp.fma(metric01, d1);
                deriv0[cnt] = tmp * wJ;

                tmp = metric01 * d0;
                tmp.fma(metric11, d1);
                deriv1[cnt] = tmp * wJ;
            }
        }
    }
    else
    { // make use of the precomputed metric
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type w1J = jac * w1[q];
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = q * nq0 + p;

                simd_type wJ = w1J * w0[p];

                simd_type d0 = deriv0[cnt];
                simd_type d1 = deriv1[cnt];

                simd_type tmp = metric00 * d0;
                tmp.fma(metric01, d1);
                deriv0[cnt] = tmp * wJ;

                tmp = metric01 * d0;
                tmp.fma(metric11, d1);
                deriv1[cnt] = tmp * wJ;
            }
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJHexKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *w2, const simd_type *df_ptr, simd_type *deriv0,
    simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type jac = {1.0};
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

    // Precompute Laplacian metrics
    if constexpr (!DEFORMED)
    {
        jac = jac_ptr[0];
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        df4 = df_ptr[4];
        df5 = df_ptr[5];
        df6 = df_ptr[6];
        df7 = df_ptr[7];
        df8 = df_ptr[8];

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
    if (DEFORMED || isVarDiff)
    {
        for (unsigned int r = 0; r < nq2; ++r)
        {
            simd_type w2J = jac * w2[r];
            for (unsigned int q = 0; q < nq1; ++q)
            {
                simd_type w21J      = w2J * w1[q];
                unsigned int cnt_qr = (r * nq1 + q);
                for (unsigned int p = 0; p < nq0; ++p)
                {
                    unsigned int cnt = cnt_qr * nq0 + p;

                    simd_type wJ = w21J * w0[p];

                    // 2. evaluate diffusion coeff if deformed
                    if constexpr (DEFORMED)
                    {
                        wJ *= jac_ptr[cnt];
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

                    simd_type d0 = deriv0[cnt];
                    simd_type d1 = deriv1[cnt];
                    simd_type d2 = deriv2[cnt];

                    // 3. apply diffusion coeff to deriv and get output
                    simd_type tmp0 = metric00 * d0;
                    tmp0.fma(metric01, d1);
                    tmp0.fma(metric02, d2);
                    deriv0[cnt] = tmp0 * wJ;

                    simd_type tmp1 = metric01 * d0;
                    tmp1.fma(metric11, d1);
                    tmp1.fma(metric12, d2);
                    deriv1[cnt] = tmp1 * wJ;

                    simd_type tmp2 = metric02 * d0;
                    tmp2.fma(metric12, d1);
                    tmp2.fma(metric22, d2);
                    deriv2[cnt] = tmp2 * wJ;
                }
            }
        }
    }
    else
    { // Make use of the precomputed metric

        for (unsigned int r = 0; r < nq2; ++r)
        {
            simd_type w2J = jac * w2[r];
            for (unsigned int q = 0; q < nq1; ++q)
            {
                simd_type w21J      = w2J * w1[q];
                unsigned int cnt_qr = (r * nq1 + q);
                for (unsigned int p = 0; p < nq0; ++p)
                {
                    unsigned int cnt = cnt_qr * nq0 + p;

                    simd_type wJ = w21J * w0[p];

                    simd_type d0 = deriv0[cnt];
                    simd_type d1 = deriv1[cnt];
                    simd_type d2 = deriv2[cnt];

                    simd_type tmp0 = metric00 * d0;
                    tmp0.fma(metric01, d1);
                    tmp0.fma(metric02, d2);
                    deriv0[cnt] = tmp0 * wJ;

                    simd_type tmp1 = metric01 * d0;
                    tmp1.fma(metric11, d1);
                    tmp1.fma(metric12, d2);
                    deriv1[cnt] = tmp1 * wJ;

                    simd_type tmp2 = metric02 * d0;
                    tmp2.fma(metric12, d1);
                    tmp2.fma(metric22, d2);
                    deriv2[cnt] = tmp2 * wJ;
                }
            }
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJTetKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    [[maybe_unused]] const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *w2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac1, const simd_type *hfac2, const simd_type *hfac3,
    simd_type *deriv0, simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type jac = {1.0};
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
        jac = jac_ptr[0];
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
        simd_type h3  = hfac3[r];
        simd_type w2J = jac * w2[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type h1   = hfac1[q];
            simd_type h2   = hfac2[q];
            simd_type h2h3 = h2 * h3;
            simd_type h1h3 = h1 * h3;
            simd_type w21J = w2J * w1[q];

            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;

                simd_type wJ = w21J * w0[p];

                // 3. apply diffusion coeff to deriv and fill output
                if constexpr (DEFORMED)
                {
                    wJ *= jac_ptr[cnt];
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

                simd_type tmp0 = h0h2h3 * (df1 + df2);
                tmp0.fma(df0, h2h3);
                simd_type tmp1 = h0h2h3 * (df4 + df5);
                tmp1.fma(df3, h2h3);
                simd_type tmp2 = h0h2h3 * (df7 + df8);
                tmp2.fma(df6, h2h3);

                simd_type tmp3 = df1 * h3;
                tmp3.fma(df2, h1h3);
                simd_type tmp4 = df4 * h3;
                tmp4.fma(df5, h1h3);
                simd_type tmp5 = df7 * h3;
                tmp5.fma(df8, h1h3);

                if (!isConstVarDiff && !isVarDiff)
                {
                    g0 = tmp0 * tmp0;
                    g0.fma(tmp1, tmp1);
                    g0.fma(tmp2, tmp2);

                    g4 = df2 * tmp0;
                    g4.fma(df5, tmp1);
                    g4.fma(df8, tmp2);

                    g3 = tmp0 * tmp3;
                    g3.fma(tmp1, tmp4);
                    g3.fma(tmp2, tmp5);

                    g1 = tmp3 * tmp3;
                    g1.fma(tmp4, tmp4);
                    g1.fma(tmp5, tmp5);

                    g5 = df2 * tmp3;
                    g5.fma(df5, tmp4);
                    g5.fma(df8, tmp5);

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

                simd_type d0 = deriv0[cnt];
                simd_type d1 = deriv1[cnt];
                simd_type d2 = deriv2[cnt];

                tmp0 = g0 * d0;
                tmp0.fma(g3, d1);
                tmp0.fma(g4, d2);
                deriv0[cnt] = tmp0 * wJ;

                tmp1 = g3 * d0;
                tmp1.fma(g1, d1);
                tmp1.fma(g5, d2);
                deriv1[cnt] = tmp1 * wJ;

                tmp2 = g4 * d0;
                tmp2.fma(g5, d1);
                tmp2.fma(g2, d2);
                deriv2[cnt] = tmp2 * wJ;
            }
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJPrismKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *w2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac3, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type jac = {1.0};
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
        jac = jac_ptr[0];
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
        simd_type h3  = hfac3[r];
        simd_type w2J = jac * w2[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            unsigned int cnt_qr = (r * nq1 + q);
            simd_type w21J      = w2J * w1[q];
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;
                simd_type wJ     = w21J * w0[p];

                // 3. apply diffusion coeff to deriv and fill output
                simd_type h0 = hfac0[p];

                if constexpr (DEFORMED)
                {
                    wJ *= jac_ptr[cnt];
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
                simd_type tmp0 = h3 * (h0 * df2 + df0);
                simd_type tmp1 = h3 * (h0 * df5 + df3);
                simd_type tmp2 = h3 * (h0 * df8 + df6);

                if (!isConstVarDiff && !isVarDiff)
                {
                    g0 = tmp0 * tmp0;
                    g0.fma(tmp1, tmp1);
                    g0.fma(tmp2, tmp2);

                    g3 = df1 * tmp0;
                    g3.fma(df4, tmp1);
                    g3.fma(df7, tmp2);

                    g4 = df2 * tmp0;
                    g4.fma(df5, tmp1);
                    g4.fma(df8, tmp2);

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

                    td0 = tmp0 * d00;
                    td0.fma(tmp1, d01);
                    td0.fma(tmp2, d02);

                    td1 = tmp0 * d01;
                    td1.fma(tmp1, d11);
                    td1.fma(tmp2, d12);

                    td2 = tmp0 * d02;
                    td2.fma(tmp1, d12);
                    td2.fma(tmp2, d22);

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

                    g0 = td0 * tmp0;
                    g0.fma(td1, tmp1);
                    g0.fma(td2, tmp2);

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

                simd_type d0 = deriv0[cnt];
                simd_type d1 = deriv1[cnt];
                simd_type d2 = deriv2[cnt];

                tmp0 = g0 * d0;
                tmp0.fma(g3, d1);
                tmp0.fma(g4, d2);
                deriv0[cnt] = tmp0 * wJ;

                tmp1 = g3 * d0;
                tmp1.fma(g1, d1);
                tmp1.fma(g5, d2);
                deriv1[cnt] = tmp1 * wJ;

                tmp2 = g4 * d0;
                tmp2.fma(g5, d1);
                tmp2.fma(g2, d2);
                deriv2[cnt] = tmp2 * wJ;
            }
        }
    }
}

template <bool DEFORMED, bool SCALE, typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJPyrKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *w2, const simd_type *df_ptr, const simd_type *hfac0,
    const simd_type *hfac1, const simd_type *hfac3, simd_type *deriv0,
    simd_type *deriv1, simd_type *deriv2)
{
    constexpr auto ndf = 9;

    simd_type jac = {1.0};
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
        jac = jac_ptr[0];
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
        simd_type h3  = hfac3[r];
        simd_type w2J = jac * w2[r];
        for (unsigned int q = 0; q < nq1; ++q)
        {
            simd_type h1        = hfac1[q];
            simd_type h1h3      = h1 * h3;
            simd_type w21J      = w2J * w1[q];
            unsigned int cnt_qr = (r * nq1 + q);
            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = cnt_qr * nq0 + p;

                // 3. apply diffusion coeff to deriv and fill output
                simd_type h0   = hfac0[p];
                simd_type h0h3 = h0 * h3;
                simd_type wJ   = w21J * w0[p];

                if constexpr (DEFORMED)
                {
                    wJ *= jac_ptr[cnt];
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
                simd_type tmp0 = h3 * df0;
                tmp0.fma(h0h3, df2);
                simd_type tmp1 = h3 * df3;
                tmp1.fma(h0h3, df5);
                simd_type tmp2 = h3 * df6;
                tmp2.fma(h0h3, df8);

                simd_type tmp3 = h3 * df1;
                tmp3.fma(h1h3, df2);
                simd_type tmp4 = h3 * df4;
                tmp4.fma(h1h3, df5);
                simd_type tmp5 = h3 * df7;
                tmp5.fma(h1h3, df8);

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

                simd_type d0 = deriv0[cnt];
                simd_type d1 = deriv1[cnt];
                simd_type d2 = deriv2[cnt];

                tmp0 = g0 * d0;
                tmp0.fma(g3, d1);
                tmp0.fma(g4, d2);
                deriv0[cnt] = tmp0 * wJ;

                tmp1 = g3 * d0;
                tmp1.fma(g1, d1);
                tmp1.fma(g5, d2);
                deriv1[cnt] = tmp1 * wJ;

                tmp2 = g4 * d0;
                tmp2.fma(g5, d1);
                tmp2.fma(g2, d2);
                deriv2[cnt] = tmp2 * wJ;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool SCALE,
          typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJ2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *df_ptr, [[maybe_unused]] const simd_type *h0,
    [[maybe_unused]] const simd_type *h1, simd_type *deriv0, simd_type *deriv1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                  SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        DiffusionCoeffwithWJTriKernel<DEFORMED, SCALE, simd_type>(
            ncoord, nq0, nq1, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, df_ptr, h0,
            h1, deriv0, deriv1);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
    {
        DiffusionCoeffwithWJQuadKernel<DEFORMED, SCALE, simd_type>(
            ncoord, nq0, nq1, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, df_ptr,
            deriv0, deriv1);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool SCALE,
          typename simd_type>
NEK_FORCE_INLINE static void DiffusionCoeffwithWJ3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool isConstVarDiff,
    const typename simd_type::scalarType *constVarDiff, const bool isVarDiff,
    const std::vector<typename simd_type::scalarType> &varD00,
    const std::vector<typename simd_type::scalarType> &varD01,
    const std::vector<typename simd_type::scalarType> &varD11,
    const std::vector<typename simd_type::scalarType> &varD02,
    const std::vector<typename simd_type::scalarType> &varD12,
    const std::vector<typename simd_type::scalarType> &varD22,
    const simd_type *jac_ptr, const simd_type *w0, const simd_type *w1,
    const simd_type *w2, const simd_type *df_ptr,
    [[maybe_unused]] const simd_type *h0, [[maybe_unused]] const simd_type *h1,
    [[maybe_unused]] const simd_type *h2, [[maybe_unused]] const simd_type *h3,
    simd_type *deriv0, simd_type *deriv1, simd_type *deriv2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        DiffusionCoeffwithWJHexKernel<DEFORMED, SCALE, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, w2, df_ptr,
            deriv0, deriv1, deriv2);
    }
    else if constexpr ((SHAPE_TYPE == LibUtilities::eTetrahedron) ||
                       (SHAPE_TYPE == LibUtilities::eNodalTet))
    {
        DiffusionCoeffwithWJTetKernel<DEFORMED, SCALE, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, w2, df_ptr,
            h0, h1, h2, h3, deriv0, deriv1, deriv2);
    }
    else if constexpr ((SHAPE_TYPE == LibUtilities::ePrism) ||
                       (SHAPE_TYPE == LibUtilities::eNodalPrism))
    {
        DiffusionCoeffwithWJPrismKernel<DEFORMED, SCALE, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, w2, df_ptr,
            h0, h3, deriv0, deriv1, deriv2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
    {
        DiffusionCoeffwithWJPyrKernel<DEFORMED, SCALE, simd_type>(
            nq0, nq1, nq2, isConstVarDiff, constVarDiff, isVarDiff, varD00,
            varD01, varD11, varD02, varD12, varD22, jac_ptr, w0, w1, w2, df_ptr,
            h0, h1, h3, deriv0, deriv1, deriv2);
    }
}

} // namespace Nektar::Operators::detail
