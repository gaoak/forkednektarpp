///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzCUDASumFacKernels.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int insize, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *__restrict__ bwd, TData *out, const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    TData metric[3];
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 1)
            {
                metric[0] = diffCoeff[0] * df[ilane];
            }
            else if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[warpsize + ilane];
                metric[1] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[warpsize + ilane];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[warpsize + ilane] +
                            diffCoeff[3] * df[2 * warpsize + ilane];
                metric[1] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[warpsize + ilane] +
                            diffCoeff[4] * df[2 * warpsize + ilane];
                metric[2] = diffCoeff[3] * df[ilane] +
                            diffCoeff[4] * df[warpsize + ilane] +
                            diffCoeff[5] * df[2 * warpsize + ilane];
            }
        }
    }

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[0] * df[dfindex];
                }
                else if (ncoord == 2)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[warpsize + dfindex];
                    metric[1] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[warpsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[warpsize + dfindex] +
                                diffCoeff[3] * df[2 * warpsize + dfindex];
                    metric[1] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[warpsize + dfindex] +
                                diffCoeff[4] * df[2 * warpsize + dfindex];
                    metric[2] = diffCoeff[3] * df[dfindex] +
                                diffCoeff[4] * df[warpsize + dfindex] +
                                diffCoeff[5] * df[2 * warpsize + dfindex];
                }
            }
        }

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            if (diffCoeff)
            {
                sum += metric[d] * in[d * insize * nq0 + index];
            }
            else
            {
                sum +=
                    df[d * warpsize + dfindex] * in[d * insize * nq0 + index];
            }
        }

        if constexpr (DEFORMED)
        {
            bwd[index] *= lambda * jac[index] * w0[i];
            out[index] = sum * jac[index] * w0[i];
        }
        else
        {
            bwd[index] *= lambda * jac[0] * w0[i];
            out[index] = sum * jac[0] * w0[i];
        }
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int insize, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *__restrict__ bwd, TData *out, const TData lambda)
{
    unsigned int dfsize = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
    }

    TData metric[3];
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 1)
            {
                metric[0] = diffCoeff[0] * df[0];
            }
            else if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1];
                metric[1] = diffCoeff[2] * df[0] + diffCoeff[3] * df[1];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1] +
                            diffCoeff[2] * df[2];
                metric[1] = diffCoeff[3] * df[0] + diffCoeff[4] * df[1] +
                            diffCoeff[5] * df[2];
                metric[2] = diffCoeff[6] * df[0] + diffCoeff[7] * df[1] +
                            diffCoeff[8] * df[2];
            }
        }
    }

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[0] * df[i];
                }
                else if (ncoord == 2)
                {
                    metric[0] =
                        diffCoeff[0] * df[i] + diffCoeff[1] * df[dfsize + i];
                    metric[1] =
                        diffCoeff[2] * df[i] + diffCoeff[3] * df[dfsize + i];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[i] +
                                diffCoeff[1] * df[dfsize + i] +
                                diffCoeff[2] * df[2 * dfsize + i];
                    metric[1] = diffCoeff[3] * df[i] +
                                diffCoeff[4] * df[dfsize + i] +
                                diffCoeff[5] * df[2 * dfsize + i];
                    metric[2] = diffCoeff[6] * df[i] +
                                diffCoeff[7] * df[dfsize + i] +
                                diffCoeff[8] * df[2 * dfsize + i];
                }
            }
        }

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            if (diffCoeff)
            {
                sum += metric[d] * in[d * insize * nq0 + i];
            }
            else
            {
                sum += df[d * dfsize + dfindex] * in[d * insize * nq0 + i];
            }
        }

        if constexpr (DEFORMED)
        {
            bwd[i] *= lambda * jac[i] * w0[i];
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
            bwd[i] *= lambda * jac[0] * w0[i];
            out[i] = sum * jac[0] * w0[i];
        }
    }

    __syncthreads();
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const unsigned int insize,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *__restrict__ bwd,
    TData *__restrict__ out0, TData *__restrict__ out1, const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    TData metric[6];
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane];
                metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                            diffCoeff[1] * df[3 * warpsize + ilane];
                metric[2] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[2 * warpsize + ilane];
                metric[3] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane] +
                            diffCoeff[3] * df[4 * warpsize + ilane];
                metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                            diffCoeff[1] * df[3 * warpsize + ilane] +
                            diffCoeff[3] * df[5 * warpsize + ilane];
                metric[2] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane] +
                            diffCoeff[3] * df[4 * warpsize + ilane];
                metric[3] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane] +
                            diffCoeff[4] * df[5 * warpsize + ilane];
                metric[4] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[2 * warpsize + ilane] +
                            diffCoeff[4] * df[4 * warpsize + ilane];
                metric[5] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane] +
                            diffCoeff[4] * df[5 * warpsize + ilane];
            }
        }
    }

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            if constexpr (DEFORMED)
            {
                if (diffCoeff)
                {
                    if (ncoord == 2)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex];
                        metric[2] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[2 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex];
                    }
                    else if (ncoord == 3)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[3] * df[4 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex] +
                                    diffCoeff[3] * df[5 * warpsize + dfindex];
                        metric[2] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[3] * df[4 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex];
                        metric[4] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[2 * warpsize + dfindex] +
                                    diffCoeff[4] * df[4 * warpsize + dfindex];
                        metric[5] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex];
                    }
                }
            }

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0; d < ncoord; ++d)
            {
                TData tmp = in[d * insize * nqTot + index];

                if (diffCoeff)
                {
                    sum1 += metric[2u * d] * tmp;
                    sum2 += metric[2u * d + 1u] * tmp;
                }
                else
                {
                    sum1 += df[(2u * d) * warpsize + dfindex] * tmp;
                    sum2 += df[(2u * d + 1u) * warpsize + dfindex] * tmp;
                }
            }

            TData tmpQ = w0[i] * w1[j];
            if constexpr (DEFORMED)
            {
                tmpQ *= jac[warpsize * cnt_ji + ilane];
            }
            else
            {
                tmpQ *= jac[0];
            }

            bwd[index] *= lambda * tmpQ;

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                out0[index] = sum1 * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                out0[index] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int insize, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *__restrict__ bwd, TData *out0,
    TData *out1, TData *metric, const TData lambda)
{
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                for (unsigned int idx = idx0; idx < 4u; idx += stride)
                {
                    metric[idx] =
                        diffCoeff[(idx / 2u) * 2u] * df[idx % 2u] +
                        diffCoeff[(idx / 2u) * 2u + 1u] * df[idx % 2u + 2u];
                }
            }
            else if (ncoord == 3)
            {
                for (unsigned int idx = idx0; idx < 6u; idx += stride)
                {
                    metric[idx] =
                        diffCoeff[(idx / 3u) * 3u] * df[idx % 2u] +
                        diffCoeff[(idx / 3u) * 3u + 1u] * df[idx % 2u + 2u] +
                        diffCoeff[(idx / 3u) * 3u + 2u] * df[idx % 2u + 4u];
                }
            }

            __syncthreads();
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 2)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex];
                    metric[2] = diffCoeff[2] * df[dfindex] +
                                diffCoeff[3] * df[2 * dfsize + dfindex];
                    metric[3] = diffCoeff[2] * df[1 * dfsize + dfindex] +
                                diffCoeff[3] * df[3 * dfsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex] +
                                diffCoeff[2] * df[4 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex] +
                                diffCoeff[2] * df[5 * dfsize + dfindex];
                    metric[2] = diffCoeff[3] * df[dfindex] +
                                diffCoeff[4] * df[2 * dfsize + dfindex] +
                                diffCoeff[5] * df[4 * dfsize + dfindex];
                    metric[3] = diffCoeff[3] * df[1 * dfsize + dfindex] +
                                diffCoeff[4] * df[3 * dfsize + dfindex] +
                                diffCoeff[5] * df[5 * dfsize + dfindex];
                    metric[4] = diffCoeff[6] * df[dfindex] +
                                diffCoeff[7] * df[2 * dfsize + dfindex] +
                                diffCoeff[8] * df[4 * dfsize + dfindex];
                    metric[5] = diffCoeff[6] * df[1 * dfsize + dfindex] +
                                diffCoeff[7] * df[3 * dfsize + dfindex] +
                                diffCoeff[8] * df[5 * dfsize + dfindex];
                }
            }
        }

        TData sum1 = 0.0, sum2 = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            TData tmp = in[d * insize * nqTot + idx];

            if (diffCoeff)
            {
                sum1 += metric[2u * d] * tmp;
                sum2 += metric[2u * d + 1u] * tmp;
            }
            else
            {
                sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
            }
        }

        TData tmpQ = w0[i] * w1[j];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        bwd[idx] *= lambda * tmpQ;

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            out0[idx] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
    }

    __syncthreads();
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int insize,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1,
    [[maybe_unused]] const TData *__restrict__ f1m,
    [[maybe_unused]] const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *__restrict__ bwd,
    TData *__restrict__ out0, TData *__restrict__ out1,
    TData *__restrict__ out2, const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData metric[9];
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            metric[0] = diffCoeff[0] * df[ilane] +
                        diffCoeff[1] * df[3 * warpsize + ilane] +
                        diffCoeff[3] * df[6 * warpsize + ilane];
            metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                        diffCoeff[1] * df[4 * warpsize + ilane] +
                        diffCoeff[3] * df[7 * warpsize + ilane];
            metric[2] = diffCoeff[0] * df[2 * warpsize + ilane] +
                        diffCoeff[1] * df[5 * warpsize + ilane] +
                        diffCoeff[3] * df[8 * warpsize + ilane];
            metric[3] = diffCoeff[1] * df[ilane] +
                        diffCoeff[2] * df[3 * warpsize + ilane] +
                        diffCoeff[4] * df[6 * warpsize + ilane];
            metric[4] = diffCoeff[1] * df[1 * warpsize + ilane] +
                        diffCoeff[2] * df[4 * warpsize + ilane] +
                        diffCoeff[4] * df[7 * warpsize + ilane];
            metric[5] = diffCoeff[1] * df[2 * warpsize + ilane] +
                        diffCoeff[2] * df[5 * warpsize + ilane] +
                        diffCoeff[4] * df[8 * warpsize + ilane];
            metric[6] = diffCoeff[3] * df[ilane] +
                        diffCoeff[4] * df[3 * warpsize + ilane] +
                        diffCoeff[5] * df[6 * warpsize + ilane];
            metric[7] = diffCoeff[3] * df[1 * warpsize + ilane] +
                        diffCoeff[4] * df[4 * warpsize + ilane] +
                        diffCoeff[5] * df[7 * warpsize + ilane];
            metric[8] = diffCoeff[3] * df[2 * warpsize + ilane] +
                        diffCoeff[4] * df[5 * warpsize + ilane] +
                        diffCoeff[5] * df[8 * warpsize + ilane];
        }
    }

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                if constexpr (DEFORMED)
                {
                    if (diffCoeff)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex] +
                                    diffCoeff[3] * df[6 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[4 * warpsize + dfindex] +
                                    diffCoeff[3] * df[7 * warpsize + dfindex];
                        metric[2] = diffCoeff[0] * df[2 * warpsize + dfindex] +
                                    diffCoeff[1] * df[5 * warpsize + dfindex] +
                                    diffCoeff[3] * df[8 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[6 * warpsize + dfindex];
                        metric[4] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[4 * warpsize + dfindex] +
                                    diffCoeff[4] * df[7 * warpsize + dfindex];
                        metric[5] = diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[2] * df[5 * warpsize + dfindex] +
                                    diffCoeff[4] * df[8 * warpsize + dfindex];
                        metric[6] = diffCoeff[3] * df[dfindex] +
                                    diffCoeff[4] * df[3 * warpsize + dfindex] +
                                    diffCoeff[5] * df[6 * warpsize + dfindex];
                        metric[7] = diffCoeff[3] * df[1 * warpsize + dfindex] +
                                    diffCoeff[4] * df[4 * warpsize + dfindex] +
                                    diffCoeff[5] * df[7 * warpsize + dfindex];
                        metric[8] = diffCoeff[3] * df[2 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex] +
                                    diffCoeff[5] * df[8 * warpsize + dfindex];
                    }
                }

                TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (unsigned int d = 0u; d < ncoord; ++d)
                {
                    TData tmp = in[d * insize * nqTot + index];

                    if (diffCoeff)
                    {
                        sum1 += metric[3u * d] * tmp;
                        sum2 += metric[3u * d + 1u] * tmp;
                        sum3 += metric[3u * d + 2u] * tmp;
                    }
                    else
                    {
                        sum1 += df[(3u * d) * warpsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * warpsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * warpsize + dfindex] * tmp;
                    }
                }

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[warpsize * cnt_kji + ilane];
                }
                else
                {
                    tmpQ *= jac[0];
                }

                bwd[index] *= lambda * tmpQ;

                if constexpr (SHAPE_TYPE == LibUtilities::Hex)
                {
                    out0[index] = sum1 * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
                {
                    out0[index] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + sum3 * f0[i]) * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric3DSumFacQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int insize, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *__restrict__ bwd, TData *out0, TData *out1, TData *out2,
    TData *metric, const TData lambda)
{
    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            for (unsigned int idx = idx0; idx < 9u; idx += stride)
            {
                metric[idx] =
                    diffCoeff[(idx / 3u) * 3u] * df[idx % 3u] +
                    diffCoeff[(idx / 3u) * 3u + 1u] * df[idx % 3u + 3u] +
                    diffCoeff[(idx / 3u) * 3u + 2u] * df[idx % 3u + 6u];
            }

            __syncthreads();
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1 * nq2; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                metric[0] = diffCoeff[0] * df[dfindex] +
                            diffCoeff[1] * df[3 * dfsize + dfindex] +
                            diffCoeff[2] * df[6 * dfsize + dfindex];
                metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                            diffCoeff[1] * df[4 * dfsize + dfindex] +
                            diffCoeff[2] * df[7 * dfsize + dfindex];
                metric[2] = diffCoeff[0] * df[2 * dfsize + dfindex] +
                            diffCoeff[1] * df[5 * dfsize + dfindex] +
                            diffCoeff[2] * df[8 * dfsize + dfindex];
                metric[3] = diffCoeff[3] * df[dfindex] +
                            diffCoeff[4] * df[3 * dfsize + dfindex] +
                            diffCoeff[5] * df[6 * dfsize + dfindex];
                metric[4] = diffCoeff[3] * df[1 * dfsize + dfindex] +
                            diffCoeff[4] * df[4 * dfsize + dfindex] +
                            diffCoeff[5] * df[7 * dfsize + dfindex];
                metric[5] = diffCoeff[3] * df[2 * dfsize + dfindex] +
                            diffCoeff[4] * df[5 * dfsize + dfindex] +
                            diffCoeff[5] * df[8 * dfsize + dfindex];
                metric[6] = diffCoeff[6] * df[dfindex] +
                            diffCoeff[7] * df[3 * dfsize + dfindex] +
                            diffCoeff[8] * df[6 * dfsize + dfindex];
                metric[7] = diffCoeff[6] * df[1 * dfsize + dfindex] +
                            diffCoeff[7] * df[4 * dfsize + dfindex] +
                            diffCoeff[8] * df[7 * dfsize + dfindex];
                metric[8] = diffCoeff[6] * df[2 * dfsize + dfindex] +
                            diffCoeff[7] * df[5 * dfsize + dfindex] +
                            diffCoeff[8] * df[8 * dfsize + dfindex];
            }
        }

        TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            TData tmp = in[d * insize * nqTot + idx];

            if (diffCoeff)
            {
                sum1 += metric[3u * d] * tmp;
                sum2 += metric[3u * d + 1u] * tmp;
                sum3 += metric[3u * d + 2u] * tmp;
            }
            else
            {
                sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
            }
        }

        TData tmpQ = w0[i] * w1[j] * w2[k];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        bwd[idx] *= lambda * tmpQ;

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            out0[idx] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + sum3 * f0[i]) * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
    }

    __syncthreads();
}

// General Launcher
template <typename Implementation, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Helmholtz1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda)
{
    const unsigned int ndf = ncoord;
    unsigned int dfsize    = 1u;
    unsigned int jacsize   = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
        jacsize *= nq0;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nm0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            TData *bwd         = wsp + nq0 * warpsize * iwarp;
            TData *deriv       = wsp + nq0 * nelmt + nq0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, bwd);
            PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt, D0,
                                              dfptr, bwd, deriv);
            ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt, w0,
                                                dfptr, jacptr, coeff, deriv,
                                                bwd, deriv, lambda);
            SumDerivTensor1DKernel<true, DEFORMED>(ilane, nq0, D0, deriv, bwd);
            IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nq0, basis0, bwd, outptr, (TData)1.0);
            e += blockDim.x * gridDim.x;
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        TData *bwd   = (TData *)shmemptr;
        TData *deriv = bwd + nq0;

        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * e;
            TData *outptr       = out + nm0 * e;

            BwdTransSegSumFacQPKernel(nm0, nq0, basis0, inptr, bwd);
            PhysDeriv1DSumFacQPKernel<DEFORMED>(ncoord, nq0, 1, D0, dfptr, bwd,
                                                deriv);
            ApplyMetric1DSumFacQPKernel<DEFORMED>(ncoord, nq0, 1, w0, dfptr,
                                                  jacptr, coeff, deriv, bwd,
                                                  deriv, lambda);
            SumDerivTensor1DQPKernel<true, DEFORMED>(nq0, D0, deriv, bwd);
            IProductWRTBaseSegSumFacQPKernel<false, false, DEFORMED>(
                nm0, nq0, basis0, bwd, outptr, (TData)1.0);

            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TData>
__global__ void Helmholtz1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz1DKernel<Implementation, DEFORMED>(ncoord, nm0, nq0, nelmt, basis0,
                                                D0, w0, df, jac, coeff, in, out,
                                                wsp, lambda);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nq0, typename TData>
__global__ void Helmholtz1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ D0,
    const TData *__restrict__ w0, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz1DKernel<Implementation, DEFORMED>(ncoord, nm0, nq0, nelmt, basis0,
                                                D0, w0, df, jac, coeff, in, out,
                                                wsp, lambda);
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Helmholtz2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    const TData lambda)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = threadIdx.x;
        const unsigned int stride = blockDim.x;

        if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            s_f0 = (TData *)shmemptr;
            s_f1 = s_f0 + nq1;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx] = f1[idx];
            }

            __syncthreads();
        }

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, bwd, wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, bwd, wsp0);
            }
            PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, D0, D1, s_f0, s_f1, dfptr, bwd,
                deriv);
            ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, w0, w1, s_f0, s_f1, dfptr,
                jacptr, coeff, deriv, bwd, deriv0, deriv1, lambda);
            SumDerivTensor2DKernel<true, DEFORMED>(ilane, nq0, nq1, D0, D1,
                                                   deriv0, deriv1, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, bwd, outptr,
                    wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                    outptr, wsp0, (TData)1.0);
            }
            e += blockDim.x * gridDim.x;
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        unsigned int offset, nmode0, nmode1;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            offset = std::max(nm0 * nq1, nm1 * nq0);
            nmode0 = nm0;
            nmode1 = nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nmTot;
        }

        TData *metric   = (TData *)shmemptr;
        TData *bwd      = metric + 6;
        TData *deriv    = bwd + nqTot;
        TData *tmp      = deriv;
        TData *deriv0   = deriv;
        TData *deriv1   = deriv0 + nqTot;
        TData *s_wsp0   = deriv + ncoord * nqTot;
        TData *s_basis0 = s_wsp0 + offset;
        TData *s_basis1 = s_basis0 + nm0 * nq0;

        // Copy to shared memory.
        const unsigned int idx0   = threadIdx.x;
        const unsigned int stride = blockDim.x;

        for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * e;
            TData *outptr       = out + nmTot * e;

            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            __syncthreads();

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                BwdTransQuadSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                           s_basis1, tmp, bwd, s_wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                BwdTransTriSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                          s_basis0, s_basis1, tmp, bwd, s_wsp0);
            }

            PhysDeriv2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, 1, D0, D1, f0, f1, dfptr, bwd, deriv);
            if constexpr (DEFORMED)
            {
                TData dmetric[6];
                ApplyMetric2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, 1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                    deriv, bwd, deriv0, deriv1, dmetric, lambda);
            }
            else
            {
                ApplyMetric2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, 1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                    deriv, bwd, deriv0, deriv1, metric, lambda);
            }
            SumDerivTensor2DQPKernel<true, DEFORMED>(nq0, nq1, D0, D1, deriv0,
                                                     deriv1, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                    outptr, s_wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, outptr, s_wsp0, (TData)1.0);
            }

            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void Helmholtz2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, df, jac, coeff, in, out, wsp, lambda);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TData>
__global__ void Helmholtz2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, df, jac, coeff, in, out, wsp, lambda);
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Helmholtz3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    [[maybe_unused]] const unsigned int *__restrict__ index3,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp, const TData lambda)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    unsigned int dfsize        = 1u;
    unsigned int jacsize       = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        TData *s_f0  = nullptr;
        TData *s_f1  = nullptr;
        TData *s_f1m = nullptr;
        TData *s_f2  = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = threadIdx.x;
        const unsigned int stride = blockDim.x;
        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            s_f0  = (TData *)shmemptr;
            s_f1  = s_f0 + nq0;
            s_f1m = s_f1 + nq1;
            s_f2  = s_f1m + nq1;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx]  = f1[idx];
                s_f1m[idx] = f1m[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            __syncthreads();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            s_f0 = (TData *)shmemptr;
            s_f2 = s_f0 + nq0;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            __syncthreads();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            s_f0 = (TData *)shmemptr;
            s_f1 = s_f0 + nq0;
            s_f2 = s_f1 + nq1;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx] = f1[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            __syncthreads();
        }

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv2 = wsp + 3 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, bwd,
                                        wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, D0, D1, D2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, bwd, deriv);
            ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, w0, w1, w2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2,
                lambda);
            SumDerivTensor3DKernel<true, DEFORMED>(
                ilane, nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (4 * nqTot + nq1 * nq2 + std::max(nq2, nm0)) * nelmt +
                    nm2 * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1) +
                               std::max(nq2, nm0)) *
                                  nelmt +
                              nm1 * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            e += blockDim.x * gridDim.x;
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        unsigned int offset0, offset1, nmode0, nmode1, nmode2;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
            offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            offset0 = nm0 * nm1 * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            offset0 = nm0 * nm1 * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }

        TData *metric   = (TData *)shmemptr;
        TData *bwd      = metric + 9;
        TData *deriv    = bwd + nqTot;
        TData *tmp      = deriv;
        TData *deriv0   = deriv;
        TData *deriv1   = deriv0 + nqTot;
        TData *deriv2   = deriv1 + nqTot;
        TData *s_wsp0   = deriv2 + nqTot;
        TData *s_wsp1   = s_wsp0 + offset0;
        TData *s_basis0 = s_wsp1 + offset1;
        TData *s_basis1 = s_basis0 + nmode0 * nq0;
        TData *s_basis2 = s_basis1 + nmode1 * nq1;

        // Copy to shared memory.
        const unsigned int idx0   = threadIdx.x;
        const unsigned int stride = blockDim.x;

        for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }

        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * e;
            TData *outptr       = out + nmTot * e;

            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            __syncthreads();

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                BwdTransHexSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          s_basis0, s_basis1, s_basis2, tmp,
                                          bwd, s_wsp0, s_wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                BwdTransTetSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, index0, index3, s_basis0,
                                          s_basis1, s_basis2, tmp, bwd, s_wsp0,
                                          s_wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                BwdTransPrismSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                            isModified, s_basis0, s_basis1,
                                            s_basis2, tmp, bwd, s_wsp0, s_wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                BwdTransPyrSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, s_basis0, s_basis1,
                                          s_basis2, tmp, bwd, s_wsp0, s_wsp1);
            }
            PhysDeriv3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, 1, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
                deriv);
            if constexpr (DEFORMED)
            {
                TData dmetric[9];
                ApplyMetric3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, 1, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2, dmetric,
                    lambda);
            }
            else
            {
                ApplyMetric3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, 1, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2, metric,
                    lambda);
            }
            SumDerivTensor3DQPKernel<true, DEFORMED>(
                nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, bwd, outptr, s_wsp0, s_wsp1,
                    (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, bwd, outptr,
                    s_wsp1, s_wsp0, (TData)1.0);
            }

            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void Helmholtz3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, df, jac, coeff, in, out, wsp, lambda);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, signed int nm0, unsigned int nm1, unsigned int nm2,
          unsigned nmTot, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TData>
__global__ void Helmholtz3DKernelLauncher(
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda)
{
    Helmholtz3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, df, jac, coeff, in, out, wsp, lambda);
}

// Kernel launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void Helmholtz1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *basis0, const TData *D0,
    const TData *w0, const TData *df, const TData *jac, const TData *coeff,
    const TData *in, TData *out, TData *wsp, const TData lambda = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz1DKernelLauncher<Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nm0, nq0, nelmt, basis0,
                                             D0, w0, df, jac, coeff, in, out,
                                             wsp, lambda);
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void Helmholtz1DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *basis0,
    const TData *D0, const TData *w0, const TData *df, const TData *jac,
    const TData *coeff, const TData *in, TData *out, TData *wsp,
    const TData lambda = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz1DKernelLauncher<Implementation, DEFORMED, nm0, nq0>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nelmt, basis0, D0, w0, df,
                                             jac, coeff, in, out, wsp, lambda);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Helmholtz2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *D0, const TData *D1, const TData *w0,
    const TData *w1, const TData *f0, const TData *f1, const TData *df,
    const TData *jac, const TData *coeff, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nm0, nm1, nmTot, nq0, nq1,
                                             nelmt, isModified, index0, basis0,
                                             basis1, D0, D1, w0, w1, f0, f1, df,
                                             jac, coeff, in, out, wsp, lambda);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void Helmholtz2DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *D0, const TData *D1, const TData *w0, const TData *w1,
    const TData *f0, const TData *f1, const TData *df, const TData *jac,
    const TData *coeff, const TData *in, TData *out, TData *wsp,
    const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED, nm0, nm1,
                              nmTot, nq0, nq1>
        <<<gridsize, blocksize, shmemsize>>>(
            ncoord, nelmt, isModified, index0, basis0, basis1, D0, D1, w0, w1,
            f0, f1, df, jac, coeff, in, out, wsp, lambda);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Helmholtz3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *df,
    const TData *jac, const TData *coeff, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
            index1, index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1,
            w2, f0, f1, f1m, f2, df, jac, coeff, in, out, wsp, lambda);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void Helmholtz3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *df,
    const TData *jac, const TData *coeff, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    Helmholtz3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED, nm0, nm1,
                              nm2, nmTot, nq0, nq1, nq2>
        <<<gridsize, blocksize, shmemsize>>>(
            nelmt, isModified, index0, index1, index2, index3, basis0, basis1,
            basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, df, jac, coeff, in,
            out, wsp, lambda);
}

} // namespace Nektar::Operators::detail

#endif
