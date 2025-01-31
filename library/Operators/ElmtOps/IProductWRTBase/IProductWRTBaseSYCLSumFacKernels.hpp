///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSYCLSumFacKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

namespace Nektar::Operators::detail
{

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacQPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int p = idx0; p < nm0; p += stride)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            sum += in[i] * basis0[p * nq0 + i];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[p] += sum;
        }
        else
        {
            out[p] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacQPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    [[maybe_unused]] const unsigned int nqTot, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = idx / nm0;
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[q * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacQPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[idx * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    // Correction for singular vertex in collpased coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            prod += basis0[nq0 + i] * basis1[nq1 + j] * in[idx];
        }

        prod = warpReduceSum(prod, threadBlock);

        if (getLaneIdx(threadBlock) == 0)
        {
            if constexpr (SCALE)
            {
                prod *= scale;
            }

            atomic_add<NektarSpaces::GlobalScope>(out + 1u, prod);
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const unsigned int nqTot, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += wsp0[cnt_pkj] * basis1[q * nq1 + j];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = (idx / nm0) % nm1;
        const unsigned int r = idx / (nm0 * nm1);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum += wsp1[cnt_pqk] * basis2[r * nq2 + k];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex1,
    const unsigned int *__restrict__ pindex2,
    const unsigned int *__restrict__ qindex2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm01 * nq2; idx += stride)
    {
        const unsigned int mode_pq = idx / nq2;
        const unsigned int p       = pindex1[mode_pq];
        const unsigned int k       = idx % nq2;
        unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[mode_pq * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex2[idx];
        const unsigned int q       = qindex2[idx];
        const unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k)
        {
            tmp += wsp1[mode_pq * nq2 + k] * basis2[mode2 * nq2 + k];
        }

        if constexpr (SCALE)
        {
            tmp *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += tmp;
        }
        else
        {
            out[idx] = tmp;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM2_MAX = 8;
        if (nm2 <= NM2_MAX)
        {
            TData prod[NM2_MAX] = {0.0};
            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                TData tmp = basis0[i] * basis1[nq1 + j];
                tmp += basis0[nq0 + i] * basis1[j];
                tmp += basis0[nq0 + i] * basis1[nq1 + j];
                tmp *= basis2[nq2 + k];
                prod[nm2 - 1u] += in[idx] * tmp;

                // singular edge
                tmp = basis1[nq1 + j] * basis0[nq0 + i] * in[idx];
#pragma unroll
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    prod[r] += basis2[(r + 1u) * nq2 + k] * tmp;
                }

                // bottom vertex
                prod[0] += basis2[k] * tmp;
            }

#pragma unroll
            for (unsigned int r = 0u; r < nm2; ++r)
            {
                prod[r] = warpReduceSum(prod[r], threadBlock);

                if (getLaneIdx(threadBlock) == 0)
                {
                    if constexpr (SCALE)
                    {
                        prod[r] *= scale;
                    }

                    if (r == nm2 - 1u)
                    {
                        atomic_add<NektarSpaces::GlobalScope>(out + 1u,
                                                              prod[nm2 - 1u]);
                    }
                    else
                    {
                        atomic_add<NektarSpaces::GlobalScope>(out + nm2 + r,
                                                              prod[r]);
                    }
                }
            }
        }
        else
        {
            TData prod0 = 0.0;
            TData prod1 = 0.0;

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                TData tmp = basis0[i] * basis1[nq1 + j];
                tmp += basis0[nq0 + i] * basis1[j];
                tmp += basis0[nq0 + i] * basis1[nq1 + j];
                tmp *= basis2[nq2 + k];
                prod0 += in[idx] * tmp;

                // bottom vertex
                prod1 +=
                    basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] * in[idx];
            }

            prod0 = warpReduceSum(prod0, threadBlock);
            prod1 = warpReduceSum(prod1, threadBlock);

            if (getLaneIdx(threadBlock) == 0)
            {
                if constexpr (SCALE)
                {
                    prod0 *= scale;
                    prod1 *= scale;
                }

                atomic_add<NektarSpaces::GlobalScope>(out + 1u, prod0);
                atomic_add<NektarSpaces::GlobalScope>(out + nm2, prod1);
            }

            // singular edge
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += basis2[(r + 1u) * nq2 + k] * basis1[nq1 + j] *
                            basis0[nq0 + i] * in[idx];
                }

                prod = warpReduceSum(prod, threadBlock);

                if (getLaneIdx(threadBlock) == 0)
                {
                    if constexpr (SCALE)
                    {
                        prod *= scale;
                    }

                    atomic_add<NektarSpaces::GlobalScope>(out + nm2 + r, prod);
                }
            }
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex,
    const unsigned int *__restrict__ rindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex[idx];
        const unsigned int q       = qindex[idx];
        const unsigned int r       = rindex[idx];
        const unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u + r;
        unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode_pr * nq2 + k] * wsp1[cnt_pqk];
        }

        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM1_MAX = 8;
        if (nm1 <= NM1_MAX)
        {
            TData prod[NM1_MAX] = {0.0};

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                TData tmp = in[idx] * basis2[nq2 + k] * basis0[nq0 + i];
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    prod[q] += tmp * basis1[q * nq1 + j];
                }
            }

#pragma unroll
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                prod[q] = warpReduceSum(prod[q], threadBlock);

                if (getLaneIdx(threadBlock) == 0)
                {
                    if constexpr (SCALE)
                    {
                        prod[q] *= scale;
                    }

                    atomic_add<NektarSpaces::GlobalScope>(out + nm2 * q + 1u,
                                                          prod[q]);
                }
            }
        }
        else
        {
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += in[idx] * basis2[nq2 + k] * basis1[q * nq1 + j] *
                            basis0[nq0 + i];
                }

                prod = warpReduceSum(prod, threadBlock);

                if (getLaneIdx(threadBlock) == 0)
                {
                    if constexpr (SCALE)
                    {
                        prod *= scale;
                    }

                    atomic_add<NektarSpaces::GlobalScope>(out + nm2 * q + 1u,
                                                          prod);
                }
            }
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        const unsigned int q = qindex[idx];
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode2 * nq2 + k] * wsp1[cnt_pqk];
        }
        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            // top vertex
            TData tmp = basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k];
            prod += in[idx] * tmp;
        }

        prod = warpReduceSum(prod, threadBlock);

        if (getLaneIdx(threadBlock) == 0)
        {
            if constexpr (SCALE)
            {
                prod *= scale;
            }

            atomic_add<NektarSpaces::GlobalScope>(out + 1, prod);
        }
    }

    localBarrier(threadBlock);
}

template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale,
    [[maybe_unused]] TData *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    unsigned int jacsize = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nq0;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nq0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            IProductWRTBaseSegSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nq0, basis0, w0, jacptr, inptr, outptr, scale);
            e += getGlobalRange(threadBlock);
        }
    }
    else
    {
        TData *s_wsp0 = (TData *)shmemptr;

        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

        unsigned int e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nq0 * e;
            TData *outptr       = out + nm0 * e;

            for (unsigned int i = idx0; i < nq0; i += stride)
            {
                if constexpr (DEFORMED)
                {
                    s_wsp0[i] = inptr[i] * jacptr[i] * w0[i];
                }
                else
                {
                    s_wsp0[i] = inptr[i] * jacptr[0] * w0[i];
                }
            }

            localBarrier(threadBlock);

            IProductWRTBaseSegSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nq0, basis0, s_wsp0, outptr, scale, threadBlock);
            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    const TData scale, [[maybe_unused]] TData *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nqTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wspptr = wsp + nq1 * warpsize * iwarp;
                IProductWRTBaseQuadSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr,
                    inptr, outptr, wspptr, scale);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wspptr = wsp + nq1 * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0,
                    w1, jacptr, inptr, outptr, wspptr, scale);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset, nmode0, nmode1;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nmTot;
        }

        TData *s_wsp0   = (TData *)shmemptr;
        TData *s_wsp1   = s_wsp0 + nqTot;
        TData *s_basis0 = s_wsp1 + offset;
        TData *s_basis1 = s_basis0 + nm0 * nq0;

        // Copy to shared memory.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

        for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        unsigned int e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nqTot * e;
            TData *outptr       = out + nmTot * e;

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = idx / nq0;
                if constexpr (DEFORMED)
                {
                    s_wsp0[idx] = inptr[idx] * jacptr[idx] * w0[i] * w1[j];
                }
                else
                {
                    s_wsp0[idx] = inptr[idx] * jacptr[0] * w0[i] * w1[j];
                }
            }

            localBarrier(threadBlock);

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1,
                    s_wsp0, outptr, s_wsp1, scale, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, s_wsp0, outptr, s_wsp1, scale,
                    threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    const TData scale, [[maybe_unused]] TData *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nqTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
                IProductWRTBaseHexSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
                TData *prod =
                    wsp + (nq1 * nq2 + nq2) * nelmt + nm2 * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0,
                    wsp1, prod, scale);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (nq1 * nq2 + nq2) * nelmt + nm1 * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0,
                    wsp1, wsp2, scale);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
                IProductWRTBasePyrSumFacKernel<SCALE, APPEND, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0,
                    wsp1, scale);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset0, offset1, nmode0, nmode1, nmode2;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
            nmode0  = nm0;
            nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }

        TData *s_wsp0   = (TData *)shmemptr;
        TData *s_wsp1   = s_wsp0 + nqTot;
        TData *s_wsp2   = s_wsp1 + offset0;
        TData *s_basis0 = s_wsp2 + offset1;
        TData *s_basis1 = s_basis0 + nmode0 * nq0;
        TData *s_basis2 = s_basis1 + nmode1 * nq1;

        // Copy to shared memory.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

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

        unsigned int e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nqTot * e;
            TData *outptr       = out + nmTot * e;

            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);
                if constexpr (DEFORMED)
                {
                    s_wsp0[idx] =
                        inptr[idx] * jacptr[idx] * w0[i] * w1[j] * w2[k];
                }
                else
                {
                    s_wsp0[idx] =
                        inptr[idx] * jacptr[0] * w0[i] * w1[j] * w2[k];
                }
            }

            localBarrier(threadBlock);

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2, scale,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2,
                    s_wsp0, outptr, s_wsp1, s_wsp2, scale, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2,
                    s_wsp0, outptr, s_wsp1, s_wsp2, scale, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacQPKernel<SCALE, APPEND, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, s_wsp0,
                    outptr, s_wsp1, s_wsp2, scale, threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

// Kernel launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *basis0, const TData *w0, const TData *jac, const TData *in,
    TData *out, const TData scale = 1.0)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<1>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<1> threadBlock) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTBase1DKernel<Implementation, SCALE, APPEND,
                                         DEFORMED>(nm0, nq0, nelmt, basis0, w0,
                                                   jac, in, out, scale,
                                                   shmemptr, threadBlock);
             });
     }).wait();
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nelmt, const TData *basis0, const TData *w0,
    const TData *jac, const TData *in, TData *out, const TData scale = 1.0)
{
    IProductWRTBase1DKernel<ExecSpace, Implementation, SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *w0, const TData *w1, const TData *jac, const TData *in,
    TData *out, TData *wsp, const TData scale = 1.0)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                    nm0, nm1);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<1>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<1> threadBlock) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTBase2DKernel<SHAPE_TYPE, Implementation, SCALE,
                                         APPEND, DEFORMED>(
                     nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0,
                     basis0, basis1, w0, w1, jac, in, out, wsp, scale, shmemptr,
                     threadBlock);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nq0,
          unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const TData *basis0, const TData *basis1, const TData *w0, const TData *w1,
    const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    IProductWRTBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation, SCALE,
                            APPEND, DEFORMED>(
        nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1, w0, w1,
        jac, in, out, wsp, scale);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<1>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<1> threadBlock) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTBase3DKernel<SHAPE_TYPE, Implementation, SCALE,
                                         APPEND, DEFORMED>(
                     nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified,
                     index0, index1, index2, basis0, basis1, basis2, w0, w1, w2,
                     jac, in, out, wsp, scale, shmemptr, threadBlock);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nq0, unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    IProductWRTBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation, SCALE,
                            APPEND, DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        basis0, basis1, basis2, w0, w1, w2, jac, in, out, wsp, scale);
}

} // namespace Nektar::Operators::detail

#endif
