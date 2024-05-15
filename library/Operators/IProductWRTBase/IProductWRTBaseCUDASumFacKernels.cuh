///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseCUDASumFacKernels.cuh
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

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseSegKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_basis0 = shared;
    TData *s_w0     = s_basis0 + nm0 * nq0;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * e;
        unsigned int outoffset = nm0 * e;

        for (unsigned int p = 0; p < nm0; ++p)
        {
            TData sum = 0.0;
            for (unsigned int i = 0; i < nq0; ++i)
            {
                unsigned int index    = inoffset + i;
                unsigned int jacindex = DEFORMED ? index : e;
                sum +=
                    in[index] * s_basis0[p * nq0 + i] * jac[jacindex] * s_w0[i];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            unsigned int index = outoffset + p;
            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseSegKernel_QP(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_wsp0 = shared;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * e;
        unsigned int outoffset = nm0 * e;

        // Copy to shared memory.
        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            unsigned int index    = inoffset + i;
            unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[i]             = in[index] * jac[jacindex];
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            TData sum = 0.0;
            for (unsigned int i = 0; i < nq0; ++i)
            {
                sum += s_wsp0[i] * basis0[p * nq0 + i] * w0[i];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            unsigned int index = outoffset + p;
            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_basis0 = shared;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_w0     = s_basis1 + nm1 * nq1;
    TData *s_w1     = s_w0 + nq0;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nm1 * nq1)
    {
        s_basis1[sIndex] = basis1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + nq1 * e;

        for (unsigned int p = 0; p < nm0; ++p)
        {
            for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                TData sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    unsigned int index    = inoffset + cnt_ji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    sum += in[index] * s_basis0[p * nq0 + i] * jac[jacindex] *
                           s_w0[i];
                }
                wsp0[j] = sum;
            }

            for (unsigned int q = 0; q < nm1; ++q)
            {
                TData sum = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    sum += wsp0[j] * s_basis1[q * nq1 + j] * s_w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                unsigned int index = outoffset + nm0 * q + p;
                if constexpr (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nq0 * nq1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji   = nq0 * j + i;
                unsigned int index    = inoffset + cnt_ji;
                unsigned int jacindex = DEFORMED ? index : e;
                s_wsp0[cnt_ji]        = in[index] * jac[jacindex];
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                unsigned int cnt_ji = nq0 * j;
                unsigned int cnt_pj = nq1 * p + j;

                TData sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    sum += s_wsp0[cnt_ji] * basis0[p * nq0 + i] * w0[i];
                }
                s_wsp1[cnt_pj] = sum;
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                unsigned int cnt_pj = nq1 * p;
                unsigned int cnt_pq = nm0 * q + p;

                TData sum = 0.0;
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_pj)
                {
                    sum += s_wsp1[cnt_pj] * basis1[q * nq1 + j] * w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                unsigned int index = outoffset + cnt_pq;
                if constexpr (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTriKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    TData *__restrict__ wsp, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_w0 = shared;
    TData *s_w1 = s_w0 + nq0;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + nq1 * e;

        for (unsigned int p = 0, mode_pq = 0; p < nm0; ++p)
        {
            for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                TData sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    unsigned int index    = inoffset + cnt_ji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    sum += in[index] * basis0[p * nq0 + i] * jac[jacindex] *
                           s_w0[i];
                }
                wsp0[j] = sum;
            }

            for (unsigned int q = 0; q < nm1 - p; ++q, ++mode_pq)
            {
                TData sum = 0.0;
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    sum += wsp0[j] * basis1[mode_pq * nq1 + j] * s_w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                unsigned int index = outoffset + mode_pq;
                if constexpr (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        }

        // Correction for singular vertex in collpased coordinates.
        // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
        // With contributions from every quadrature point
        if (correct)
        {
            TData iprod_01 = 0.0;
            for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                unsigned int jacindex = DEFORMED ? inoffset : e;

                TData tmp = s_w1[j] * basis1[nq1 + j];
                if constexpr (!DEFORMED)
                {
                    tmp *= jac[jacindex];
                }

                for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    unsigned int index    = inoffset + cnt_ji;
                    unsigned int jacindex = DEFORMED ? index : e;

                    TData prod = in[index] * tmp * s_w0[i];
                    if constexpr (DEFORMED)
                    {
                        prod *= jac[jacindex];
                    }
                    iprod_01 += prod * basis0[nq0 + i];
                }
            }

            unsigned int index = outoffset + 1;
            if constexpr (SCALE)
            {
                out[index] += iprod_01 * scale;
            }
            else
            {
                out[index] += iprod_01;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTriKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_wsp0     = shared;
    TData *s_wsp1     = s_wsp0 + nq0 * nq1;
    TData *s_iprod_01 = s_wsp1 + nm0 * nq1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji   = nq0 * j + i;
                unsigned int index    = inoffset + cnt_ji;
                unsigned int jacindex = DEFORMED ? index : e;
                s_wsp0[cnt_ji]        = in[index] * jac[jacindex];
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                unsigned int cnt_ji = nq0 * j;
                unsigned int cnt_pj = nq1 * p + j;

                TData sum = 0.0;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    sum += s_wsp0[cnt_ji] * basis0[p * nq0 + i] * w0[i];
                }
                s_wsp1[cnt_pj] = sum;
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1 - p; q += blockDim.y)
            {
                unsigned int cnt_pj  = nq1 * p;
                unsigned int mode_pq = (2 * nm1 - p + 1) * p / 2 + q;

                TData sum = 0.0;
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_pj)
                {
                    sum += s_wsp1[cnt_pj] * basis1[mode_pq * nq1 + j] * w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                unsigned int index = outoffset + mode_pq;
                if constexpr (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        }

        __syncthreads();

        // Correction for singular vertex in collpased coordinates.
        // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
        // With contributions from every quadrature point
        if (correct)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                *s_iprod_01 = 0.0;
            }

            __syncthreads();

            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                TData tmp = w1[j] * basis1[nq1 + j];
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_ji = nq0 * j + i;
                    TData prod          = s_wsp0[cnt_ji] * tmp * w0[i];
                    atomicAdd(s_iprod_01, prod * basis0[nq0 + i]);
                }
            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                unsigned int index = outoffset + 1;
                if constexpr (SCALE)
                {
                    out[index] += (*s_iprod_01) * scale;
                }
                else
                {
                    out[index] += (*s_iprod_01);
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_basis0 = shared;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm1 * nq1;
    TData *s_w0     = s_basis2 + nm2 * nq2;
    TData *s_w1     = s_w0 + nq0;
    TData *s_w2     = s_w1 + nq1;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nm1 * nq1)
    {
        s_basis1[sIndex] = basis1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nm2 * nq2)
    {
        s_basis2[sIndex] = basis2[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq2)
    {
        s_w2[sIndex] = w2[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + (nq2 * nq1 + nq2) * e;
        TData *wsp1            = wsp0 + nq2 * nq1;

        for (unsigned int p = 0; p < nm0; ++p)
        {
            for (unsigned int k = 0, cnt_kj = 0, cnt_kji = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index    = inoffset + cnt_kji;
                        unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * s_basis0[i + nq0 * p] *
                                  jac[jacindex] * s_w0[i];
                    }
                    wsp0[cnt_kj] = sum_kj;
                }
            }

            for (unsigned int q = 0; q < nm1; ++q)
            {
                for (unsigned int k = 0, cnt_kj = 0; k < nq2; ++k)
                {
                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k += wsp0[cnt_kj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }
                    wsp1[k] = sum_k;
                }

                for (unsigned int r = 0; r < nm2; ++r)
                {
                    unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                    TData sum = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        sum += wsp1[k] * s_basis2[r * nq2 + k] * s_w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum *= scale;
                    }

                    unsigned int index = outoffset + cnt_rqp;
                    if constexpr (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji  = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int index    = inoffset + cnt_kji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    s_wsp0[cnt_kji]       = in[index] * jac[jacindex];
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;
                    unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k + j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj += s_wsp0[cnt_kji] * basis0[i + nq0 * p] * w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
                {
                    unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;
                    unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k += s_wsp1[cnt_pkj] * basis1[q * nq1 + j] * w1[j];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                for (unsigned int r = threadIdx.z; r < nm2; r += blockDim.z)
                {
                    unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;
                    unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                    TData sum = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum += s_wsp2[cnt_pqk] * basis2[r * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum *= scale;
                    }

                    unsigned int index = outoffset + cnt_rqp;
                    if constexpr (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTetKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_w0 = shared;
    TData *s_w1 = s_w0 + nq0;
    TData *s_w2 = s_w1 + nq1;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq2)
    {
        s_w2[sIndex] = w2[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + (nq2 * nq1 + nq2 + nm2) * e;
        TData *wsp1            = wsp0 + nq2 * nq1;
        TData *prod            = wsp1 + nq2;

        for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0; ++p)
        {
            for (unsigned int k = 0, cnt_kj = 0, cnt_kji = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index    = inoffset + cnt_kji;
                        unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[i + nq0 * p] *
                                  jac[jacindex] * s_w0[i];
                    }
                    wsp0[cnt_kj] = sum_kj;
                }
            }

            for (unsigned int q = 0; q < nm1 - p; ++q, ++mode_pq)
            {
                for (unsigned int k = 0, cnt_kj = 0; k < nq2; ++k)
                {
                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k +=
                            basis1[mode_pq * nq1 + j] * wsp0[cnt_kj] * s_w1[j];
                    }
                    wsp1[k] = sum_k;
                }

                for (unsigned int r = 0; r < nm2 - p - q; ++r, ++mode_pqr)
                {
                    TData tmp = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        tmp += wsp1[k] * basis2[mode_pqr * nq2 + k] * s_w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        tmp *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr;
                    if constexpr (APPEND)
                    {
                        out[index] += tmp;
                    }
                    else
                    {
                        out[index] = tmp;
                    }
                }
            }
        }

        // Add correction for collapsed coordinate.
        if (correct)
        {
            for (unsigned int r = 0; r < nm2; ++r)
            {
                prod[r] = 0.0;
            }

            for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
            {
                TData tmpQ2 = s_w2[k];
                if constexpr (!DEFORMED)
                {
                    tmpQ2 *= jac[e];
                }

                for (unsigned int j = 0; j < nq1; ++j)
                {
                    TData tmpQ1 = tmpQ2 * s_w1[j];
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index = inoffset + cnt_kji;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];
                        if constexpr (DEFORMED)
                        {
                            tmpQ *= jac[inoffset + cnt_kji];
                        }

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= in[index] * tmpQ;
                        prod[nm2 - 1] += tmp;

                        // bottom vertex
                        tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                              in[index] * tmpQ;
                        prod[0] += tmp;

                        // singular edge
                        for (unsigned int r = 1; r < nm2 - 1; ++r)
                        {
                            tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                                  basis0[nq0 + i] * in[index] * tmpQ;
                            prod[r] += tmp;
                        }
                    }
                }
            }

            if constexpr (SCALE)
            {
                out[outoffset + 1] += prod[nm2 - 1] * scale;
                for (unsigned int r = 0; r < nm2 - 1; ++r)
                {
                    out[outoffset + nm2 + r] += prod[r] * scale;
                }
            }
            else
            {
                out[outoffset + 1] += prod[nm2 - 1];
                for (unsigned int r = 0; r < nm2 - 1; ++r)
                {
                    out[outoffset + nm2 + r] += prod[r];
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTetKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_prod = shared;
    TData *s_wsp0 = s_prod + nm2;
    TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji  = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int index    = inoffset + cnt_kji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    s_wsp0[cnt_kji]       = in[index] * jac[jacindex];
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj += s_wsp0[cnt_kji] * basis0[i + nq0 * p] * w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1 - p; q += blockDim.y)
            {
                for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
                {
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;
                    unsigned int mode_pq = (2 * nm1 - p + 1) * p / 2 + q;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            basis1[mode_pq * nq1 + j] * s_wsp1[cnt_pkj] * w1[j];
                    }
                    s_wsp2[mode_pq * nq2 + k] = sum_k;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1 - p; q += blockDim.y)
            {
                for (unsigned int r = threadIdx.z; r < nm2 - p - q;
                     r += blockDim.z)
                {
                    unsigned int mode_pq  = (2 * nm1 - p + 1) * p / 2 + q;
                    unsigned int mode_pqr = (2 * (nm2 - p) - q + 1) * q;
                    mode_pqr += nm2 * (nm2 + 1) * p;
                    mode_pqr -= (2 * nm2 + 1) * (p - 1) * p / 2;
                    mode_pqr += (p - 1) * p * (2 * p - 1) / 6;
                    mode_pqr /= 2;

                    TData tmp = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        tmp += s_wsp2[mode_pq * nq2 + k] *
                               basis2[(mode_pqr + r) * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        tmp *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr + r;
                    if constexpr (APPEND)
                    {
                        out[index] += tmp;
                    }
                    else
                    {
                        out[index] = tmp;
                    }
                }
            }
        }

        __syncthreads();

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                for (unsigned int r = threadIdx.z; r < nm2; r += blockDim.z)
                {
                    s_prod[r] = 0.0;
                }
            }

            __syncthreads();

            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                TData tmpQ2 = w2[k];
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    TData tmpQ1 = tmpQ2 * w1[j];
                    for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                    {
                        unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j + i;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * w0[i];

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= s_wsp0[cnt_kji] * tmpQ;
                        atomicAdd(s_prod + nm2 - 1, tmp);

                        // bottom vertex
                        tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                              s_wsp0[cnt_kji] * tmpQ;
                        atomicAdd(s_prod, tmp);

                        // singular edge
                        for (unsigned int r = 1; r < nm2 - 1; ++r)
                        {
                            tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                                  basis0[nq0 + i] * s_wsp0[cnt_kji] * tmpQ;
                            atomicAdd(s_prod + r, tmp);
                        }
                    }
                }
            }

            __syncthreads();

            if constexpr (SCALE)
            {
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1] * scale;
                }
                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    for (unsigned int r = threadIdx.z; r < nm2 - 1;
                         r += blockDim.z)
                    {
                        out[outoffset + nm2 + r] += s_prod[r] * scale;
                    }
                }
            }
            else
            {
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1];
                }
                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    for (unsigned int r = threadIdx.z; r < nm2 - 1;
                         r += blockDim.z)
                    {
                        out[outoffset + nm2 + r] += s_prod[r];
                    }
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePrismKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_w0 = shared;
    TData *s_w1 = s_w0 + nq0;
    TData *s_w2 = s_w1 + nq1;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq2)
    {
        s_w2[sIndex] = w2[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + (nq2 * nq1 + nq2 + nm1) * e;
        TData *wsp1            = wsp0 + nq2 * nq1;
        TData *wsp2            = wsp1 + nq2;

        for (unsigned int p = 0, mode_pqr = 0; p < nm0; ++p)
        {
            for (unsigned int k = 0, cnt_kj = 0, cnt_kji = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index    = inoffset + cnt_kji;
                        unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[nq0 * p + i] *
                                  jac[jacindex] * s_w0[i];
                    }
                    wsp0[cnt_kj] = sum_kj;
                }
            }

            for (unsigned int q = 0; q < nm1; ++q)
            {
                for (unsigned int k = 0, cnt_kj = 0; k < nq2; ++k)
                {
                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k += basis1[q * nq1 + j] * s_w1[j] * wsp0[cnt_kj];
                    }
                    wsp1[k] = sum_k;
                }

                for (int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                {
                    unsigned int mode_pr = (2 * nm2 - p + 1) * p / 2;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        sum_k +=
                            basis2[(mode_pr + r) * nq2 + k] * s_w2[k] * wsp1[k];
                    }

                    if (SCALE)
                    {
                        sum_k *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr;
                    if constexpr (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                }
            }
        }

        // Add correction for collapsed coordinate.
        if (correct)
        {
            for (unsigned int q = 0; q < nm1; ++q)
            {
                wsp2[q] = 0.0;
            }

            for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
            {
                TData k_weight = s_w2[k];
                if constexpr (!DEFORMED)
                {
                    k_weight *= jac[e];
                }

                for (unsigned int j = 0; j < nq1; ++j)
                {
                    TData kj_weight = k_weight * s_w1[j];
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index = inoffset + cnt_kji;
                        TData prod         = kj_weight * s_w0[i] * in[index];
                        if constexpr (DEFORMED)
                        {
                            prod *= jac[index];
                        }

                        for (unsigned int q = 0; q < nm1; ++q)
                        {
                            wsp2[q] += prod * basis2[nq2 + k] *
                                       basis1[q * nq1 + j] * basis0[nq0 + i];
                        }
                    }
                }
            }

            for (unsigned int q = 0; q < nm1; ++q)
            {
                unsigned int index = outoffset + nm2 * q + 1;
                if constexpr (SCALE)
                {
                    out[index] += wsp2[q] * scale;
                }
                else
                {
                    out[index] += wsp2[q];
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePrismKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_wsp3 = s_wsp2 + nm0 * nm1 * nq2;

    __syncthreads();

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji  = nq1 * nq0 * k + nq0 * j + i;
                    unsigned int index    = inoffset + cnt_kji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    s_wsp0[cnt_kji]       = in[index] * jac[jacindex];
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj += s_wsp0[cnt_kji] * basis0[nq0 * p + i] * w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
                {
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;
                    unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k += basis1[q * nq1 + j] * w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                for (int r = threadIdx.z; r < nm2 - p; r += blockDim.z)
                {
                    unsigned int cnt_pqk  = nm1 * nq2 * p + nq2 * q;
                    unsigned int mode_pr  = (2 * nm2 - p + 1) * p / 2;
                    unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q + r;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum_k += basis2[(mode_pr + r) * nq2 + k] * w2[k] *
                                 s_wsp2[cnt_pqk];
                    }

                    if (SCALE)
                    {
                        sum_k *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr;
                    if constexpr (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                }
            }
        }

        __syncthreads();

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (threadIdx.x == 0 && threadIdx.z == 0)
            {
                for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
                {
                    s_wsp2[q] = 0.0;
                }
            }

            __syncthreads();

            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                TData k_weight = w2[k];
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    TData kj_weight = k_weight * w1[j];
                    for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                    {
                        unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j + i;
                        TData prod = kj_weight * w0[i] * s_wsp0[cnt_kji];
                        for (unsigned int q = 0; q < nm1; ++q)
                        {
                            atomicAdd(s_wsp2 + q, prod * basis2[nq2 + k] *
                                                      basis1[q * nq1 + j] *
                                                      basis0[nq0 + i]);
                        }
                    }
                }
            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.z == 0)
            {
                for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
                {
                    unsigned int index = outoffset + nm2 * q + 1;
                    if constexpr (SCALE)
                    {
                        out[index] += s_wsp2[q] * scale;
                    }
                    else
                    {
                        out[index] += s_wsp2[q];
                    }
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePyrKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_w0 = shared;
    TData *s_w1 = s_w0 + nq0;
    TData *s_w2 = s_w1 + nq1;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0)
    {
        s_w0[sIndex] = w0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1)
    {
        s_w1[sIndex] = w1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq2)
    {
        s_w2[sIndex] = w2[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;
        TData *wsp0            = wsp + (nq2 * nq1 + nq2) * e;
        TData *wsp1            = wsp0 + nq2 * nq1;

        for (unsigned int p = 0, mode_pqr = 0; p < nm0; ++p)
        {
            for (unsigned int k = 0, cnt_kj = 0, cnt_kji = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index    = inoffset + cnt_kji;
                        unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[nq0 * p + i] *
                                  jac[jacindex] * s_w0[i];
                    }
                    wsp0[cnt_kj] = sum_kj;
                }
            }

            for (unsigned int q = 0; q < p; ++q)
            {
                for (unsigned int k = 0, cnt_kj = 0; k < nq2; ++k)
                {
                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k += basis1[q * nq1 + j] * s_w1[j] * wsp0[cnt_kj];
                    }
                    wsp1[k] = sum_k;
                }

                for (unsigned int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                {
                    TData sum_k = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        sum_k += basis2[mode_pqr * nq2 + k] * s_w2[k] * wsp1[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum_k *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr;
                    if constexpr (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                }
            }

            for (unsigned int q = p; q < nm1; ++q)
            {
                for (unsigned int k = 0, cnt_kj = 0; k < nq2; ++k)
                {
                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k += basis1[q * nq1 + j] * s_w1[j] * wsp0[cnt_kj];
                    }
                    wsp1[k] = sum_k;
                }

                for (unsigned int r = 0; r < nm2 - q; ++r, ++mode_pqr)
                {
                    TData sum_k = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        sum_k += basis2[mode_pqr * nq2 + k] * s_w2[k] * wsp1[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum_k *= scale;
                    }

                    unsigned int index = outoffset + mode_pqr;
                    if constexpr (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                }
            }
        }

        // Add correction for collapsed coordinate.
        if (correct)
        {
            TData prod = 0.0;
            for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
            {
                TData tmpQ2 = s_w2[k];
                if constexpr (!DEFORMED)
                {
                    tmpQ2 *= jac[e];
                }

                for (unsigned int j = 0; j < nq1; ++j)
                {
                    TData tmpQ1 = tmpQ2 * s_w1[j];
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index = inoffset + cnt_kji;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];
                        if constexpr (DEFORMED)
                        {
                            tmpQ *= jac[index];
                        }

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= in[index] * tmpQ;
                        prod += tmp;
                    }
                }
            }

            // add to existing entry
            if constexpr (SCALE)
            {
                out[outoffset + 1] += prod * scale;
            }
            else
            {
                out[outoffset + 1] += prod;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePyrKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData scale = 1.0)
{
    extern __shared__ TData shared[];
    TData *s_prod = shared;
    TData *s_wsp0 = s_prod + 1;
    TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int inoffset  = nq0 * nq1 * nq2 * e;
        unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji  = k * nq1 * nq0 + j * nq0 + i;
                    unsigned int index    = inoffset + cnt_kji;
                    unsigned int jacindex = DEFORMED ? index : e;
                    s_wsp0[cnt_kji]       = in[index] * jac[jacindex];
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj += s_wsp0[cnt_kji] * basis0[nq0 * p + i] * w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
                {
                    unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;
                    unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k += basis1[q * nq1 + j] * w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        __syncthreads();

        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                unsigned int mode_pq = nm1 * (2 * nm2 + 1 - nm1) * p;
                mode_pq -= (p - 1) * p / 2;
                mode_pq -= (p - 1) * p * (2 * p - 1) / 6;
                mode_pq /= 2;

                if (q < p)
                {
                    for (unsigned int r = threadIdx.z; r < nm2 - p;
                         r += blockDim.z)
                    {
                        unsigned int cnt_pqk  = nm1 * nq2 * p + nq2 * q;
                        unsigned int mode_pqr = mode_pq + q * (nm2 - p) + r;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += basis2[mode_pqr * nq2 + k] * w2[k] *
                                     s_wsp2[cnt_pqk];
                        }

                        if constexpr (SCALE)
                        {
                            sum_k *= scale;
                        }

                        unsigned int index = outoffset + mode_pqr;
                        if constexpr (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                }
                else
                {
                    for (unsigned int r = threadIdx.z; r < nm2 - q;
                         r += blockDim.z)
                    {
                        unsigned int cnt_pqk  = nm1 * nq2 * p + nq2 * q;
                        unsigned int mode_pqr = mode_pq + p * (nm2 - p);
                        mode_pqr +=
                            ((2 * (nm2 - p) - (q - p) + 1) * (q - p)) / 2 + r;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += basis2[mode_pqr * nq2 + k] * w2[k] *
                                     s_wsp2[cnt_pqk];
                        }
                        if constexpr (SCALE)
                        {
                            sum_k *= scale;
                        }

                        unsigned int index = outoffset + mode_pqr;
                        if constexpr (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                (*s_prod) = 0.0;
            }

            __syncthreads();

            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                TData tmpQ2 = w2[k];
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    TData tmpQ1 = tmpQ2 * w1[j];
                    for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                    {
                        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * w0[i];

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= s_wsp0[cnt_kji] * tmpQ;
                        atomicAdd(s_prod, tmp);
                    }
                }
            }

            __syncthreads();

            // add to existing entry
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                if constexpr (SCALE)
                {
                    out[outoffset + 1] += (*s_prod) * scale;
                }
                else
                {
                    out[outoffset + 1] += (*s_prod);
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTBase1DKernel(const unsigned int gridSize,
                            const unsigned int blockSize,
                            const unsigned int nm0, const unsigned int nq0,
                            const unsigned int nElmts, const TData *basis0,
                            const TData *w0, const TData *jac, const TData *in,
                            TData *out, TData scale = 1.0)
{
    if constexpr (FLAG_QP)
    {
        unsigned int nshared = sizeof(TData) * (nq0);
        IProductWRTBaseSegKernel<ExecSpace, TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, dim3(32), nshared>>>(nm0, nq0, nElmts, basis0, w0, jac,
                                              in, out, scale);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nm0 * nq0 + nq0);
        IProductWRTBaseSegKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize, nshared>>>(nm0, nq0, nElmts, basis0, w0,
                                               jac, in, out, scale);
    }
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTBase2DKernel(
        const unsigned int gridSize, const unsigned int blockSize,
        LibUtilities::ShapeType shapetype, const unsigned int nm0,
        const unsigned int nm1, const unsigned int nq0, const unsigned int nq1,
        const unsigned int nElmts, const bool correct, const TData *basis0,
        const TData *basis1, const TData *w0, const TData *w1, const TData *jac,
        TData *wsp, const TData *in, TData *out, TData scale = 1.0)
{
    if (shapetype == LibUtilities::Quad)
    {
        unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);

        if constexpr (FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 * nq1 + nm0 * nq1);
            IProductWRTBaseQuadKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(8, 8), nshared>>>(nm0, nm1, nmTot, nq0, nq1,
                                                    nElmts, basis0, basis1, w0,
                                                    w1, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nm0 * nq0 + nm1 * nq1 + nq0 + nq1);
            IProductWRTBaseQuadKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nElmts, basis0, basis1, w0, w1,
                    jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        if constexpr (FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 * nq1 + nm0 * nq1 + 1);
            IProductWRTBaseTriKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(8, 8), nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nElmts, correct, basis0, basis1,
                    w0, w1, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            IProductWRTBaseTriKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nElmts, correct, basis0, basis1,
                    w0, w1, jac, wsp, in, out, scale);
        }
    }
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTBase3DKernel(
        const unsigned int gridSize, const unsigned int blockSize,
        LibUtilities::ShapeType shapetype, const unsigned int nm0,
        const unsigned int nm1, const unsigned int nm2, const unsigned int nq0,
        const unsigned int nq1, const unsigned int nq2,
        const unsigned int nElmts, const bool correct, const TData *basis0,
        const TData *basis1, const TData *basis2, const TData *w0,
        const TData *w1, const TData *w2, const TData *jac, TData *wsp,
        const TData *in, TData *out, TData scale = 1.0)
{
    if (shapetype == LibUtilities::Hex)
    {
        unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);

        if constexpr (FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2);
            IProductWRTBaseHexKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                    basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) *
                (nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 + nq1 + nq2);
            IProductWRTBaseHexKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                    basis2, w0, w1, w2, jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);

        if constexpr (FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) * (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 +
                                 ((2 * nm1 - nm0 + 1) * nm0 / 2) * nq2 + nm2);
            IProductWRTBaseTetKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBaseTetKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out,
                    scale);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);

        if constexpr (FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nm1);
            IProductWRTBasePrismKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBasePrismKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out,
                    scale);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);

        if constexpr (FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 1);
            IProductWRTBasePyrKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBasePyrKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out,
                    scale);
        }
    }
}

#endif

} // namespace Nektar::Operators::detail
