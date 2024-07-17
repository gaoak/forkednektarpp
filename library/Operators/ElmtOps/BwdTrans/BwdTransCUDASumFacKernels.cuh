///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransCUDASumFacKernels.cuh
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, bool SHMEM = true>
__global__ void BwdTransSegKernel(const unsigned int nm0,
                                  const unsigned int nq0,
                                  const unsigned int nelmt,
                                  const TData *__restrict__ basis0,
                                  const TData *__restrict__ in,
                                  TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += in[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                       s_basis0[p * nq0 + i];
            }
            out[nq0 * warpsize * iwarp + warpsize * i + ilane] = tmp;
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransSegKernel_QP(const unsigned int nm0,
                                     const unsigned int nq0,
                                     const unsigned int nelmt,
                                     const TData *__restrict__ basis0,
                                     const TData *__restrict__ in,
                                     TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    TData *s_wsp0   = shared;
    TData *s_basis0 = SHMEM ? s_wsp0 + nm0 : (TData *)basis0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nm0 * e;
        TData *outptr      = out + nq0 * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nm0; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; p++)
            {
                tmp += s_wsp0[p] * s_basis0[p * nq0 + i];
            }
            outptr[i] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    TData *__restrict__ wsp, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            // direction 0
            for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize * cnt_qp +
                              ilane] *
                           s_basis0[p * nq0 + i];
                }
                wsp[nm1 * warpsize * iwarp + warpsize * q + ilane] = tmp;
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += wsp[nm1 * warpsize * iwarp + warpsize * q + ilane] *
                           s_basis1[q * nq1 + j];
                }
                out[nqTot * warpsize * iwarp + warpsize * (nq0 * j + i) +
                    ilane] = tmp;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm1 * nq0 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int tid0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nm1 * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
        {
            for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
            {
                const unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp       = nm0 * q;

                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
                }
                s_wsp1[cnt_iq] = tmp;
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                const unsigned int cnt_ji = nq0 * j + i;
                unsigned int cnt_iq       = nm1 * i;

                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
                {
                    tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
                }
                outptr[cnt_ji] = tmp;
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransQuadKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm1 * nq0 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int i = tid / nm1;
            unsigned int cnt_qp  = nm0 * q;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
            {
                tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = tid / nq0;
            unsigned int cnt_iq  = nm1 * i;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
            {
                tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransTriKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq1;
             tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            // direction 1
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < (nm1 - p); ++q, ++mode_pq)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize * mode_pq +
                              ilane] *
                           s_basis1[mode_pq * nq1 + j];
                }
                wsp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += wsp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                           s_basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize + ilane] *
                           s_basis0[nq0 + i] * s_basis1[nq1 + j];
                }

                out[nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane] = tmp;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransTriKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int tid0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nmTot * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
            {
                const unsigned int cnt_jp = nm0 * j + p;
                unsigned int mode_pq      = (2u * nm1 - p + 1u) * p / 2u;

                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    tmp += s_basis1[mode_pq * nq1 + j] * s_wsp0[mode_pq];
                }
                s_wsp1[cnt_jp] = tmp;
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                const unsigned int cnt_ij = nq0 * j + i;
                unsigned int cnt_jp       = nm0 * j;

                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
                {
                    tmp += s_wsp1[cnt_jp] * s_basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += s_wsp0[1] * s_basis0[nq0 + i] * s_basis1[nq1 + j];
                }

                outptr[cnt_ij] = tmp;
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransTriKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq1;
             tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq1; tid += blockDim.x)
        {
            const unsigned int p = tid % nm0;
            const unsigned int j = tid / nm0;
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                tmp += s_basis1[mode_pq * nq1 + j] * s_wsp0[mode_pq];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = tid / nq0;
            unsigned int cnt_jp  = nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
            {
                tmp += s_wsp1[cnt_jp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                tmp += s_wsp0[1] * s_basis0[nq0 + i] * s_basis1[nq1 + j];
            }

            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm2 * nq2; tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *wsp1        = wsp;
        TData *wsp2        = wsp1 + (nm1 * nm2) * nelmt;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            // direction 0
            for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u; r < nm2; ++r)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * cnt_rqp + ilane] *
                               s_basis0[p * nq0 + i];
                    }
                    wsp1[nm1 * nm2 * warpsize * iwarp + warpsize * cnt_rq +
                         ilane] = tmp;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                    {
                        tmp += wsp1[nm1 * nm2 * warpsize * iwarp +
                                    warpsize * cnt_rq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    wsp2[nm2 * warpsize * iwarp + warpsize * r + ilane] = tmp;
                }

                // direction 2
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r)
                    {
                        tmp += wsp2[nm2 * warpsize * iwarp + warpsize * r +
                                    ilane] *
                               s_basis2[r * nq2 + k];
                    }
                    out[nqTot * warpsize * iwarp +
                        warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nq0 * nm1 * nm2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq1 * nq0 * nm2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nm1 * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = tid0; tid < nm2 * nq2; tid += stride)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int i = threadIdx.z; i < nq0; i += blockDim.z)
        {
            for (unsigned int r = threadIdx.y; r < nm2; r += blockDim.y)
            {
                for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
                {
                    const unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r + q;
                    unsigned int cnt_rqp       = nm1 * nm0 * r + nm0 * q;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
                    }
                    s_wsp1[cnt_irq] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.z; j < nq1; j += blockDim.z)
        {
            for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
            {
                for (unsigned int r = threadIdx.x; r < nm2; r += blockDim.x)
                {
                    const unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i + r;
                    unsigned int cnt_irq       = nm1 * nm2 * i + nm1 * r;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                    {
                        tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[cnt_jir] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int cnt_jir       = nq0 * nm2 * j + nm2 * i;

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                    {
                        tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
                    }
                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nq0 * nm1 * nm2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq1 * nq0 * nm2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm2 * nq2; tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int r = (tid / nm1) % nm2;
            const unsigned int i = tid / (nm1 * nm2);
            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
            {
                tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int r = tid % nm2;
            const unsigned int i = (tid / nm2) % nq0;
            const unsigned int j = tid / (nm2 * nq0);
            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
            {
                tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[tid] = tmp;
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = (tid / nq0) % nq1;
            const unsigned int k = tid / (nq1 * nq0);
            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
            {
                tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransTetKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm01 * nq1;
             tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm01 * nelmt;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode_pq = 0u, mode_pqr = 0u; p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[nq2 * mode_pqr + k];
                    }
                    fpq[nm01 * warpsize * iwarp + warpsize * mode_pq + ilane] =
                        tmp;
                }

                // increment mode in case order1!=order2
                for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
                {
                    mode_pqr += nm2 - p - q;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm01 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[mode_pq * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 *
                               in[nmTot * warpsize * iwarp + warpsize + ilane];

                        // bottom vertex
                        tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[k];
                        tmp += tmp1 * in[nmTot * warpsize * iwarp +
                                         warpsize * nm2 + ilane];

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                            tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                            tmp += tmp1 * in[nmTot * warpsize * iwarp +
                                             warpsize * (nm2 + r) + ilane];
                        }
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransTetKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + nm01 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nm01 * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = tid0; tid < nmTot * nq2; tid += stride)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int p = threadIdx.y; p < nm0; p += blockDim.y)
            {
                for (unsigned int q = threadIdx.x; q < nm1 - p; q += blockDim.x)
                {
                    const unsigned int cnt_kpq =
                        nm01 * k + (2u * nm1 - p + 1u) * p / 2u + q;
                    unsigned int mode_pqr = (2u * (nm2 - p) - q + 1u) * q;
                    mode_pqr += nm2 * (nm2 + 1u) * p;
                    mode_pqr -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
                    mode_pqr += (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode_pqr /= 2u;

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode_pqr)
                    {
                        tmp += s_wsp0[mode_pqr] * s_basis2[k + nq2 * mode_pqr];
                    }
                    s_wsp1[cnt_kpq] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_pq        = (2u * nm1 - p + 1u) * p / 2u;
                    unsigned int cnt_kpq        = nm01 * k + mode_pq;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1 - p;
                         ++q, ++cnt_kpq, ++mode_pq)
                    {
                        tmp += s_wsp1[cnt_kpq] * s_basis1[mode_pq * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];

                        // bottom vertex
                        tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[k];
                        tmp += tmp1 * s_wsp0[nm2];

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                            tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                            tmp += tmp1 * s_wsp0[nm2 + r];
                        }
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransTetKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + nm01 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

    // Temporary solution, to be removed - TODO
    unsigned int *vpindex = (unsigned int *)pindex;
    unsigned int *vqindex = (unsigned int *)qindex;
    for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
    {
        for (unsigned int q = 0; q < nm1 - p; q++)
        {
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
            vpindex[mode_pq]     = p;
            vqindex[mode_pq]     = q;
        }
    }

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm01 * nq1;
             tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int k  = tid / nm01;
            const unsigned int p  = pindex[tid % nm01];
            const unsigned int q  = qindex[tid % nm01];
            unsigned int mode_pqr = (2u * (nm2 - p) - q + 1u) * q;
            mode_pqr += nm2 * (nm2 + 1u) * p;
            mode_pqr -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
            mode_pqr += (p - 1u) * p * (2u * p - 1u) / 6u;
            mode_pqr /= 2u;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode_pqr)
            {
                tmp += s_wsp0[mode_pqr] * s_basis2[k + nq2 * mode_pqr];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int p = tid % nm0;
            const unsigned int j = (tid / nm0) % nq1;
            const unsigned int k = tid / (nm0 * nq1);
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;
            unsigned int cnt_kpq = nm01 * k + mode_pq;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++cnt_kpq, ++mode_pq)
            {
                tmp += s_wsp1[cnt_kpq] * s_basis1[mode_pq * nq1 + j];
            }
            s_wsp2[tid] = tmp;
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i  = tid % nq0;
            const unsigned int j  = (tid / nq0) % nq1;
            const unsigned int k  = tid / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                // top vertex
                TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[nq2 + k];
                tmp += tmp1 * s_wsp0[1];

                // bottom vertex
                tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[k];
                tmp += tmp1 * s_wsp0[nm2];

                // singular edge
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                    tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                    tmp += tmp1 * s_wsp0[nm2 + r];
                }
            }

            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransPrismKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm12 * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm0 * nm1 * nelmt;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode_pr = 0u, mode_pq = 0u, mode_pqr = 0u;
                 p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[(mode_pr + r) * nq2 + k];
                    }
                    fpq[nm0 * nm1 * warpsize * iwarp + warpsize * mode_pq +
                        ilane] = tmp;
                }
                mode_pr += nm2 - p;
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm0 * nm1 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                                   s_basis0[nq0 + i] *
                                   in[nmTot * warpsize * iwarp +
                                      warpsize * (nm2 * q + 1u) + ilane];
                        }
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransPrismKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nm1 * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = tid0; tid < nm12 * nq2; tid += stride)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int p = threadIdx.y; p < nm0; p += blockDim.y)
            {
                for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
                {
                    const unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;
                    unsigned int mode_pr        = (2u * nm2 - p + 1u) * p / 2u;
                    unsigned int mode_pqr       = mode_pr * nm1 + (nm2 - p) * q;

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p;
                         ++r, ++mode_pqr, ++mode_pr)
                    {
                        tmp += s_wsp0[mode_pqr] * s_basis2[mode_pr * nq2 + k];
                    }
                    s_wsp1[mode_kpq] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_kpq       = nm0 * nm1 * k + nm1 * p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                                   s_basis0[nq0 + i] * s_wsp0[q * nm2 + 1u];
                        }
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true>
__global__ void BwdTransPrismKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm12 * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int q  = tid % nm1;
            const unsigned int p  = (tid / nm1) % nm0;
            const unsigned int k  = tid / (nm1 * nm0);
            unsigned int mode_pr  = (2u * nm2 - p + 1u) * p / 2u;
            unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr, ++mode_pr)
            {
                tmp += s_wsp0[mode_pqr] * s_basis2[mode_pr * nq2 + k];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int p  = tid % nm0;
            const unsigned int j  = (tid / nm0) % nq1;
            const unsigned int k  = tid / (nm0 * nq1);
            unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
            {
                tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[tid] = tmp;
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i  = tid % nq0;
            const unsigned int j  = (tid / nq0) % nq1;
            const unsigned int k  = tid / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                           s_basis0[nq0 + i] * s_wsp0[q * nm2 + 1u];
                }
            }

            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransPyrKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm0 * nm1 * nelmt;

        for (int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode_pq = 0u, mode_pqr = 0u; p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < p; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[mode_pqr * nq2 + k];
                    }
                    fpq[nm0 * nm1 * warpsize * iwarp + warpsize * mode_pq +
                        ilane] = tmp;
                }

                for (unsigned int q = p; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[mode_pqr * nq2 + k];
                    }
                    fpq[nm0 * nm1 * warpsize * iwarp + warpsize * mode_pq +
                        ilane] = tmp;
                }

                // increment mode in case nm2>nm1
                for (unsigned int q = nm1; q < nm2 - p; ++q)
                {
                    mode_pqr += nm2 - q;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm0 * nm1 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 *
                               in[nmTot * warpsize * iwarp + warpsize + ilane];
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransPyrKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nm0 * nq0; tid += stride)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = tid0; tid < nm1 * nq1; tid += stride)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = tid0; tid < nmTot * nq2; tid += stride)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int tid0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int tid = tid0; tid < nmTot; tid += stride)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int p = threadIdx.y; p < nm0; p += blockDim.y)
            {
                for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
                {
                    const unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;
                    unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
                    mode_pqr -= (p - 1u) * p / 2u;
                    mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode_pqr /= 2u;

                    if (q < p)
                    {
                        mode_pqr += q * (nm2 - p);
                        TData tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                        {
                            tmp +=
                                s_wsp0[mode_pqr] * s_basis2[mode_pqr * nq2 + k];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                    else
                    {
                        mode_pqr += p * (nm2 - p);
                        mode_pqr +=
                            ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;

                        TData tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode_pqr)
                        {
                            tmp +=
                                s_wsp0[mode_pqr] * s_basis2[mode_pqr * nq2 + k];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_kpq       = nm0 * nm1 * k + nm1 * p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, bool SHMEM = true> // not working for nm2 > nm1
__global__ void BwdTransPyrKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
        {
            s_basis0[tid] = basis0[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
        {
            s_basis1[tid] = basis1[tid];
        }

        for (unsigned int tid = threadIdx.x; tid < nmTot * nq2;
             tid += blockDim.x)
        {
            s_basis2[tid] = basis2[tid];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nmTot; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int q  = tid % nm1;
            const unsigned int p  = (tid / nm1) % nm0;
            const unsigned int k  = tid / (nm1 * nm0);
            unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
            mode_pqr -= (p - 1u) * p / 2u;
            mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
            mode_pqr /= 2u;

            if (q < p)
            {
                mode_pqr += q * (nm2 - p);
                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                {
                    tmp += s_wsp0[mode_pqr] * s_basis2[mode_pqr * nq2 + k];
                }
                s_wsp1[tid] = tmp;
            }
            else
            {
                mode_pqr += p * (nm2 - p);
                mode_pqr += ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;

                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode_pqr)
                {
                    tmp += s_wsp0[mode_pqr] * s_basis2[mode_pqr * nq2 + k];
                }
                s_wsp1[tid] = tmp;
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nm0 * nq1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int p  = tid % nm0;
            const unsigned int j  = (tid / nm0) % nq1;
            const unsigned int k  = tid / (nm0 * nq1);
            unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
            {
                tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[tid] = tmp;
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nqTot; tid += blockDim.x)
        {
            const unsigned int i  = tid % nq0;
            const unsigned int j  = (tid / nq0) % nq1;
            const unsigned int k  = tid / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                // top vertex
                TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[nq2 + k];
                tmp += tmp1 * s_wsp0[1];
            }

            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

// Kernel launchers
template <typename ExecSpace, typename TData, bool MULTILEVEL = true,
          bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    BwdTrans1DKernel(const unsigned int nm0, const unsigned int nq0,
                     const unsigned int nelmt, const TData *basis0,
                     const TData *in, TData *out)
{
    const unsigned int gridsize  = MULTILEVEL ? std::min(nq0, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if constexpr (MULTILEVEL)
    {
        unsigned int nshared = sizeof(TData) * (nm0 + SHMEM ? nm0 * nq0 : 0u);
        BwdTransSegKernel_QP<TData, SHMEM>
            <<<gridsize, dim3(32), nshared>>>(nm0, nq0, nelmt, basis0, in, out);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nm0 * nq0);
        BwdTransSegKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
            nm0, nq0, nelmt, basis0, in, out);
    }
}

template <typename ExecSpace, typename TData, bool MULTILEVEL = true,
          bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    BwdTrans2DKernel(LibUtilities::ShapeType shapetype, const unsigned int nm0,
                     const unsigned int nm1, const unsigned int nq0,
                     const unsigned int nq1, const unsigned int nelmt,
                     const bool correct, const TData *basis0,
                     const TData *basis1, TData *wsp, const TData *in,
                     TData *out)
{
    const dim3 blocksize2d      = dim3(std::min(nq0, 16u), std::min(nq1, 16u));
    const unsigned int gridsize = MULTILEVEL ? std::min(nq0 * nq1, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Quad)
    {
        const unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nq0 * nm0 + nq1 * nm1) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nmTot + nq0 * nm1);
            BwdTransQuadKernel_QP<TData, SHMEM>
                <<<gridsize, blocksize2d, nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, in, out);
            // BwdTransQuadKernel_QP_1D<TData, SHMEM>
            //    <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, in,
            //         out);
        }
        else
        {
            BwdTransQuadKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, wsp, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nm0 * nq0 + nmTot * nq1) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nmTot + nm0 * nq1);
            BwdTransTriKernel_QP<TData, SHMEM><<<nelmt, blocksize2d, nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, correct, basis0, basis1, in,
                out);
            // BwdTransTriKernel_QP_1D<TData, SHMEM>
            //     <<<nelmt, blocksize, nshared>>>(nm0, nm1, nmTot, nq0, nq1,
            //                                        nelmt, correct, basis0,
            //                                        basis1, in, out);
        }
        else
        {
            BwdTransTriKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, correct, basis0, basis1, wsp,
                in, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool MULTILEVEL = true,
          bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    BwdTrans3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nm0,
                     const unsigned int nm1, const unsigned int nm2,
                     const unsigned int nq0, const unsigned int nq1,
                     const unsigned int nq2, const unsigned int nelmt,
                     const bool correct, const TData *basis0,
                     const TData *basis1, const TData *basis2, TData *wsp,
                     const TData *in, TData *out)
{
    const dim3 blocksize3d =
        dim3(std::min(nq0, 8u), std::min(nq1, 8u), std::min(nq2, 8u));
    const unsigned int gridsize =
        MULTILEVEL ? std::min(nq0 * nq1 * nq2, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Hex)
    {
        const unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nq0 * nm0 + nq1 * nm1 + nq2 * nm2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared +=
                sizeof(TData) * (nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot +
                                 (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2));
            BwdTransHexKernel_QP<TData, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(nm0, nm1, nm2, nmTot, nq0,
                                                     nq1, nq2, nelmt, basis0,
                                                     basis1, basis2, in, out);
            // BwdTransHexKernel_QP_1D<TData, SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(nm0, nm1, nm2, nmTot, nq0,
            //                                          nq1, nq2, nelmt,
            //                                          basis0, basis1, basis2,
            //                                          in, out);
        }
        else
        {
            BwdTransHexKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0, basis1,
                basis2, wsp, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nm0 * nq0 + nm01 * nq1 + nmTot * nq2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) *
                       (nmTot + ((2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2) +
                        (nm0 * nq1 * nq2));
            BwdTransTetKernel_QP<TData, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                    basis1, basis2, in, out);
            // unsigned int *pindex;
            // unsigned int *qindex;
            // cudaMalloc((void **)&pindex, sizeof(unsigned int)*nm01);
            // cudaMalloc((void **)&qindex, sizeof(unsigned int)*nm01);
            // BwdTransTetKernel_QP_1D<TData, SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         pindex, qindex, basis0, basis1, basis2, in, out);
        }
        else
        {
            BwdTransTetKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                basis1, basis2, wsp, in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        const unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nm0 * nq0 + nm1 * nq1 + nm12 * nq2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared +=
                sizeof(TData) * (nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2));
            BwdTransPrismKernel_QP<TData, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                    basis1, basis2, in, out);
            // BwdTransPrismKernel_QP_1D<TData, SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         basis0, basis1, basis2, in, out);
        }
        else
        {
            BwdTransPrismKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                basis1, basis2, wsp, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared =
            SHMEM ? sizeof(TData) * (nm0 * nq0 + nm1 * nq1 + nmTot * nq2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared +=
                sizeof(TData) * (nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2));
            BwdTransPyrKernel_QP<TData, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                    basis1, basis2, in, out);
            // BwdTransPyrKernel_QP_1D<TData, SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         basis0, basis1, basis2, in, out);
        }
        else
        {
            BwdTransPyrKernel<TData, SHMEM><<<gridsize, blocksize, nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct, basis0,
                basis1, basis2, wsp, in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
