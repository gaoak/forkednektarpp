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

namespace Nektar::Operators::detail
{

template <typename TData>
__device__ __forceinline__ void BwdTransSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        TData tmp = 0.0;
        for (unsigned int p = 0u; p < nm0; ++p)
        {
            tmp += in[warpsize * p + ilane] * basis0[p * nq0 + i];
        }
        out[warpsize * i + ilane] = tmp;
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransSegSumFacQPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
    {
        TData tmp = 0.0;
        for (unsigned int p = 0u; p < nm0; p++)
        {
            tmp += in[p] * basis0[p * nq0 + i];
        }
        out[i] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // direction 0
        for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
            {
                tmp += in[warpsize * cnt_qp + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * q + ilane] = tmp;
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += wsp[warpsize * q + ilane] * basis1[q * nq1 + j];
            }
            out[warpsize * (nq0 * j + i) + ilane] = tmp;
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransQuadSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nq0 * nm1; idx += blockDim.x)
    {
        const unsigned int q = idx % nm1;
        const unsigned int i = idx / nm1;
        unsigned int cnt_qp  = nm0 * q;

        TData tmp = 0.0;
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
        {
            tmp += in[cnt_qp] * basis0[p * nq0 + i];
        }
        wsp[idx] = tmp;
    }

    __syncthreads();

    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_iq  = nm1 * i;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
        {
            tmp += wsp[cnt_iq] * basis1[q * nq1 + j];
        }
        out[idx] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        // direction 1
        for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
        {
            TData tmp = 0.0;
            for (unsigned int q = 0u; q < (nm1 - p); ++q, ++mode_pq)
            {
                tmp +=
                    in[warpsize * mode_pq + ilane] * basis1[mode_pq * nq1 + j];
            }
            wsp[warpsize * p + ilane] = tmp;
        }

        // direction 0
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            TData tmp = 0.0;

            if (isModified)
            {
                tmp += in[warpsize + ilane] * basis0[nq0 + i] * basis1[nq1 + j];
            }

            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += wsp[warpsize * p + ilane] * basis0[p * nq0 + i];
            }

            out[warpsize * cnt_ji + ilane] = tmp;
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransTriSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nm0 * nq1; idx += blockDim.x)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = idx / nm0;
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            tmp += in[mode_pq] * basis1[mode_pq * nq1 + j];
        }
        wsp[idx] = tmp;
    }

    __syncthreads();

    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_jp  = nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            tmp += in[1] * basis0[nq0 + i] * basis1[nq1 + j];
        }

        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
        {
            tmp += wsp[cnt_jp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp1, TData *__restrict__ wsp2)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

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
                    tmp += in[warpsize * cnt_rqp + ilane] * basis0[p * nq0 + i];
                }
                wsp1[warpsize * cnt_rq + ilane] = tmp;
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
                    tmp +=
                        wsp1[warpsize * cnt_rq + ilane] * basis1[q * nq1 + j];
                }
                wsp2[warpsize * r + ilane] = tmp;
            }

            // direction 2
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    tmp += wsp2[warpsize * r + ilane] * basis2[r * nq2 + k];
                }
                out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransHexSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp1, TData *__restrict__ wsp2)
{
    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nq0 * nm1 * nm2;
         idx += blockDim.x)
    {
        const unsigned int q = idx % nm1;
        const unsigned int r = (idx / nm1) % nm2;
        const unsigned int i = idx / (nm1 * nm2);
        unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

        TData tmp = 0.0;
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
        {
            tmp += in[cnt_rqp] * basis0[p * nq0 + i];
        }
        wsp1[idx] = tmp;
    }

    __syncthreads();

    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nq0 * nq1 * nm2;
         idx += blockDim.x)
    {
        const unsigned int r = idx % nm2;
        const unsigned int i = (idx / nm2) % nq0;
        const unsigned int j = idx / (nm2 * nq0);
        unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
        {
            tmp += wsp1[cnt_irq] * basis1[q * nq1 + j];
        }
        wsp2[idx] = tmp;
    }

    __syncthreads();

    // direction 2
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq1 * nq0);
        unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

        TData tmp = 0.0;
        for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
        {
            tmp += wsp2[cnt_jir] * basis2[r * nq2 + k];
        }
        out[idx] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - p - q;
                     ++r, ++mode2, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[nq2 * mode2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case order1!=order2
            for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
            {
                mode2 += nm2 - p - q;
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
                    tmp += fpq[warpsize * mode_pq + ilane] *
                           basis1[mode_pq * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];

                    // bottom vertex
                    TData tmp1 = basis2[k] * in[warpsize * nm2 + ilane];

                    // singular edge
                    for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                    {
                        tmp1 += basis2[(r + 1u) * nq2 + k] *
                                in[warpsize * (nm2 + r) + ilane];
                    }
                    tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
                }

                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransTetSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp1, TData *__restrict__ wsp2)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    // direction 2
    for (unsigned int idx = threadIdx.x; idx < nm01 * nq2; idx += blockDim.x)
    {
        const unsigned int k = idx / nm01;
        const unsigned int p = pindex[idx % nm01];
        const unsigned int q = qindex[idx % nm01];
        unsigned int mode2   = (2u * (nm2 - p) - q + 1u) * q;
        mode2 += nm2 * (nm2 + 1u) * p;
        mode2 -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
        mode2 += (p - 1u) * p * (2u * p - 1u) / 6u;
        mode2 /= 2u;
        unsigned int mode_pqr =
            mode2 -
            ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

        TData tmp = 0.0;
        for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
        {
            tmp += in[mode_pqr] * basis2[k + nq2 * mode2];
        }
        wsp1[idx] = tmp;
    }

    __syncthreads();

    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nm0 * nq1 * nq2;
         idx += blockDim.x)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = (idx / nm0) % nq1;
        const unsigned int k = idx / (nm0 * nq1);
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;
        unsigned int cnt_kpq = nm01 * k + mode_pq;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++cnt_kpq, ++mode_pq)
        {
            tmp += wsp1[cnt_kpq] * basis1[mode_pq * nq1 + j];
        }
        wsp2[idx] = tmp;
    }

    __syncthreads();

    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];

            // bottom vertex
            TData tmp1 = basis2[k] * in[nm2];

            // singular edge
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                tmp1 += basis2[(r + 1u) * nq2 + k] * in[nm2 + r];
            }
            tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
        }

        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp2[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransPrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

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
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[(mode_pr + r) * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
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
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        tmp += basis1[q * nq1 + j] *
                               in[warpsize * (nm2 * q + 1u) + ilane];
                    }
                    tmp *= basis2[nq2 + k] * basis0[nq0 + i];
                }

                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransPrismSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp1, TData *__restrict__ wsp2)
{
    // direction 2
    for (unsigned int idx = threadIdx.x; idx < nm0 * nm1 * nq2;
         idx += blockDim.x)
    {
        const unsigned int q  = idx % nm1;
        const unsigned int p  = (idx / nm1) % nm0;
        const unsigned int k  = idx / (nm1 * nm0);
        unsigned int mode_pr  = (2u * nm2 - p + 1u) * p / 2u;
        unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

        TData tmp = 0.0;
        for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr, ++mode_pr)
        {
            tmp += in[mode_pqr] * basis2[mode_pr * nq2 + k];
        }
        wsp1[idx] = tmp;
    }

    __syncthreads();

    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nm0 * nq1 * nq2;
         idx += blockDim.x)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp1[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp2[idx] = tmp;
    }

    __syncthreads();

    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += in[q * nm2 + 1u] * basis1[q * nq1 + j];
            }
            tmp *= basis2[nq2 + k] * basis0[nq0 + i];
        }

        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp2[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    __syncthreads();
}

template <typename TData>
__device__ __forceinline__ void BwdTransPyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u, mode_pq = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - max(p, q);
                     ++r, ++mode_pqr, ++mode2)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[mode2 * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case nm2>nm1
            for (unsigned int q = nm1; q < nm2; ++q)
            {
                mode2 += nm2 - q;
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
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];
                }

                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
__device__ __forceinline__ void BwdTransPyrSumFacQPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp1, TData *__restrict__ wsp2)
{
    // direction 2
    for (unsigned int idx = threadIdx.x; idx < nm0 * nm1 * nq2;
         idx += blockDim.x)
    {
        const unsigned int q = idx % nm1;
        const unsigned int p = (idx / nm1) % nm0;
        const unsigned int k = idx / (nm1 * nm0);
        unsigned int mode2 =
            (nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u;
        unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
        mode_pqr -= (p - 1u) * p / 2u;
        mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
        mode_pqr /= 2u;

        if (q < p)
        {
            mode_pqr += q * (nm2 - p);
            mode2 += mode_pqr;
            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
            {
                tmp += in[mode_pqr] * basis2[mode2 * nq2 + k];
            }
            wsp1[idx] = tmp;
        }
        else
        {
            mode_pqr += p * (nm2 - p);
            mode_pqr += ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;
            mode2 += mode_pqr;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
            {
                tmp += in[mode_pqr] * basis2[mode2 * nq2 + k];
            }
            wsp1[idx] = tmp;
        }
    }

    __syncthreads();

    // direction 1
    for (unsigned int idx = threadIdx.x; idx < nm0 * nq1 * nq2;
         idx += blockDim.x)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp1[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp2[idx] = tmp;
    }

    __syncthreads();

    // direction 0
    for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];
        }

        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp2[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    __syncthreads();
}

// General Launcher
template <typename Implementation, typename TData>
__device__ __forceinline__ void BwdTrans1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *inptr       = in + nm0 * warpsize * iwarp;
            TData *outptr            = out + nq0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, outptr);
            e += blockDim.x * gridDim.x;
        }
    }
    else
    {
        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *inptr = in + nm0 * e;
            TData *outptr      = out + nq0 * e;
            BwdTransSegSumFacQPKernel(nm0, nq0, basis0, inptr, outptr);
            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <typename Implementation, typename TData>
__global__ void BwdTrans1DKernelLauncher(const unsigned int nm0,
                                         const unsigned int nq0,
                                         const unsigned int nelmt,
                                         const TData *__restrict__ basis0,
                                         const TData *__restrict__ in,
                                         TData *__restrict__ out)
{
    BwdTrans1DKernel<Implementation>(nm0, nq0, nelmt, basis0, in, out);
}

// Size based template version.
template <typename Implementation, unsigned int nm0, unsigned int nq0,
          typename TData>
__global__ void BwdTrans1DKernelLauncher(const unsigned int nelmt,
                                         const TData *__restrict__ basis0,
                                         const TData *__restrict__ in,
                                         TData *__restrict__ out)
{
    BwdTrans1DKernel<Implementation>(nm0, nq0, nelmt, basis0, in, out);
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData>
__device__ __forceinline__ void BwdTrans2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp)
{
    const unsigned int nqTot = nq0 * nq1;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *inptr       = in + nmTot * warpsize * iwarp;
            TData *outptr            = out + nqTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wspptr = wsp + nm1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, outptr, wspptr);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wspptr = wsp + nm0 * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, outptr, wspptr);
            }
            e += blockDim.x * gridDim.x;
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        unsigned int nmode0, nmode1;
        TData *s_wsp0, *s_wsp1;
        TData *s_basis0, *s_basis1;

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            nmode0 = nm0;
            nmode1 = nm1;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_basis0 = s_wsp1 + nm1 * nq0;
            s_basis1 = s_basis0 + nm0 * nq0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            nmode0 = nm0;
            nmode1 = nmTot;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_basis0 = s_wsp1 + nm0 * nq1;
            s_basis1 = s_basis0 + nm0 * nq0;
        }

        // Copy to shared memory.
        for (unsigned int idx = threadIdx.x; idx < nmode0 * nq0;
             idx += blockDim.x)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nmode1 * nq1;
             idx += blockDim.x)
        {
            s_basis1[idx] = basis1[idx];
        }

        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *inptr = in + nmTot * e;
            TData *outptr      = out + nqTot * e;

            // Copy to shared memory.
            for (unsigned int idx = threadIdx.x; idx < nmTot; idx += blockDim.x)
            {
                s_wsp0[idx] = inptr[idx];
            }

            __syncthreads();

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                BwdTransQuadSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                           s_basis1, s_wsp0, outptr, s_wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                BwdTransTriSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                          s_basis0, s_basis1, s_wsp0, outptr,
                                          s_wsp1);
            }

            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData>
__global__ void BwdTrans2DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp)
{
    BwdTrans2DKernel<SHAPE_TYPE, Implementation>(nm0, nm1, nmTot, nq0, nq1,
                                                 nelmt, isModified, basis0,
                                                 basis1, in, out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          unsigned int nm0, unsigned int nm1, const unsigned nmTot,
          unsigned int nq0, unsigned int nq1, typename TData>
__global__ void BwdTrans2DKernelLauncher(const unsigned int nelmt,
                                         const bool isModified,
                                         const TData *__restrict__ basis0,
                                         const TData *__restrict__ basis1,
                                         const TData *__restrict__ in,
                                         TData *__restrict__ out,
                                         TData *__restrict__ wsp)
{
    BwdTrans2DKernel<SHAPE_TYPE, Implementation>(nm0, nm1, nmTot, nq0, nq1,
                                                 nelmt, isModified, basis0,
                                                 basis1, in, out, wsp);
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData>
__device__ __forceinline__ void BwdTrans3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *inptr       = in + nmTot * warpsize * iwarp;
            TData *outptr            = out + nqTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp1 = wsp + nm1 * nm2 * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (nm1 * nm2) * nelmt + nm2 * warpsize * iwarp;

                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, outptr,
                                        wsp1, wsp2);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

                TData *fpq = wsp + nm01 * warpsize * iwarp;
                TData *fp  = wsp + nm01 * nelmt + nm0 * warpsize * iwarp;

                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, outptr, fpq, fp);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *fpq = wsp + nm0 * nm1 * warpsize * iwarp;
                TData *fp  = wsp + nm0 * nm1 * nelmt + nm0 * warpsize * iwarp;

                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, outptr, fpq, fp);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *fpq = wsp + nm0 * nm1 * warpsize * iwarp;
                TData *fp  = wsp + nm0 * nm1 * nelmt + nm0 * warpsize * iwarp;

                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, outptr, fpq, fp);
            }
            e += blockDim.x * gridDim.x;
        }
    }
    else
    {
        extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

        unsigned int nmode0, nmode1, nmode2;
        TData *s_wsp0, *s_wsp1, *s_wsp2;
        TData *s_basis0, *s_basis1, *s_basis2;

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            nmode0 = nm0;
            nmode1 = nm1;
            nmode2 = nm2;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_wsp2   = s_wsp1 + nq0 * nm1 * nm2;
            s_basis0 = s_wsp2 + nq1 * nq0 * nm2;
            s_basis1 = s_basis0 + nmode0 * nq0;
            s_basis2 = s_basis1 + nmode1 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            nmode0 = nm0;
            nmode1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nmode2 = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_wsp2   = s_wsp1 + nmode1 * nq2;
            s_basis0 = s_wsp2 + nq2 * nq1 * nm0;
            s_basis1 = s_basis0 + nmode0 * nq0;
            s_basis2 = s_basis1 + nmode1 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            nmode0 = nm0;
            nmode1 = nm1;
            nmode2 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_wsp2   = s_wsp1 + nm0 * nm1 * nq2;
            s_basis0 = s_wsp2 + nq2 * nq1 * nm0;
            s_basis1 = s_basis0 + nmode0 * nq0;
            s_basis2 = s_basis1 + nmode1 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            nmode0 = nm0;
            nmode1 = nm1;
            nmode2 = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

            s_wsp0   = (TData *)shmemptr;
            s_wsp1   = s_wsp0 + nmTot;
            s_wsp2   = s_wsp1 + nm0 * nm1 * nq2;
            s_basis0 = s_wsp2 + nq2 * nq1 * nm0;
            s_basis1 = s_basis0 + nmode0 * nq0;
            s_basis2 = s_basis1 + nmode1 * nq1;
        }

        // Copy to shared memory.
        for (unsigned int idx = threadIdx.x; idx < nmode0 * nq0;
             idx += blockDim.x)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nmode1 * nq1;
             idx += blockDim.x)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nmode2 * nq2;
             idx += blockDim.x)
        {
            s_basis2[idx] = basis2[idx];
        }

        unsigned int e = blockIdx.x;
        while (e < nelmt)
        {
            const TData *inptr = in + nmTot * e;
            TData *outptr      = out + nqTot * e;

            // Copy to shared memory.
            for (unsigned int idx = threadIdx.x; idx < nmTot; idx += blockDim.x)
            {
                s_wsp0[idx] = inptr[idx];
            }

            __syncthreads();

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                BwdTransHexSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          s_basis0, s_basis1, s_basis2, s_wsp0,
                                          outptr, s_wsp1, s_wsp2);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                BwdTransTetSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, index0, index1, s_basis0,
                                          s_basis1, s_basis2, s_wsp0, outptr,
                                          s_wsp1, s_wsp2);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                BwdTransPrismSumFacQPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                BwdTransPyrSumFacQPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2);
            }

            e += gridDim.x;
        }
    }
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData>
__global__ void BwdTrans3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const unsigned int *index1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp)
{
    BwdTrans3DKernel<SHAPE_TYPE, Implementation>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        basis0, basis1, basis2, in, out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          unsigned int nm0, unsigned int nm1, unsigned int nm2,
          const unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TData>
__global__ void BwdTrans3DKernelLauncher(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    BwdTrans3DKernel<SHAPE_TYPE, Implementation>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        basis0, basis1, basis2, in, out, wsp);
}

// Kernel launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, typename TData>
NEK_FORCE_INLINE static void BwdTrans1DKernel(const unsigned int nm0,
                                              const unsigned int nq0,
                                              const unsigned int nelmt,
                                              const TData *basis0,
                                              const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans1DKernelLauncher<Implementation>
        <<<gridsize, blocksize, shmemsize>>>(nm0, nq0, nelmt, basis0, in, out);
}

// Size based template version.
template <typename ExecSpace, typename Implementation, unsigned int nm0,
          unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void BwdTrans1DKernel(const unsigned int nelmt,
                                              const TData *basis0,
                                              const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans1DKernelLauncher<Implementation, nm0, nq0>
        <<<gridsize, NektarSpaces::vector_width<TData>::value, shmemsize>>>(
            nelmt, basis0, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, typename TData>
NEK_FORCE_INLINE static void BwdTrans2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    const TData *basis0, const TData *basis1, TData *wsp, const TData *in,
    TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans2DKernelLauncher<SHAPE_TYPE, Implementation>
        <<<gridsize, blocksize, shmemsize>>>(nm0, nm1, nmTot, nq0, nq1, nelmt,
                                             isModified, basis0, basis1, in,
                                             out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, unsigned int nm0, unsigned int nm1,
          unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void BwdTrans2DKernel(const unsigned int nelmt,
                                              const bool isModified,
                                              const TData *basis0,
                                              const TData *basis1, TData *wsp,
                                              const TData *in, TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans2DKernelLauncher<SHAPE_TYPE, Implementation, nm0, nm1, nmTot, nq0,
                             nq1><<<gridsize, blocksize, shmemsize>>>(
        nelmt, isModified, basis0, basis1, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, typename TData>
NEK_FORCE_INLINE static void BwdTrans3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const TData *basis0, const TData *basis1,
    const TData *basis2, TData *wsp, const TData *in, TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans3DKernelLauncher<SHAPE_TYPE, Implementation>
        <<<gridsize, blocksize, shmemsize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
            index1, basis0, basis1, basis2, in, out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, unsigned int nm0, unsigned int nm1,
          unsigned int nm2, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void BwdTrans3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const TData *basis0, const TData *basis1,
    const TData *basis2, TData *wsp, const TData *in, TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    BwdTrans3DKernelLauncher<SHAPE_TYPE, Implementation, nm0, nm1, nm2, nmTot,
                             nq0, nq1, nq2><<<gridsize, blocksize, shmemsize>>>(
        nelmt, isModified, index0, index1, basis0, basis1, basis2, in, out,
        wsp);
}

} // namespace Nektar::Operators::detail

#endif
