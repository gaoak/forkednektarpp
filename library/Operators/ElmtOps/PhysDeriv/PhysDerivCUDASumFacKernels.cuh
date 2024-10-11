///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivCUDASumFacKernels.cuh
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

template <typename TData, bool DEFORMED>
__global__ void PhysDeriv1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *__restrict__ D0,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::CUDA::width;

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            const unsigned int index =
                nq0 * warpsize * iwarp + warpsize * i + ilane;
            const unsigned int dfindex =
                DEFORMED ? nq0 * ncoord * warpsize * iwarp +
                               warpsize * i * ncoord + ilane
                         : ncoord * warpsize * iwarp + ilane;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] *
                      in[nq0 * warpsize * iwarp + warpsize * q + ilane];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nsize + index] = d0 * df[d * warpsize + dfindex];
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDeriv1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *__restrict__ D0,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            const unsigned int index   = offset + i;
            const unsigned int dfindex = DEFORMED ? ncoord * index : ncoord * e;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[offset + q];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nsize + index] = d0 * df[d + dfindex];
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int warpsize = NektarSpaces::CUDA::width;

    const unsigned int nqTot = nq0 * nq1;
    const unsigned int ndf   = 2 * ncoord;

    TData *s_D0 = SHMEM ? (TData *)shared : (TData *)D0;
    TData *s_D1 = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_xfrm0, *s_xfrm1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = threadIdx.x; idx < nq0 * nq0; idx += blockDim.x)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq1 * nq1; idx += blockDim.x)
        {
            s_D1[idx] = D1[idx];
        }

        // Precompute geometric factors.
        if constexpr (SHAPETYPE == LibUtilities::Tri)
        {
            s_xfrm0 = s_D1 + nq1 * nq1;
            s_xfrm1 = s_xfrm0 + nq1;

            for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
            {
                s_xfrm0[idx] = 2.0 / (1.0 - Z1[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
            {
                s_xfrm1[idx] = 0.5 * (1.0 + Z0[idx]);
            }
        }
    }

    if constexpr (SHAPETYPE == LibUtilities::Tri || SHMEM)
    {
        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index =
                    nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                   warpsize * cnt_ji * ndf + ilane
                             : ndf * warpsize * iwarp + ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += s_D0[q * nq0 + i] *
                          in[nqTot * warpsize * iwarp +
                             warpsize * (nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += s_D1[q * nq1 + j] *
                          in[nqTot * warpsize * iwarp +
                             warpsize * (nq0 * q + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    if constexpr (SHMEM)
                    {
                        d0 *= s_xfrm0[j];
                        d1 += d0 * s_xfrm1[i];
                    }
                    else
                    {
                        d0 *= 2.0 / (1.0 - Z1[j]);
                        d1 += d0 * 0.5 * (1.0 + Z0[i]);
                    }
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * nsize + index] =
                        d0 * df[(2u * d) * warpsize + dfindex] +
                        d1 * df[(2u * d + 1u) * warpsize + dfindex];
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    const unsigned int nqTot = nq0 * nq1;
    const unsigned int ndf   = 2 * ncoord;

    TData *s_wsp = (TData *)shared;
    TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData xfrm0, xfrm1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
        {
            s_D1[idx] = D1[idx];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0   = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y;
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp[idx] = in[offset + idx];
        }

        __syncthreads();

        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                const unsigned int cnt_ji  = nq0 * j + i;
                const unsigned int index   = offset + cnt_ji;
                const unsigned int dfindex = DEFORMED ? ndf * index : ndf * e;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * j + q];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * q + i];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    xfrm0 = 2.0 / (1.0 - Z1[j]);
                    xfrm1 = 0.5 * (1.0 + Z0[i]);
                    d0 *= xfrm0;
                    d1 += d0 * xfrm1;
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * nsize + index] = d0 * df[(2u * d) + dfindex] +
                                             d1 * df[(2u * d + 1u) + dfindex];
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv2DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    const unsigned int nqTot = nq0 * nq1;
    const unsigned int ndf   = 2 * ncoord;

    TData *s_wsp = (TData *)shared;
    TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData xfrm0, xfrm1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = threadIdx.x; idx < nq0 * nq0; idx += blockDim.x)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq1 * nq1; idx += blockDim.x)
        {
            s_D1[idx] = D1[idx];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            s_wsp[idx] = in[offset + idx];
        }

        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? ndf * index : ndf * e;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * q + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                xfrm0 = 2.0 / (1.0 - Z1[j]);
                xfrm1 = 0.5 * (1.0 + Z0[i]);
                d0 *= xfrm0;
                d1 += d0 * xfrm1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nsize + index] = d0 * df[(2u * d) + dfindex] +
                                         d1 * df[(2u * d + 1u) + dfindex];
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int warpsize = NektarSpaces::CUDA::width;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData *s_D0 = SHMEM ? (TData *)shared : (TData *)D0;
    TData *s_D1 = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2 = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData *s_xfrm_eta0, *s_xfrm_eta1, *s_xfrm_eta1m, *s_xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = threadIdx.x; idx < nq0 * nq0; idx += blockDim.x)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq1 * nq1; idx += blockDim.x)
        {
            s_D1[idx] = D1[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq2 * nq2; idx += blockDim.x)
        {
            s_D2[idx] = D2[idx];
        }

        // Precompute geometric factors.
        if constexpr (SHAPETYPE == LibUtilities::Tet)
        {
            s_xfrm_eta0  = s_D2 + nq2 * nq2;
            s_xfrm_eta1  = s_xfrm_eta0 + nq0;
            s_xfrm_eta1m = s_xfrm_eta1 + nq1;
            s_xfrm_eta2  = s_xfrm_eta1m + nq1;

            for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
            {
                s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
            {
                s_xfrm_eta1m[idx] = 2.0 / (1.0 - Z1[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
        else if constexpr (SHAPETYPE == LibUtilities::Prism)
        {
            s_xfrm_eta0 = s_D2 + nq2 * nq2;
            s_xfrm_eta2 = s_xfrm_eta0 + nq0;

            for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
        else if constexpr (SHAPETYPE == LibUtilities::Pyr)
        {
            s_xfrm_eta0 = s_D2 + nq2 * nq2;
            s_xfrm_eta1 = s_xfrm_eta0 + nq0;
            s_xfrm_eta2 = s_xfrm_eta1 + nq1;

            for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
            {
                s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
            }

            for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
    }

    if constexpr (SHAPETYPE != LibUtilities::Hex || SHMEM)
    {
        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
        {
            for (unsigned int j = 0u; j < nq1; j++)
            {
                for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane;
                    const unsigned int dfindex =
                        DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                       warpsize * cnt_kji * ndf + ilane
                                 : ndf * warpsize * iwarp + ilane;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * j + q) +
                                 ilane];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * q + i) +
                                 ilane];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += s_D2[q * nq2 + k] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * q + nq0 * j + i) +
                                 ilane];
                    }

                    // Moving from standard to collapsed coordinates.
                    if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        if constexpr (SHMEM)
                        {
                            TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                            TData tmp0 = xfrm * d0;
                            TData tmp1 = s_xfrm_eta0[i] * tmp0;
                            TData tmp2 = s_xfrm_eta2[k] * d1;
                            d0         = tmp0;
                            d1         = tmp1 + tmp2;
                            d2 += tmp1 + s_xfrm_eta1[j] * tmp2;
                        }
                        else
                        {
                            TData xfrm =
                                2.0 / (1.0 - Z1[j]) * 2.0 / (1.0 - Z2[k]);
                            TData tmp0 = xfrm * d0;
                            TData tmp1 = 0.5 * (1.0 + Z0[i]) * tmp0;
                            TData tmp2 = 2.0 / (1.0 - Z2[k]) * d1;
                            d0         = tmp0;
                            d1         = tmp1 + tmp2;
                            d2 += tmp1 + 0.5 * (1.0 + Z1[j]) * tmp2;
                        }
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        if constexpr (SHMEM)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0;
                        }
                        else
                        {
                            d0 *= 2.0 / (1.0 - Z2[k]);
                            d2 += 0.5 * (1.0 + Z0[i]) * d0;
                        }
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        if constexpr (SHMEM)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d1 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0 + s_xfrm_eta1[j] * d1;
                        }
                        else
                        {
                            d0 *= 2.0 / (1.0 - Z2[k]);
                            d1 *= 2.0 / (1.0 - Z2[k]);
                            d2 += 0.5 * (1.0 + Z0[i]) * d0 +
                                  0.5 * (1.0 + Z1[j]) * d1;
                        }
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(3u * d) * warpsize + dfindex] +
                            d1 * df[(3u * d + 1u) * warpsize + dfindex] +
                            d2 * df[(3u * d + 2u) * warpsize + dfindex];
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp             = (TData *)shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2              = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
        {
            s_D1[idx] = D1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2 * nq2; idx += stride)
        {
            s_D2[idx] = D2[idx];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 = blockDim.x * blockDim.y * threadIdx.z +
                                  blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp[idx] = in[offset + idx];
        }

        __syncthreads();

        // Compute tensorial derivative.
        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int index   = offset + cnt_kji;
                    const unsigned int dfindex =
                        DEFORMED ? ndf * index : ndf * e;

                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] *
                              s_wsp[nq0 * nq1 * k + nq0 * j + q];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] *
                              s_wsp[nq0 * nq1 * k + nq0 * q + i];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += s_D2[q * nq2 + k] *
                              s_wsp[nq0 * nq1 * q + nq0 * j + i];
                    }

                    // Moving from standard to collapsed coordinates.
                    if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                        xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                        TData xfrm = xfrm_eta1m * xfrm_eta2;
                        TData tmp0 = xfrm * d0;
                        TData tmp1 = xfrm_eta0 * tmp0;
                        TData tmp2 = xfrm_eta2 * d1;
                        d0         = tmp0;
                        d1         = tmp1 + tmp2;
                        d2 += tmp1 + xfrm_eta1 * tmp2;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d1 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(3u * d) + dfindex] +
                            d1 * df[(3u * d + 1u) + dfindex] +
                            d2 * df[(3u * d + 2u) + dfindex];
                    }
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
__global__ void PhysDeriv3DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp             = (TData *)shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2              = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = threadIdx.x; idx < nq0 * nq0; idx += blockDim.x)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq1 * nq1; idx += blockDim.x)
        {
            s_D1[idx] = D1[idx];
        }

        for (unsigned int idx = threadIdx.x; idx < nq2 * nq2; idx += blockDim.x)
        {
            s_D2[idx] = D2[idx];
        }
    }

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            s_wsp[idx] = in[offset + idx];
        }

        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            unsigned int index         = offset + idx;
            const unsigned int dfindex = DEFORMED ? index : e;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 += s_D2[q * nq2 + k] * s_wsp[nq0 * nq1 * q + nq0 * j + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tet)
            {
                xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                TData xfrm = xfrm_eta1m * xfrm_eta2;
                TData tmp0 = xfrm * d0;
                TData tmp1 = xfrm_eta0 * tmp0;
                TData tmp2 = xfrm_eta2 * d1;
                d0         = tmp0;
                d1         = tmp1 + tmp2;
                d2 += tmp1 + xfrm_eta1 * tmp2;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Prism)
            {
                xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Pyr)
            {
                xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d1 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nsize + index] = d0 * df[(3u * d) + dfindex] +
                                         d1 * df[(3u * d + 1u) + dfindex] +
                                         d2 * df[(3u * d + 2u) + dfindex];
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv1DKernel(const unsigned int nq0, const unsigned int ncoord,
                      const unsigned int nelmts, const unsigned int nsize,
                      const TData *D0, const TData *df, const TData *in,
                      TData *out)
{
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0, NektarSpaces::CUDA::defaultBlockSize)
                   : NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    if constexpr (MULTILEVEL)
    {
        PhysDeriv1DKernel_QP<TData, DEFORMED>
            <<<gridsize, dim3(NektarSpaces::CUDA::width)>>>(
                nq0, ncoord, nelmts, nsize, D0, df, in, out);
    }
    else
    {
        PhysDeriv1DKernel<TData, DEFORMED><<<gridsize, blocksize>>>(
            nq0, ncoord, nelmts, nsize, D0, df, in, out);
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv2DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int ncoord,
                      const unsigned int nelmts, const unsigned int nsize,
                      const TData *D0, const TData *D1, const TData *Z0,
                      const TData *Z1, const TData *df, const TData *in,
                      TData *out)
{
    const dim3 blocksize2d = dim3(std::min(nq0, 16u), std::min(nq1, 16u));
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0 * nq1, NektarSpaces::CUDA::defaultBlockSize)
                   : NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    unsigned int nshared = SHMEM ? sizeof(TData) * (nq0 * nq0 + nq1 * nq1) : 0u;

    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1);
            PhysDeriv2DKernel_QP<TData, LibUtilities::Quad, DEFORMED, SHMEM>
                <<<gridsize, blocksize2d, nshared>>>(nq0, nq1, ncoord, nelmts,
                                                     nsize, D0, D1, nullptr,
                                                     nullptr, df, in, out);
        }
        else
        {
            PhysDeriv2DKernel<TData, LibUtilities::Quad, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, ncoord, nelmts,
                                                   nsize, D0, D1, nullptr,
                                                   nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1);
            PhysDeriv2DKernel_QP<TData, LibUtilities::Tri, DEFORMED, SHMEM>
                <<<gridsize, blocksize2d, nshared>>>(nq0, nq1, ncoord, nelmts,
                                                     nsize, D0, D1, Z0, Z1, df,
                                                     in, out);
        }
        else
        {
            nshared += sizeof(TData) * (nq0 + nq1);
            PhysDeriv2DKernel<TData, LibUtilities::Tri, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, ncoord, nelmts,
                                                   nsize, D0, D1, Z0, Z1, df,
                                                   in, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nq2,
                      const unsigned int nelmts, const unsigned int nsize,
                      const TData *D0, const TData *D1, const TData *D2,
                      const TData *Z0, const TData *Z1, const TData *Z2,
                      const TData *df, const TData *in, TData *out)
{
    const dim3 blocksize3d =
        dim3(std::min(nq0, 8u), std::min(nq1, 8u), std::min(nq2, 8u));
    const unsigned int blocksize =
        MULTILEVEL
            ? std::min(nq0 * nq1 * nq2, NektarSpaces::CUDA::defaultBlockSize)
            : NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    unsigned int nshared =
        SHMEM ? sizeof(TData) * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) : 0u;

    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1 * nq2);
            PhysDeriv3DKernel_QP<TData, LibUtilities::Hex, DEFORMED, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(
                    nq0, nq1, nq2, nelmts, nsize, D0, D1, D2, nullptr, nullptr,
                    nullptr, df, in, out);
        }
        else
        {
            PhysDeriv3DKernel<TData, LibUtilities::Hex, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, nelmts, nsize,
                                                   D0, D1, D2, nullptr, nullptr,
                                                   nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1 * nq2);
            PhysDeriv3DKernel_QP<TData, LibUtilities::Tet, DEFORMED, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(nq0, nq1, nq2, nelmts,
                                                     nsize, D0, D1, D2, Z0, Z1,
                                                     Z2, df, in, out);
        }
        else
        {
            nshared += sizeof(TData) * (nq0 + 2u * nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Tet, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, nelmts, nsize,
                                                   D0, D1, D2, Z0, Z1, Z2, df,
                                                   in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1 * nq2);
            PhysDeriv3DKernel_QP<TData, LibUtilities::Prism, DEFORMED, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(nq0, nq1, nq2, nelmts,
                                                     nsize, D0, D1, D2, Z0,
                                                     nullptr, Z2, df, in, out);
        }
        else
        {
            nshared += sizeof(TData) * (nq0 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Prism, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, nelmts, nsize,
                                                   D0, D1, D2, Z0, nullptr, Z2,
                                                   df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += sizeof(TData) * (nq0 * nq1 * nq2);
            PhysDeriv3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED, SHMEM>
                <<<gridsize, blocksize3d, nshared>>>(nq0, nq1, nq2, nelmts,
                                                     nsize, D0, D1, D2, Z0, Z1,
                                                     Z2, df, in, out);
        }
        else
        {
            nshared += sizeof(TData) * (nq0 + nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Pyr, DEFORMED, SHMEM>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, nelmts, nsize,
                                                   D0, D1, D2, Z0, Z1, Z2, df,
                                                   in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
