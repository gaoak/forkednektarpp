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

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData, bool DEFORMED>
__global__ void PhysDeriv1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict__ df, TData *__restrict__ inout)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            unsigned int index   = offset + i;
            unsigned int dfindex = DEFORMED ? index : e;

            TData d0 = inout[index];
            for (unsigned int d = 0; d < ncoord; d++)
            {
                inout[index + d * nSize] = d0 * df[dfindex + d * dfSize];
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDeriv1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict__ df, TData *__restrict__ inout)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            unsigned int index   = offset + i;
            unsigned int dfindex = DEFORMED ? index : e;

            TData d0 = inout[index];
            for (unsigned int d = 0; d < ncoord; d++)
            {
                inout[index + d * nSize] = d0 * df[dfindex + d * dfSize];
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    TData *__restrict__ inout)
{
    extern __shared__ TData shared[];
    TData *s_xfrm0, *s_xfrm1;

    // Copy to shared memory.
    if constexpr (SHAPETYPE == LibUtilities::Tri)
    {
        s_xfrm0 = shared;
        s_xfrm1 = s_xfrm0 + nq1;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_xfrm0[sIndex] = 2.0 / (1.0 - Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_xfrm1[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
        {
            for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
            {
                unsigned int index   = offset + cnt_ji;
                unsigned int dfindex = DEFORMED ? index : e;

                TData d0 = inout[index];
                TData d1 = inout[index + nSize];

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    d0 *= s_xfrm0[j];
                    d1 += d0 * s_xfrm1[i];
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0; d < ncoord; d++)
                {
                    inout[index + d * nSize] =
                        d0 * df[dfindex + (2 * d) * dfSize] +
                        d1 * df[dfindex + (2 * d + 1) * dfSize];
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void PhysDeriv2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    TData *__restrict__ inout)
{
    TData xfrm0, xfrm1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                xfrm0 = 2.0 / (1.0 - Z1[j]);
            }

            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji  = nq0 * j + i;
                unsigned int index   = offset + cnt_ji;
                unsigned int dfindex = DEFORMED ? index : e;

                TData d0 = inout[index];
                TData d1 = inout[index + nSize];

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    xfrm1 = 0.5 * (1.0 + Z0[i]);
                    d0 *= xfrm0;
                    d1 += d0 * xfrm1;
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0; d < ncoord; d++)
                {
                    inout[index + d * nSize] =
                        d0 * df[dfindex + (2 * d) * dfSize] +
                        d1 * df[dfindex + (2 * d + 1) * dfSize];
                }
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, TData *__restrict__ inout)
{
    extern __shared__ TData shared[];
    TData *s_xfrm_eta0, *s_xfrm_eta1, *s_xfrm_eta1m, *s_xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHAPETYPE == LibUtilities::Tet)
    {
        s_xfrm_eta0  = shared;
        s_xfrm_eta1  = s_xfrm_eta0 + nq0;
        s_xfrm_eta1m = s_xfrm_eta1 + nq1;
        s_xfrm_eta2  = s_xfrm_eta1m + nq1;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_xfrm_eta0[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_xfrm_eta1[sIndex] = 0.5 * (1.0 + Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_xfrm_eta1m[sIndex] = 2.0 / (1.0 - Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_xfrm_eta2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Prism)
    {
        s_xfrm_eta0 = shared;
        s_xfrm_eta2 = s_xfrm_eta0 + nq0;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_xfrm_eta0[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_xfrm_eta2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
    {
        s_xfrm_eta0 = shared;
        s_xfrm_eta1 = s_xfrm_eta0 + nq0;
        s_xfrm_eta2 = s_xfrm_eta1 + nq1;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_xfrm_eta0[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_xfrm_eta1[sIndex] = 0.5 * (1.0 + Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_xfrm_eta2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }

    constexpr unsigned int ncoord = 3;

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPETYPE == LibUtilities::Tet)
        {
            for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        unsigned int index = offset + cnt_kji;

                        TData tmp0   = xfrm * inout[index];
                        TData tmp1   = s_xfrm_eta0[i] * tmp0;
                        TData tmp2   = s_xfrm_eta2[k] * inout[index + nSize];
                        inout[index] = tmp0;
                        inout[index + nSize] = tmp1 + tmp2;
                        inout[index + 2 * nSize] +=
                            tmp1 + s_xfrm_eta1[j] * tmp2;
                    }
                }
            }
        }

        for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
        {
            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    unsigned int index   = offset + cnt_kji;
                    unsigned int dfindex = DEFORMED ? index : e;

                    TData d0 = inout[index];
                    TData d1 = inout[index + nSize];
                    TData d2 = inout[index + 2 * nSize];

                    // Chain-rule for eta_0 and eta_2.
                    if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        d0 *= s_xfrm_eta2[k];
                        d2 += s_xfrm_eta0[i] * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        d0 *= s_xfrm_eta2[k];
                        d1 *= s_xfrm_eta2[k];
                        d2 += s_xfrm_eta0[i] * d0 + s_xfrm_eta1[j] * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0; d < ncoord; d++)
                    {
                        inout[index + d * nSize] =
                            d0 * df[dfindex + (3 * d) * dfSize] +
                            d1 * df[dfindex + (3 * d + 1) * dfSize] +
                            d2 * df[dfindex + (3 * d + 2) * dfSize];
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void PhysDeriv3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, TData *__restrict__ inout)
{
    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

    constexpr unsigned int ncoord = 3;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPETYPE == LibUtilities::Tet)
        {
            for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
            {
                xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
                {
                    xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                    xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                    TData xfrm = xfrm_eta1m * xfrm_eta2;
                    for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                    {
                        unsigned int index =
                            offset + nq0 * nq1 * k + nq0 * j + i;

                        xfrm_eta0            = 0.5 * (1.0 + Z0[i]);
                        TData tmp0           = xfrm * inout[index];
                        TData tmp1           = xfrm_eta0 * tmp0;
                        TData tmp2           = xfrm_eta2 * inout[index + nSize];
                        inout[index]         = tmp0;
                        inout[index + nSize] = tmp1 + tmp2;
                        inout[index + 2 * nSize] += tmp1 + xfrm_eta1 * tmp2;
                    }
                }
            }

            __syncthreads();
        }

        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            if constexpr (SHAPETYPE == LibUtilities::Prism ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
            }

            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                if constexpr (SHAPETYPE == LibUtilities::Pyr)
                {
                    xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                }

                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int index   = offset + cnt_kji;
                    unsigned int dfindex = DEFORMED ? index : e;

                    TData d0 = inout[index];
                    TData d1 = inout[index + nSize];
                    TData d2 = inout[index + 2 * nSize];

                    if constexpr (SHAPETYPE == LibUtilities::Prism ||
                                  SHAPETYPE == LibUtilities::Pyr)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                    }

                    // Chain-rule for eta_0 and eta_2.
                    if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        d0 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        d0 *= xfrm_eta2;
                        d1 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0; d < ncoord; d++)
                    {
                        inout[index + d * nSize] =
                            d0 * df[dfindex + (3 * d) * dfSize] +
                            d1 * df[dfindex + (3 * d + 1) * dfSize] +
                            d2 * df[dfindex + (3 * d + 2) * dfSize];
                    }
                }
            }
        }

        e += gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor1DKernel(const unsigned int nq0,
                                        const unsigned int nelmt,
                                        const TData *__restrict__ D0,
                                        const TData *__restrict__ in,
                                        TData *__restrict__ out)
{
    extern __shared__ TData shared[];
    TData *s_D0 = shared;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0 * nq0)
    {
        s_D0[sIndex] = D0[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            TData sum = 0.0;
            for (unsigned int q = 0; q < nq0; ++q)
            {
                sum += s_D0[q * nq0 + i] * in[offset + q];
            }
            out[offset + i] = sum;
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor1DKernel_QP(const unsigned int nq0,
                                           const unsigned int nelmt,
                                           const TData *__restrict__ D0,
                                           const TData *__restrict__ in,
                                           TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            TData sum = 0.0;
            for (unsigned int q = 0; q < nq0; ++q)
            {
                sum += D0[q * nq0 + i] * in[offset + q];
            }
            out[offset + i] = sum;
        }

        e += gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const unsigned int nSize, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];
    TData *s_D0 = shared;
    TData *s_D1 = s_D0 + nq0 * nq0;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0 * nq0)
    {
        s_D0[sIndex] = D0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1 * nq1)
    {
        s_D1[sIndex] = D1[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = 0; j < nq1; ++j)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                unsigned int cnt_ji = nq0 * j + i;

                // Direction 1
                TData sum = 0.0;
                for (unsigned int q = 0; q < nq0; ++q)
                {
                    sum += s_D0[q * nq0 + i] * in[offset + nq0 * j + q];
                }
                out[offset + cnt_ji] = sum;

                // Direction 2
                sum = 0.0;
                for (unsigned int q = 0; q < nq1; ++q)
                {
                    sum += s_D1[q * nq1 + j] * in[offset + nq0 * q + i];
                }
                out[offset + cnt_ji + nSize] = sum;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const unsigned int nSize, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji = nq0 * j + i;

                // Direction 1
                TData sum = 0.0;
                for (unsigned int q = 0; q < nq0; ++q)
                {
                    sum += D0[q * nq0 + i] * in[offset + nq0 * j + q];
                }
                out[offset + cnt_ji] = sum;

                // Direction 2
                sum = 0.0;
                for (unsigned int q = 0; q < nq1; ++q)
                {
                    sum += D1[q * nq1 + j] * in[offset + nq0 * q + i];
                }
                out[offset + cnt_ji + nSize] = sum;
            }
        }

        e += gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nSize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ TData shared[];
    TData *s_D0 = shared;
    TData *s_D1 = s_D0 + nq0 * nq0;
    TData *s_D2 = s_D1 + nq1 * nq1;

    // Copy to shared memory.
    unsigned int sIndex = threadIdx.x;
    while (sIndex < nq0 * nq0)
    {
        s_D0[sIndex] = D0[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq1 * nq1)
    {
        s_D1[sIndex] = D1[sIndex];
        sIndex += blockDim.x;
    }

    sIndex = threadIdx.x;
    while (sIndex < nq2 * nq2)
    {
        s_D2[sIndex] = D2[sIndex];
        sIndex += blockDim.x;
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = 0; k < nq2; k++)
        {
            unsigned int cnt_k = nq0 * nq1 * k;
            for (unsigned int j = 0; j < nq1; j++)
            {
                unsigned int cnt_kj = cnt_k + nq0 * j;
                for (unsigned int i = 0; i < nq0; i++)
                {
                    unsigned int cnt_ji  = nq0 * j + i;
                    unsigned int cnt_kji = cnt_k + cnt_ji;

                    // Direction 1
                    TData sum = 0.0;
                    for (unsigned int q = 0; q < nq0; ++q)
                    {
                        sum += s_D0[q * nq0 + i] * in[offset + cnt_kj + q];
                    }
                    out[offset + cnt_kji] = sum;

                    // Direction 2
                    sum = 0.0;
                    for (unsigned int q = 0; q < nq1; ++q)
                    {
                        sum += s_D1[q * nq1 + j] *
                               in[offset + cnt_k + nq0 * q + i];
                    }
                    out[offset + cnt_kji + nSize] = sum;

                    // Direction 3
                    sum = 0.0;
                    for (unsigned int q = 0; q < nq2; ++q)
                    {
                        sum += s_D2[q * nq2 + k] *
                               in[offset + nq0 * nq1 * q + cnt_ji];
                    }
                    out[offset + cnt_kji + 2 * nSize] = sum;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void PhysDerivTensor3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nSize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;

                    // Direction 1
                    TData sum = 0.0;
                    for (unsigned int q = 0; q < nq0; ++q)
                    {
                        sum += D0[q * nq0 + i] *
                               in[offset + nq0 * nq1 * k + nq0 * j + q];
                    }
                    out[offset + cnt_kji] = sum;

                    // Direction 2
                    sum = 0.0;
                    for (unsigned int q = 0; q < nq1; ++q)
                    {
                        sum += D1[q * nq1 + j] *
                               in[offset + nq0 * nq1 * k + nq0 * q + i];
                    }
                    out[offset + cnt_kji + nSize] = sum;

                    // Direction 3
                    sum = 0.0;
                    for (unsigned int q = 0; q < nq2; ++q)
                    {
                        sum += D2[q * nq2 + k] *
                               in[offset + nq0 * nq1 * q + nq0 * j + i];
                    }
                    out[offset + cnt_kji + 2 * nSize] = sum;
                }
            }
        }

        e += gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData, bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv1DKernel(const size_t gridSize, const size_t blockSize,
                      const unsigned int nq0, const unsigned int nCoord,
                      const unsigned int nElmts, const unsigned int nSize,
                      const unsigned int dfSize, const TData *D0,
                      const TData *df, const TData *in, TData *out)
{
    // Compute tensorial derivative.
    if constexpr (FLAG_QP)
    {
        PhysDerivTensor1DKernel_QP<TData>
            <<<gridSize, dim3(32)>>>(nq0, nElmts, D0, in, out);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nq0 * nq0);
        PhysDerivTensor1DKernel<TData>
            <<<gridSize, blockSize, nshared>>>(nq0, nElmts, D0, in, out);
    }

    // Compute physical derivative.
    if constexpr (FLAG_QP)
    {
        PhysDeriv1DKernel_QP<TData, DEFORMED><<<gridSize, dim3(32)>>>(
            nq0, nCoord, nElmts, nSize, dfSize, df, out);
    }
    else
    {
        PhysDeriv1DKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nCoord, nElmts, nSize, dfSize, df, out);
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv2DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nCoord,
                      const unsigned int nElmts, const unsigned int nSize,
                      const unsigned int dfSize, const TData *D0,
                      const TData *D1, const TData *Z0, const TData *Z1,
                      const TData *df, const TData *in, TData *out)
{
    // Compute tensorial derivative.
    if constexpr (FLAG_QP)
    {
        PhysDerivTensor2DKernel_QP<TData><<<gridSize, dim3(8, 8)>>>(
            nq0, nq1, nElmts, nSize, D0, D1, in, out);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nq0 * nq0 + nq1 * nq1);
        PhysDerivTensor2DKernel<TData><<<gridSize, blockSize, nshared>>>(
            nq0, nq1, nElmts, nSize, D0, D1, in, out);
    }

    // Compute physical derivative.
    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv2DKernel_QP<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, nullptr, nullptr, df, out);
        }
        else
        {
            PhysDeriv2DKernel<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nCoord, nElmts, nSize,
                                          dfSize, nullptr, nullptr, df, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, Z0, Z1, df, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            PhysDeriv2DKernel<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nq0, nq1, nCoord, nElmts, nSize, dfSize, Z0, Z1, df, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    PhysDeriv3DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nq2,
                      const unsigned int nElmts, const unsigned int nSize,
                      const unsigned int dfSize, const TData *D0,
                      const TData *D1, const TData *D2, const TData *Z0,
                      const TData *Z1, const TData *Z2, const TData *df,
                      const TData *in, TData *out)
{
    // Compute tensorial derivative.
    if constexpr (FLAG_QP)
    {
        PhysDerivTensor3DKernel_QP<TData><<<gridSize, dim3(4, 4, 4)>>>(
            nq0, nq1, nq2, nElmts, nSize, D0, D1, D2, in, out);
    }
    else
    {
        unsigned int nshared =
            sizeof(TData) * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2);
        PhysDerivTensor3DKernel<TData><<<gridSize, blockSize, nshared>>>(
            nq0, nq1, nq2, nElmts, nSize, D0, D1, D2, in, out);
    }

    // Compute physical derivative.
    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, nullptr, nullptr, nullptr,
                                              df, out);
        }
        else
        {
            PhysDeriv3DKernel<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nq2, nElmts, nSize, dfSize,
                                          nullptr, nullptr, nullptr, df, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, Z1, Z2, df, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + 2 * nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, Z1, Z2, df, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Prism, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, nullptr, Z2, df, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Prism, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, nullptr, Z2, df,
                                                   out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (FLAG_QP)
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, Z1, Z2, df, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, Z1, Z2, df, out);
        }
    }
}

#endif

} // namespace Nektar::Operators::detail
