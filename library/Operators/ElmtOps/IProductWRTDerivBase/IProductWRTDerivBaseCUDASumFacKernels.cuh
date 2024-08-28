///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseCUDASumFacKernels.cuh
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
__global__ void IProductWRTDerivBase1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
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
            const unsigned int dfindex = DEFORMED ? index : e;

            TData sum = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                sum += df[d * dfsize + dfindex] * in[d * nsize + index];
            }
            out[index] = sum;
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBase1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
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
            const unsigned int dfindex = DEFORMED ? index : e;

            TData sum = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                sum += df[d * dfsize + dfindex] * in[d * nsize + index];
            }
            out[index] = sum;
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = NektarSpaces::CUDA::width;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_f0, *s_f1;

    // Copy to shared memory.
    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;
    if constexpr (SHAPETYPE == LibUtilities::Tri)
    {
        s_f0 = shared;
        s_f1 = s_f0 + nq1;

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f0[idx] = 2.0 / (1.0 - Z1[idx]);
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

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
                const unsigned int dfindex = DEFORMED ? index : e;

                TData sum1 = 0.0, sum2 = 0.0;
                for (unsigned int d = 0; d < ncoord; ++d)
                {
                    TData tmp = in[d * nsize + index];
                    sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                    sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
                }

                if constexpr (SHAPETYPE == LibUtilities::Quad)
                {
                    out[index]         = sum1;
                    out[nsize + index] = sum2;
                }
                else if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    out[index]         = (sum1 + sum2 * s_f1[i]) * s_f0[j];
                    out[nsize + index] = sum2;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int nqTot = nq0 * nq1;
    TData f0, f1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
            }

            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                const unsigned int cnt_ji  = nq0 * j + i;
                const unsigned int index   = offset + cnt_ji;
                const unsigned int dfindex = DEFORMED ? index : e;

                TData sum1 = 0.0, sum2 = 0.0;
                for (unsigned int d = 0u; d < ncoord; ++d)
                {
                    TData tmp = in[d * nsize + index];
                    sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                    sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    f1 = 0.5 * (1.0 + Z0[i]);
                }

                if constexpr (SHAPETYPE == LibUtilities::Quad)
                {
                    out[index]         = sum1;
                    out[nsize + index] = sum2;
                }
                else if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    out[index]         = (sum1 + sum2 * f1) * f0;
                    out[nsize + index] = sum2;
                }
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase2DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int nqTot = nq0 * nq1;
    TData f0, f1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nq0 * nq1; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? index : e;

            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
            }

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                TData tmp = in[d * nsize + index];
                sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                f1 = 0.5 * (1.0 + Z0[i]);
            }

            if constexpr (SHAPETYPE == LibUtilities::Quad)
            {
                out[index]         = sum1;
                out[nsize + index] = sum2;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                out[index]         = (sum1 + sum2 * f1) * f0;
                out[nsize + index] = sum2;
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ Z2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ TData shared[];

    constexpr unsigned int warpsize = NektarSpaces::CUDA::width;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_f0, *s_f1, *s_f2, *s_f3;

    // Copy to shared memory.
    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;
    if constexpr (SHAPETYPE == LibUtilities::Tet)
    {
        s_f0 = shared;
        s_f1 = s_f0 + nq1;
        s_f2 = s_f1 + nq0;
        s_f3 = s_f2 + nq2;

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f0[idx] = 2.0 / (1.0 - Z1[idx]);
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f3[idx] = 0.5 * (1.0 + Z1[idx]);
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Prism)
    {
        s_f1 = shared;
        s_f2 = s_f1 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
    {
        s_f1 = shared;
        s_f2 = s_f1 + nq0;
        s_f3 = s_f2 + nq2;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f3[idx] = 0.5 * (1.0 + Z1[idx]);
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * s_f1[i]) * s_f0[j];
                        out[nsize + index] = (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]         = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[nsize + index] = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]         = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[nsize + index] = (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[2u * nsize + index] = sum3;
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ Z2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData f0, f1, f2, f3;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            if constexpr (SHAPETYPE == LibUtilities::Tet ||
                          SHAPETYPE == LibUtilities::Prism ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                f2 = 2.0 / (1.0 - Z2[k]);
            }

            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                if constexpr (SHAPETYPE == LibUtilities::Tet ||
                              SHAPETYPE == LibUtilities::Pyr)
                {
                    f3 = 0.5 * (1.0 + Z1[j]);
                }

                if constexpr (SHAPETYPE == LibUtilities::Tet)
                {
                    f0 = 2.0 * f2 / (1.0 - Z1[j]);
                }

                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    const unsigned int index =
                        offset + nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Tet ||
                                  SHAPETYPE == LibUtilities::Prism ||
                                  SHAPETYPE == LibUtilities::Pyr)
                    {
                        f1 = 0.5 * (1.0 + Z0[i]);
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index]         = (sum1 + (sum2 + sum3) * f1) * f0;
                        out[nsize + index] = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                }
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase3DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ Z2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData f0, f1, f2, f3;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nq0 * nq1 * nq2;
             idx += blockDim.x)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            if constexpr (SHAPETYPE == LibUtilities::Tet ||
                          SHAPETYPE == LibUtilities::Prism ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                f2 = 2.0 / (1.0 - Z2[k]);
            }

            if constexpr (SHAPETYPE == LibUtilities::Tet ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                f3 = 0.5 * (1.0 + Z1[j]);
            }

            if constexpr (SHAPETYPE == LibUtilities::Tet)
            {
                f0 = 2.0 * f2 / (1.0 - Z1[j]);
            }

            if constexpr (SHAPETYPE == LibUtilities::Tet ||
                          SHAPETYPE == LibUtilities::Prism ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                f1 = 0.5 * (1.0 + Z0[i]);
            }

            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? index : e;

            TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                TData tmp = in[d * nsize + index];
                sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
            }

            if constexpr (SHAPETYPE == LibUtilities::Hex)
            {
                out[index]              = sum1;
                out[nsize + index]      = sum2;
                out[2u * nsize + index] = sum3;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Tet)
            {
                out[index]              = (sum1 + (sum2 + sum3) * f1) * f0;
                out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                out[2u * nsize + index] = sum3;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Prism)
            {
                out[index]              = (sum1 + sum3 * f1) * f2;
                out[nsize + index]      = sum2;
                out[2u * nsize + index] = sum3;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Pyr)
            {
                out[index]              = (sum1 + sum3 * f1) * f2;
                out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                out[2u * nsize + index] = sum3;
            }
        }

        e += gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTDerivBase1DKernel(const unsigned int nq0,
                                 const unsigned int ncoord,
                                 const unsigned int nelmts,
                                 const unsigned int nsize,
                                 const unsigned int dfsize, const TData *df,
                                 const TData *in, TData *out)
{
    const unsigned int gridsize  = std::min(nq0, 256u);
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmts : (nelmts + gridsize - 1u) / gridsize, 2147483647u);

    if constexpr (MULTILEVEL)
    {
        IProductWRTDerivBase1DKernel_QP<TData, DEFORMED>
            <<<gridsize, blocksize>>>(nq0, ncoord, nelmts, nsize, dfsize, df,
                                      in, out);
    }
    else
    {
        IProductWRTDerivBase1DKernel<TData, DEFORMED><<<gridsize, blocksize>>>(
            nq0, ncoord, nelmts, nsize, dfsize, df, in, out);
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTDerivBase2DKernel(LibUtilities::ShapeType shapetype,
                                 const unsigned int nq0, const unsigned int nq1,
                                 const unsigned int ncoord,
                                 const unsigned int nelmts,
                                 const unsigned int nsize,
                                 const unsigned int dfsize, const TData *Z0,
                                 const TData *Z1, const TData *df,
                                 const TData *in, TData *out)
{
    const dim3 blocksize2d       = dim3(std::min(nq0, 16u), std::min(nq1, 16u));
    const unsigned int gridsize  = std::min(nq0 * nq1, 256u);
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmts : (nelmts + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Quad, DEFORMED>
                <<<gridsize, blocksize2d>>>(nq0, nq1, ncoord, nelmts, nsize,
                                            dfsize, nullptr, nullptr, df, in,
                                            out);
        }
        else
        {
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Quad, DEFORMED>
                <<<gridsize, blocksize>>>(nq0, nq1, ncoord, nelmts, nsize,
                                          dfsize, nullptr, nullptr, df, in,
                                          out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>
                <<<gridsize, blocksize2d>>>(nq0, nq1, ncoord, nelmts, nsize,
                                            dfsize, Z0, Z1, df, in, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Tri, DEFORMED>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, ncoord, nelmts,
                                                   nsize, dfsize, Z0, Z1, df,
                                                   in, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    IProductWRTDerivBase3DKernel(LibUtilities::ShapeType shapetype,
                                 const unsigned int nq0, const unsigned int nq1,
                                 const unsigned int nq2,
                                 const unsigned int ncoord,
                                 const unsigned int nelmts,
                                 const unsigned int nsize,
                                 const unsigned int dfsize, const TData *Z0,
                                 const TData *Z1, const TData *Z2,
                                 const TData *df, const TData *in, TData *out)
{
    const dim3 blocksize3d =
        dim3(std::min(nq0, 8u), std::min(nq1, 8u), std::min(nq2, 8u));
    const unsigned int gridsize  = std::min(nq0 * nq1 * nq2, 256u);
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmts : (nelmts + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>
                <<<gridsize, blocksize3d>>>(nq0, nq1, nq2, ncoord, nelmts,
                                            nsize, dfsize, nullptr, nullptr,
                                            nullptr, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Hex, DEFORMED>
                <<<gridsize, blocksize>>>(nq0, nq1, nq2, ncoord, nelmts, nsize,
                                          dfsize, nullptr, nullptr, nullptr, df,
                                          in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>
                <<<gridsize, blocksize3d>>>(nq0, nq1, nq2, ncoord, nelmts,
                                            nsize, dfsize, Z0, Z1, Z2, df, in,
                                            out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + 2 * nq1 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Tet, DEFORMED>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, ncoord,
                                                   nelmts, nsize, dfsize, Z0,
                                                   Z1, Z2, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Prism,
                                            DEFORMED>
                <<<gridsize, blocksize3d>>>(nq0, nq1, nq2, ncoord, nelmts,
                                            nsize, dfsize, Z0, nullptr, Z2, df,
                                            in, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Prism, DEFORMED>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, ncoord,
                                                   nelmts, nsize, dfsize, Z0,
                                                   nullptr, Z2, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridsize, blocksize3d>>>(nq0, nq1, nq2, ncoord, nelmts,
                                            nsize, dfsize, Z0, Z1, Z2, df, in,
                                            out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridsize, blocksize, nshared>>>(nq0, nq1, nq2, ncoord,
                                                   nelmts, nsize, dfsize, Z0,
                                                   Z1, Z2, df, in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
