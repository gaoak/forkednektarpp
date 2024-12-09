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

namespace Nektar::Operators::detail
{

template <typename Implementation, bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFac>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

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

            TData sum = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                sum += df[d * warpsize + dfindex] * in[d * nelmt * nq0 + index];
            }
            out[index] = sum;
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename Implementation, bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFacQP>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nq0 : 1;
        const unsigned int dfoffset = ncoord * dfsize * e;
        const unsigned int offset   = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            const unsigned int index   = offset + i;
            const unsigned int dfindex = DEFORMED ? dfoffset + i : dfoffset;

            TData sum = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                sum += df[d * dfsize + dfindex] * in[d * nelmt * nq0 + index];
            }
            out[index] = sum;
        }

        e += gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFac>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = ncoord * 2;
    const unsigned int nqTot = nq0 * nq1;

    TData *s_f0, *s_f1;

    // Pre-compute factor.
    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;
    if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        s_f0 = (TData *)shmemptr;
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
                const unsigned int dfindex =
                    DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                   warpsize * cnt_ji * ndf + ilane
                             : ndf * warpsize * iwarp + ilane;

                TData sum1 = 0.0, sum2 = 0.0;
                for (unsigned int d = 0; d < ncoord; ++d)
                {
                    TData tmp = in[d * nelmt * nqTot + index];
                    sum1 += df[(2u * d) * warpsize + dfindex] * tmp;
                    sum2 += df[(2u * d + 1u) * warpsize + dfindex] * tmp;
                }

                if constexpr (SHAPE_TYPE == LibUtilities::Quad)
                {
                    out[index]                 = sum1;
                    out[nelmt * nqTot + index] = sum2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
                {
                    out[index] = (sum1 + sum2 * s_f1[i]) * s_f0[j];
                    out[nelmt * nqTot + index] = sum2;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFacQP>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    TData f0, f1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nq0 * nq1; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
            }

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                TData tmp = in[d * nelmt * nqTot + index];
                sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                f1 = 0.5 * (1.0 + Z0[i]);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                out[index]                 = sum1;
                out[nelmt * nqTot + index] = sum2;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                out[index]                 = (sum1 + sum2 * f1) * f0;
                out[nelmt * nqTot + index] = sum2;
            }
        }

        e += gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFac>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData *s_f0, *s_f1, *s_f2, *s_f3;

    // Pre-compute factor.
    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;
    if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        s_f0 = (TData *)shmemptr;
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
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        s_f1 = (TData *)shmemptr;
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
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f1 = (TData *)shmemptr;
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
                    const unsigned int dfindex =
                        DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                       warpsize * cnt_kji * ndf + ilane
                                 : ndf * warpsize * iwarp + ilane;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nelmt * nqTot + index];
                        sum1 += df[(3u * d) * warpsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * warpsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * warpsize + dfindex] * tmp;
                    }

                    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
                    {
                        out[index]                      = sum1;
                        out[nelmt * nqTot + index]      = sum2;
                        out[2u * nelmt * nqTot + index] = sum3;
                    }
                    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * s_f1[i]) *
                                     s_f0[j] * s_f2[k];
                        out[nelmt * nqTot + index] =
                            (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[2u * nelmt * nqTot + index] = sum3;
                    }
                    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
                    {
                        out[index] = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[nelmt * nqTot + index]      = sum2;
                        out[2u * nelmt * nqTot + index] = sum3;
                    }
                    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                    {
                        out[index] = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[nelmt * nqTot + index] =
                            (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[2u * nelmt * nqTot + index] = sum3;
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData,
          typename std::enable_if_t<
              std::is_same_v<Implementation, Operators::SumFacQP>, bool> = true>
__device__ __forceinline__ void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData f0, f1, f2, f3;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nq0 * nq1 * nq2;
             idx += blockDim.x)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                          SHAPE_TYPE == LibUtilities::Prism ||
                          SHAPE_TYPE == LibUtilities::Pyr)
            {
                f2 = 2.0 / (1.0 - Z2[k]);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                          SHAPE_TYPE == LibUtilities::Pyr)
            {
                f3 = 0.5 * (1.0 + Z1[j]);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                          SHAPE_TYPE == LibUtilities::Prism ||
                          SHAPE_TYPE == LibUtilities::Pyr)
            {
                f1 = 0.5 * (1.0 + Z0[i]);
            }

            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                TData tmp = in[d * nelmt * nqTot + index];
                sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                out[index]                      = sum1;
                out[nelmt * nqTot + index]      = sum2;
                out[2u * nelmt * nqTot + index] = sum3;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                out[index] = (sum1 + (sum2 + sum3) * f1) * f0 * f2;
                out[nelmt * nqTot + index]      = (sum2 + sum3 * f3) * f2;
                out[2u * nelmt * nqTot + index] = sum3;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                out[index]                      = (sum1 + sum3 * f1) * f2;
                out[nelmt * nqTot + index]      = sum2;
                out[2u * nelmt * nqTot + index] = sum3;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                out[index]                      = (sum1 + sum3 * f1) * f2;
                out[nelmt * nqTot + index]      = (sum2 + sum3 * f3) * f2;
                out[2u * nelmt * nqTot + index] = sum3;
            }
        }

        e += gridDim.x;
    }
}

// Non-size based operator.
template <typename Implementation, bool DEFORMED, typename TData>
__global__ void IProductWRTDerivBase1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    IProductWRTDerivBase1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt,
                                                           df, in, out);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int nq0,
          typename TData>
__global__ void IProductWRTDerivBase1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    IProductWRTDerivBase1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt,
                                                           df, in, out);
}

// Non-size based operator.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void IProductWRTDerivBase2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    IProductWRTDerivBase2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, Z0, Z1, df, in, out);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, typename TData>
__global__ void IProductWRTDerivBase2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    IProductWRTDerivBase2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, Z0, Z1, df, in, out);
}

// Non-size based operator.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void IProductWRTDerivBase3DKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    IProductWRTDerivBase3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, Z0, Z1, Z2, df, in, out);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TData>
__global__ void IProductWRTDerivBase3DKernelLauncher(
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    IProductWRTDerivBase3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, Z0, Z1, Z2, df, in, out);
}

// Launchers
// Non-size based operator.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *df, const TData *in, TData *out)
{
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase1DKernelLauncher<Implementation, DEFORMED>
        <<<gridsize, blocksize>>>(ncoord, nq0, nelmt, df, in, out);
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *df,
    const TData *in, TData *out)
{
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase1DKernelLauncher<Implementation, DEFORMED, nq0>
        <<<gridsize, blocksize>>>(ncoord, nelmt, df, in, out);
}

// Non-size based operator.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *Z0, const TData *Z1, const TData *df,
    const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0,
                                                                         nq1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nq0, nq1, nelmt, Z0, Z1,
                                             df, in, out);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *Z0,
    const TData *Z1, const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0,
                                                                         nq1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                         nq0, nq1>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nelmt, Z0, Z1, df, in,
                                             out);
}

// Non-size based operator.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2);
    const unsigned int blocksize =
        GetCUDABlockSize<Implementation>(nq0 * nq1 * nq2);
    const unsigned int gridsize = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(nq0, nq1, nq2, nelmt, Z0, Z1, Z2,
                                             df, in, out);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2);
    const unsigned int blocksize =
        GetCUDABlockSize<Implementation>(nq0 * nq1 * nq2);
    const unsigned int gridsize = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTDerivBase3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                         nq0, nq1, nq2>
        <<<gridsize, blocksize, shmemsize>>>(nelmt, Z0, Z1, Z2, df, in, out);
}

} // namespace Nektar::Operators::detail

#endif
