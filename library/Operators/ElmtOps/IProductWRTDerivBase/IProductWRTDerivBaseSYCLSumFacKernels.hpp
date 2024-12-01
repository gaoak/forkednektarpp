///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSYCLSumFacKernels.hpp
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

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    unsigned int e = item_ct1.get_global_id(2);

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

        e += item_ct1.get_global_range(2);
    }
}

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nq0 : 1;
        const unsigned int dfoffset = ncoord * dfsize * e;
        const unsigned int offset   = nq0 * e;

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
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

        e += item_ct1.get_group_range(2);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ shmemptr, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    TData *s_f0, *s_f1;

    // Pre-compute factor.
    const unsigned int idx0   = item_ct1.get_local_id(2);
    const unsigned int stride = item_ct1.get_local_range(2);
    if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        s_f0 = shmemptr;
        s_f1 = s_f0 + nq1;

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f0[idx] = 2.0 / (1.0 - Z1[idx]);
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_global_id(2);

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

        e += item_ct1.get_global_range(2);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    TData f0, f1;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
                f1 = 0.5 * (1.0 + Z0[i]);
            }

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0u; d < ncoord; ++d)
            {
                TData tmp = in[d * nelmt * nqTot + index];
                sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
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

        e += item_ct1.get_group_range(2);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData *s_f0, *s_f1, *s_f2, *s_f3;

    // Pre-compute factor.
    const unsigned int idx0   = item_ct1.get_local_id(2);
    const unsigned int stride = item_ct1.get_local_range(2);
    if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        s_f0 = shmemptr;
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

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        s_f1 = shmemptr;
        s_f2 = s_f1 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f1 = shmemptr;
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

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_global_id(2);

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

        e += item_ct1.get_global_range(2);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ Z2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData f0, f1, f2, f3;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                          SHAPE_TYPE == LibUtilities::Prism ||
                          SHAPE_TYPE == LibUtilities::Pyr)
            {
                f1 = 0.5 * (1.0 + Z0[i]);
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

        e += item_ct1.get_group_range(2);
    }
}

// Launchers
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *df, const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     IProductWRTDerivBase1DKernel_QP<DEFORMED>(
                         nq0, ncoord, nelmt, df, in, out, item);
                 });
         }).wait();
    }
    else
    {
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     IProductWRTDerivBase1DKernel<DEFORMED>(nq0, ncoord, nelmt,
                                                            df, in, out, item);
                 });
         }).wait();
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *Z0, const TData *Z1, const TData *df,
    const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0 * nq1, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item_ct1) {
                     IProductWRTDerivBase2DKernel_QP<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, ncoord, nelmt, Z0, Z1, df, in, out,
                         item_ct1);
                 });
         }).wait();
    }
    else
    {
        const unsigned int shmemsize =
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE>(nq0, nq1);
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize),
                                                  cgh);
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item_ct1) {
                     TData *shmemptr = shmem
                                           .template get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get();
                     IProductWRTDerivBase2DKernel<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, ncoord, nelmt, Z0, Z1, df, in, out, shmemptr,
                         item_ct1);
                 });
         }).wait();
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const TData *df, const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0 * nq1 * nq2, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item_ct1) {
                     IProductWRTDerivBase3DKernel_QP<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, nq2, nelmt, Z0, Z1, Z2, df, in, out,
                         item_ct1);
                 });
         }).wait();
    }
    else
    {
        const unsigned int shmemsize =
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE>(nq0, nq1, nq2);
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize),
                                                  cgh);
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item_ct1) {
                     TData *shmemptr = shmem
                                           .template get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get();
                     IProductWRTDerivBase3DKernel<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, nq2, nelmt, Z0, Z1, Z2, df, in, out,
                         shmemptr, item_ct1);
                 });
         }).wait();
    }
}

} // namespace Nektar::Operators::detail

#endif
