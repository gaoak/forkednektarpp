///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFacTOPKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1)
{
    return nq0 * nq1;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1,
                                                        const unsigned int nq2)
{
    return nq0 * nq1 * nq2;
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t outoffset,
    const TData *__restrict__ D0, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[q];
        }

        // Multiply by derivative factors.
        for (unsigned int d = 0u; d < ncoord; d++)
        {
            out[d * outoffset + i] = d0 * df[d * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor1DSumFacTOPKernel(
    const unsigned int nq0, const TData *__restrict__ D0,
    const TData *__restrict__ in0, TData *__restrict__ out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[q];
        }

        if constexpr (APPEND)
        {
            out[i] += d0;
        }
        else
        {
            out[i] = d0;
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t outoffset, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * q + i];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            d0 *= f1[j];
            d1 += d0 * f0[i];
        }

        // Multiply by derivative factors.
        out[idx] =
            d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex];
        out[outoffset + idx] =
            d0 * df[2u * dfsize + dfindex] + d1 * df[3u * dfsize + dfindex];
        if (ncoord == 3u)
        {
            out[2u * outoffset + idx] =
                d0 * df[4u * dfsize + dfindex] + d1 * df[5u * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor2DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ in0, const TData *__restrict__ in1,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[j * nq1 + q] * in1[nq0 * q + i];
        }

        if constexpr (APPEND)
        {
            out[idx] += d0 + d1;
        }
        else
        {
            out[idx] = d0 + d1;
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t outoffset, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[q * nq2 + k] * in[nq0 * nq1 * q + nq0 * j + i];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData tmp0 = f1m[j] * f2[k] * d0;
            TData tmp1 = f0[i] * tmp0;
            TData tmp2 = f2[k] * d1;
            d0         = tmp0;
            d1         = tmp1 + tmp2;
            d2 += tmp1 + f1[j] * tmp2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            d0 *= f2[k];
            d2 += f0[i] * d0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            d0 *= f2[k];
            d1 *= f2[k];
            d2 += f0[i] * d0 + f1[j] * d1;
        }

        // Multiply by derivative factors.
        out[idx] = d0 * df[0u * dfsize + dfindex] +
                   d1 * df[1u * dfsize + dfindex] +
                   d2 * df[2u * dfsize + dfindex];
        out[outoffset + idx] = d0 * df[3u * dfsize + dfindex] +
                               d1 * df[4u * dfsize + dfindex] +
                               d2 * df[5u * dfsize + dfindex];
        out[2u * outoffset + idx] = d0 * df[6u * dfsize + dfindex] +
                                    d1 * df[7u * dfsize + dfindex] +
                                    d2 * df[8u * dfsize + dfindex];
    }

    localBarrier(threadBlock);
}

template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ in0,
    const TData *__restrict__ in1, const TData *__restrict__ in2,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq0 * nq1);

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[j * nq1 + q] * in1[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[k * nq2 + q] * in2[nq0 * nq1 * q + nq0 * j + i];
        }

        if constexpr (APPEND)
        {
            out[idx] += d0 + d1 + d2;
        }
        else
        {
            out[idx] = d0 + d1 + d2;
        }
    }

    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivSumFacTOP1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t nelmt,
    const unsigned int outoffset, const TData *__restrict__ D0,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nq0 * e;
        TData *outptr      = out + nq0 * e;
        PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, outoffset, D0, dfptr,
                                             inptr, outptr, threadBlock);
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivSumFacTOP2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const unsigned int outoffset,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, outoffset, D0, D1, f0, f1, dfptr, s_wsp0, outptr,
            threadBlock);

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivSumFacTOP3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const unsigned int outoffset,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e = getBlockIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, outoffset, D0, D1, D2, f0, f1, f1m, f2, dfptr,
            s_wsp0, outptr, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    PhysDeriv1DKernelLauncher(const unsigned int ncoord, const unsigned int nq0,
                              const size_t nelmt, const unsigned int outoffset,
                              const TData *__restrict__ D0,
                              const TData *__restrict__ df,
                              const TData *__restrict__ in,
                              TData *__restrict__ out,
                              const TthreadBlock &threadBlock)
{
    PhysDerivSumFacTOP1DKernel<DEFORMED>(ncoord, nq0, nelmt, outoffset, D0, df,
                                         in, out, threadBlock);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, typename TthreadBlock,
          typename TData /*,
unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv1DKernelLauncher(const size_t nelmt, const unsigned int outoffset,
                              const TData *__restrict__ D0,
                              const TData *__restrict__ df,
                              const TData *__restrict__ in,
                              TData *__restrict__ out,
                              const TthreadBlock &threadBlock)
{
    PhysDerivSumFacTOP1DKernel<DEFORMED>(ncoord, nq0, nelmt, outoffset, D0, df,
                                         in, out, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    PhysDeriv2DKernelLauncher(
        const unsigned int ncoord, const unsigned int nq0,
        const unsigned int nq1, const size_t nelmt,
        const unsigned int outoffset, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ df,
        const TData *__restrict__ in, TData *__restrict__ out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, nelmt, outoffset, D0, D1, f0, f1, df, in, out,
        shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData /*,
           unsigned int maxThreadPerBlock =
               GetDeviceBlockSize<Implementation>(nq0 *nq1)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv2DKernelLauncher(const size_t nelmt, const unsigned int outoffset,
                              const TData *__restrict__ D0,
                              const TData *__restrict__ D1,
                              const TData *__restrict__ f0,
                              const TData *__restrict__ f1,
                              const TData *__restrict__ df,
                              const TData *__restrict__ in,
                              TData *__restrict__ out, unsigned char *shmemptr,
                              const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, nelmt, outoffset, D0, D1, f0, f1, df, in, out,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    PhysDeriv3DKernelLauncher(
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
        const size_t nelmt, const unsigned int outoffset,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ D2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ df,
        const TData *__restrict__ in, TData *__restrict__ out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, nelmt, outoffset, D0, D1, D2, f0, f1, f1m, f2, df, in,
        out, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TthreadBlock,
          typename TData /*,
           unsigned int maxThreadPerBlock =
               GetDeviceBlockSize<Implementation>(nq0 *nq1 *nq2)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv3DKernelLauncher(
        const size_t nelmt, const unsigned int outoffset,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ D2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ df,
        const TData *__restrict__ in, TData *__restrict__ out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, nelmt, outoffset, D0, D1, D2, f0, f1, f1m, f2, df, in,
        out, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
