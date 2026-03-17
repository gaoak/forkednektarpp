///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceSumFacTOPKernels.hpp
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
inline constexpr unsigned int AdvectionSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1)
{
    return nq0 * nq1;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline constexpr unsigned int AdvectionSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1,
                                                        const unsigned int nq2)
{
    return nq0 * nq1 * nq2;
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const TData *__restrict__ D0, const TData *__restrict__ df,
    const TData *__restrict__ advVel_ptr, const size_t adVecoffset,
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
        TData tmp = 0.0;
        for (unsigned int d = 0u; d < ncoord; d++)
        {
            tmp +=
                d0 * df[d * dfsize + dfindex] * advVel_ptr[d * adVecoffset + i];
        }
        out[i] = tmp;
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ df, const TData *__restrict__ advVel_ptr,
    const size_t adVecoffset, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
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

        // Get advection velocity.
        TData vx, vy;
        vx = advVel_ptr[idx];
        vy = advVel_ptr[idx + 1u * adVecoffset];

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
        TData tmp;
        tmp = vx *
              (d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex]);
        tmp += vy * (d0 * df[2u * dfsize + dfindex] +
                     d1 * df[3u * dfsize + dfindex]);
        if (ncoord == 3u)
        {
            TData vz = advVel_ptr[idx + 2u * adVecoffset];
            tmp += vz * (d0 * df[4u * dfsize + dfindex] +
                         d1 * df[5u * dfsize + dfindex]);
        }
        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ advVel_ptr, const size_t adVecoffset,
    const TData *__restrict__ in, TData *__restrict__ out,
    const TthreadBlock &threadBlock)
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

        // Get advection velocity.
        TData vx, vy, vz;
        vx = advVel_ptr[idx];
        vy = advVel_ptr[idx + 1u * adVecoffset];
        vz = advVel_ptr[idx + 2u * adVecoffset];

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
        TData tmp;
        tmp = vx *
              (d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex] +
               d2 * df[2u * dfsize + dfindex]);
        tmp += vy * (d0 * df[3u * dfsize + dfindex] +
                     d1 * df[4u * dfsize + dfindex] +
                     d2 * df[5u * dfsize + dfindex]);
        tmp += vz * (d0 * df[6u * dfsize + dfindex] +
                     d1 * df[7u * dfsize + dfindex] +
                     d2 * df[8u * dfsize + dfindex]);
        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionSumFacTOP1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t nelmt,
    const TData *__restrict__ D0, const TData *__restrict__ df,
    const TData *__restrict__ advVel_ptr, const size_t adVecoffset,
    const TData *__restrict__ in, TData *__restrict__ out,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr     = df + ndf * dfsize * e;
        const TData *inptr     = in + nq0 * e;
        TData *outptr          = out + nq0 * e;
        const TData *advVelPtr = advVel_ptr + nq0 * e;
        Advection1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, D0, dfptr, advVelPtr,
                                             adVecoffset, inptr, outptr,
                                             threadBlock);
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionSumFacTOP2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ advVel_ptr, const size_t adVecoffset,
    const TData *__restrict__ in, TData *__restrict__ out,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
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
        const TData *dfptr     = df + ndf * dfsize * e;
        const TData *inptr     = in + nqTot * e;
        TData *outptr          = out + nqTot * e;
        const TData *advVelPtr = advVel_ptr + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        Advection2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, D0, D1, f0, f1, dfptr, advVelPtr, adVecoffset,
            s_wsp0, outptr, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionSumFacTOP3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ advVel_ptr,
    const size_t adVecoffset, const TData *__restrict__ in,
    TData *__restrict__ out, unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
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
        const TData *dfptr     = df + ndf * dfsize * e;
        const TData *inptr     = in + nqTot * e;
        TData *outptr          = out + nqTot * e;
        const TData *advVelPtr = advVel_ptr + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        Advection3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, advVelPtr,
            adVecoffset, s_wsp0, outptr, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Advection1DKernelLauncher(NonTemplated1DPhysSizeParameters sizeParam1D,
                              const size_t nelmt, const TData *__restrict__ D0,
                              const TData *__restrict__ df,
                              const TData *__restrict__ advVel_ptr,
                              const size_t adVecoffset,
                              const TData *__restrict__ in,
                              TData *__restrict__ out,
                              const TthreadBlock &threadBlock)
{
    AdvectionSumFacTOP1DKernel<DEFORMED>(
        sizeParam1D.ncoord(), sizeParam1D.nq0(), nelmt, D0, df, advVel_ptr,
        adVecoffset, in, out, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int ncoord,
    unsigned int nq0, typename TthreadBlock, typename TData/*,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)*/>
NEK_DEVICE_KERNEL 
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type 
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    Advection1DKernelLauncher(
        [[maybe_unused]] Templated1DPhysSizeParameters<ncoord, nq0>,
  const size_t nelmt, const TData *__restrict__ D0,
                              const TData *__restrict__ df,
                              const TData *__restrict__ advVel_ptr, 
                              const size_t adVecoffset, 
                              const TData *__restrict__ in,
                              TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    AdvectionSumFacTOP1DKernel<DEFORMED>(ncoord, nq0, nelmt, D0, df, advVel_ptr,
                                         adVecoffset, in, out, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Advection2DKernelLauncher(
        NonTemplated2DPhysSizeParameters sizeParam2D, const size_t nelmt,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ df, const TData *__restrict__ advVel_ptr,
        const size_t adVecoffset, const TData *__restrict__ in,
        TData *__restrict__ out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    AdvectionSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam2D.ncoord(), sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, D0,
        D1, f0, f1, df, advVel_ptr, adVecoffset, in, out, shmemptr,
        threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData/*,
          unsigned int maxThreadPerBlock =
              GetDeviceBlockSize<Implementation>(nq0 *nq1)*/>
NEK_DEVICE_KERNEL 
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type 
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    Advection2DKernelLauncher(
        [[maybe_unused]] Templated2DPhysSizeParameters<ncoord, nq0, nq1>,
const size_t nelmt, const TData *__restrict__ D0,
                              const TData *__restrict__ D1,
                              const TData *__restrict__ f0,
                              const TData *__restrict__ f1,
                              const TData *__restrict__ df,
                              const TData *__restrict__ advVel_ptr, 
                              const size_t adVecoffset, 
                              const TData *__restrict__ in,
                              TData *__restrict__ out, unsigned char* shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    AdvectionSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, advVel_ptr, adVecoffset,
        in, out, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Advection3DKernelLauncher(
        NonTemplated3DPhysSizeParameters sizeParam3D, const size_t nelmt,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ D2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ df,
        const TData *__restrict__ advVel_ptr, const size_t adVecoffset,
        const TData *__restrict__ in, TData *__restrict__ out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    AdvectionSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nq0(), sizeParam3D.nq1(), sizeParam3D.nq2(), nelmt, D0, D1,
        D2, f0, f1, f1m, f2, df, advVel_ptr, adVecoffset, in, out, shmemptr,
        threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2, typename TthreadBlock,
          typename TData/*,
          unsigned int maxThreadPerBlock =
              GetDeviceBlockSize<Implementation>(nq0 *nq1 *nq2)*/>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type 
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/ 
    Advection3DKernelLauncher(
        [[maybe_unused]] Templated3DPhysSizeParameters<nq0, nq1, nq2>,
    const size_t nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df,  const TData *__restrict__ advVel_ptr, 
    const size_t adVecoffset, const TData *__restrict__ in,
    TData *__restrict__ out, unsigned char*shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    AdvectionSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, advVel_ptr,
        adVecoffset, in, out, shmemptr, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
