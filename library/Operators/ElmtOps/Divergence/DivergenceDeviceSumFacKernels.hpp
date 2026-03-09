///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceDeviceSumFacKernels.hpp
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline constexpr unsigned int DivergenceSharedMemorySize(const unsigned int nq0,
                                                         const unsigned int nq1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nq0 + nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline constexpr unsigned int DivergenceSharedMemorySize(const unsigned int nq0,
                                                         const unsigned int nq1,
                                                         const unsigned int nq2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nq0 + 2u * nq1 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nq0 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nq0 + nq1 + nq2;
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Divergence1DSumFacKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t nelmt,
    [[maybe_unused]] const unsigned int inoffset, const TData *__restrict__ D0,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * warpsize * iwarp;
        TData *outptr      = out + nq0 * warpsize * iwarp;
        PhysDerivDir1DSumFacKernel<DEFORMED, 0, false>(ilane, 1, nq0, D0, dfptr,
                                                       inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Divergence2DSumFacKernel(
    const unsigned ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const size_t inoffset, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;

        PhysDerivDir2DSumFacKernel<SHAPE_TYPE, DEFORMED, 0, false>(
            ilane, ncoord, nq0, nq1, D0, D1, s_f0, s_f1, dfptr, inptr, outptr);
        PhysDerivDir2DSumFacKernel<SHAPE_TYPE, DEFORMED, 1, true>(
            ilane, ncoord, nq0, nq1, D0, D1, s_f0, s_f1, dfptr,
            inptr + inoffset, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Divergence3DSumFacKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const size_t inoffset, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, 0, false>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr, outptr);
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, 1, true>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr + inoffset, outptr);
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, 2, true>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr + 2 * inoffset, outptr);
        e += getGlobalRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Divergence1DKernelLauncher(NonTemplated1DPhysSizeParameters sizeParam1D,
                               const size_t nelmt, const size_t inoffset,
                               const TData *__restrict__ D0,
                               const TData *__restrict__ df,
                               const TData *__restrict__ in,
                               TData *__restrict__ out,
                               const TthreadBlock &threadBlock)
{
    Divergence1DSumFacKernel<DEFORMED>(sizeParam1D.ncoord(), sizeParam1D.nq0(),
                                       nelmt, inoffset, D0, df, in, out,
                                       threadBlock);
}

// Size based template version.
template <typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          typename TthreadBlock, typename TData/*,
                                                                   unsigned int maxThreadPerBlock =
                                                                   GetDeviceBlockSize<Implementation>(nq0 *nq1)*/>
NEK_DEVICE_KERNEL 
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
 /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
Divergence1DKernelLauncher(
        [[maybe_unused]] Templated1DPhysSizeParameters<ncoord, nq0>,
const size_t nelmt, const size_t inoffset, const TData *__restrict__ D0,
                             const TData *__restrict__ df,
                             const TData *__restrict__ in,
                             TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    Divergence1DSumFacKernel<DEFORMED>(ncoord, nq0, nelmt, inoffset, D0, df, in,
                                       out, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Divergence2DKernelLauncher(
        NonTemplated2DPhysSizeParameters sizeParam2D, const size_t nelmt,
        const size_t inoffset, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ df,
        const TData *__restrict__ in, TData *__restrict__ out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Divergence2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam2D.ncoord(), sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt,
        inoffset, D0, D1, f0, f1, df, in, out, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData/*,
                                                                   unsigned int maxThreadPerBlock =
                                                                   GetDeviceBlockSize<Implementation>(nq0 *nq1)*/>
NEK_DEVICE_KERNEL 
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
 /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
Divergence2DKernelLauncher(
        [[maybe_unused]] Templated2DPhysSizeParameters<ncoord, nq0, nq1>,
        const size_t nelmt, const size_t inoffset, const TData *__restrict__ D0,
                             const TData *__restrict__ D1,
                             const TData *__restrict__ f0,
                             const TData *__restrict__ f1,
                             const TData *__restrict__ df,
                             const TData *__restrict__ in,
                             TData *__restrict__ out, unsigned char* shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Divergence2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, nelmt, inoffset, D0, D1, f0, f1, df, in, out,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Divergence3DKernelLauncher(
        NonTemplated3DPhysSizeParameters sizeParam3D, const size_t nelmt,
        const size_t inoffset, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ D2,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ f1m, const TData *__restrict__ f2,
        const TData *__restrict__ df, const TData *__restrict__ in,
        TData *__restrict__ out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Divergence3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nq0(), sizeParam3D.nq1(), sizeParam3D.nq2(), nelmt,
        inoffset, D0, D1, D2, f0, f1, f1m, f2, df, in, out, shmemptr,
        threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Divergence3DKernelLauncher(
        [[maybe_unused]] Templated3DPhysSizeParameters<nq0, nq1, nq2>,
        const size_t nelmt, const size_t inoffset, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ D2,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ f1m, const TData *__restrict__ f2,
        const TData *__restrict__ df, const TData *__restrict__ in,
        TData *__restrict__ out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Divergence3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, nelmt, inoffset, D0, D1, D2, f0, f1, f1m, f2, df, in,
        out, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
