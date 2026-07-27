///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2DeviceKernels.hpp
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

#include "ElmtOps/ElmtBlockOp.hpp"
#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void NormSumFac1DKernel(
    const unsigned int nq0, const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + nq0 * warpsize * iwarp;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            const unsigned int index = warpsize * i + ilane;
            const TData jacobian     = DEFORMED ? jacptr[index] : jacptr[ilane];

            acc += inptr[index] * inptr[index] * w0[i] * jacobian;
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void VolumeSumFac1DKernel(
    const unsigned int nq0, const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            const unsigned int index = warpsize * i + ilane;
            const TData jacobian     = DEFORMED ? jacptr[index] : jacptr[ilane];

            acc += w0[i] * jacobian;
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void NormSumFac2DKernel(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + nqTot * warpsize * iwarp;

        unsigned int q = 0;
        for (unsigned int j = 0; j < nq1; ++j)
        {
            const TData w1j = w1[j];
            for (unsigned int i = 0; i < nq0; ++i, ++q)
            {
                const unsigned int index = warpsize * q + ilane;
                const TData jacobian = DEFORMED ? jacptr[index] : jacptr[ilane];

                acc += inptr[index] * inptr[index] * w0[i] * w1j * jacobian;
            }
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void VolumeSumFac2DKernel(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;

        unsigned int q = 0;
        for (unsigned int j = 0; j < nq1; ++j)
        {
            const TData w1j = w1[j];
            for (unsigned int i = 0; i < nq0; ++i, ++q)
            {
                const unsigned int index = warpsize * q + ilane;
                const TData jacobian = DEFORMED ? jacptr[index] : jacptr[ilane];

                acc += w0[i] * w1j * jacobian;
            }
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void NormSumFac3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1 * nq2;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + nqTot * warpsize * iwarp;

        unsigned int q = 0;
        for (unsigned int k = 0; k < nq2; ++k)
        {
            const TData w2k = w2[k];
            for (unsigned int j = 0; j < nq1; ++j)
            {
                const TData w12 = w1[j] * w2k;
                for (unsigned int i = 0; i < nq0; ++i, ++q)
                {
                    const unsigned int index = warpsize * q + ilane;
                    const TData jacobian =
                        DEFORMED ? jacptr[index] : jacptr[ilane];

                    acc += inptr[index] * inptr[index] * w0[i] * w12 * jacobian;
                }
            }
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void VolumeSumFac3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1 * nq2;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;

        unsigned int q = 0;
        for (unsigned int k = 0; k < nq2; ++k)
        {
            const TData w2k = w2[k];
            for (unsigned int j = 0; j < nq1; ++j)
            {
                const TData w12 = w1[j] * w2k;
                for (unsigned int i = 0; i < nq0; ++i, ++q)
                {
                    const unsigned int index = warpsize * q + ilane;
                    const TData jacobian =
                        DEFORMED ? jacptr[index] : jacptr[ilane];

                    acc += w0[i] * w12 * jacobian;
                }
            }
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

// Non-size based versions.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Norm1DKernelLauncher(
    NonTemplatedPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT norm,
    const TthreadBlock &threadBlock)
{
    NormSumFac1DKernel<DEFORMED>(sizeParam1D.nq0(), nelmt, w0, jac, in, norm,
                                 threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Volume1DKernelLauncher(
    NonTemplatedPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
    TData *NEK_RESTRICT volume, const TthreadBlock &threadBlock)
{
    VolumeSumFac1DKernel<DEFORMED>(sizeParam1D.nq0(), nelmt, w0, jac, volume,
                                   threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Norm2DKernelLauncher(
    NonTemplatedPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    NormSumFac2DKernel<DEFORMED>(sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt,
                                 w0, w1, jac, in, norm, threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Volume2DKernelLauncher(
    NonTemplatedPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    VolumeSumFac2DKernel<DEFORMED>(sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt,
                                   w0, w1, jac, volume, threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Norm3DKernelLauncher(
    NonTemplatedPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT norm,
    const TthreadBlock &threadBlock)
{
    NormSumFac3DKernel<DEFORMED>(sizeParam3D.nq0(), sizeParam3D.nq1(),
                                 sizeParam3D.nq2(), nelmt, w0, w1, w2, jac, in,
                                 norm, threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Volume3DKernelLauncher(
    NonTemplatedPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT jac,
    TData *NEK_RESTRICT volume, const TthreadBlock &threadBlock)
{
    VolumeSumFac3DKernel<DEFORMED>(sizeParam3D.nq0(), sizeParam3D.nq1(),
                                   sizeParam3D.nq2(), nelmt, w0, w1, w2, jac,
                                   volume, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
