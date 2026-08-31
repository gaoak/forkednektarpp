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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// 1D Case
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nq0;

    TData acc            = 0.0;
    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + compSize * c + nq0 * warpsize * iwarp;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            const unsigned int index = warpsize * i + ilane;
            const TData jacobian     = DEFORMED ? jacptr[index] : jacptr[ilane];

            acc += inptr[index] * inptr[index] * w0[i] * jacobian;
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nq0;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    const unsigned int c      = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + compSize * c + nq0 * e;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += inptr[idx] * inptr[idx] * w0[idx] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(const unsigned int nq0,
                                            const size_t nelmt,
                                            const TData *NEK_RESTRICT w0,
                                            const TData *NEK_RESTRICT jac,
                                            TData *NEK_RESTRICT volume,
                                            const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx<0>(threadBlock);
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

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(const unsigned int nq0,
                                            const size_t nelmt,
                                            const TData *NEK_RESTRICT w0,
                                            const TData *NEK_RESTRICT jac,
                                            TData *NEK_RESTRICT volume,
                                            const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += w0[idx] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

// 2D Case
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nqTot;

    TData acc            = 0.0;
    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + compSize * c + nqTot * warpsize * iwarp;

        for (unsigned int j = 0, cnt = 0; j < nq1; ++j)
        {
            const TData w1j = w1[j];
            for (unsigned int i = 0; i < nq0; ++i, ++cnt)
            {
                const unsigned int index = warpsize * cnt + ilane;
                const TData jacobian = DEFORMED ? jacptr[index] : jacptr[ilane];

                acc += inptr[index] * inptr[index] * w0[i] * w1j * jacobian;
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nqTot;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    const unsigned int c      = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + compSize * c + nqTot * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += inptr[idx] * inptr[idx] * w0[i] * w1[j] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData acc = 0.0;
    size_t e  = getGlobalIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;

        for (unsigned int j = 0, cnt = 0; j < nq1; ++j)
        {
            const TData w1j = w1[j];
            for (unsigned int i = 0; i < nq0; ++i, ++cnt)
            {
                const unsigned int index = warpsize * cnt + ilane;
                const TData jacobian = DEFORMED ? jacptr[index] : jacptr[ilane];

                acc += w0[i] * w1j * jacobian;
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += w0[i] * w1[j] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

// 3D Case
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1 * nq2;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nqTot;

    TData acc            = 0.0;
    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;
        const TData *inptr  = in + compSize * c + nqTot * warpsize * iwarp;

        for (unsigned int k = 0, cnt = 0; k < nq2; ++k)
        {
            const TData w2k = w2[k];
            for (unsigned int j = 0; j < nq1; ++j)
            {
                const TData w12 = w1[j] * w2k;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt)
                {
                    const unsigned int index = warpsize * cnt + ilane;
                    const TData jacobian =
                        DEFORMED ? jacptr[index] : jacptr[ilane];

                    acc += inptr[index] * inptr[index] * w0[i] * w12 * jacobian;
                }
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void NormKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT norm, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1 * nq2;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const size_t compSize =
        ((nelmt + warpsize - 1) / warpsize) * warpsize * nqTot;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    const unsigned int c      = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + compSize * c + nqTot * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += inptr[idx] * inptr[idx] * w0[i] * w1[j] * w2[k] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, norm + c);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(
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
    size_t e  = getGlobalIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const TData *jacptr = jac + jacsize * warpsize * iwarp;

        for (unsigned int k = 0, cnt = 0; k < nq2; ++k)
        {
            const TData w2k = w2[k];
            for (unsigned int j = 0; j < nq1; ++j)
            {
                const TData w12 = w1[j] * w2k;
                for (unsigned int i = 0; i < nq0; ++i, ++cnt)
                {
                    const unsigned int index = warpsize * cnt + ilane;
                    const TData jacobian =
                        DEFORMED ? jacptr[index] : jacptr[ilane];

                    acc += w0[i] * w12 * jacobian;
                }
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void VolumeKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT volume,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            const TData jacobian = DEFORMED ? jacptr[idx] : jacptr[0];

            acc += w0[i] * w1[j] * w2[k] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, volume);
}

#endif

} // namespace Nektar::Operators::detail
