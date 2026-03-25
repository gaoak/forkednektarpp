///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsDeviceKernels.hpp
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

#include "Operators/Common/Memory/MemoryAlloc.hpp"
#include "Operators/Common/Spaces.hpp"
#include "Operators/Utils/UtilsDeviceKernelsHelper.hpp"

namespace Nektar
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <bool APPEND = false, bool TRANSPOSE = false, typename TData>
NEK_DEVICE_INLINE static void MatVecKernel(const unsigned int ilane,
                                           const unsigned int nmTot,
                                           const TData *NEK_RESTRICT nodToMod,
                                           const TData *NEK_RESTRICT in,
                                           TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nmTot; ++i)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nmTot; ++j)
        {
            if constexpr (TRANSPOSE)
            {
                tmp += nodToMod[j * nmTot + i] * in[warpsize * j + ilane];
            }
            else
            {
                tmp += nodToMod[i * nmTot + j] * in[warpsize * j + ilane];
            }
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += tmp;
        }
        else
        {
            out[warpsize * i + ilane] = tmp;
        }
    }
}

template <bool APPEND = false, bool TRANSPOSE = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void MatVecSumFacTOPKernel(
    const unsigned int nmTot, const TData *NEK_RESTRICT nodToMod,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nmTot; i += stride)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nmTot; j++)
        {
            if constexpr (TRANSPOSE)
            {
                tmp += in[j] * nodToMod[j * nmTot + i];
            }
            else
            {
                tmp += in[j] * nodToMod[i * nmTot + j];
            }
        }

        if constexpr (APPEND)
        {
            out[i] += tmp;
        }
        else
        {
            out[i] = tmp;
        }
    }

    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByJacobianKernel(
    const size_t nelmt, const unsigned int nqTot, const unsigned int nhomo,
    const TData *jacptr, const TData *inptr, TData *outptr, const TData scale,
    const TthreadBlock &threadBlock)
{
    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nelmt * nqTot * nhomo; idx += stride)
    {
        if constexpr (DEFORMED)
        {
            size_t e    = idx % (nelmt * nqTot);
            outptr[idx] = scale * jacptr[e] * inptr[idx];
        }
        else
        {
            size_t e    = (idx % (nelmt * nqTot)) / nqTot;
            outptr[idx] = scale * jacptr[e] * inptr[idx];
        }
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void DivideByJacobianKernel(
    const size_t nelmt, const unsigned int nqTot, const unsigned int nhomo,
    const TData *jacptr, const TData *inptr, TData *outptr,
    const TthreadBlock &threadBlock)
{
    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nelmt * nqTot * nhomo; idx += stride)
    {
        if constexpr (DEFORMED)
        {
            size_t e    = idx % (nelmt * nqTot);
            outptr[idx] = inptr[idx] / jacptr[e];
        }
        else
        {
            size_t e    = (idx % (nelmt * nqTot)) / nqTot;
            outptr[idx] = inptr[idx] / jacptr[e];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void interleave(const unsigned int interleaveWidth,
                                         const size_t numElmtGroups,
                                         const unsigned int npts,
                                         TData *NEK_RESTRICT inout,
                                         unsigned char *shmem,
                                         const TthreadBlock &threadBlock)
{
    TData *shmemptr = (TData *)shmem;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *inoutptr = inout + npts * interleaveWidth * e;

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            shmemptr[idx] = inoutptr[idx];
        }

        localBarrier(threadBlock);

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            unsigned int iElem = idx % interleaveWidth;
            unsigned int iPts  = idx / interleaveWidth;
            inoutptr[idx]      = shmemptr[iElem * npts + iPts];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void interleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *NEK_RESTRICT inout, unsigned char *shmem,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmem);

    interleave(interleaveWidth, numElmtGroups, npts, inout, shmem, threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void interleaveKernel(const size_t numElmtGroups,
                                               const unsigned int npts,
                                               TData *NEK_RESTRICT inout,
                                               unsigned char *shmem,
                                               const TthreadBlock &threadBlock)
{
    static constexpr unsigned int interleaveWidth =
        NektarSpaces::Device::warpSize;

    FETCH_SHARED_MEMORY(shmem);

    interleave(interleaveWidth, numElmtGroups, npts, inout, shmem, threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void interleave(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *NEK_RESTRICT inout, TData *NEK_RESTRICT wsp,
    unsigned char *shmem, const TthreadBlock &threadBlock)
{
    TData *shmemptr = (TData *)shmem;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *inoutptr = inout + npts * interleaveWidth * e;
        TData *wspptr   = wsp + npts * interleaveWidth * e;
        for (unsigned int s = 0;
             s < (npts + interleaveWidth - 1u) / interleaveWidth; ++s)
        {
            for (unsigned int idx = idx0;
                 idx < interleaveWidth * interleaveWidth; idx += stride)
            {
                unsigned int iElem = idx / interleaveWidth;
                unsigned int iPts  = idx % interleaveWidth;
                if (s * interleaveWidth + iPts < npts)
                {
                    shmemptr[idx] = inoutptr[iElem * npts + iPts];
                }
            }

            localBarrier(threadBlock);

            for (unsigned int idx = idx0;
                 idx < interleaveWidth * interleaveWidth; idx += stride)
            {
                unsigned int iElem = idx % interleaveWidth;
                unsigned int iPts  = idx / interleaveWidth;
                if (s * interleaveWidth + iPts < npts)
                {
                    wspptr[idx] = shmemptr[iElem * interleaveWidth + iPts];
                }
            }

            localBarrier(threadBlock);

            inoutptr += interleaveWidth;
            wspptr += interleaveWidth * interleaveWidth;
        }

        inoutptr = inout + npts * interleaveWidth * e;
        wspptr   = wsp + npts * interleaveWidth * e;
        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            inoutptr[idx] = wspptr[idx];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void interleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *NEK_RESTRICT inout, TData *NEK_RESTRICT wsp,
    unsigned char *shmem, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmem);

    interleave(interleaveWidth, numElmtGroups, npts, inout, wsp, shmem,
               threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void interleaveKernel(const size_t numElmtGroups,
                                               const unsigned int npts,
                                               TData *NEK_RESTRICT inout,
                                               TData *NEK_RESTRICT wsp,
                                               unsigned char *shmem,
                                               const TthreadBlock &threadBlock)
{
    static constexpr unsigned int interleaveWidth =
        NektarSpaces::Device::warpSize;

    FETCH_SHARED_MEMORY(shmem);

    interleave(interleaveWidth, numElmtGroups, npts, inout, wsp, shmem,
               threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void deInterleave(const unsigned int interleaveWidth,
                                           const size_t numElmtGroups,
                                           const unsigned int npts,
                                           TData *inout, unsigned char *shmem,
                                           const TthreadBlock &threadBlock)
{
    TData *shmemptr = (TData *)shmem;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *inoutptr = inout + npts * interleaveWidth * e;

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            shmemptr[idx] = inoutptr[idx];
        }

        localBarrier(threadBlock);

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            unsigned int iPts  = idx % npts;
            unsigned int iElem = idx / npts;
            inoutptr[idx]      = shmemptr[iPts * interleaveWidth + iElem];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void deInterleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *inout, unsigned char *shmem,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmem);

    deInterleave(interleaveWidth, numElmtGroups, npts, inout, shmem,
                 threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void deInterleaveKernel(
    const size_t numElmtGroups, const unsigned int npts, TData *inout,
    unsigned char *shmem, const TthreadBlock &threadBlock)
{
    static constexpr unsigned int interleaveWidth =
        NektarSpaces::Device::warpSize;

    FETCH_SHARED_MEMORY(shmem);

    deInterleave(interleaveWidth, numElmtGroups, npts, inout, shmem,
                 threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void deInterleave(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *NEK_RESTRICT inout, TData *NEK_RESTRICT wsp,
    unsigned char *shmem, const TthreadBlock &threadBlock)
{
    TData *shmemptr = (TData *)shmem;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *inoutptr = inout + npts * interleaveWidth * e;
        TData *wspptr   = wsp + npts * interleaveWidth * e;
        for (unsigned int s = 0;
             s < (npts + interleaveWidth - 1u) / interleaveWidth; ++s)
        {
            for (unsigned int idx = idx0;
                 idx < interleaveWidth * interleaveWidth; idx += stride)
            {
                unsigned int iPts = idx / interleaveWidth;
                if (s * interleaveWidth + iPts < npts)
                {
                    shmemptr[idx] = inoutptr[idx];
                }
            }

            localBarrier(threadBlock);

            for (unsigned int idx = idx0;
                 idx < interleaveWidth * interleaveWidth; idx += stride)
            {
                unsigned int iPts  = idx % interleaveWidth;
                unsigned int iElem = idx / interleaveWidth;
                if (s * interleaveWidth + iPts < npts)
                {
                    wspptr[iElem * npts + iPts] =
                        shmemptr[iPts * interleaveWidth + iElem];
                }
            }

            localBarrier(threadBlock);

            inoutptr += interleaveWidth * interleaveWidth;
            wspptr += interleaveWidth;
        }

        inoutptr = inout + npts * interleaveWidth * e;
        wspptr   = wsp + npts * interleaveWidth * e;
        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            inoutptr[idx] = wspptr[idx];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void deInterleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *NEK_RESTRICT inout, TData *NEK_RESTRICT wsp,
    unsigned char *shmem, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmem);

    deInterleave(interleaveWidth, numElmtGroups, npts, inout, wsp, shmem,
                 threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void deInterleaveKernel(
    const size_t numElmtGroups, const unsigned int npts,
    TData *NEK_RESTRICT inout, TData *NEK_RESTRICT wsp, unsigned char *shmem,
    const TthreadBlock &threadBlock)
{
    static constexpr unsigned int interleaveWidth =
        NektarSpaces::Device::warpSize;

    FETCH_SHARED_MEMORY(shmem);

    deInterleave(interleaveWidth, numElmtGroups, npts, inout, wsp, shmem,
                 threadBlock);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
interleave(const unsigned int interleaveWidth, const size_t numElmtGroups,
           const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;

    const unsigned int shmemsize =
        sizeof(TData) * interleaveWidth * std::min(interleaveWidth, npts);
    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    if (npts <= interleaveWidth)
    {
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(interleaveKernel<>, gridSize,
                                          blockSize, shmemsize, 0,
                                          numElmtGroups, npts, inout);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                interleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                interleaveWidth, numElmtGroups, npts, inout);
        }
    }
    else
    {
        const size_t bufferSize =
            sizeof(TData) * interleaveWidth * numElmtGroups * npts;

        if (internalInterleaveDeviceBufferSize < bufferSize)
        {
            deviceFree(internalInterleaveDeviceBuffer, bufferSize);
            deviceMalloc(&internalInterleaveDeviceBuffer, bufferSize);
            internalInterleaveDeviceBufferSize = bufferSize;
        }
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                interleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                numElmtGroups, npts, inout,
                (TData *)internalInterleaveDeviceBuffer);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                interleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                interleaveWidth, numElmtGroups, npts, inout,
                (TData *)internalInterleaveDeviceBuffer);
        }
    }
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
deInterleave(const unsigned int interleaveWidth, size_t numElmtGroups,
             const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;

    const unsigned int shmemsize =
        sizeof(TData) * interleaveWidth * std::min(interleaveWidth, npts);
    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    if (npts <= interleaveWidth)
    {
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(deInterleaveKernel<>, gridSize,
                                          blockSize, shmemsize, 0,
                                          numElmtGroups, npts, inout);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                deInterleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                interleaveWidth, numElmtGroups, npts, inout);
        }
    }
    else
    {
        const size_t bufferSize =
            sizeof(TData) * interleaveWidth * numElmtGroups * npts;

        if (internalInterleaveDeviceBufferSize < bufferSize)
        {
            deviceFree(internalInterleaveDeviceBuffer, bufferSize);
            deviceMalloc(&internalInterleaveDeviceBuffer, bufferSize);
            internalInterleaveDeviceBufferSize = bufferSize;
        }
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                deInterleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                numElmtGroups, npts, inout,
                (TData *)internalInterleaveDeviceBuffer);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                deInterleaveKernel<>, gridSize, blockSize, shmemsize, 0,
                interleaveWidth, numElmtGroups, npts, inout,
                (TData *)internalInterleaveDeviceBuffer);
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    MultiplyByJacobian(const size_t nelmt, const unsigned int nqTot,
                       const unsigned int nhomo, const TData *jacptr,
                       const TData *inptr, TData *outptr, const TData scale)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM((MultiplyByJacobianKernel<DEFORMED>),
                                          gridSize, blockSize, 0, nelmt, nqTot,
                                          nhomo, jacptr, inptr, outptr, scale);
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    DivideByJacobian(const size_t nelmt, const unsigned int nqTot,
                     const unsigned int nhomo, const TData *jacptr,
                     const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM((DivideByJacobianKernel<DEFORMED>),
                                          gridSize, blockSize, 0, nelmt, nqTot,
                                          nhomo, jacptr, inptr, outptr);
}

#endif

} // namespace Nektar
