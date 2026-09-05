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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernelsHelper.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

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

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void interleave(const unsigned int interleaveWidth,
                       const size_t numElmtGroups, const unsigned int npts,
                       TData *inout, const unsigned int streamID)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;

    const unsigned int shmemsize = sizeof(TData) * interleaveWidth * npts;
    const unsigned int maxShemsize =
        GetDeviceProperties::SharedMemoryPerBlock();

    if (shmemsize <= maxShemsize)
    {
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(interleaveKernel<>, gridSize,
                                          blockSize, shmemsize, streamID,
                                          numElmtGroups, npts, inout);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                interleaveKernel<>, gridSize, blockSize, shmemsize, streamID,
                interleaveWidth, numElmtGroups, npts, inout);
        }
    }
    else
    {
        const size_t bufferSize =
            sizeof(TData) * interleaveWidth * numElmtGroups * npts;

        if (internalInterleaveDeviceBufferSize.find(streamID) ==
            internalInterleaveDeviceBufferSize.end())
        {
            internalInterleaveDeviceBuffer[streamID] = nullptr;
            deviceMalloc(&internalInterleaveDeviceBuffer[streamID], bufferSize,
                         streamID);
            internalInterleaveDeviceBufferSize[streamID] = bufferSize;
        }
        else if (internalInterleaveDeviceBufferSize[streamID] < bufferSize)
        {
            deviceFree(internalInterleaveDeviceBuffer[streamID], bufferSize,
                       streamID);
            deviceMalloc(&internalInterleaveDeviceBuffer[streamID], bufferSize,
                         streamID);
            internalInterleaveDeviceBufferSize[streamID] = bufferSize;
        }

#if defined(NEKTAR_ENABLE_HIP)
        // HIP specific optimisation
        if constexpr (std::is_floating_point_v<TData>)
        {
#if defined(NEKTAR_USE_MAGMA)
            auto queue =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);
            auto handle = magma_queue_get_hipblas_handle(queue);
#else
            auto handle =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);
#endif

            TData alpha   = 1.0;
            TData beta    = 0.0;
            TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
            if constexpr (std::is_same_v<TData, float>)
            {
                // Use SgeamStridedBatched.
                HIPBLAS_CHECK(hipblasSgeamStridedBatched(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_T, interleaveWidth, npts,
                    &alpha, inout, npts, interleaveWidth * npts, &beta,
                    (TData *)nullptr, npts, interleaveWidth * npts, buffer,
                    interleaveWidth, interleaveWidth * npts, numElmtGroups));
            }
            else if constexpr (std::is_same_v<TData, double>)
            {
                // Use SgeamStridedBatched.
                HIPBLAS_CHECK(hipblasDgeamStridedBatched(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_T, interleaveWidth, npts,
                    &alpha, inout, npts, interleaveWidth * npts, &beta,
                    (TData *)nullptr, npts, interleaveWidth * npts, buffer,
                    interleaveWidth, interleaveWidth * npts, numElmtGroups));
            }
            deviceMemcpy<DeviceToDevice>(inout, buffer, bufferSize, streamID);
            return;
        }
#endif

        const unsigned int shmemsize =
            sizeof(TData) * interleaveWidth * interleaveWidth;
        TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(interleaveKernel<>, gridSize,
                                          blockSize, shmemsize, streamID,
                                          numElmtGroups, npts, inout, buffer);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                interleaveKernel<>, gridSize, blockSize, shmemsize, streamID,
                interleaveWidth, numElmtGroups, npts, inout, buffer);
        }
    }
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void deInterleave(const unsigned int interleaveWidth,
                         size_t numElmtGroups, const unsigned int npts,
                         TData *inout, const unsigned int streamID)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;

    const unsigned int shmemsize = sizeof(TData) * interleaveWidth * npts;
    const unsigned int maxShemsize =
        GetDeviceProperties::SharedMemoryPerBlock();

    if (shmemsize <= maxShemsize)
    {
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(deInterleaveKernel<>, gridSize,
                                          blockSize, shmemsize, streamID,
                                          numElmtGroups, npts, inout);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                deInterleaveKernel<>, gridSize, blockSize, shmemsize, streamID,
                interleaveWidth, numElmtGroups, npts, inout);
        }
    }
    else
    {
        const size_t bufferSize =
            sizeof(TData) * interleaveWidth * numElmtGroups * npts;

        if (internalInterleaveDeviceBufferSize.find(streamID) ==
            internalInterleaveDeviceBufferSize.end())
        {
            internalInterleaveDeviceBuffer[streamID] = nullptr;
            deviceMalloc(&internalInterleaveDeviceBuffer[streamID], bufferSize,
                         streamID);
            internalInterleaveDeviceBufferSize[streamID] = bufferSize;
        }
        else if (internalInterleaveDeviceBufferSize[streamID] < bufferSize)
        {
            deviceFree(internalInterleaveDeviceBuffer[streamID], bufferSize,
                       streamID);
            deviceMalloc(&internalInterleaveDeviceBuffer[streamID], bufferSize,
                         streamID);
            internalInterleaveDeviceBufferSize[streamID] = bufferSize;
        }

#if defined(NEKTAR_ENABLE_HIP)
        // HIP specific optimisation
        if constexpr (std::is_floating_point_v<TData>)
        {
#if defined(NEKTAR_USE_MAGMA)
            auto queue =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);
            auto handle = magma_queue_get_hipblas_handle(queue);
#else
            auto handle =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);
#endif

            TData alpha   = 1.0;
            TData beta    = 0.0;
            TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
            if constexpr (std::is_same_v<TData, float>)
            {
                // Use SgeamStridedBatched.
                HIPBLAS_CHECK(hipblasSgeamStridedBatched(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_T, npts, interleaveWidth,
                    &alpha, inout, interleaveWidth, interleaveWidth * npts,
                    &beta, (TData *)nullptr, interleaveWidth,
                    interleaveWidth * npts, buffer, npts,
                    interleaveWidth * npts, numElmtGroups));
            }
            else if constexpr (std::is_same_v<TData, double>)
            {
                // Use SgeamStridedBatched.
                HIPBLAS_CHECK(hipblasDgeamStridedBatched(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_T, npts, interleaveWidth,
                    &alpha, inout, interleaveWidth, interleaveWidth * npts,
                    &beta, (TData *)nullptr, interleaveWidth,
                    interleaveWidth * npts, buffer, npts,
                    interleaveWidth * npts, numElmtGroups));
            }
            deviceMemcpy<DeviceToDevice>(inout, buffer, bufferSize, streamID);
            return;
        }
#endif

        const unsigned int shmemsize =
            sizeof(TData) * interleaveWidth * interleaveWidth;
        TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
        if (interleaveWidth == NektarSpaces::Device::warpSize)
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(deInterleaveKernel<>, gridSize,
                                          blockSize, shmemsize, streamID,
                                          numElmtGroups, npts, inout, buffer);
        }
        else
        {
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                deInterleaveKernel<>, gridSize, blockSize, shmemsize, streamID,
                interleaveWidth, numElmtGroups, npts, inout, buffer);
        }
    }
}

template <
    typename ExecSpace, bool DEFORMED, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void MultiplyByJacobian(
    const size_t nelmt, const unsigned int nqTot, const unsigned int nhomo,
    const TData *jacptr, const TData *inptr, TData *outptr, const TData scale,
    const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        (MultiplyByJacobianKernel<DEFORMED>), gridSize, blockSize, streamID,
        nelmt, nqTot, nhomo, jacptr, inptr, outptr, scale);
}

template <
    typename ExecSpace, bool DEFORMED, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void DivideByJacobian(const size_t nelmt,
                                              const unsigned int nqTot,
                                              const unsigned int nhomo,
                                              const TData *jacptr,
                                              const TData *inptr, TData *outptr,
                                              const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM((DivideByJacobianKernel<DEFORMED>),
                                          gridSize, blockSize, streamID, nelmt,
                                          nqTot, nhomo, jacptr, inptr, outptr);
}

// Quadrature approximation of the element volume (integral of 1 over each
// element): sum of w0[i]*w1[j]*w2[k]*jac, reduced across the thread block.
// DEFORMED selects a per-quadrature-point Jacobian versus a single constant
// one. Shared by NormL2 (normalisation) and MeanRemoval (mean division). The
// two families differ only in the thread<->element mapping and Jacobian memory
// layout.

// 1D Case - Non-interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
    typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
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
#endif

// 1D Case - Non-interleaved
template <unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
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

// 2D Case - Interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
    typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
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
#endif

// 2D Case - Non-interleaved
template <unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
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

// 3D Case - Interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
    typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
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
#endif

// 3D Case - Non-interleaved
template <unsigned int interleaveWidth, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
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

// The IntegralKernelLauncher family strides between components with the
// compSize the caller passes, taken from the block's own CompSize(). Do not
// recompute it here as roundUp(nelmt, warpSize) * nqTot: a block is padded to
// NektarSpaces::max_vector_width, which is max(AVX width, warpSize), so that
// re-derivation is only correct where the two happen to be equal. It was wrong
// for every block with an element count not a multiple of the SIMD width, and
// every component after the first was then read from the wrong offset.
//
// 1D Case - Interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
    bool DEFORMED, typename TthreadBlock, typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
        Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const size_t nelmt, const size_t compSize,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT integral,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize      = DEFORMED ? nq0 : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

            acc += INTEGRALOP<TData>()(inptr[index]) * w0[i] * jacobian;
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}
#endif

// 1D Case - Non-interleaved
template <template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
          bool DEFORMED, typename TthreadBlock, typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const size_t nelmt, const size_t compSize,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT integral,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

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

            acc += INTEGRALOP<TData>()(inptr[idx]) * w0[idx] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}

// 2D Case - Interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
    bool DEFORMED, typename TthreadBlock, typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
        Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const size_t compSize, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT integral,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                acc +=
                    INTEGRALOP<TData>()(inptr[index]) * w0[i] * w1j * jacobian;
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}
#endif

// 2D Case - Non-interleaved
template <template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
          bool DEFORMED, typename TthreadBlock, typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const size_t compSize, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT integral,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

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

            acc += INTEGRALOP<TData>()(inptr[idx]) * w0[i] * w1[j] * jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}

// 3D Case - Interleaved
#if !(defined(SYCL_ENABLE_CPU) && defined(__ADAPTIVECPP__))
template <
    template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
    bool DEFORMED, typename TthreadBlock, typename TData,
    std::enable_if_t<interleaveWidth == NektarSpaces::Device::warpSize, bool>
        Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const size_t compSize, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT integral, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot        = nq0 * nq1 * nq2;
    const unsigned int jacsize      = DEFORMED ? nqTot : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                    acc += INTEGRALOP<TData>()(inptr[index]) * w0[i] * w12 *
                           jacobian;
                }
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}
#endif

// 3D Case - Non-interleaved
template <template <typename> typename INTEGRALOP, unsigned int interleaveWidth,
          bool DEFORMED, typename TthreadBlock, typename TData,
          std::enable_if_t<interleaveWidth == 1u, bool> Enable = true>
NEK_DEVICE_KERNEL void IntegralKernelLauncher(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const size_t compSize, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT integral, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

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

            acc += INTEGRALOP<TData>()(inptr[idx]) * w0[i] * w1[j] * w2[k] *
                   jacobian;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceSum(acc, threadBlock, integral + c);
}

#endif

} // namespace Nektar
