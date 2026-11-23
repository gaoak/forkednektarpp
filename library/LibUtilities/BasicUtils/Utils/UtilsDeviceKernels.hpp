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
#include <LibUtilities/LoopExecution/LoopExecution.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

namespace Nektar
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LoadToSharedMemory(
    const unsigned int size, const TData *NEK_RESTRICT src,
    TData *NEK_RESTRICT dst, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < size; idx += stride)
    {
        dst[idx] = src[idx];
    }
}

// Padded shared memory: rows of ncol values, each read contiguously by one
// thread, are free of bank conflicts at a stride of SharedMemoryRowStride.
//
// 1. Specify the stride:
//        const unsigned int ldi = SharedMemoryRowStride<TData>(ncol);
//        const bool isPadded    = (ldi != ncol);
// 2. Round each shared-memory region up with SharedMemoryAlignedSize so the
//    next one starts 16-byte aligned; a padded region holds ldi * nrow
//    values.
// 3. Choose the array to read once, outside the element loop:
//        TData *wsp = (isPadded) ? s_padded : s_src;
// 4. Load only when padded, then synchronise:
//        if (isPadded)
//        {
//            LoadWithPaddingToSharedMemory(ncol, nrow, ldi, s_src, wsp,
//                                          threadBlock);
//            localBarrier(threadBlock);
//        }
// 5. Read wsp with row stride ldi.

/**
 * @brief Round a shared-memory region of @p n values up to whole 16-byte
 * words, so that the region after it starts 16-byte aligned and contiguous
 * walks over it can be loaded 16 bytes at a time.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE constexpr unsigned int SharedMemoryAlignedSize(
    const unsigned int n)
{
    constexpr unsigned int v = 16u / sizeof(TData);
    return (n + v - 1u) / v * v;
}

/**
 * @brief Row stride for a shared-memory array with rows of @p n values,
 * where each thread reads its row contiguously in 16-byte vector loads.
 *
 * With v = 16 / sizeof(TData), a stride that is an odd multiple of v puts
 * the rows of consecutive threads in distinct banks. Rows of a multiple of
 * 2v values, from 4v on, are padded by v; other rows are left unpadded.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE constexpr unsigned int SharedMemoryRowStride(
    const unsigned int n)
{
    constexpr unsigned int v = 16u / sizeof(TData);
    return (n % (2u * v) == 0u && n >= 4u * v) ? n + v : n;
}

/**
 * @brief Whether rows of stride @p ld, each read contiguously by one thread
 * in 16-byte vector loads, are free of bank conflicts: @p ld is an odd
 * multiple of 16 / sizeof(TData).
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE constexpr bool IsConflictFreeSharedMemoryRowStride(
    const unsigned int ld)
{
    constexpr unsigned int v = 16u / sizeof(TData);
    return ld % (2u * v) == v;
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LoadWithPaddingToSharedMemory(
    const unsigned int ncol, const unsigned int nrow, const unsigned int ldi,
    const TData *NEK_RESTRICT src, TData *NEK_RESTRICT dst,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < ncol * nrow; idx += stride)
    {
        dst[ldi * (idx / ncol) + idx % ncol] = src[idx];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LoadTransposeToSharedMemory(
    const unsigned int ncol, const unsigned int nrow,
    const TData *NEK_RESTRICT src, TData *NEK_RESTRICT dst,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < ncol * nrow; idx += stride)
    {
        dst[(idx % ncol) * nrow + idx / ncol] = src[idx];
    }
}

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

    // On the SYCL CPU back-ends a sub-group need not be warpSize
    // consecutive work-items, so the row-per-warp form below is kept to GPUs.
#if defined(SYCL_ENABLE_CPU) &&                                                \
    (defined(__ADAPTIVECPP__) || defined(__DPCPP_COMPILER_NATIVE_CPU))
    constexpr bool rowPerThread = true;
#else
    constexpr bool rowPerThread = TRANSPOSE;
#endif

    // One output per warp needs the threads to be a whole number of
    // warps, which they need not be when a thread block takes several
    // elements.
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (rowPerThread || stride % warpsize != 0u)
    {
        // One output per thread: with TRANSPOSE consecutive threads read
        // consecutive entries of each matrix row.
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
    }
    else
    {
        // One output per warp, so that the lanes read consecutive entries
        // of its matrix row; every lane of a warp takes part in its
        // reduction.
        const unsigned int lane  = getLaneIdx(threadBlock);
        const unsigned int warp  = idx0 / warpsize;
        const unsigned int nwarp = stride / warpsize;

        for (unsigned int i = warp; i < nmTot; i += nwarp)
        {
            TData tmp = 0.0;
            for (unsigned int j = lane; j < nmTot; j += warpsize)
            {
                tmp += in[j] * nodToMod[i * nmTot + j];
            }
            tmp = warpReduceSum(tmp, threadBlock);

            if (lane == 0u)
            {
                if constexpr (APPEND)
                {
                    out[i] += tmp;
                }
                else
                {
                    out[i] = tmp;
                }
            }
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
        // Elements whose Jacobian is zero, the padding of a block, are set
        // to zero rather than divided by it.
        if constexpr (DEFORMED)
        {
            size_t e = idx % (nelmt * nqTot);
            outptr[idx] =
                (jacptr[e] == TData(0)) ? TData(0) : inptr[idx] / jacptr[e];
        }
        else
        {
            size_t e = (idx % (nelmt * nqTot)) / nqTot;
            outptr[idx] =
                (jacptr[e] == TData(0)) ? TData(0) : inptr[idx] / jacptr[e];
        }
    }
}

// Batched in-place transpose: each column-major rows x cols matrix of inout is
// overwritten by its cols x rows transpose, through shared memory. A non-zero
// ROWS or COLS fixes that size at compile time.
template <unsigned int ROWS = 0u, unsigned int COLS = 0u, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL static void transposeBatchedInPlaceKernel(
    const size_t nbatch, const unsigned int nrows, const unsigned int ncols,
    TData *NEK_RESTRICT inout, unsigned char *shmem,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmem);
    TData *shmemptr = (TData *)shmem;

    const unsigned int rows = ROWS ? ROWS : nrows;
    const unsigned int cols = COLS ? COLS : ncols;

    // Odd column stride makes the strided reads free of bank conflicts.
    const unsigned int ld = rows | 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t b = getBlockIdx(threadBlock); b < nbatch;
         b += getBlockRange(threadBlock))
    {
        TData *inoutptr = inout + rows * cols * b;

        for (unsigned int idx = idx0; idx < rows * cols; idx += stride)
        {
            unsigned int r       = idx % rows;
            unsigned int c       = idx / rows;
            shmemptr[c * ld + r] = inoutptr[idx];
        }

        localBarrier(threadBlock);

        for (unsigned int idx = idx0; idx < rows * cols; idx += stride)
        {
            unsigned int r = idx / cols;
            unsigned int c = idx % cols;
            inoutptr[idx]  = shmemptr[c * ld + r];
        }

        localBarrier(threadBlock);
    }
}

template <typename TData>
inline void transposeBatchedInPlace(const size_t nbatch,
                                    const unsigned int rows,
                                    const unsigned int cols, TData *inout,
                                    const unsigned int streamID)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = nbatch;
    const unsigned int shmemsize = sizeof(TData) * cols * (rows | 1u);

    if (rows == warpsize)
    {
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (transposeBatchedInPlaceKernel<warpsize, 0u>), gridSize, blockSize,
            shmemsize, streamID, nbatch, rows, cols, inout);
    }
    else if (cols == warpsize)
    {
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (transposeBatchedInPlaceKernel<0u, warpsize>), gridSize, blockSize,
            shmemsize, streamID, nbatch, rows, cols, inout);
    }
    else
    {
        DEVICE_1DGRID_KERNEL_LAUNCHER(transposeBatchedInPlaceKernel<>, gridSize,
                                      blockSize, shmemsize, streamID, nbatch,
                                      rows, cols, inout);
    }
}

static constexpr unsigned int transposeTileSize = 32u;

// Batched out-of-place transpose: each column-major rows x cols matrix of in is
// written as its cols x rows transpose to out, through a padded shared tile.
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void transposeBatchedKernel(
    const size_t nbatch, const unsigned int rows, const unsigned int cols,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, unsigned char *shmem,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int tile = transposeTileSize;

    FETCH_SHARED_MEMORY(shmem);
    TData *shmemptr = (TData *)shmem;

    const unsigned int tx       = getLocalIdx<0>(threadBlock);
    const unsigned int ty0      = getLocalIdx<1>(threadBlock);
    const unsigned int tyStride = getLocalRange<1>(threadBlock);

    const unsigned int nTileRows = (rows + tile - 1u) / tile;
    const unsigned int nTiles    = nTileRows * ((cols + tile - 1u) / tile);
    const size_t matSize         = static_cast<size_t>(rows) * cols;

    for (size_t b = getBlockIdx<0>(threadBlock); b < nbatch;
         b += getBlockRange<0>(threadBlock))
    {
        const TData *inptr = in + matSize * b;
        TData *outptr      = out + matSize * b;

        for (unsigned int t = getBlockIdx<1>(threadBlock); t < nTiles;
             t += getBlockRange<1>(threadBlock))
        {
            const unsigned int r0 = (t % nTileRows) * tile;
            const unsigned int c0 = (t / nTileRows) * tile;

            // Coalesced read along the columns of in.
            for (unsigned int j = ty0; j < tile; j += tyStride)
            {
                const unsigned int r = r0 + tx;
                const unsigned int c = c0 + j;
                if (r < rows && c < cols)
                {
                    shmemptr[j * (tile + 1u) + tx] =
                        inptr[r + static_cast<size_t>(c) * rows];
                }
            }

            localBarrier(threadBlock);

            // Coalesced write along the columns of out.
            for (unsigned int j = ty0; j < tile; j += tyStride)
            {
                const unsigned int c = c0 + tx;
                const unsigned int r = r0 + j;
                if (r < rows && c < cols)
                {
                    outptr[c + static_cast<size_t>(r) * cols] =
                        shmemptr[tx * (tile + 1u) + j];
                }
            }

            localBarrier(threadBlock);
        }
    }
}

template <typename TData>
inline void transposeBatched(const size_t nbatch, const unsigned int rows,
                             const unsigned int cols, const TData *in,
                             TData *out, const unsigned int streamID)
{
    constexpr unsigned int tile = transposeTileSize;

    const unsigned int blockSizeY =
        NektarSpaces::Device::defaultBlockSize / tile;
    const unsigned int gridSizeX = nbatch;
    const unsigned int nTiles =
        ((rows + tile - 1u) / tile) * ((cols + tile - 1u) / tile);
    // The kernel strides over tiles, so cap at the CUDA gridDim.y limit.
    const unsigned int gridSizeY = nTiles < 65535u ? nTiles : 65535u;
    const unsigned int shmemsize = sizeof(TData) * tile * (tile + 1u);

    DEVICE_2DGRID_KERNEL_LAUNCHER(transposeBatchedKernel<>, gridSizeX,
                                  gridSizeY, tile, blockSizeY, shmemsize,
                                  streamID, nbatch, rows, cols, in, out);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void interleave(const unsigned int interleaveWidth,
                       const size_t numElmtGroups, const unsigned int npts,
                       TData *inout, const unsigned int streamID)
{
    const unsigned int shmemsize =
        sizeof(TData) * interleaveWidth * (npts | 1u);
    const unsigned int maxShemsize =
        GetDeviceProperties::SharedMemoryPerBlock();

    if (shmemsize <= maxShemsize)
    {
        transposeBatchedInPlace(numElmtGroups, npts, interleaveWidth, inout,
                                streamID);
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

#if defined(NEKTAR_ENABLE_HIP) || defined(NEKTAR_ENABLE_CUDA)
        // Batched transpose through the BLAS geam
        if constexpr (std::is_floating_point_v<TData>)
        {
            auto handle =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);

            TData alpha   = 1.0;
            TData beta    = 0.0;
            TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
            NekBlas::GeamStridedBatched(
                handle, "T", "T", interleaveWidth, npts, alpha, inout, npts,
                interleaveWidth * npts, beta, (TData *)nullptr, npts,
                interleaveWidth * npts, buffer, interleaveWidth,
                interleaveWidth * npts, numElmtGroups);
            deviceMemcpy<eDeviceToDevice>(inout, buffer, bufferSize, streamID);
            return;
        }
#endif

        TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
        transposeBatched(numElmtGroups, npts, interleaveWidth, inout, buffer,
                         streamID);
        deviceMemcpy<eDeviceToDevice>(inout, buffer, bufferSize, streamID);
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
    const unsigned int shmemsize =
        sizeof(TData) * (interleaveWidth | 1u) * npts;
    const unsigned int maxShemsize =
        GetDeviceProperties::SharedMemoryPerBlock();

    if (shmemsize <= maxShemsize)
    {
        transposeBatchedInPlace(numElmtGroups, interleaveWidth, npts, inout,
                                streamID);
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

#if defined(NEKTAR_ENABLE_HIP) || defined(NEKTAR_ENABLE_CUDA)
        // Batched transpose through the BLAS geam
        if constexpr (std::is_floating_point_v<TData>)
        {
            auto handle =
                NekBlas::Handle<NektarSpaces::Device>::GetInstance(streamID);

            TData alpha   = 1.0;
            TData beta    = 0.0;
            TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
            NekBlas::GeamStridedBatched(
                handle, "T", "T", npts, interleaveWidth, alpha, inout,
                interleaveWidth, interleaveWidth * npts, beta, (TData *)nullptr,
                interleaveWidth, interleaveWidth * npts, buffer, npts,
                interleaveWidth * npts, numElmtGroups);
            deviceMemcpy<eDeviceToDevice>(inout, buffer, bufferSize, streamID);
            return;
        }
#endif

        TData *buffer = (TData *)internalInterleaveDeviceBuffer[streamID];
        transposeBatched(numElmtGroups, interleaveWidth, npts, inout, buffer,
                         streamID);
        deviceMemcpy<eDeviceToDevice>(inout, buffer, bufferSize, streamID);
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

/**
 * @brief Add the volume of a block's elements, the sum of its quadrature
 * metric, to @p out. See IntegralKernelLauncher().
 *
 * @param n         Entries of the metric, padding included.
 * @param metric    Quadrature weight times Jacobian at every entry.
 * @param out       The volume is added to this value.
 * @param streamID  The stream the launches run on; reset to the null stream
 *                  afterwards.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void VolumeKernelLauncher(const size_t n,
                                                  const TData *metric,
                                                  TData *out,
                                                  const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>>(
        0, n, NEKTAR_LAMBDA(const size_t idx) { return metric[idx]; }, out);
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Add the integrals of INTEGRALOP of @p NRED consecutive components
 * of a block to out[0, NRED).
 *
 * The components and the quadrature metric share the block's storage layout,
 * the caller passing the metric at the interleave width of the input, so one
 * storage index addresses the same point in all of them whatever that width,
 * and every entry is one term of a parallel sum. The sum is the two-stage
 * reduction of parallel_reduce(): a grid of the reduction's own size, each
 * thread block storing its partials, and one add to @p out. A thread block
 * per element, each adding to @p out, spends its time contending for that
 * one address on some devices. The components are taken @p NRED at a time
 * because each reduction is two launches, whose fixed cost is most of what
 * the integrals of a few components cost.
 *
 * The metric is zero on the padding, which holds no solution and may hold
 * anything; an entry with no metric is left out, not multiplied by it, so
 * that a non-finite padding value cannot reach the sum.
 *
 * @tparam INTEGRALOP Applied to every value before it is weighted.
 * @tparam NRED     Components summed in the one reduction.
 * @param n         Entries of a component, padding included; the stride
 *                  between components.
 * @param metric    Quadrature weight times Jacobian at every entry.
 * @param in        The first of the components.
 * @param out       The integrals are added to these values.
 * @param streamID  The stream the launches run on; reset to the null stream
 *                  afterwards.
 */
template <typename ExecSpace, template <typename> typename INTEGRALOP,
          unsigned int NRED, typename TData>
NEK_FORCE_INLINE static void IntegralKernelLauncher(const size_t n,
                                                    const TData *metric,
                                                    const TData *in, TData *out,
                                                    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, NRED>(
        0, n,
        NEKTAR_LAMBDA(const size_t idx, TData *vals) {
            const TData weight = metric[idx];
            for (unsigned int s = 0; s < NRED; ++s)
            {
                const TData value = in[s * n + idx];
                vals[s]           = weight != TData(0.0)
                                        ? weight * INTEGRALOP<TData>()(value)
                                        : TData(0.0);
            }
        },
        out);
    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Add the integrals of INTEGRALOP of @p NRED consecutive components
 * of a block, each over its @p nPlanes homogeneous planes, to out[0, NRED).
 *
 * As IntegralKernelLauncher(), with the planes of a component stored one
 * after the other and each weighted by its entry of @p planeWeights, so that
 * all the planes of a component are summed in the one reduction.
 *
 * @tparam INTEGRALOP Applied to every value before it is weighted.
 * @tparam NRED     Components summed in the one reduction.
 * @param n         Entries of a plane, padding included; the stride between
 *                  planes.
 * @param nPlanes   Planes of a component.
 * @param metric    Quadrature weight times Jacobian at every entry of a plane.
 * @param planeWeights  Quadrature weight of every plane.
 * @param in        The first of the components.
 * @param out       The integrals are added to these values.
 * @param streamID  The stream the launches run on; reset to the null stream
 *                  afterwards.
 */
template <typename ExecSpace, template <typename> typename INTEGRALOP,
          unsigned int NRED, typename TData>
NEK_FORCE_INLINE static void IntegralKernelLauncher(const size_t n,
                                                    const unsigned int nPlanes,
                                                    const TData *metric,
                                                    const TData *planeWeights,
                                                    const TData *in, TData *out,
                                                    const unsigned int streamID)
{
    const size_t compSize = n * nPlanes;

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>, NRED>(
        0, compSize,
        NEKTAR_LAMBDA(const size_t idx, TData *vals) {
            const size_t plane = idx / n;
            const TData weight = metric[idx - plane * n] * planeWeights[plane];
            for (unsigned int s = 0; s < NRED; ++s)
            {
                const TData value = in[s * compSize + idx];
                vals[s]           = weight != TData(0.0)
                                        ? weight * INTEGRALOP<TData>()(value)
                                        : TData(0.0);
            }
        },
        out);
    Nektar::LoopExecutionSetStreamID(0);
}

#endif

} // namespace Nektar
