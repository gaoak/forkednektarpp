///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrCUDAKernelLaunchers.hpp
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

template <typename TData>
__global__ void AssembleKernelLauncher(const unsigned int nsize,
                                       const int *__restrict__ assmbPtr,
                                       const TData *__restrict__ signPtr,
                                       const TData *__restrict__ inptr,
                                       TData *__restrict__ outptr,
                                       const cudaBlock1D &threadBlock)
{
    AssembleKernel<>(nsize, assmbPtr, signPtr, inptr, outptr, threadBlock);
}

template <typename TData>
__global__ void AssembleKernelLauncher(const unsigned int nsize,
                                       const int *__restrict__ assmbPtr,
                                       const TData sign,
                                       const TData *__restrict__ inptr,
                                       TData *__restrict__ outptr,
                                       const cudaBlock1D &threadBlock)
{
    AssembleKernel<>(nsize, assmbPtr, sign, inptr, outptr, threadBlock);
}

template <typename TData>
__global__ void AssembleKernelLauncher(const unsigned int nsize,
                                       const int *__restrict__ assmbPtr,
                                       const TData *__restrict__ inptr,
                                       TData *__restrict__ outptr,
                                       const cudaBlock1D &threadBlock)
{
    AssembleKernel<>(nsize, assmbPtr, inptr, outptr, threadBlock);
}

template <typename TData>
__global__ void GlobalToLocalKernelLauncher(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData *__restrict__ signPtr,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const cudaBlock1D &threadBlock)
{
    GlobalToLocalKernel<>(nsize, assmbPtr, signPtr, inptr, outptr, threadBlock);
}

template <typename TData>
__global__ void GlobalToLocalKernelLauncher(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData sign,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const cudaBlock1D &threadBlock)
{
    GlobalToLocalKernel<>(nsize, assmbPtr, sign, inptr, outptr, threadBlock);
}

template <typename TData>
__global__ void GlobalToLocalKernelLauncher(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const cudaBlock1D &threadBlock)
{
    GlobalToLocalKernel<>(nsize, assmbPtr, inptr, outptr, threadBlock);
}

// Kernel Launchers.
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *signPtr, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernelLauncher<><<<gridSize, blockSize>>>(
        nsize, assmbPtr, signPtr, inptr, outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData sign, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernelLauncher<><<<gridSize, blockSize>>>(
        nsize, assmbPtr, sign, inptr, outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernelLauncher<><<<gridSize, blockSize>>>(nsize, assmbPtr, inptr,
                                                      outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *signPtr, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernelLauncher<><<<gridSize, blockSize>>>(
        nsize, assmbPtr, signPtr, inptr, outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData sign, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernelLauncher<><<<gridSize, blockSize>>>(
        nsize, assmbPtr, sign, inptr, outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernelLauncher<><<<gridSize, blockSize>>>(
        nsize, assmbPtr, inptr, outptr, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

} // namespace Nektar::Operators::detail

#endif
