///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsHIPCUDAKernels.hpp
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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__))

namespace Nektar
{

template <typename TData>
__global__ void interleaveKernelLauncher(const unsigned int VectorWidth,
                                         const unsigned int numMetaBlocks,
                                         const unsigned int npts, TData *buffer,
                                         TData *inout,
                                         const hipcudaBlock1D &threadBlock)
{
    interleaveKernel(VectorWidth, numMetaBlocks, npts, buffer, inout,
                     threadBlock);
}

template <typename TData>
__global__ void deInterleaveKernelLauncher(const unsigned int VectorWidth,
                                           const unsigned int numMetaBlocks,
                                           const unsigned int npts,
                                           TData *buffer, TData *inout,
                                           const hipcudaBlock1D &threadBlock)
{
    deInterleaveKernel(VectorWidth, numMetaBlocks, npts, buffer, inout,
                       threadBlock);
}

template <typename TData>
__global__ void BuildInterleaveMapKernelLauncher(const unsigned int npts,
                                                 const unsigned int newVecWidth,
                                                 const unsigned int offset,
                                                 TData *deInterleaveMapPtr,
                                                 TData *interleaveMapPtr,
                                                 TData *buffer)
{
    BuildInterleaveMapKernel(npts, newVecWidth, offset, deInterleaveMapPtr,
                             interleaveMapPtr, buffer, hipcudaBlock1D());
}

template <unsigned int VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int npts,
           TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMalloc(&buffer, bufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMalloc(&buffer, bufferSize));
#endif
    interleaveKernelLauncher<<<gridSize, blockSize>>>(
        VectorWidth, numMetaBlocks, npts, buffer, inout, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(buffer));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(buffer));
#endif
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMalloc(&buffer, bufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMalloc(&buffer, bufferSize));
#endif
    deInterleaveKernelLauncher<<<gridSize, blockSize>>>(
        VectorWidth, numMetaBlocks, npts, buffer, inout, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(buffer));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(buffer));
#endif
}

template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
BuildInterleaveMap(const unsigned int numMetaBlocks, const unsigned int npts,
                   const unsigned int newVecWidth, const unsigned int offset,
                   int *deInterleaveMapPtr, int *interleaveMapPtr)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize = sizeof(int) * newVecWidth * numMetaBlocks * npts;

    int *buffer;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMalloc(&buffer, bufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMalloc(&buffer, bufferSize));
#endif
    BuildInterleaveMapKernelLauncher<<<gridSize, blockSize>>>(
        npts, newVecWidth, offset, deInterleaveMapPtr, interleaveMapPtr,
        buffer);
    CHECK_LAST_HIPCUDA_ERROR();
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(buffer));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(buffer));
#endif
}

} // namespace Nektar

#endif
