///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsCUDAKernels.cu
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include "UtilsCUDALaunchers.hpp"

namespace Nektar
{

template <typename TData>
__global__ void interleave(const unsigned int VectorWidth,
                           const unsigned int numMetaBlocks,
                           const unsigned int metaBlockSize,
                           const unsigned int dataLen, TData *buffer,
                           TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = dataLen * VectorWidth * metaBlock;

    for (size_t idx = threadIdx.x; idx < dataLen; idx += blockDim.x)
    {
        for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
        {
            buffer[offset + vecElem * dataLen + idx] =
                inout[offset + vecElem * dataLen + idx];
        }
    }

    __syncthreads();

    for (size_t idx = threadIdx.x; idx < dataLen; idx += blockDim.x)
    {
        for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
        {
            inout[offset + idx * VectorWidth + vecElem] =
                buffer[offset + vecElem * dataLen + idx];
        }
    }
}

template <typename TData>
__global__ void deInterleave(const unsigned int VectorWidth,
                             const unsigned int numMetaBlocks,
                             const unsigned int metaBlockSize,
                             const unsigned int dataLen, TData *buffer,
                             TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = dataLen * VectorWidth * metaBlock;

    for (size_t idx = threadIdx.x; idx < dataLen; idx += blockDim.x)
    {
        for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
        {
            buffer[offset + vecElem * dataLen + idx] =
                inout[offset + vecElem * dataLen + idx];
        }
    }

    __syncthreads();

    for (size_t idx = threadIdx.x; idx < dataLen; idx += blockDim.x)
    {
        for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
        {
            inout[offset + vecElem * dataLen + idx] =
                buffer[offset + idx * VectorWidth + vecElem];
        }
    }
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int metaBlockSize,
                            const unsigned int dataLen, double *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(double) * VectorWidth * numMetaBlocks * metaBlockSize;

    double *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks,
                                        metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int metaBlockSize,
                            const unsigned int dataLen, float *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(float) * VectorWidth * numMetaBlocks * metaBlockSize;

    float *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks,
                                        metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int metaBlockSize,
                            const unsigned int dataLen, int *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * VectorWidth * numMetaBlocks * metaBlockSize;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks,
                                        metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int metaBlockSize,
                              const unsigned int dataLen, double *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(double) * VectorWidth * numMetaBlocks * metaBlockSize;

    double *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(
        VectorWidth, numMetaBlocks, metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int metaBlockSize,
                              const unsigned int dataLen, float *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(float) * VectorWidth * numMetaBlocks * metaBlockSize;

    float *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(
        VectorWidth, numMetaBlocks, metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int metaBlockSize,
                              const unsigned int dataLen, int *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * VectorWidth * numMetaBlocks * metaBlockSize;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(
        VectorWidth, numMetaBlocks, metaBlockSize, dataLen, buffer, inout);

    cudaFree(buffer);
}

} // namespace Nektar

#endif
