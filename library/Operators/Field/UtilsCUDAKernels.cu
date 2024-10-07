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
                           const unsigned int dataLen, TData *buffer,
                           TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = dataLen * VectorWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < dataLen * VectorWidth;
         idx += blockDim.x)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < dataLen * VectorWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx % VectorWidth;
        unsigned int iElem   = idx / VectorWidth;
        inout[offset + idx]  = buffer[offset + vecElem * dataLen + iElem];
    }
}

template <typename TData>
__global__ void deInterleave(const unsigned int VectorWidth,
                             const unsigned int numMetaBlocks,
                             const unsigned int dataLen, TData *buffer,
                             TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = dataLen * VectorWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < dataLen * VectorWidth;
         idx += blockDim.x)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < dataLen * VectorWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx / dataLen;
        unsigned int iElem   = idx % dataLen;
        inout[offset + idx]  = buffer[offset + iElem * VectorWidth + vecElem];
    }
}

__global__ void BuildInterleaveMap(const unsigned int numMetaBlocks,
                                   const unsigned int ncoeff,
                                   const unsigned int newVecWidth,
                                   const unsigned int offset,
                                   int *deInterleaveMapPtr,
                                   int *interleaveMapPtr, int *buffer)
{
    // rewrite UtilsSYCL.hpp BuildInterleaveMapKernel into CUDA codes
    const unsigned int metaBlock   = blockIdx.x;
    const unsigned int groupOffset = ncoeff * newVecWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < ncoeff * newVecWidth;
         idx += blockDim.x)
    {
        buffer[groupOffset + idx] = offset + groupOffset + idx;
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < ncoeff * newVecWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx % newVecWidth;
        unsigned int iElem   = idx / newVecWidth;
        deInterleaveMapPtr[groupOffset + idx] =
            buffer[groupOffset + newVecWidth * ncoeff + iElem];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < ncoeff * newVecWidth;
         idx += blockDim.x)
    {
        interleaveMapPtr[deInterleaveMapPtr[groupOffset + idx]] =
            groupOffset + idx;
    }
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int dataLen, double *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(double) * VectorWidth * numMetaBlocks * dataLen;

    double *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                        buffer, inout);

    cudaFree(buffer);
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int dataLen, float *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(float) * VectorWidth * numMetaBlocks * dataLen;

    float *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                        buffer, inout);

    cudaFree(buffer);
}

void interleaveCUDAlauncher(const unsigned int VectorWidth,
                            const unsigned int numMetaBlocks,
                            const unsigned int dataLen, int *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * VectorWidth * numMetaBlocks * dataLen;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                        buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int dataLen, double *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(double) * VectorWidth * numMetaBlocks * dataLen;

    double *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                          buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int dataLen, float *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(float) * VectorWidth * numMetaBlocks * dataLen;

    float *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                          buffer, inout);

    cudaFree(buffer);
}

void deInterleaveCUDAlauncher(const unsigned int VectorWidth,
                              const unsigned int numMetaBlocks,
                              const unsigned int dataLen, int *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * VectorWidth * numMetaBlocks * dataLen;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleave<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, dataLen,
                                          buffer, inout);

    cudaFree(buffer);
}

void BuildInterleaveMapCUDAlauncher(const unsigned int numMetaBlocks,
                                    const unsigned int ncoeff,
                                    const unsigned int newVecWidth,
                                    const unsigned int offset,
                                    int *deInterleaveMapPtr,
                                    int *interleaveMapPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * ncoeff;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    BuildInterleaveMap<<<gridSize, blockSize>>>(
        numMetaBlocks, ncoeff, newVecWidth, offset, deInterleaveMapPtr,
        interleaveMapPtr, buffer);

    cudaFree(buffer);
}

} // namespace Nektar

#endif
