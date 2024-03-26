///////////////////////////////////////////////////////////////////////////////
//
// File: CUDAMathKernels.cuh
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

namespace Nektar::Operators
{

template <typename TData>
__global__ void negKernel(const unsigned int nsize, const TData *x, TData *y)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        y[i] = -x[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void addKernel(const unsigned int nsize, const TData *x,
                          const TData *y, TData *z)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void subKernel(const unsigned int nsize, const TData *x,
                          const TData *y, TData *z)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] - y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void daxpyKernel(const unsigned int nsize, const TData alpha,
                            const TData *x, const TData *y, TData *z)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = alpha * x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void vdivKernel(const unsigned int nsize, const TData *x,
                           const TData *y, TData *z)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] / y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <int blockSize, typename TData>
__device__ inline void reduce(const unsigned int nsize, TData *s, TData *out)
{
    // Implementation based on reduce6 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."
    // and https://developer.nvidia.com/blog/using-cuda-warp-level-primitives

    int id = threadIdx.x;

    if (blockSize > 512 && id < 512 && id + 512 < blockSize)
    {
        s[id] += s[id + 512];
    }
    __syncthreads();
    if (blockSize > 256 && id < 256 && id + 256 < blockSize)
    {
        s[id] += s[id + 256];
    }
    __syncthreads();
    if (blockSize > 128 && id < 128 && id + 128 < blockSize)
    {
        s[id] += s[id + 128];
    }
    __syncthreads();
    if (blockSize > 64 && id < 64 && id + 64 < blockSize)
    {
        s[id] += s[id + 64];
    }
    __syncthreads();
    if (blockSize > 32 && id < 32 && id + 32 < blockSize)
    {
        s[id] += s[id + 32];
    }
    __syncthreads();

    if (id < 32)
    {
        TData val = (id < nsize) ? s[id] : 0.0;

        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        if (id == 0)
        {
            out[blockIdx.x] = val;
        }
    }
}

template <int blockSize, typename TData>
__global__ void reduceKernel(const unsigned int nsize, const TData *x,
                             TData *out)
{
    // kernel assumes that blockDim.x = blockSize,
    // and blockSize is power of 2 between 64 and 1024
    __shared__ TData s[blockSize];
    int id     = threadIdx.x;
    double tmp = 0.0;
    for (int tid = blockSize * blockIdx.x + threadIdx.x; tid < nsize;
         tid += blockSize * gridDim.x)
    {
        tmp += x[tid];
    }
    s[id] = tmp;

    __syncthreads();

    reduce<blockSize, TData>(nsize, s, out);
}

template <int blockSize, typename TData>
__global__ void dotKernel(const unsigned int nsize, const TData *x,
                          const TData *y, TData *out)
{
    // kernel assumes that blockDim.x = blockSize,
    // and blockSize is power of 2 between 64 and 1024
    __shared__ TData s[blockSize];
    int id     = threadIdx.x;
    double tmp = 0.0;
    for (int tid = blockSize * blockIdx.x + threadIdx.x; tid < nsize;
         tid += blockSize * gridDim.x)
    {
        tmp += x[tid] * y[tid];
    }
    s[id] = tmp;

    __syncthreads();

    reduce<blockSize, TData>(nsize, s, out);
}

} // namespace Nektar::Operators
