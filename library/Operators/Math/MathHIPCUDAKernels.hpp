///////////////////////////////////////////////////////////////////////////////
//
// File: MathHIPCUDAKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar
{

__device__ inline float4 &operator+=(float4 &a, const float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__device__ inline double2 &operator+=(double2 &a, const double2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ inline float4 &operator+=(float4 &a, const float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
}

__device__ inline double2 &operator+=(double2 &a, const double b)
{
    a.x += b;
    a.y += b;
    return a;
}

__device__ inline float4 &operator-=(float4 &a, const float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

__device__ inline double2 &operator-=(double2 &a, const double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__device__ inline float4 &operator-=(float4 &a, const float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    return a;
}

__device__ inline double2 &operator-=(double2 &a, const double b)
{
    a.x -= b;
    a.y -= b;
    return a;
}

__device__ inline float4 &operator*=(float4 &a, const float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

__device__ inline double2 &operator*=(double2 &a, const double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

__device__ inline float4 &operator*=(float4 &a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

__device__ inline double2 &operator*=(double2 &a, const double b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

__device__ inline float4 operator*(const float4 &a, const float4 &b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ inline double2 operator*(const double2 &a, const double2 &b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}

__device__ inline float4 operator*(const float &a, const float4 &b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__device__ inline double2 operator*(const double &a, const double2 &b)
{
    return make_double2(a * b.x, a * b.y);
}

__device__ inline float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline double2 operator+(const double2 &a, const double2 &b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ inline float4 operator+(const float4 &a, const float &b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__device__ inline double2 operator+(const double2 &a, const double &b)
{
    return make_double2(a.x + b, a.y + b);
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void absKernel(const size_t nsize,
                                                       const TData *x, TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = std::abs(x[idx]);
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void negKernel(const size_t nsize,
                                                       const TData *x, TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = -x[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void sqrtKernel(const size_t nsize,
                                                        const TData *x,
                                                        TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = std::sqrt(x[idx]);
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void addKernel(const size_t nsize,
                                                       const TData *x,
                                                       const TData *y, TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = x[idx] + y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void subKernel(const size_t nsize,
                                                       const TData *x,
                                                       const TData *y, TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = x[idx] - y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void mulKernel(const size_t nsize,
                                                       const TData alpha,
                                                       const TData *x, TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = alpha * x[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void mulKernel(const size_t nsize,
                                                       const TData *x,
                                                       const TData *y, TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = x[idx] * y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void divKernel(const size_t nsize,
                                                       const TData alpha,
                                                       const TData *x, TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = alpha / x[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void divKernel(const size_t nsize,
                                                       const TData *x,
                                                       const TData *y, TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = x[idx] / y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void daxpyKernel(const size_t nsize,
                                                         const TData alpha,
                                                         const TData *x,
                                                         const TData *y,
                                                         TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = alpha * x[idx] + y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void sumNMatrixKernel(
    const size_t nsize, const size_t n, const TData *x, TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {

        TData sum = TData(0);

        for (int i = 0; i < n; i++)
        {
            sum += x[idx + i * nsize];
        }

        y[idx] = sum;
    }
}

template <bool init, typename TData>
__global__ void reduceSumKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            v4 += reinterpret_cast<const float4 *>(x)[tid];
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            v2 += reinterpret_cast<const double2 *>(x)[tid];
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v += x[tid];
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v += x[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void reduceSumKernel(const size_t nsize, const unsigned int *mask,
                                const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v += mask[tid] * x[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void reduceMaxKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = min;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v               = std::max(v,
                                       std::max(std::max(v4.x, v4.y), std::max(v4.z, v4.w)));
        }
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v                = std::max(v, std::max(v2.x, v2.y));
        }
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v = std::max(v, x[tid]);
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v          = std::max(v, x[tid]);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::max(v, warp.shfl_down(v, 32)); // |
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMax_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMax(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void reduceMaxKernel(const size_t nsize, const unsigned int *mask,
                                const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = min;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v = mask[tid] ? std::max(v, x[tid]) : v;
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::max(v, warp.shfl_down(v, 32)); // |
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMax_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMax(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void reduceMinKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData max             = std::numeric_limits<TData>::max();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = max;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = max;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v               = std::min(v,
                                       std::min(std::min(v4.x, v4.y), std::min(v4.z, v4.w)));
        }
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v                = std::min(v, std::min(v2.x, v2.y));
        }
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v = std::min(v, x[tid]);
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v          = std::min(v, x[tid]);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::min(v, warp.shfl_down(v, 32)); // |
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMin_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMin(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void reduceMinKernel(const size_t nsize, const unsigned int *mask,
                                const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData max             = std::numeric_limits<TData>::max();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = max;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = max;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v = mask[tid] ? std::min(v, x[tid]) : v;
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::min(v, warp.shfl_down(v, 32)); // |
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMin_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMin(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void ddotKernel(const size_t nsize, const TData *x, const TData *y,
                           TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 x4 = reinterpret_cast<const float4 *>(x)[tid];
            const float4 y4 = reinterpret_cast<const float4 *>(y)[tid];
            v4 += x4 * y4;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 x2 = reinterpret_cast<const double2 *>(x)[tid];
            const double2 y2 = reinterpret_cast<const double2 *>(y)[tid];
            v2 += x2 * y2;
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v += x[tid] * y[tid];
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v += x[tid] * y[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void ddotKernel(const size_t nsize, const unsigned int *mask,
                           const TData *x, const TData *y, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    // constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v += mask[tid] * x[tid] * y[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void l1normKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = 0.0;
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += make_float4(std::abs(tmp.x), std::abs(tmp.y), std::abs(tmp.z),
                              std::abs(tmp.w));
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += make_double2(std::abs(tmp.x), std::abs(tmp.y));
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v += std::abs(x[tid]);
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v += std::abs(x[tid]);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void l1normKernel(const size_t nsize, const unsigned int *mask,
                             const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = 0.0;
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v += mask[tid] * std::abs(x[tid]);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void l2normKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int vecsize  = (16u / sizeof(TData));
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += tmp * tmp;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += tmp * tmp;
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v += x[tid] * x[tid];
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v += x[tid] * x[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void l2normKernel(const size_t nsize, const unsigned int *mask,
                             const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v += mask[tid] * x[tid] * x[tid];
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void lpnormKernel(const size_t nsize, const unsigned int p,
                             const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += make_float4(
                std::pow(std::abs(tmp.x), p), std::pow(std::abs(tmp.y), p),
                std::pow(std::abs(tmp.z), p), std::pow(std::abs(tmp.w), p));
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += make_double2(std::pow(std::abs(tmp.x), p),
                               std::pow(std::abs(tmp.y), p));
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v += std::pow(std::abs(x[tid]), p);
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v += std::pow(std::abs(x[tid]), p);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void lpnormKernel(const size_t nsize, const unsigned int p,
                             const unsigned int *mask, const TData *x,
                             TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = 0.0;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v += mask[tid] * std::pow(std::abs(x[tid]), p);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32); // |
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicAdd_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicAdd(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void linfnormKernel(const size_t nsize, const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = min;
        }
    }

    block.sync();

    if constexpr (std::is_same_v<TData, float>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v = std::max(v, std::max(std::max(std::abs(v4.x), std::abs(v4.y)),
                                     std::max(std::abs(v4.z), std::abs(v4.w))));
        }
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v = std::max(v, std::max(std::abs(v2.x), std::abs(v2.y)));
        }
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v = std::max(v, std::abs(x[tid]));
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecsize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v          = std::max(v, std::abs(x[tid]));
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::max(v, warp.shfl_down(v, 32)); // |
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMax_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMax(&out[block.group_index().x], v);
#endif
    }
}

template <bool init, typename TData>
__global__ void linfnormKernel(const size_t nsize, const unsigned int *mask,
                               const TData *x, TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = min;
        }
    }

    block.sync();

    for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
    {
        v = mask[tid] ? std::max(v, std::abs(x[tid])) : v;
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::max(v, warp.shfl_down(v, 32)); // |
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        atomicMax_block(&out[block.group_index().x], v);
#elif defined(NEKTAR_ENABLE_HIP)
        atomicMax(&out[block.group_index().x], v);
#endif
    }
}

// Launchers for the kernels

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
absKernel(const size_t nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    absKernel<<<gridSize, blockSize>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
negKernel(const size_t nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    negKernel<<<gridSize, blockSize>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
sqrtKernel(const size_t nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sqrtKernel<<<gridSize, blockSize>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
addKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    addKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
subKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    subKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    mulKernel<<<gridSize, blockSize>>>(nsize, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    mulKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    divKernel<<<gridSize, blockSize>>>(nsize, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    divKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    daxpyKernel<<<gridSize, blockSize>>>(nsize, alpha, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
sumNMatrixKernel(const size_t nsize, const size_t n, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sumNMatrixKernel<<<gridSize, blockSize>>>(nsize, n, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceSumKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceSumKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceSumKernel(const size_t nsize, const unsigned int *mask, const TData *x,
                TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceSumKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMaxKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceMaxKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMaxKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMaxKernel(const size_t nsize, const unsigned int *mask, const TData *x,
                TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceMaxKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMaxKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMinKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceMinKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMinKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMinKernel(const size_t nsize, const unsigned int *mask, const TData *x,
                TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    reduceMinKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMinKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
ddotKernel(const size_t nsize, const TData *x, const TData *y, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    ddotKernel<true><<<gridSize, blockSize>>>(nsize, x, y, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
ddotKernel(const size_t nsize, const unsigned int *mask, const TData *x,
           const TData *y, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    ddotKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, y, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l1normKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    l1normKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l1normKernel(const size_t nsize, const unsigned int *mask, const TData *x,
             TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    l1normKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l2normKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    l2normKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l2normKernel(const size_t nsize, const unsigned int *mask, const TData *x,
             TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    l2normKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
lpnormKernel(const size_t nsize, const unsigned int p, const TData *x,
             TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    lpnormKernel<true><<<gridSize, blockSize>>>(nsize, p, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
lpnormKernel(const size_t nsize, const unsigned int p, const unsigned int *mask,
             const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    lpnormKernel<true><<<gridSize, blockSize>>>(nsize, p, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceSumKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
linfnormKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    linfnormKernel<true><<<gridSize, blockSize>>>(nsize, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMaxKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
linfnormKernel(const size_t nsize, const unsigned int *mask, const TData *x,
               TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize = sizeof(TData) * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    linfnormKernel<true><<<gridSize, blockSize>>>(nsize, mask, x, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceMaxKernel<init><<<1, gridSize>>>(gridSize, buffer, out);
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar

#endif
