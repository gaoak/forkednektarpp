///////////////////////////////////////////////////////////////////////////////
//
// File: MathCUDAKernels.cuh
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

template <typename TData, bool vl = true>
__global__ void reduceSumKernel(const unsigned int nsize, const TData *x,
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

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = 0.0;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            v4 += reinterpret_cast<const float4 *>(x)[tid];
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            v2 += reinterpret_cast<const double2 *>(x)[tid];
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v += x[tid];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v += x[tid];
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void reduceMaxKernel(const unsigned int nsize, const TData *x,
                                TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = ::cuda::std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = min;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v               = max(v, max(max(v4.x, v4.y), max(v4.z, v4.w)));
        }
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v                = max(v, max(v2.x, v2.y));
        }
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v = max(v, x[tid]);
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v                = max(v, x[tid]);
        }
    }

    warp.sync();
    v = max(v, warp.shfl_down(v, 16)); // |
    v = max(v, warp.shfl_down(v, 8));  // | warp level
    v = max(v, warp.shfl_down(v, 4));  // | reduce here
    v = max(v, warp.shfl_down(v, 2));  // |
    v = max(v, warp.shfl_down(v, 1));  // |

    if (warp.thread_rank() == 0)
    {
        atomicMax(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void reduceMinKernel(const unsigned int nsize, const TData *x,
                                TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData max             = ::cuda::std::numeric_limits<TData>::max();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = max;

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = max;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v               = min(v, min(min(v4.x, v4.y), min(v4.z, v4.w)));
        }
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v                = min(v, min(v2.x, v2.y));
        }
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v = min(v, x[tid]);
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v                = min(v, x[tid]);
        }
    }

    warp.sync();
    v = min(v, warp.shfl_down(v, 16)); // |
    v = min(v, warp.shfl_down(v, 8));  // | warp level
    v = min(v, warp.shfl_down(v, 4));  // | reduce here
    v = min(v, warp.shfl_down(v, 2));  // |
    v = min(v, warp.shfl_down(v, 1));  // |

    if (warp.thread_rank() == 0)
    {
        atomicMin(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void ddotKernel(const unsigned int nsize, const TData *x,
                           const TData *y, TData *out)
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

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 x4 = reinterpret_cast<const float4 *>(x)[tid];
            const float4 y4 = reinterpret_cast<const float4 *>(y)[tid];
            v4 += x4 * y4;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
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
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v += x[tid] * y[tid];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v += x[tid] * y[tid];
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void l1normKernel(const unsigned int nsize, const TData *x,
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

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = 0.0;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += make_float4(abs(tmp.x), abs(tmp.y), abs(tmp.z), abs(tmp.w));
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += make_double2(abs(tmp.x), abs(tmp.y));
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v += abs(x[tid]);
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v += abs(x[tid]);
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void l2normKernel(const unsigned int nsize, const TData *x,
                             TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int vecsize  = (16u / sizeof(TData));
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

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += tmp * tmp;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += tmp * tmp;
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v += x[tid] * x[tid];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v += x[tid] * x[tid];
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void lpnormKernel(const unsigned int nsize, const int p,
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

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = 0.0;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(x)[tid];
            v4 += make_float4(pow(abs(tmp.x), p), pow(abs(tmp.y), p),
                              pow(abs(tmp.z), p), pow(abs(tmp.w), p));
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(x)[tid];
            v2 += make_double2(pow(abs(tmp.x), p), pow(abs(tmp.y), p));
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v += pow(abs(x[tid]), p);
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v += pow(abs(x[tid]), p);
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&out[block.group_index().x], v);
    }
}

template <typename TData, bool vl = true>
__global__ void linfnormKernel(const unsigned int nsize, const TData *x,
                               TData *out)
{
    // Implementation based on reduce7_vl of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = ::cuda::std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;
    constexpr unsigned int vecsize  = (16u / sizeof(TData));

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if (block.thread_rank() == 0)
    {
        out[block.group_index().x] = min;
    }

    block.sync();

    if constexpr (vl && std::is_same_v<TData, float>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const float4 v4 = reinterpret_cast<const float4 *>(x)[tid];
            v               = max(v,
                                  max(max(abs(v4.x), abs(v4.y)), max(abs(v4.z), abs(v4.w))));
        }
    }
    else if constexpr (vl && std::is_same_v<TData, double>)
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize / vecsize;
             tid += grid.size())
        {
            const double2 v2 = reinterpret_cast<const double2 *>(x)[tid];
            v                = max(v, max(abs(v2.x), abs(v2.y)));
        }
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < nsize;
             tid += grid.size())
        {
            v = max(v, abs(x[tid]));
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < nsize % vecsize)
        {
            unsigned int tid = nsize - 1u - grid.thread_rank();
            v                = max(v, abs(x[tid]));
        }
    }

    warp.sync();
    v = max(v, warp.shfl_down(v, 16)); // |
    v = max(v, warp.shfl_down(v, 8));  // | warp level
    v = max(v, warp.shfl_down(v, 4));  // | reduce here
    v = max(v, warp.shfl_down(v, 2));  // |
    v = max(v, warp.shfl_down(v, 1));  // |

    if (warp.thread_rank() == 0)
    {
        atomicMax(&out[block.group_index().x], v);
    }
}

// Launchers for the kernels

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    negKernel(const unsigned int nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    negKernel<<<gridSize, blockSize>>>(nsize, x, y);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    addKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    addKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    subKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    subKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    daxpyKernel(const unsigned int nsize, const TData alpha, const TData *x,
                const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    daxpyKernel<<<gridSize, blockSize>>>(nsize, alpha, x, y, z);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    divKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    vdivKernel<<<gridSize, blockSize>>>(nsize, x, y, z);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceSumKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    cudaMemset(cudaBuffer, 0, sizeof(TData) * gridSize);
    cudaMemset(out, 0, sizeof(TData));
    reduceSumKernel<TData>
        <<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceSumKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceMaxKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    reduceMaxKernel<TData>
        <<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceMaxKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    reduceMinKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    reduceMinKernel<TData>
        <<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceMinKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    ddotKernel(const unsigned int nsize, const TData *x, const TData *y,
               TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    ddotKernel<TData>
        <<<gridSize, blockSize>>>(nsize, x, y, (TData *)cudaBuffer);
    reduceSumKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    l1normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    l1normKernel<TData><<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceSumKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    l2normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    l2normKernel<TData><<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceSumKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    lpnormKernel(const unsigned int nsize, const unsigned int p, const TData *x,
                 TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    lpnormKernel<TData>
        <<<gridSize, blockSize>>>(nsize, p, x, (TData *)cudaBuffer);
    reduceSumKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    linfnormKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::CUDA::defaultGridSize;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }
    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    linfnormKernel<TData>
        <<<gridSize, blockSize>>>(nsize, x, (TData *)cudaBuffer);
    reduceMaxKernel<TData>
        <<<1, gridSize>>>(gridSize, (TData *)cudaBuffer, d_out);
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

} // namespace Nektar

#endif
