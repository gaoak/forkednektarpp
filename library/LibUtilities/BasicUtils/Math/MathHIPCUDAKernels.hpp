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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)) ||           \
    (defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY))

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

namespace Nektar::Math
{

// The vecSize entries of x from vecSize * tid in one load, for float or
// double.
template <typename TData>
NEK_DEVICE_INLINE static auto vecLoad(const TData *x, const size_t tid)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        return reinterpret_cast<const float4 *>(x)[tid];
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return reinterpret_cast<const double2 *>(x)[tid];
    }
}

// The vecSize mask entries matching a vecLoad of TData.
template <typename TData>
using VecMask =
    std::conditional_t<std::is_same_v<TData, float>, uchar4, uchar2>;

// The vecSize entries of mask matching vecLoad of TData at tid, in one load.
template <typename TData>
NEK_DEVICE_INLINE static VecMask<TData> vecLoadMask(const uint8_t *mask,
                                                    const size_t tid)
{
    return reinterpret_cast<const VecMask<TData> *>(mask)[tid];
}

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
__global__ __launch_bounds__(blockSize) void addScalarKernel(const size_t nsize,
                                                             const TData alpha,
                                                             const TData *x,
                                                             TData *y)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        y[idx] = x[idx] + alpha;
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void addScalarKernel(
    const size_t compSize, const size_t prefix, const size_t nelmt,
    const unsigned int ndata, const unsigned int width, const TData alpha,
    const TData *x, TData *y)
{
    // One component per row of the grid.
    const size_t offset = blockIdx.y * compSize;
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx0; i < compSize; i += stride)
    {
        const bool padding =
            i >= prefix && (i / (ndata * width)) * width + i % width >= nelmt;
        y[offset + i] = padding ? TData(0) : x[offset + i] + alpha;
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
        y[idx] = (x[idx] == TData(0)) ? TData(0) : alpha / x[idx];
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
        z[idx] = (y[idx] == TData(0)) ? TData(0) : x[idx] / y[idx];
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
__global__ __launch_bounds__(blockSize) void daxpbyKernel(
    const size_t nsize, const TData alpha, const TData *x, const TData beta,
    const TData *y, TData *z)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        z[idx] = alpha * x[idx] + beta * y[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void daxpbypzKernel(
    const size_t nsize, const TData alpha, const TData *x, const TData beta,
    const TData *y, const TData *z, TData *w)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        w[idx] = alpha * x[idx] + beta * y[idx] + z[idx];
    }
}

template <typename TData,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void daxpbypczKernel(
    const size_t nsize, const TData alpha, const TData *x, const TData beta,
    const TData *y, const TData gamma, const TData *z, TData *w)
{
    const size_t idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        w[idx] = alpha * x[idx] + beta * y[idx] + gamma * z[idx];
    }
}

// Reductions of reduceKernel.
enum class ReduceOp
{
    Sum,
    Max,
    Min,
    AbsMax
};

// Identity of a reduction.
template <ReduceOp op, typename TData>
NEK_HOSTDEVICE_INLINE constexpr TData reduceIdentity()
{
    if constexpr (op == ReduceOp::Sum)
    {
        return 0;
    }
    else if constexpr (op == ReduceOp::Max)
    {
        return std::numeric_limits<TData>::lowest();
    }
    else if constexpr (op == ReduceOp::Min)
    {
        return std::numeric_limits<TData>::max();
    }
    else if constexpr (op == ReduceOp::AbsMax)
    {
        return 0;
    }
}

// Atomic combination of val into *dest by a reduction, within a thread block.
template <ReduceOp op, typename TData>
NEK_DEVICE_INLINE static void reduceAtomic(TData *const dest, const TData val)
{
    if constexpr (op == ReduceOp::Sum)
    {
        atomic_add<NektarSpaces::LocalScope>(dest, val);
    }
    else if constexpr (op == ReduceOp::Max)
    {
        atomic_max<NektarSpaces::LocalScope>(dest, val);
    }
    else if constexpr (op == ReduceOp::Min)
    {
        atomic_min<NektarSpaces::LocalScope>(dest, val);
    }
    else if constexpr (op == ReduceOp::AbsMax)
    {
        atomic_max<NektarSpaces::LocalScope>(dest, val);
    }
}

// Transforms of the entries reduceKernel reduces.
template <typename TData> struct IdentityOp
{
    NEK_DEVICE_INLINE TData operator()(const TData x)
    {
        return x;
    }
};

template <typename TData> struct AbsOp
{
    NEK_DEVICE_INLINE TData operator()(const TData x)
    {
        return std::abs(x);
    }
};

template <typename TData> struct SquareOp
{
    NEK_DEVICE_INLINE TData operator()(const TData x)
    {
        return x * x;
    }
};

template <typename TData> struct MultiplyOp
{
    NEK_DEVICE_INLINE TData operator()(const TData x, const TData y)
    {
        return x * y;
    }
};

// Transforms of the entries reduceNKernel reduces, writing value s of an entry
// to vals[s].
template <typename TData> struct IdentityEachOp
{
    template <typename... TIn>
    NEK_DEVICE_INLINE void operator()(TData *vals, const TIn... x)
    {
        unsigned int s = 0;
        ((vals[s++] = x), ...);
    }
};

template <typename TData> struct MultiplyFirstOp
{
    template <typename... TIn>
    NEK_DEVICE_INLINE void operator()(TData *vals, const TData x,
                                      const TIn... y)
    {
        unsigned int s = 0;
        ((vals[s++] = x * y), ...);
    }
};

// Combination of two values by a reduction.
template <ReduceOp op, typename TData>
NEK_DEVICE_INLINE static TData reduceJoin(const TData a, const TData b)
{
    if constexpr (op == ReduceOp::Sum)
    {
        return a + b;
    }
    else if constexpr (op == ReduceOp::Max)
    {
        return std::max(a, b);
    }
    else if constexpr (op == ReduceOp::Min)
    {
        return std::min(a, b);
    }
    else if constexpr (op == ReduceOp::AbsMax)
    {
        return std::max(std::abs(a), std::abs(b));
    }
}

// Combination of v by reduction op with TransformOp<TData>() of each lane of
// the vecLoad results xv.
template <ReduceOp op, template <typename> typename TransformOp, typename TData,
          typename... TVec>
NEK_DEVICE_INLINE static TData vecReduceJoin(const TData v, const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        return reduceJoin<op>(
            v, reduceJoin<op>(reduceJoin<op>(TransformOp<TData>()(xv.x...),
                                             TransformOp<TData>()(xv.y...)),
                              reduceJoin<op>(TransformOp<TData>()(xv.z...),
                                             TransformOp<TData>()(xv.w...))));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return reduceJoin<op>(v, reduceJoin<op>(TransformOp<TData>()(xv.x...),
                                                TransformOp<TData>()(xv.y...)));
    }
}

// Combination of v[0, nred) by reduction op with the values
// TransformOp<TData>()(vals, x...) writes to vals[0, nred).
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
NEK_DEVICE_INLINE static void reduceNJoin(TData *v, const TIn... x)
{
    TData vals[nred];
    TransformOp<TData>()(vals, x...);
    for (unsigned int s = 0; s < nred; ++s)
    {
        v[s] = reduceJoin<op>(v[s], vals[s]);
    }
}

// As reduceNJoin, with each lane of the vecLoad results xv.
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TVec>
NEK_DEVICE_INLINE static void vecReduceNJoin(TData *v, const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        TData a[nred], b[nred], c[nred], d[nred];
        TransformOp<TData>()(a, xv.x...);
        TransformOp<TData>()(b, xv.y...);
        TransformOp<TData>()(c, xv.z...);
        TransformOp<TData>()(d, xv.w...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s],
                                  reduceJoin<op>(reduceJoin<op>(a[s], b[s]),
                                                 reduceJoin<op>(c[s], d[s])));
        }
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        TData a[nred], b[nred];
        TransformOp<TData>()(a, xv.x...);
        TransformOp<TData>()(b, xv.y...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s], reduceJoin<op>(a[s], b[s]));
        }
    }
}

// As vecReduceJoin, over the lanes the vecLoadMask result m selects.
template <ReduceOp op, template <typename> typename TransformOp, typename TData,
          typename... TVec>
NEK_DEVICE_INLINE static TData vecReduceJoinMasked(const TData v,
                                                   const VecMask<TData> m,
                                                   const TVec... xv)
{
    constexpr TData identity = reduceIdentity<op, TData>();

    if constexpr (std::is_same_v<TData, float>)
    {
        return reduceJoin<op>(
            v,
            reduceJoin<op>(
                reduceJoin<op>(m.x ? TransformOp<TData>()(xv.x...) : identity,
                               m.y ? TransformOp<TData>()(xv.y...) : identity),
                reduceJoin<op>(m.z ? TransformOp<TData>()(xv.z...) : identity,
                               m.w ? TransformOp<TData>()(xv.w...)
                                   : identity)));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return reduceJoin<op>(
            v, reduceJoin<op>(m.x ? TransformOp<TData>()(xv.x...) : identity,
                              m.y ? TransformOp<TData>()(xv.y...) : identity));
    }
}

// TransformOp<TData>()(vals, x...) if m, otherwise the identity of op, in
// vals[0, nred).
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
NEK_DEVICE_INLINE static void maskTransformN(TData *vals, const uint8_t m,
                                             const TIn... x)
{
    TransformOp<TData>()(vals, x...);
    for (unsigned int s = 0; s < nred; ++s)
    {
        vals[s] = m ? vals[s] : reduceIdentity<op, TData>();
    }
}

// As vecReduceNJoin, over the lanes the vecLoadMask result m selects.
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TVec>
NEK_DEVICE_INLINE static void vecReduceNJoinMasked(TData *v,
                                                   const VecMask<TData> m,
                                                   const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        TData a[nred], b[nred], c[nred], d[nred];
        maskTransformN<op, nred, TransformOp>(a, m.x, xv.x...);
        maskTransformN<op, nred, TransformOp>(b, m.y, xv.y...);
        maskTransformN<op, nred, TransformOp>(c, m.z, xv.z...);
        maskTransformN<op, nred, TransformOp>(d, m.w, xv.w...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s],
                                  reduceJoin<op>(reduceJoin<op>(a[s], b[s]),
                                                 reduceJoin<op>(c[s], d[s])));
        }
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        TData a[nred], b[nred];
        maskTransformN<op, nred, TransformOp>(a, m.x, xv.x...);
        maskTransformN<op, nred, TransformOp>(b, m.y, xv.y...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s], reduceJoin<op>(a[s], b[s]));
        }
    }
}

// Partial result of reduction op over TransformOp<TData>()(x[i]...) per
// thread block, in out[b] for block b, which is set to the identity first if
// init.
template <bool init, ReduceOp op, template <typename> typename TransformOp,
          typename TData,
          unsigned int blockSize = NektarSpaces::Device::reductionBlockSize(),
          typename... TIn>
__global__ __launch_bounds__(blockSize) void reduceKernel(const size_t nsize,
                                                          TData *out,
                                                          const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceKernel: inputs of the output type.");

    constexpr TData identity        = reduceIdentity<op, TData>();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = identity;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = identity;
        }
    }

    block.sync();

    if constexpr (vecSize > 1u)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecSize;
             tid += grid.size())
        {
            v = vecReduceJoin<op, TransformOp>(v, vecLoad(x, tid)...);
        }
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v = reduceJoin<op>(v, TransformOp<TData>()(x[tid]...));
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecSize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v          = reduceJoin<op>(v, TransformOp<TData>()(x[tid]...));
    }

    warp.sync();
#pragma unroll
    for (unsigned int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        v = reduceJoin<op>(v, warp.shfl_down(v, offset));
    }

    // use atomics to combine over warps
    if (warp.thread_rank() == 0)
    {
        reduceAtomic<op>(&out[block.group_index().x], v);
    }
}

// As reduceKernel, over the entries mask selects.
template <bool init, ReduceOp op, template <typename> typename TransformOp,
          typename TData,
          unsigned int blockSize = NektarSpaces::Device::reductionBlockSize(),
          typename... TIn>
__global__ __launch_bounds__(blockSize) void reduceKernel(const size_t nsize,
                                                          const uint8_t *mask,
                                                          TData *out,
                                                          const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceKernel: inputs of the output type.");

    constexpr TData identity        = reduceIdentity<op, TData>();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = identity;

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            out[block.group_index().x] = identity;
        }
    }

    block.sync();

    if constexpr (vecSize > 1u)
    {
        for (size_t tid = grid.thread_rank(); tid < nsize / vecSize;
             tid += grid.size())
        {
            v = vecReduceJoinMasked<op, TransformOp>(
                v, vecLoadMask<TData>(mask, tid), vecLoad(x, tid)...);
        }
    }
    else
    {
        for (size_t tid = grid.thread_rank(); tid < nsize; tid += grid.size())
        {
            v = mask[tid] ? reduceJoin<op>(v, TransformOp<TData>()(x[tid]...))
                          : v;
        }
    }

    // process final elements (if there are any)
    if (grid.thread_rank() < nsize % vecSize)
    {
        size_t tid = nsize - 1u - grid.thread_rank();
        v = mask[tid] ? reduceJoin<op>(v, TransformOp<TData>()(x[tid]...)) : v;
    }

    warp.sync();
#pragma unroll
    for (unsigned int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        v = reduceJoin<op>(v, warp.shfl_down(v, offset));
    }

    // use atomics to combine over warps
    if (warp.thread_rank() == 0)
    {
        reduceAtomic<op>(&out[block.group_index().x], v);
    }
}

// Partial results of nred reductions op per thread block, of the values
// TransformOp<TData>()(vals, x[i]...) writes to vals[0, nred) for each
// entry i: result s of block b is in out[s * gridDim.x + b], which is set to
// the identity first if init.
template <bool init, ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          unsigned int blockSize = NektarSpaces::Device::reductionBlockSize(),
          typename... TIn>
__global__ __launch_bounds__(blockSize) void reduceNKernel(const size_t nsize,
                                                           TData *out,
                                                           const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceNKernel: inputs of the output type.");

    constexpr TData identity        = reduceIdentity<op, TData>();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v[nred];

    for (unsigned int s = 0; s < nred; ++s)
    {
        v[s] = identity;
    }

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            for (unsigned int s = 0; s < nred; ++s)
            {
                out[s * gridDim.x + block.group_index().x] = identity;
            }
        }
    }

    block.sync();

    // Vector loads need every input 16-byte aligned, which the final stage's
    // partial results past the first sum need not be.
    size_t nvec = 0;
    if constexpr (vecSize > 1u)
    {
        if (((reinterpret_cast<uintptr_t>(x) % 16u == 0u) && ...))
        {
            nvec = nsize / vecSize;
        }

        for (size_t tid = grid.thread_rank(); tid < nvec; tid += grid.size())
        {
            vecReduceNJoin<op, nred, TransformOp>(v, vecLoad(x, tid)...);
        }
    }

    // process the remaining entries
    for (size_t tid = nvec * vecSize + grid.thread_rank(); tid < nsize;
         tid += grid.size())
    {
        reduceNJoin<op, nred, TransformOp>(v, x[tid]...);
    }

    warp.sync();
#pragma unroll
    for (unsigned int s = 0; s < nred; ++s)
    {
#pragma unroll
        for (unsigned int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            v[s] = reduceJoin<op>(v[s], warp.shfl_down(v[s], offset));
        }
    }

    // use atomics to combine over warps
    if (warp.thread_rank() == 0)
    {
        for (unsigned int s = 0; s < nred; ++s)
        {
            reduceAtomic<op>(&out[s * gridDim.x + block.group_index().x], v[s]);
        }
    }
}

// As reduceNKernel, over the entries mask selects.
template <bool init, ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          unsigned int blockSize = NektarSpaces::Device::reductionBlockSize(),
          typename... TIn>
__global__ __launch_bounds__(blockSize) void reduceNKernel(const size_t nsize,
                                                           const uint8_t *mask,
                                                           TData *out,
                                                           const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceNKernel: inputs of the output type.");

    constexpr TData identity        = reduceIdentity<op, TData>();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v[nred];

    for (unsigned int s = 0; s < nred; ++s)
    {
        v[s] = identity;
    }

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            for (unsigned int s = 0; s < nred; ++s)
            {
                out[s * gridDim.x + block.group_index().x] = identity;
            }
        }
    }

    block.sync();

    size_t nvec = 0;
    if constexpr (vecSize > 1u)
    {
        nvec = nsize / vecSize;

        for (size_t tid = grid.thread_rank(); tid < nvec; tid += grid.size())
        {
            vecReduceNJoinMasked<op, nred, TransformOp>(
                v, vecLoadMask<TData>(mask, tid), vecLoad(x, tid)...);
        }
    }

    // process the remaining entries
    for (size_t tid = nvec * vecSize + grid.thread_rank(); tid < nsize;
         tid += grid.size())
    {
        if (mask[tid])
        {
            reduceNJoin<op, nred, TransformOp>(v, x[tid]...);
        }
    }

    warp.sync();
#pragma unroll
    for (unsigned int s = 0; s < nred; ++s)
    {
#pragma unroll
        for (unsigned int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            v[s] = reduceJoin<op>(v[s], warp.shfl_down(v[s], offset));
        }
    }

    // use atomics to combine over warps
    if (warp.thread_rank() == 0)
    {
        for (unsigned int s = 0; s < nred; ++s)
        {
            reduceAtomic<op>(&out[s * gridDim.x + block.group_index().x], v[s]);
        }
    }
}

// Launchers for the kernels

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void zeroKernel(const size_t nsize, TData *x,
                       const unsigned int streamID = 0)
{
    deviceMemset(x, 0, nsize * sizeof(TData), streamID);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void fillKernel(const size_t nsize, const TData &val, TData *x,
                       const unsigned int streamID = 0)
{
    deviceFill(x, val, nsize, streamID);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void copyKernel(const size_t nsize, const TData *x, TData *y,
                       const unsigned int streamID = 0)
{
    deviceMemcpy<DeviceToDevice>(y, x, nsize * sizeof(TData), streamID);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void absKernel(const size_t nsize, const TData *x, TData *y,
                      const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    absKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void negKernel(const size_t nsize, const TData *x, TData *y,
                      const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    negKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void sqrtKernel(const size_t nsize, const TData *x, TData *y,
                       const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    sqrtKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void addKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    addKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void addScalarKernel(const size_t nsize, const TData alpha,
                            const TData *x, TData *y,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    addScalarKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

// addScalarKernel over the ncomp components of compSize values of a block,
// stored as element groups of width lanes by ndata points of which the first
// nelmt elements are not padding: y = x + alpha, and zero in the padding.
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void addScalarKernel(const unsigned int ncomp, const size_t compSize,
                            const size_t nelmt, const unsigned int ndata,
                            const unsigned int width, const TData alpha,
                            const TData *x, TData *y,
                            const unsigned int streamID = 0)
{
    const size_t prefix          = (nelmt / width) * width * ndata;
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const dim3 gridSize((compSize + blockSize - 1u) / blockSize, ncomp);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    addScalarKernel<<<gridSize, blockSize, 0, stream>>>(
        compSize, prefix, nelmt, ndata, width, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void subKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    subKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    mulKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    mulKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    divKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    divKernel<<<gridSize, blockSize, 0, stream>>>(nsize, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
                        const TData *y, TData *z,
                        const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    daxpyKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, y, z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpbyKernel(const size_t nsize, const TData alpha, const TData *x,
                         const TData beta, const TData *y, TData *z,
                         const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    daxpbyKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, beta, y,
                                                     z);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpbypzKernel(const size_t nsize, const TData alpha,
                           const TData *x, const TData beta, const TData *y,
                           const TData *z, TData *w,
                           const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    daxpbypzKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, beta, y,
                                                       z, w);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpbypczKernel(const size_t nsize, const TData alpha,
                            const TData *x, const TData beta, const TData *y,
                            const TData gamma, const TData *z, TData *w,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    daxpbypczKernel<<<gridSize, blockSize, 0, stream>>>(nsize, alpha, x, beta,
                                                        y, gamma, z, w);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Max, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Max, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Max, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Max, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Min, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Min, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Min, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Min, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const TData *x, const TData *y,
                       TData *out, const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, MultiplyOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                       const TData *y, TData *out,
                       const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, MultiplyOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x, y);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

// ddot2Kernel computes, in one pass over x, out[0] (+)= (x, y) and out[1] (+)=
// (x, z): two dot products sharing x, over the entries mask selects unless it
// is null.
template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void ddot2Kernel(const size_t nsize, const uint8_t *mask, const TData *x,
                        const TData *y, const TData *z, TData *out,
                        const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    // Two partial sums per thread block, which internalReductionBufferSize
    // holds.
    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    if (mask)
    {
        reduceNKernel<true, ReduceOp::Sum, 2, MultiplyFirstOp>
            <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x, y, z);
    }
    else
    {
        reduceNKernel<true, ReduceOp::Sum, 2, MultiplyFirstOp>
            <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x, y, z);
    }
    CHECK_LAST_HIPCUDA_ERROR();
    reduceNKernel<init, ReduceOp::Sum, 2, IdentityEachOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer, buffer + gridSize);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, AbsOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, AbsOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, SquareOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, SquareOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::Sum, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const TData *x, TData *out,
                           const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::AbsMax, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::AbsMax, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const uint8_t *mask,
                           const TData *x, TData *out,
                           const unsigned int streamID = 0)
{
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::AbsMax, IdentityOp>
        <<<gridSize, blockSize, 0, stream>>>(nsize, mask, buffer, x);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceKernel<init, ReduceOp::AbsMax, IdentityOp>
        <<<1, blockSize, 0, stream>>>(gridSize, out, buffer);
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar::Math

#endif
