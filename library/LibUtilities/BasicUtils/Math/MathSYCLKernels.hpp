///////////////////////////////////////////////////////////////////////////////
//
// File: MathSYCLKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace Nektar::Math
{

// The 16 / sizeof(TData) entries of x from 16 / sizeof(TData) * tid in one
// load, for float or double.
template <typename TData>
static inline sycl::vec<TData, 16 / sizeof(TData)> vecLoad(const TData *x,
                                                           const size_t tid)
{
    return reinterpret_cast<const sycl::vec<TData, 16 / sizeof(TData)> *>(
        x)[tid];
}

// The vecSize mask entries matching a vecLoad of TData.
template <typename TData>
using VecMask = sycl::vec<uint8_t, 16 / sizeof(TData)>;

// The vecSize entries of mask matching vecLoad of TData at tid, in one load.
template <typename TData>
static inline VecMask<TData> vecLoadMask(const uint8_t *mask, const size_t tid)
{
    return reinterpret_cast<const VecMask<TData> *>(mask)[tid];
}

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

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                             NektarSpaces::Device::warpSize) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 y[idx] = sycl::fabs(x[idx]);
                             }
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void negKernel(const size_t nsize, const TData *x, TData *y,
                      const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                          [=](sycl::id<1> indx) { y[indx] = -x[indx]; });
    });
    SYCLQueue::SetEvent(streamID, e);
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

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                             NektarSpaces::Device::warpSize) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 // sycl::sqrt selects the device overload
                                 // (float/double)
                                 y[idx] = sycl::sqrt(x[idx]);
                             }
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void addKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = x[indx] + y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void addScalarKernel(const size_t nsize, const TData alpha,
                            const TData *x, TData *y,
                            const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                          [=](sycl::id<1> indx) { y[indx] = x[indx] + alpha; });
    });
    SYCLQueue::SetEvent(streamID, e);
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
    const size_t prefix = (nelmt / width) * width * ndata;

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(
            sycl::range<2>(ncomp, compSize), [=](sycl::id<2> indx) {
                const size_t i   = indx[1];
                const size_t idx = indx[0] * compSize + i;
                const bool padding =
                    i >= prefix &&
                    (i / (ndata * width)) * width + i % width >= nelmt;
                y[idx] = padding ? TData(0) : x[idx] + alpha;
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void subKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = x[indx] - y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                          [=](sycl::id<1> indx) { y[indx] = alpha * x[indx]; });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = x[indx] * y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            y[indx] = (x[indx] == TData(0)) ? TData(0) : alpha / x[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z, const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = (y[indx] == TData(0)) ? TData(0) : x[indx] / y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
                        const TData *y, TData *z,
                        const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = alpha * x[indx] + y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void daxpbyKernel(const size_t nsize, const TData alpha, const TData *x,
                         const TData beta, const TData *y, TData *z,
                         const unsigned int streamID = 0)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            z[indx] = alpha * x[indx] + beta * y[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
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
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            w[indx] = alpha * x[indx] + beta * y[indx] + z[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
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
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize), [=](sycl::id<1> indx) {
            w[indx] = alpha * x[indx] + beta * y[indx] + gamma * z[indx];
        });
    });
    SYCLQueue::SetEvent(streamID, e);
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
template <ReduceOp op, typename TData> inline constexpr TData reduceIdentity()
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

// Atomic combination of val into *dest by a reduction, within a work-group.
template <ReduceOp op, typename TData>
static inline void reduceAtomic(TData *const dest, const TData val)
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
    inline TData operator()(const TData x)
    {
        return x;
    }
};

template <typename TData> struct AbsOp
{
    inline TData operator()(const TData x)
    {
        return sycl::fabs(x);
    }
};

template <typename TData> struct SquareOp
{
    inline TData operator()(const TData x)
    {
        return x * x;
    }
};

template <typename TData> struct MultiplyOp
{
    inline TData operator()(const TData x, const TData y)
    {
        return x * y;
    }
};

// Transforms of the entries reduceNKernel reduces, writing value s of an entry
// to vals[s].
template <typename TData> struct IdentityEachOp
{
    template <typename... TIn>
    inline void operator()(TData *vals, const TIn... x)
    {
        unsigned int s = 0;
        ((vals[s++] = x), ...);
    }
};

template <typename TData> struct MultiplyFirstOp
{
    template <typename... TIn>
    inline void operator()(TData *vals, const TData x, const TIn... y)
    {
        unsigned int s = 0;
        ((vals[s++] = x * y), ...);
    }
};

// Combination of two values by a reduction.
template <ReduceOp op, typename TData>
static inline TData reduceJoin(const TData a, const TData b)
{
    if constexpr (op == ReduceOp::Sum)
    {
        return a + b;
    }
    else if constexpr (op == ReduceOp::Max)
    {
        return sycl::fmax(a, b);
    }
    else if constexpr (op == ReduceOp::Min)
    {
        return sycl::fmin(a, b);
    }
    else if constexpr (op == ReduceOp::AbsMax)
    {
        return sycl::fmax(sycl::fabs(a), sycl::fabs(b));
    }
}

// Combination of v by reduction op with TransformOp<TData>() of each lane of
// the vecLoad results xv.
template <ReduceOp op, template <typename> typename TransformOp, typename TData,
          typename... TVec>
static inline TData vecReduceJoin(const TData v, const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        return reduceJoin<op>(
            v, reduceJoin<op>(reduceJoin<op>(TransformOp<TData>()(xv[0]...),
                                             TransformOp<TData>()(xv[1]...)),
                              reduceJoin<op>(TransformOp<TData>()(xv[2]...),
                                             TransformOp<TData>()(xv[3]...))));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return reduceJoin<op>(v,
                              reduceJoin<op>(TransformOp<TData>()(xv[0]...),
                                             TransformOp<TData>()(xv[1]...)));
    }
}

// Combination of v[0, nred) by reduction op with the values
// TransformOp<TData>()(vals, x...) writes to vals[0, nred).
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
static inline void reduceNJoin(TData *v, const TIn... x)
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
static inline void vecReduceNJoin(TData *v, const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        TData a[nred], b[nred], c[nred], d[nred];
        TransformOp<TData>()(a, xv[0]...);
        TransformOp<TData>()(b, xv[1]...);
        TransformOp<TData>()(c, xv[2]...);
        TransformOp<TData>()(d, xv[3]...);
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
        TransformOp<TData>()(a, xv[0]...);
        TransformOp<TData>()(b, xv[1]...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s], reduceJoin<op>(a[s], b[s]));
        }
    }
}

// As vecReduceJoin, over the lanes the vecLoadMask result m selects.
template <ReduceOp op, template <typename> typename TransformOp, typename TData,
          typename... TVec>
static inline TData vecReduceJoinMasked(const TData v, const VecMask<TData> m,
                                        const TVec... xv)
{
    constexpr TData identity = reduceIdentity<op, TData>();

    if constexpr (std::is_same_v<TData, float>)
    {
        return reduceJoin<op>(
            v, reduceJoin<op>(
                   reduceJoin<op>(
                       m[0] ? TransformOp<TData>()(xv[0]...) : identity,
                       m[1] ? TransformOp<TData>()(xv[1]...) : identity),
                   reduceJoin<op>(
                       m[2] ? TransformOp<TData>()(xv[2]...) : identity,
                       m[3] ? TransformOp<TData>()(xv[3]...) : identity)));
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return reduceJoin<op>(
            v,
            reduceJoin<op>(m[0] ? TransformOp<TData>()(xv[0]...) : identity,
                           m[1] ? TransformOp<TData>()(xv[1]...) : identity));
    }
}

// TransformOp<TData>()(vals, x...) if m, otherwise the identity of op, in
// vals[0, nred).
template <ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
static inline void maskTransformN(TData *vals, const uint8_t m, const TIn... x)
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
static inline void vecReduceNJoinMasked(TData *v, const VecMask<TData> m,
                                        const TVec... xv)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        TData a[nred], b[nred], c[nred], d[nred];
        maskTransformN<op, nred, TransformOp>(a, m[0], xv[0]...);
        maskTransformN<op, nred, TransformOp>(b, m[1], xv[1]...);
        maskTransformN<op, nred, TransformOp>(c, m[2], xv[2]...);
        maskTransformN<op, nred, TransformOp>(d, m[3], xv[3]...);
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
        maskTransformN<op, nred, TransformOp>(a, m[0], xv[0]...);
        maskTransformN<op, nred, TransformOp>(b, m[1], xv[1]...);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduceJoin<op>(v[s], reduceJoin<op>(a[s], b[s]));
        }
    }
}

// Partial result of reduction op over TransformOp<TData>()(x[i]...) per
// work-group, in out[g] for work-group g, which is set to the identity first if
// init.
template <bool init, ReduceOp op, template <typename> typename TransformOp,
          typename TData, typename... TIn>
void reduceKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize, TData *out,
                  const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceKernel: inputs of the output type.");

    constexpr TData identity = reduceIdentity<op, TData>();
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(sycl::range<1>(blockSize),
                                                   cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                NektarSpaces::Device::warpSize) {
                auto scratch =
                    scratchpad
                        .template get_multi_ptr<sycl::access::decorated::yes>()
                        .get();
                const size_t lid = indx.get_local_id(0);
                const size_t gid = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = identity;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = identity;
                if constexpr (vecSize > 1u)
                {
                    for (size_t tid = gid; tid < nsize / vecSize;
                         tid += indx.get_global_range(0))
                    {
                        tmp = vecReduceJoin<op, TransformOp>(
                            tmp, vecLoad(x, tid)...);
                    }
                }
                else
                {
                    for (size_t tid = gid; tid < nsize;
                         tid += indx.get_global_range(0))
                    {
                        tmp = reduceJoin<op>(tmp,
                                              TransformOp<TData>()(x[tid]...));
                    }
                }

                // process final elements (if there are any)
                if (gid < nsize % vecSize)
                {
                    const size_t tid = nsize - 1u - gid;
                    tmp = reduceJoin<op>(tmp, TransformOp<TData>()(x[tid]...));
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            reduceJoin<op>(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    reduceAtomic<op>(out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

// As reduceKernel, over the entries mask selects.
template <bool init, ReduceOp op, template <typename> typename TransformOp,
          typename TData, typename... TIn>
void reduceKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize,
                  const uint8_t *mask, TData *out, const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceKernel: inputs of the output type.");

    constexpr TData identity = reduceIdentity<op, TData>();
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(sycl::range<1>(blockSize),
                                                   cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                NektarSpaces::Device::warpSize) {
                auto scratch =
                    scratchpad
                        .template get_multi_ptr<sycl::access::decorated::yes>()
                        .get();
                const size_t lid = indx.get_local_id(0);
                const size_t gid = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = identity;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = identity;
                if constexpr (vecSize > 1u)
                {
                    for (size_t tid = gid; tid < nsize / vecSize;
                         tid += indx.get_global_range(0))
                    {
                        tmp = vecReduceJoinMasked<op, TransformOp>(
                            tmp, vecLoadMask<TData>(mask, tid),
                            vecLoad(x, tid)...);
                    }
                }
                else
                {
                    for (size_t tid = gid; tid < nsize;
                         tid += indx.get_global_range(0))
                    {
                        tmp = mask[tid]
                                   ? reduceJoin<op>(
                                        tmp, TransformOp<TData>()(x[tid]...))
                                   : tmp;
                    }
                }

                // process final elements (if there are any)
                if (gid < nsize % vecSize)
                {
                    const size_t tid = nsize - 1u - gid;
                    tmp              = mask[tid] ? reduceJoin<op>(
                                          tmp, TransformOp<TData>()(x[tid]...))
                                                  : tmp;
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            reduceJoin<op>(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    reduceAtomic<op>(out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

// Partial results of nred reductions op per work-group, of the values
// TransformOp<TData>()(vals, x[i]...) writes to vals[0, nred) for each
// entry i: result s of work-group g is in out[s * gridSize + g], which is set
// to the identity first if init.
template <bool init, ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
void reduceNKernel(const unsigned int gridSize, const unsigned int blockSize,
                   const unsigned int streamID, const size_t nsize, TData *out,
                   const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceNKernel: inputs of the output type.");

    constexpr TData identity = reduceIdentity<op, TData>();
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(
            sycl::range<1>(nred * blockSize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                NektarSpaces::Device::warpSize) {
                auto scratch =
                    scratchpad
                        .template get_multi_ptr<sycl::access::decorated::yes>()
                        .get();
                const size_t lid = indx.get_local_id(0);
                const size_t gid = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        out[s * gridSize + indx.get_group(0)] = identity;
                    }
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp[nred];
                for (unsigned int s = 0; s < nred; ++s)
                {
                    tmp[s] = identity;
                }

                // Vector loads need every input 16-byte aligned, which the
                // final stage's partial results past the first sum need not
                // be.
                size_t nvec = 0;
                if constexpr (vecSize > 1u)
                {
                    if (((reinterpret_cast<uintptr_t>(x) % 16u == 0u) && ...))
                    {
                        nvec = nsize / vecSize;
                    }

                    for (size_t tid = gid; tid < nvec;
                         tid += indx.get_global_range(0))
                    {
                        vecReduceNJoin<op, nred, TransformOp>(
                            tmp, vecLoad(x, tid)...);
                    }
                }

                // process the remaining entries
                for (size_t tid = nvec * vecSize + gid; tid < nsize;
                     tid += indx.get_global_range(0))
                {
                    reduceNJoin<op, nred, TransformOp>(tmp, x[tid]...);
                }
                for (unsigned int s = 0; s < nred; ++s)
                {
                    scratch[s * blockSize + lid] = tmp[s];
                }

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        for (unsigned int s = 0; s < nred; ++s)
                        {
                            scratch[s * blockSize + lid] = reduceJoin<op>(
                                scratch[s * blockSize + lid],
                                scratch[s * blockSize + lid + n]);
                        }
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        reduceAtomic<op>(out + s * gridSize + indx.get_group(0),
                                          scratch[s * blockSize]);
                    }
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

// As reduceNKernel, over the entries mask selects.
template <bool init, ReduceOp op, unsigned int nred,
          template <typename> typename TransformOp, typename TData,
          typename... TIn>
void reduceNKernel(const unsigned int gridSize, const unsigned int blockSize,
                   const unsigned int streamID, const size_t nsize,
                   const uint8_t *mask, TData *out, const TIn *...x)
{
    static_assert((std::is_same_v<TIn, TData> && ...),
                  "reduceNKernel: inputs of the output type.");

    constexpr TData identity = reduceIdentity<op, TData>();
    constexpr unsigned int vecSize =
        std::is_same_v<TData, float> || std::is_same_v<TData, double>
            ? (16u / sizeof(TData))
            : 1u;

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(
            sycl::range<1>(nred * blockSize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                NektarSpaces::Device::warpSize) {
                auto scratch =
                    scratchpad
                        .template get_multi_ptr<sycl::access::decorated::yes>()
                        .get();
                const size_t lid = indx.get_local_id(0);
                const size_t gid = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        out[s * gridSize + indx.get_group(0)] = identity;
                    }
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp[nred];
                for (unsigned int s = 0; s < nred; ++s)
                {
                    tmp[s] = identity;
                }

                size_t nvec = 0;
                if constexpr (vecSize > 1u)
                {
                    nvec = nsize / vecSize;

                    for (size_t tid = gid; tid < nvec;
                         tid += indx.get_global_range(0))
                    {
                        vecReduceNJoinMasked<op, nred, TransformOp>(
                            tmp, vecLoadMask<TData>(mask, tid),
                            vecLoad(x, tid)...);
                    }
                }

                // process the remaining entries
                for (size_t tid = nvec * vecSize + gid; tid < nsize;
                     tid += indx.get_global_range(0))
                {
                    if (mask[tid])
                    {
                        reduceNJoin<op, nred, TransformOp>(tmp, x[tid]...);
                    }
                }
                for (unsigned int s = 0; s < nred; ++s)
                {
                    scratch[s * blockSize + lid] = tmp[s];
                }

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        for (unsigned int s = 0; s < nred; ++s)
                        {
                            scratch[s * blockSize + lid] = reduceJoin<op>(
                                scratch[s * blockSize + lid],
                                scratch[s * blockSize + lid + n]);
                        }
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        reduceAtomic<op>(out + s * gridSize + indx.get_group(0),
                                          scratch[s * blockSize]);
                    }
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(
            sycl::range<1>(nsize),
            sycl::reduction(out, sycl::plus<>(), initializer),
            [=](sycl::id<1> indx, auto &reducer) { reducer += x[indx]; });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += mask[indx] * x[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(
            sycl::range<1>(nsize),
            sycl::reduction(out, sycl::maximum<>(), initializer),
            [=](sycl::id<1> indx, auto &reducer) { reducer.combine(x[indx]); });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Max, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, buffer, x);
    reduceKernel<init, ReduceOp::Max, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::maximum<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             if (mask[indx])
                             {
                                 reducer.combine(x[indx]);
                             }
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Max, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::Max, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(
            sycl::range<1>(nsize),
            sycl::reduction(out, sycl::minimum<>(), initializer),
            [=](sycl::id<1> indx, auto &reducer) { reducer.combine(x[indx]); });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Min, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, buffer, x);
    reduceKernel<init, ReduceOp::Min, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::minimum<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             if (mask[indx])
                             {
                                 reducer.combine(x[indx]);
                             }
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Min, IdentityOp>(gridSize, blockSize, streamID,
                                                  nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::Min, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const TData *x, const TData *y,
                       TData *out, const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += x[indx] * y[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, MultiplyOp>(gridSize, blockSize, streamID,
                                                  nsize, buffer, x, y);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                       const TData *y, TData *out,
                       const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += mask[indx] * x[indx] * y[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, MultiplyOp>(gridSize, blockSize, streamID,
                                                  nsize, mask, buffer, x, y);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
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
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         sycl::reduction(out + 1, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &xy, auto &xz) {
                             const TData xi =
                                 mask ? mask[indx] * x[indx] : x[indx];
                             xy += xi * y[indx];
                             xz += xi * z[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    // Two partial sums per work-group, which internalReductionBufferSize
    // holds.
    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    if (mask)
    {
        reduceNKernel<true, ReduceOp::Sum, 2, MultiplyFirstOp>(
            gridSize, blockSize, streamID, nsize, mask, buffer, x, y, z);
    }
    else
    {
        reduceNKernel<true, ReduceOp::Sum, 2, MultiplyFirstOp>(
            gridSize, blockSize, streamID, nsize, buffer, x, y, z);
    }
    reduceNKernel<init, ReduceOp::Sum, 2, IdentityEachOp>(
        1, blockSize, streamID, gridSize, out, buffer, buffer + gridSize);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += sycl::fabs(x[indx]);
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, AbsOp>(gridSize, blockSize, streamID,
                                             nsize, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += mask[indx] * sycl::fabs(x[indx]);
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, AbsOp>(gridSize, blockSize, streamID,
                                             nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += x[indx] * x[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, SquareOp>(gridSize, blockSize, streamID,
                                                nsize, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::plus<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer += mask[indx] * x[indx] * x[indx];
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::Sum, SquareOp>(gridSize, blockSize, streamID,
                                                nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::Sum, IdentityOp>(1, gridSize, streamID,
                                                  gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const TData *x, TData *out,
                           const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::maximum<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             reducer.combine(sycl::fabs(x[indx]));
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::AbsMax, IdentityOp>(
        gridSize, blockSize, streamID, nsize, buffer, x);
    reduceKernel<init, ReduceOp::AbsMax, IdentityOp>(1, gridSize, streamID,
                                                     gridSize, out, buffer);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const uint8_t *mask,
                           const TData *x, TData *out,
                           const unsigned int streamID = 0)
{
#if defined(SYCL_ENABLE_CPU)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    sycl::property_list initializer =
        init ? sycl::property_list{sycl::property::reduction::
                                       initialize_to_identity{}}
             : sycl::property_list{};
    sycl::event e = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(nsize),
                         sycl::reduction(out, sycl::maximum<>(), initializer),
                         [=](sycl::id<1> indx, auto &reducer) {
                             if (mask[indx])
                             {
                                 reducer.combine(sycl::fabs(x[indx]));
                             }
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(nsize);

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceKernel<true, ReduceOp::AbsMax, IdentityOp>(
        gridSize, blockSize, streamID, nsize, mask, buffer, x);
    reduceKernel<init, ReduceOp::AbsMax, IdentityOp>(1, gridSize, streamID,
                                                     gridSize, out, buffer);
#endif
}

} // namespace Nektar::Math

#endif
