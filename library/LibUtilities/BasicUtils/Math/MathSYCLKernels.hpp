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

#include "LibUtilities/Memory/MemoryAlloc.hpp"
#include "Operators/Common/Spaces.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Nektar::Math
{

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
        cgh.parallel_for(sycl::range<1>(nsize),
                          [=](sycl::id<1> indx) { y[indx] = alpha / x[indx]; });
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
            z[indx] = x[indx] / y[indx];
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

template <bool init, typename TData>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += x[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const uint8_t *mask, const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += mask[gid] * x[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::lowest();

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < nsize)
                {
                    tmp = sycl::fmax(tmp, x[gid]);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_max<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const uint8_t *mask, const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::lowest();

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < nsize)
                {
                    tmp = mask[gid] ? sycl::fmax(tmp, x[gid]) : tmp;
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_max<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const TData *x, TData *out)
{
    constexpr TData max = std::numeric_limits<TData>::max();

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = max;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = max;
                while (gid < nsize)
                {
                    tmp = sycl::fmin(tmp, x[gid]);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmin(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_min<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int streamID, const size_t nsize,
                     const uint8_t *mask, const TData *x, TData *out)
{
    constexpr TData max = std::numeric_limits<TData>::max();

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = max;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = max;
                while (gid < nsize)
                {
                    tmp = mask[gid] ? sycl::fmin(tmp, x[gid]) : tmp;
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmin(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_min<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void ddotKernel(const unsigned int gridSize, const unsigned int blockSize,
                const unsigned int streamID, const size_t nsize, const TData *x,
                const TData *y, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += x[gid] * y[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void ddotKernel(const unsigned int gridSize, const unsigned int blockSize,
                const unsigned int streamID, const size_t nsize,
                const uint8_t *mask, const TData *x, const TData *y, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += mask[gid] * x[gid] * y[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void l1normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize,
                  const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += sycl::fabs(x[gid]);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void l1normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize,
                  const uint8_t *mask, const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += mask[gid] * sycl::fabs(x[gid]);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void l2normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize,
                  const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += x[gid] * x[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void l2normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize,
                  const uint8_t *mask, const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += mask[gid] * x[gid] * x[gid];
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void lpnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize, const int p,
                  const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
#if defined(__ADAPTIVECPP__) || defined(__DPCPP_COMPILER)
                    tmp += sycl::pown(sycl::fabs(x[gid]), p);
#else
                    tmp += sycl::pow(sycl::fabs(x[gid]), p);
#endif
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void lpnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int streamID, const size_t nsize, const int p,
                  const uint8_t *mask, const TData *x, TData *out)
{
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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
#if defined(__ADAPTIVECPP__) || defined(__DPCPP_COMPILER)
                    tmp += mask[gid] * sycl::pown(sycl::fabs(x[gid]), p);
#else
                    tmp += mask[gid] * sycl::pow(sycl::fabs(x[gid]), p);
#endif
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_add<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void linfnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                    const unsigned int streamID, const size_t nsize,
                    const TData *x, TData *out)
{
    constexpr TData min = 0.0;

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < nsize)
                {
                    tmp = sycl::fmax(tmp, sycl::fabs(x[gid]));
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_max<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData>
void linfnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                    const unsigned int streamID, const size_t nsize,
                    const uint8_t *mask, const TData *x, TData *out)
{
    constexpr TData min = 0.0;

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
                size_t gid       = indx.get_global_id(0);

                if (init && lid == 0)
                {
                    out[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < nsize)
                {
                    tmp = mask[gid] ? sycl::fmax(tmp, sycl::fabs(x[gid])) : tmp;
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    atomic_max<NektarSpaces::LocalScope>(
                        out + indx.get_group(0), scratch[0]);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceSumKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceSumKernel<true>(gridSize, blockSize, streamID, nsize, mask, x,
                          buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceMaxKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceMaxKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceMaxKernel<true>(gridSize, blockSize, streamID, nsize, mask, x,
                          buffer);
    reduceMaxKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceMinKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceMinKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceMinKernel<true>(gridSize, blockSize, streamID, nsize, mask, x,
                          buffer);
    reduceMinKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    ddotKernel<true>(gridSize, blockSize, streamID, nsize, x, y, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    ddotKernel<true>(gridSize, blockSize, streamID, nsize, mask, x, y, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    l1normKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    l1normKernel<true>(gridSize, blockSize, streamID, nsize, mask, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    l2normKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    l2normKernel<true>(gridSize, blockSize, streamID, nsize, mask, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
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
                             reducer += std::pow(sycl::fabs(x[indx]), p);
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    lpnormKernel<true>(gridSize, blockSize, streamID, nsize, p, x, buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
#endif
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
                         const uint8_t *mask, const TData *x, TData *out,
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
                             reducer +=
                                 mask[indx] * std::pow(sycl::fabs(x[indx]), p);
                         });
    });
    SYCLQueue::SetEvent(streamID, e);
#else
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    lpnormKernel<true>(gridSize, blockSize, streamID, nsize, p, mask, x,
                       buffer);
    reduceSumKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    linfnormKernel<true>(gridSize, blockSize, streamID, nsize, x, buffer);
    reduceMaxKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
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
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize = sizeof(TData) * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    linfnormKernel<true>(gridSize, blockSize, streamID, nsize, mask, x, buffer);
    reduceMaxKernel<init>(1, gridSize, streamID, gridSize, buffer, out);
#endif
}

} // namespace Nektar::Math

#endif
