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

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Nektar
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
negKernel(const size_t nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 y[idx] = -x[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
addKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 z[idx] = x[idx] + y[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
subKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 z[idx] = x[idx] - y[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 y[idx] = alpha * x[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 z[idx] = x[idx] * y[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 y[idx] = alpha / x[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 z[idx] = x[idx] / y[idx];
                             }
                         });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t idx0   = indx.get_global_id(0);
                             size_t stride = indx.get_global_range(0);

                             for (size_t idx = idx0; idx < nsize; idx += stride)
                             {
                                 z[idx] = alpha * x[idx] + y[idx];
                             }
                         });
    });
}

template <typename TData>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t nsize, const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t nsize, const TData *x, TData *out)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void ddotKernel(const unsigned int gridSize, const unsigned int blockSize,
                const size_t nsize, const TData *x, const TData *y, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void l1normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const size_t nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void l2normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const size_t nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void lpnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const size_t nsize, const int p, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
                {
                    out[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < nsize)
                {
                    tmp += sycl::pown(sycl::fabs(x[gid]), p);
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData>
void linfnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                    const size_t nsize, const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = indx.get_global_id(0);

                if (lid == 0)
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
                    out[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceSumKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    reduceSumKernel(gridSize, blockSize, nsize, x, buffer);
    reduceSumKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMaxKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    reduceMaxKernel(gridSize, blockSize, nsize, x, buffer);
    reduceMaxKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMinKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    reduceMinKernel(gridSize, blockSize, nsize, x, buffer);
    reduceMinKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
ddotKernel(const size_t nsize, const TData *x, const TData *y, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    ddotKernel(gridSize, blockSize, nsize, x, y, buffer);
    reduceSumKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l1normKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    l1normKernel(gridSize, blockSize, nsize, x, buffer);
    reduceSumKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l2normKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    l2normKernel(gridSize, blockSize, nsize, x, buffer);
    reduceSumKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
lpnormKernel(const size_t nsize, const unsigned int p, const TData *x,
             TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    lpnormKernel(gridSize, blockSize, nsize, p, x, buffer);
    reduceSumKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
linfnormKernel(const size_t nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = buffer + gridSize;
    linfnormKernel(gridSize, blockSize, nsize, x, buffer);
    reduceMaxKernel(1, gridSize, gridSize, buffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

} // namespace Nektar

#endif
