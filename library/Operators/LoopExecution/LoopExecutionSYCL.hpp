///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionSYCL.hpp
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

namespace Nektar
{

static void *syclBuffer = nullptr;

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    Nektar::atomic_add<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    Nektar::atomic_sub<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    Nektar::atomic_max<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    Nektar::atomic_min<Scope>(dest, val);
}

template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const size_t begin, const size_t end, const Functor &functor)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> indx) {
                             size_t i = begin + indx.get_global_id(0);

                             while (i < end)
                             {
                                 functor(i);
                                 i += indx.get_global_range(0);
                             }
                         });
    });

#if defined(SYCL_ENABLE_SERIAL)
    Q.wait();
#endif
}

template <typename TData, typename Functor>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < end)
                {
                    functor(gid, tmp);
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
                    buffer[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData, typename Functor>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < end)
                {
                    functor(gid, tmp);
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
                    buffer[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename TData, typename Functor>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] = max;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = max;
                while (gid < end)
                {
                    functor(gid, tmp);
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
                    buffer[indx.get_group(0)] = scratch[0];
                }
            });
    });
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    using TData = typename Reduction::value_type;

    if (syclBuffer == nullptr)
    {
        const unsigned int syclBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(syclBufferSize);
        syclBuffer =
            sycl::malloc_device(syclBufferSize, SYCLQueue::GetInstance());
        GetDeviceProperties::TotalGlobalMemory() -= syclBufferSize;
    }

    TData *buffer = (TData *)syclBuffer;
    TData *d_out  = (TData *)syclBuffer + gridSize;
    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        reduceSumKernel(gridSize, blockSize, begin, end, buffer, functor);
        reduceSumKernel(1, gridSize, 0, gridSize, out,
                        [=](const size_t i, TData &ans) { ans += buffer[i]; });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        reduceMaxKernel(gridSize, blockSize, begin, end, buffer, functor);
        reduceMaxKernel(1, gridSize, 0, gridSize, out,
                        [=](const size_t i, TData &ans) {
                            ans = sycl::max(ans, buffer[i]);
                        });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        reduceMinKernel(gridSize, blockSize, begin, end, buffer, functor);
        reduceMinKernel(1, gridSize, 0, gridSize, out,
                        [=](const size_t i, TData &ans) {
                            ans = sycl::min(ans, buffer[i]);
                        });
    }
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
}

} // namespace Nektar

#endif
