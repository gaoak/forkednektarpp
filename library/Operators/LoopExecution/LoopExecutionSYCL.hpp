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

#include "Operators/SYCLQueue.hpp"

namespace Nektar
{

static unsigned int syclBlockSize  = 256u;
static unsigned int syclGridSize   = 1024u;
static unsigned int syclBufferSize = 0u;
static void *syclBuffer            = nullptr;

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*dest)
        .fetch_add(val);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*dest)
        .fetch_sub(val);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*dest)
        .fetch_max(val);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*dest)
        .fetch_min(val);
}

// Simple 1D Range parallel_for
template <typename ExecSpace, typename Functor>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    parallel_for(const int begin, const int end, const Functor &functor)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<1>(syclGridSize * syclBlockSize, syclBlockSize),
             [=](sycl::nd_item<1> indx) {
                 unsigned int i = begin + indx.get_global_id(0);

                 while (i < end)
                 {
                     functor(i);
                     i += indx.get_global_range(0);
                 }
             });
     }).wait();
}

// Simple 1D Range parallel_reduce
template <typename TData, typename Functor>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int begin, const unsigned int end,
                     TData *buffer, const Functor &functor)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = begin + indx.get_global_id(0);

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

                 unsigned int n = 512;
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
     }).wait();
}

template <typename TData, typename Functor>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int begin, const unsigned int end,
                     TData *buffer, const Functor &functor)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = begin + indx.get_global_id(0);

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

                 unsigned int n = 512;
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
     }).wait();
}

template <typename TData, typename Functor>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int begin, const unsigned int end,
                     TData *buffer, const Functor &functor)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = begin + indx.get_global_id(0);

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

                 unsigned int n = 512;
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
     }).wait();
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    parallel_reduce(const unsigned int begin, const unsigned int end,
                    const Functor &functor, typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

    if (syclBuffer == nullptr)
    {
        syclBuffer = (void *)sycl::malloc_device<TData>(
            syclGridSize, SYCLQueue::GetInstance());
    }

    TData *buffer = (TData *)syclBuffer;

    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());

    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        reduceSumKernel<TData>(syclGridSize, syclBlockSize, begin, end, buffer,
                               functor);
        reduceSumKernel<TData>(
            1, syclGridSize, 0, syclGridSize, out,
            [=](const unsigned int i, TData &ans) { ans += buffer[i]; });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        reduceMaxKernel<TData>(syclGridSize, syclBlockSize, begin, end, buffer,
                               functor);
        reduceMaxKernel<TData>(1, syclGridSize, 0, syclGridSize, out,
                               [=](const unsigned int i, TData &ans) {
                                   ans = sycl::max(ans, buffer[i]);
                               });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        reduceMinKernel<TData>(syclGridSize, syclBlockSize, begin, end, buffer,
                               functor);
        reduceMinKernel<TData>(1, syclGridSize, 0, syclGridSize, out,
                               [=](const unsigned int i, TData &ans) {
                                   ans = sycl::min(ans, buffer[i]);
                               });
    }
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

} // namespace Nektar

#endif
