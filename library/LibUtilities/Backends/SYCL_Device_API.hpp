///////////////////////////////////////////////////////////////////////////////
//
// File: SYCL_Device_API.hpp
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

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_local_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_local_id(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_local_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_local_id(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_local_id(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_local_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_local_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_local_range(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_local_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_local_range(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_local_range(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_local_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_global_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_global_id(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_global_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_global_id(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_global_id(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_global_id(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_global_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_global_range(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_global_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_global_range(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_global_range(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_global_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_group(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_group(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_group(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_group(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_group(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_group(0);
    }
    else
    {
        return 0;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    const sycl::nd_item<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return threadBlock.get_group_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    const sycl::nd_item<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_group_range(1);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_group_range(0);
    }
    else
    {
        return 1;
    }
}

template <int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    const sycl::nd_item<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadBlock.get_group_range(2);
    }
    else if constexpr (dim == 1)
    {
        return threadBlock.get_group_range(1);
    }
    else if constexpr (dim == 2)
    {
        return threadBlock.get_group_range(0);
    }
    else
    {
        return 1;
    }
}

template <int ndim>
NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    const sycl::nd_item<ndim> &threadBlock)
{
    return threadBlock.get_sub_group().get_group_id();
}

template <int ndim>
NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    const sycl::nd_item<ndim> &threadBlock)
{
    return threadBlock.get_sub_group().get_local_id();
}

template <typename T> NEK_DEVICE_INLINE T getBit(T mask, T lane)
{
#if defined(SYCL_ENABLE_CUDA)
    static_assert(std::is_same_v<T, unsigned int>, "Mask must be unsigned int");
#elif defined(SYCL_ENABLE_HIP)
    static_assert(std::is_same_v<T, size_t>, "Mask must be size_t");
#endif

    // Shift the bit corresponding to the lane index completely to
    // the right and check if its value is 0 or 1.
    // e.g.: mask = 0b1001100010101110
    //                    ^
    //                    |
    //                   lane bit
    //       mask = 0b0000000000010011
    //        shift right ---------> ^
    //                               |
    //                   check last bit value (0 or 1)
    return (mask >> lane) & 0x00000001;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_add(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_add(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_add(val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_sub(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_sub(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_sub(val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_max(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_max(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_max(val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_min(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_min(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_min(val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_absmax(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        auto atomic =
            sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(*dest);
        TData ret = atomic.load();
        while (sycl::fabs(val) > sycl::fabs(ret) &&
               !atomic.compare_exchange_weak(ret, val))
        {
        }
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        auto atomic =
            sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                             sycl::memory_scope_work_group,
                             sycl::access::address_space::generic_space>(*dest);
        TData ret = atomic.load();
        while (sycl::fabs(val) > sycl::fabs(ret) &&
               !atomic.compare_exchange_weak(ret, val))
        {
        }
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_or(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_or(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_or(val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_and(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dest)
            .fetch_and(val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        sycl::atomic_ref<TData, sycl::memory_order::relaxed,
                         sycl::memory_scope_work_group,
                         sycl::access::address_space::generic_space>(*dest)
            .fetch_and(val);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, const sycl::nd_item<ndim> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::plus<>());
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, const sycl::nd_item<ndim> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::maximum<>());
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, const sycl::nd_item<ndim> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::minimum<>());
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceOr(
    const TData val, const sycl::nd_item<ndim> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::bit_or<>());
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceAnd(
    const TData val, const sycl::nd_item<ndim> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::bit_and<>());
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceSum(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_add<NektarSpaces::LocalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceMax(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_max<NektarSpaces::LocalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::LocalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceOr(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceOr(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_or<NektarSpaces::LocalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceAnd(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceAnd(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_and<NektarSpaces::LocalScope>(red, tmp);
    }
}

template <int ndim>
NEK_DEVICE_INLINE static int warpVoteAll(int predictate,
                                         const sycl::nd_item<ndim> &threadBlock)
{
    auto warp = threadBlock.get_sub_group();
    return sycl::all_of_group(warp, predictate);
}

template <int ndim>
NEK_DEVICE_INLINE static int warpVoteAny(int predictate,
                                         const sycl::nd_item<ndim> &threadBlock)
{
    auto warp = threadBlock.get_sub_group();
    return sycl::any_of_group(warp, predictate);
}

#if defined(SYCL_ENABLE_CUDA)
template <int ndim>
NEK_DEVICE_INLINE static unsigned int warpBallot(
    int predictate, const sycl::nd_item<ndim> &threadBlock)
{
    unsigned int tmp =
        predictate ? (unsigned int)1 << getLaneIdx(threadBlock) : 0;
    return warpReduceOr(tmp, threadBlock);
}
#elif defined(SYCL_ENABLE_HIP)
template <int ndim>
NEK_DEVICE_INLINE static unsigned long long warpBallot(
    int predictate, const sycl::nd_item<ndim> &threadBlock)
{
    unsigned long long tmp =
        predictate ? (unsigned long long)1 << getLaneIdx(threadBlock) : 0;
    return warpReduceOr(tmp, threadBlock);
}
#else
template <int ndim>
NEK_DEVICE_INLINE static unsigned int warpBallot(
    int predictate, [[maybe_unused]] const sycl::nd_item<ndim> &threadBlock)
{
    unsigned int tmp =
        predictate ? (unsigned int)1 << getLaneIdx(threadBlock) : 0;
    return warpReduceOr(tmp, threadBlock);
}
#endif

template <int ndim>
NEK_DEVICE_INLINE void localBarrier(const sycl::nd_item<ndim> &threadBlock)
{
#if defined(__ADAPTIVECPP__)
    threadBlock.barrier(sycl::access::fence_space::local_space);
#else
    sycl::group_barrier(threadBlock.get_group(),
                        sycl::memory_scope::work_group);
#endif
}

template <int ndim>
NEK_DEVICE_INLINE int localBarrier_and(int predictate,
                                       const sycl::nd_item<ndim> &threadBlock)
{
    threadBlock.barrier(sycl::access::fence_space::local_space);
    return sycl::all_of_group(threadBlock.get_group(), predictate);
}

template <int ndim>
NEK_DEVICE_INLINE int localBarrier_or(int predictate,
                                      const sycl::nd_item<ndim> &threadBlock)
{
    threadBlock.barrier(sycl::access::fence_space::local_space);
    return sycl::any_of_group(threadBlock.get_group(), predictate);
}

template <int ndim>
NEK_DEVICE_INLINE int localBarrier_count(int predictate,
                                         const sycl::nd_item<ndim> &threadBlock)
{
    threadBlock.barrier(sycl::access::fence_space::local_space);
    return sycl::reduce_over_group(threadBlock.get_group(), predictate ? 1 : 0,
                                   sycl::plus<>());
}

} // namespace Nektar
#endif
