///////////////////////////////////////////////////////////////////////////////
//
// File: Spaces.hpp
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

#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <type_traits>

#include <float.h>
#include <limits.h>

#if defined(NEKTAR_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <thrust/fill.h>
#define CHECK_LAST_CUDA_ERROR()                                                \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                                \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
        }                                                                      \
    }
#define CHECK_CUDA_ERROR(err)                                                  \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
        std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":" << __LINE__  \
                  << std::endl;                                                \
        std::cerr << cudaGetErrorString(err) << std::endl;                     \
    }
#elif defined(NEKTAR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Utils/SYCLQueue.hpp"
#endif

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__) ||                  \
    defined(__SYCL_DEVICE_ONLY__)
#define DEVICE_COMPILE_ONLY
#endif

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

using default_fp_type = double;

namespace NektarSpaces
{

// Device vector width
template <typename TData>
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
struct vector_width
{
    static constexpr unsigned int value = tinysimd::simd<TData>::width;
};
#elif defined(NEKTAR_ENABLE_CUDA)
struct vector_width
{
    static constexpr unsigned int value = 32u;
};
#elif defined(NEKTAR_ENABLE_HIP)
struct vector_width
{
    static constexpr unsigned int value = 64u;
};
#elif defined(SYCL_ENABLE_CUDA)
struct vector_width
{
    static constexpr unsigned int value = 32u;
};
#elif defined(SYCL_ENABLE_HIP)
struct vector_width
{
    static constexpr unsigned int value = 64u;
};
#else
struct vector_width
{
    static constexpr unsigned int value = 1u;
};
#endif

// Memory space.
// Used to refer to any data in host memory.
struct HostSpace
{
};
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
// Used to refer to any data in device memory.
struct DeviceSpace
{
};
#else
// No specific GPU so the device is the host.
using DeviceSpace = HostSpace;
#endif

// Execution space.
struct Serial
{
    static constexpr char name[]      = "Serial";
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t alignment = tinysimd::simd<double>::alignment;
};

struct AVX
{
    static constexpr char name[]      = "AVX";
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t alignment = tinysimd::simd<double>::alignment;
};

struct CUDA
{
    static constexpr char name[]      = "CUDA";
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
};

struct HIP
{
    static constexpr char name[]      = "HIP";
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 2048u;
};

struct SYCL
{
    static constexpr char name[] = "SYCL";
    using memory_space           = NektarSpaces::DeviceSpace;
#if defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP)
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#if defined(NEKTAR_DEBUG)
    static constexpr unsigned int defaultBlockSize = 128u;
#else
    static constexpr unsigned int defaultBlockSize = 256u;
#endif
    static constexpr unsigned int maximumBlockSize = 1024u;
#else
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 16u;
    static constexpr unsigned int maximumBlockSize = 16u;
#endif
};

struct DeviceOnHost
{
    static constexpr char name[]      = "DeviceOnHost";
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 1u;
    static constexpr unsigned int maximumBlockSize = 1u;
};

// Specify execution for CMakeList.txt
#define NEKTAR_DEFAULT_HOST_TAG NektarSpaces::Serial

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::AVX
#elif defined(NEKTAR_ENABLE_CUDA)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::CUDA
#elif defined(NEKTAR_ENABLE_HIP)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::HIP
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::SYCL
#elif defined(NEKTAR_ENABLE_DEVICEONHOST)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::DeviceOnHost
#endif

// These are used for LoopExecution.hpp
#if (defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)) &&             \
    defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEKTAR_LAMBDA [=]
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#else
#define NEKTAR_LAMBDA [&]
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#endif

// Memory scope for atomic.
struct GlobalScope
{
};

struct LocalScope
{
};

} // namespace NektarSpaces

namespace Nektar
{

// Helper function
[[maybe_unused]] static unsigned int GetExecSpaceAlignment(
    const std::string &execspace)
{
    if (execspace == "AVX")
    {
        return NektarSpaces::AVX::alignment;
    }
    else if (execspace == "CUDA")
    {
        return NektarSpaces::CUDA::alignment;
    }
    else if (execspace == "SYCL")
    {
        return NektarSpaces::SYCL::alignment;
    }
    else if (execspace == "DeviceOnHost")
    {
        return NektarSpaces::DeviceOnHost::alignment;
    }
    else
    {
        return NektarSpaces::Serial::alignment;
    }
}

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

namespace cg = cooperative_groups;

class cudaBlock1D
{
};

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return threadIdx.x;
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return blockDim.x;
}

NEK_DEVICE_INLINE static unsigned int getGlobalIdx(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

NEK_DEVICE_INLINE static unsigned int getGlobalRange(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return gridDim.x * blockDim.x;
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return blockIdx.x;
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    return gridDim.x;
}

NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<double>::value;
    return threadIdx.x % warpsize;
}

NEK_DEVICE_INLINE float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

NEK_DEVICE_INLINE float atomicMax_block(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
    }
    return __int_as_float(ret);
}

NEK_DEVICE_INLINE double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

NEK_DEVICE_INLINE double atomicMax_block(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

NEK_DEVICE_INLINE float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

NEK_DEVICE_INLINE float atomicMin_block(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
    }
    return __int_as_float(ret);
}

NEK_DEVICE_INLINE double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

NEK_DEVICE_INLINE double atomicMin_block(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_add(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicAdd(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicAdd_block(dest, val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_sub(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicAdd(dest, -val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicAdd_block(dest, -val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_max(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicMax(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicMax_block(dest, val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_min(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicMin(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicMin_block(dest, val);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::plus<TData>());

    // Warp-level primitives (keep it for now)
    /* val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;*/
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::greater<TData>());

    // Warp-level primitives (keep it for now)
    /*val = std::max(val, __shfl_down_sync(0xffffffff, val, 16));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 8));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 4));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 2));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;*/
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::less<TData>());

    // Warp-level primitives (keep it for now)
    /*val = std::min(val, __shfl_down_sync(0xffffffff, val, 16));
    val = std::min(val, __shfl_down_sync(0xffffffff, val, 8));
    val = std::min(val, __shfl_down_sync(0xffffffff, val, 4));
    val = std::min(val, __shfl_down_sync(0xffffffff, val, 2));
    val = std::min(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;*/
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock,
    TData *red)
{
    auto tmp = warpReduceSum(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_add<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock,
    TData *red)
{
    auto tmp = warpReduceMax(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_max<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, [[maybe_unused]] const cudaBlock1D &threadBlock,
    TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::GlobalScope>(red, tmp);
    }
}

NEK_DEVICE_INLINE void localBarrier(const cudaBlock1D &threadBlock)
{
    __syncthreads();
}

#elif defined(NEKTAR_ENABLE_SYCL)

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_local_id(0);
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_local_range(0);
}

NEK_DEVICE_INLINE static unsigned int getGlobalIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_global_id(0);
}

NEK_DEVICE_INLINE static unsigned int getGlobalRange(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_global_range(0);
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_group(0);
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_group_range(0);
}

NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_sub_group().get_local_id();
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
                         sycl::access::address_space::local_space>(*dest)
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
                         sycl::access::address_space::local_space>(*dest)
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
                         sycl::access::address_space::local_space>(*dest)
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
                         sycl::access::address_space::local_space>(*dest)
            .fetch_min(val);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, const sycl::nd_item<1> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::plus<>());
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, const sycl::nd_item<1> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::maximum<>());
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, const sycl::nd_item<1> &threadBlock)
{
    return sycl::reduce_over_group(threadBlock.get_sub_group(), val,
                                   sycl::minimum<>());
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, const sycl::nd_item<1> &threadBlock, TData *red)
{
    auto tmp = warpReduceSum(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_add<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, const sycl::nd_item<1> &threadBlock, TData *red)
{
    auto tmp = warpReduceMax(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_max<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, const sycl::nd_item<1> &threadBlock, TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::GlobalScope>(red, tmp);
    }
}

NEK_DEVICE_INLINE void localBarrier(const sycl::nd_item<1> &threadBlock)
{
    threadBlock.barrier(sycl::access::fence_space::local_space);
}

#elif defined(NEKTAR_ENABLE_DEVICEONHOST)
class deviceOnHostBlock1D
{
};

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 1;
}

NEK_DEVICE_INLINE static unsigned int getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static unsigned int getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 1;
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 1;
}

NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 0;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_add(TData *const dest, const TData val)
{
    *dest += val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_sub(TData *const dest, const TData val)
{
    *dest -= val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_max(TData *const dest, const TData val)
{
    *dest = std::max(*dest, val);
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_min(TData *const dest, const TData val)
{
    *dest = std::min(*dest, val);
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return val;
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return val;
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return val;
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock,
    TData *red)
{
    *red += val;
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock,
    TData *red)
{
    *red = std::max(*red, val);
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, [[maybe_unused]] const deviceOnHostBlock1D &threadBlock,
    TData *red)
{
    *red = std::min(*red, val);
}

NEK_DEVICE_INLINE void localBarrier(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
}

#endif

} // namespace Nektar
