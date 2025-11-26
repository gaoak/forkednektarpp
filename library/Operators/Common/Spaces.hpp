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
#include <iostream>
#include <limits.h>
#include <string>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

#if defined(NEKTAR_ENABLE_CUDA)
#include <cuda_runtime.h>
#define CHECK_LAST_HIPCUDA_ERROR()                                             \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                                \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
            exit(0);                                                           \
        }                                                                      \
    }
#define CHECK_HIPCUDA_ERROR(err)                                               \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
        std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":" << __LINE__  \
                  << std::endl;                                                \
        std::cerr << cudaGetErrorString(err) << std::endl;                     \
        exit(0);                                                               \
    }
#elif defined(NEKTAR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#define CHECK_LAST_HIPCUDA_ERROR()                                             \
    {                                                                          \
        hipError_t err = hipGetLastError();                                    \
        if (err != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "HIP Runtime Error at: " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;                                \
            std::cerr << hipGetErrorString(err) << std::endl;                  \
            exit(0);                                                           \
        }                                                                      \
    }
#define CHECK_HIPCUDA_ERROR(err)                                               \
    if (err != hipSuccess)                                                     \
    {                                                                          \
        std::cerr << "HIP Runtime Error at: " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                                                \
        std::cerr << hipGetErrorString(err) << std::endl;                      \
        exit(0);                                                               \
    }
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Common/SYCLQueue.hpp"
#endif

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#elif defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__SYCL_DEVICE_ONLY__)
#define DEVICE_COMPILE_ONLY
#endif

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

using default_fp_type = double;

namespace NektarSpaces
{

// Alignment
struct memory_alignment
{
#if defined(NEKTAR_ENABLE_CUDA)
    static constexpr size_t value = 256u;
#elif defined(NEKTAR_ENABLE_HIP)
    static constexpr size_t value       = 128u;
#elif defined(SYCL_ENABLE_CUDA)
    static constexpr size_t value                  = 256u;
#elif defined(SYCL_ENABLE_HIP)
    static constexpr size_t value                  = 128u;
#elif defined(SYCL_ENABLE_INTEL)
    static constexpr size_t value                  = 128u;
#elif defined(SYCL_ENABLE_CPU)
    static constexpr size_t value       = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#elif defined(NEKTAR_ENABLE_SIMD)
    static constexpr size_t value       = tinysimd::simd<double>::alignment;
#else
    static constexpr size_t value       = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#endif
};

// Vector width
template <typename TData> struct vector_width
{
#if defined(NEKTAR_ENABLE_CUDA)
    static constexpr unsigned int value = 32u;
#elif defined(NEKTAR_ENABLE_HIP)
    static constexpr unsigned int value = 64u;
#elif defined(SYCL_ENABLE_CUDA)
    static constexpr unsigned int value            = 32u;
#elif defined(SYCL_ENABLE_HIP)
    static constexpr unsigned int value            = 64u;
#elif defined(SYCL_ENABLE_INTEL)
    static constexpr unsigned int value            = 32u;
#elif defined(SYCL_ENABLE_CPU)
    static constexpr unsigned int value = 1u;
#elif defined(NEKTAR_ENABLE_SIMD)
    static constexpr unsigned int value = tinysimd::simd<TData>::width;
#else
    static constexpr unsigned int value = 1u;
#endif
};

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
    static inline const std::string name = "Serial";
    using memory_space                   = NektarSpaces::HostSpace;
};

struct AVX
{
    static inline const std::string name = "AVX";
    using memory_space                   = NektarSpaces::HostSpace;
};

struct Device
{
    static inline const std::string name = "Device";
    using memory_space                   = NektarSpaces::DeviceSpace;
#if defined(NEKTAR_ENABLE_CUDA)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(NEKTAR_ENABLE_HIP)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(SYCL_ENABLE_CUDA)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(SYCL_ENABLE_HIP)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(SYCL_ENABLE_INTEL)
    static constexpr unsigned int defaultBlockSize = 128u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(SYCL_ENABLE_CPU)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#else
    static constexpr unsigned int defaultBlockSize = 1u;
    static constexpr unsigned int maximumBlockSize = 1u;
#endif
};

// These are used for LoopExecution.hpp
// NEKTAR_LAMBDA
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __host__ __device__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEKTAR_LAMBDA [=]
#else
#define NEKTAR_LAMBDA [&]
#endif

// NEK_DEVICE_INLINE
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __host__ __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#else
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

[[maybe_unused]] static inline void nekDeviceSynchronize(void)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().wait();
#endif
}

class hipcudaBlock1D
{
};

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__))

namespace cg = cooperative_groups;

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return threadIdx.x;
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return blockDim.x;
}

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return gridDim.x * blockDim.x;
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return blockIdx.x;
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    return gridDim.x;
}

NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<double>::value;
    return getGlobalIdx(threadBlock) / warpsize;
}

NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<double>::value;
    return getLocalIdx(threadBlock) % warpsize;
}

template <typename T> __device__ __inline__ T getBit(T mask, T lane)
{
#if defined(__CUDACC__)
    static_assert(std::is_same_v<T, unsigned int>, "Mask must be unsigned int");
#elif defined(__HIPCC__)
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
#if defined(__CUDACC__)
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
#elif defined(__HIPCC__)
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
#endif
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
#if defined(__CUDACC__)
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
#elif defined(__HIPCC__)
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
#endif
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
#if defined(__CUDACC__)
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
#elif defined(__HIPCC__)
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
#endif
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
#if defined(__CUDACC__)
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
#elif defined(__HIPCC__)
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
#endif
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
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::plus<TData>());

    /*// Warp-level primitives (keep it for now)
    auto tmp = val;
    tmp += __shfl_down_sync(0xffffffff, tmp, 16);
    tmp += __shfl_down_sync(0xffffffff, tmp, 8);
    tmp += __shfl_down_sync(0xffffffff, tmp, 4);
    tmp += __shfl_down_sync(0xffffffff, tmp, 2);
    tmp += __shfl_down_sync(0xffffffff, tmp, 1);
    return tmp;*/
#elif defined(__HIPCC__)
    // Warp-level primitives
    auto tmp = val;
    tmp += __shfl_down(tmp, 32);
    tmp += __shfl_down(tmp, 16);
    tmp += __shfl_down(tmp, 8);
    tmp += __shfl_down(tmp, 4);
    tmp += __shfl_down(tmp, 2);
    tmp += __shfl_down(tmp, 1);
    return tmp;
#endif
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::greater<TData>());

    /*// Warp-level primitives (keep it for now)
    auto tmp = val;
    tmp = std::max(tmp, __shfl_down_sync(0xffffffff, tmp, 16));
    tmp = std::max(tmp, __shfl_down_sync(0xffffffff, tmp, 8));
    tmp = std::max(tmp, __shfl_down_sync(0xffffffff, tmp, 4));
    tmp = std::max(tmp, __shfl_down_sync(0xffffffff, tmp, 2));
    tmp = std::max(tmp, __shfl_down_sync(0xffffffff, tmp, 1));
    return tmp;*/
#elif defined(__HIPCC__)
    // Warp-level primitives
    auto tmp = val;
    tmp      = std::max(tmp, __shfl_down(tmp, 32));
    tmp      = std::max(tmp, __shfl_down(tmp, 16));
    tmp      = std::max(tmp, __shfl_down(tmp, 8));
    tmp      = std::max(tmp, __shfl_down(tmp, 4));
    tmp      = std::max(tmp, __shfl_down(tmp, 2));
    tmp      = std::max(tmp, __shfl_down(tmp, 1));
    return tmp;
#endif
}

template <typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::less<TData>());

    /*// Warp-level primitives (keep it for now)
    auto tmp = val;
    tmp = std::min(tmp, __shfl_down_sync(0xffffffff, tmp, 16));
    tmp = std::min(tmp, __shfl_down_sync(0xffffffff, tmp, 8));
    tmp = std::min(tmp, __shfl_down_sync(0xffffffff, tmp, 4));
    tmp = std::min(tmp, __shfl_down_sync(0xffffffff, tmp, 2));
    tmp = std::min(tmp, __shfl_down_sync(0xffffffff, tmp, 1));
    return tmp;*/
#elif defined(__HIPCC__)
    // Warp-level primitives
    auto tmp = val;
    tmp      = std::min(tmp, __shfl_down(tmp, 32));
    tmp      = std::min(tmp, __shfl_down(tmp, 16));
    tmp      = std::min(tmp, __shfl_down(tmp, 8));
    tmp      = std::min(tmp, __shfl_down(tmp, 4));
    tmp      = std::min(tmp, __shfl_down(tmp, 2));
    tmp      = std::min(tmp, __shfl_down(tmp, 1));
    return tmp;
#endif
}

template <typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock,
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
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock,
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
    const TData val, [[maybe_unused]] const hipcudaBlock1D &threadBlock,
    TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::GlobalScope>(red, tmp);
    }
}

NEK_DEVICE_INLINE void localBarrier(
    [[maybe_unused]] const hipcudaBlock1D &threadBlock)
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

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_global_id(0);
}

NEK_DEVICE_INLINE static size_t getGlobalRange(
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

NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_sub_group().get_group_id();
}

NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const sycl::nd_item<1> &threadBlock)
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

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock1D &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static size_t getGlobalRange(
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
