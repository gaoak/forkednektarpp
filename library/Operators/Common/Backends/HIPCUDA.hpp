///////////////////////////////////////////////////////////////////////////////
//
// File: HIPCUDA.hpp
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
namespace Nektar
{

template <unsigned int ndim> class hipcudaBlock
{
};

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x) __launch_bounds__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)                                               \
    extern __shared__ __align__(sizeof(TData)) unsigned char __shmemptr[];     \
    ptr = __shmemptr

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<1>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<2>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<3>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<1>.
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                    \
                                               hipcudaBlock<1>());             \
    CHECK_LAST_HIPCUDA_ERROR();

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<2>.
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                    \
                                               hipcudaBlock<2>());             \
    CHECK_LAST_HIPCUDA_ERROR();

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<3>.
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                    \
                                               hipcudaBlock<3>());             \
    CHECK_LAST_HIPCUDA_ERROR();

namespace cg = cooperative_groups;

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return threadIdx.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return threadIdx.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return threadIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return threadIdx.y;
    }
    else if constexpr (dim == 2)
    {
        return threadIdx.z;
    }
    else
    {
        return 0;
    }
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return blockDim.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockDim.x;
    }
    else if constexpr (dim == 1)
    {
        return blockDim.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockDim.x;
    }
    else if constexpr (dim == 1)
    {
        return blockDim.y;
    }
    else if constexpr (dim == 2)
    {
        return blockDim.z;
    }
    else
    {
        return 0;
    }
}

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockDim.x * blockIdx.x + threadIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return blockDim.y * blockIdx.y + threadIdx.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockDim.x * blockIdx.x + threadIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return blockDim.y * blockIdx.y + threadIdx.y;
    }
    else if constexpr (dim == 2)
    {
        return blockDim.z * blockIdx.z + threadIdx.z;
    }
    else
    {
        return 0;
    }
}

NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return gridDim.x * blockDim.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return gridDim.x * blockDim.x;
    }
    else if constexpr (dim == 1)
    {
        return gridDim.y * blockDim.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return gridDim.x * blockDim.x;
    }
    else if constexpr (dim == 1)
    {
        return gridDim.y * blockDim.y;
    }
    else if constexpr (dim == 2)
    {
        return gridDim.z * blockDim.z;
    }
    else
    {
        return 0;
    }
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return blockIdx.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return blockIdx.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return blockIdx.x;
    }
    else if constexpr (dim == 1)
    {
        return blockIdx.y;
    }
    else if constexpr (dim == 2)
    {
        return blockIdx.z;
    }
    else
    {
        return 0;
    }
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const hipcudaBlock<1> &threadBlock)
{
    return gridDim.x;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const hipcudaBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return gridDim.x;
    }
    else if constexpr (dim == 1)
    {
        return gridDim.y;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const hipcudaBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return gridDim.x;
    }
    else if constexpr (dim == 1)
    {
        return gridDim.y;
    }
    else if constexpr (dim == 2)
    {
        return gridDim.z;
    }
    else
    {
        return 0;
    }
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    return getGlobalIdx(threadBlock) / warpsize;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
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

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_or(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicOr(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicOr_block(dest, val);
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_and(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicAnd(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicAnd_block(dest, val);
    }
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    // Broadcast
    tmp = __shfl_sync(0xffffffff, tmp, 0);
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
    // Broadcast
    tmp = __shfl(tmp, 0);
    return tmp;
#endif
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    // Broadcast
    tmp = __shfl_sync(0xffffffff, tmp, 0);
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
    // Broadcast
    tmp = __shfl(tmp, 0);
    return tmp;
#endif
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    // Broadcast
    tmp = __shfl_sync(0xffffffff, tmp, 0);
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
    // Broadcast
    tmp = __shfl(tmp, 0);
    return tmp;
#endif
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceOr(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::bit_or<TData>());

    /*// Warp-level primitives (keep it for now)
    auto tmp = val;
    tmp |= __shfl_down_sync(0xffffffff, tmp, 16);
    tmp |= __shfl_down_sync(0xffffffff, tmp, 8);
    tmp |= __shfl_down_sync(0xffffffff, tmp, 4);
    tmp |= __shfl_down_sync(0xffffffff, tmp, 2);
    tmp |= __shfl_down_sync(0xffffffff, tmp, 1);
    // Broadcast
    tmp = __shfl_sync(0xffffffff, tmp, 0);
    return tmp;*/
#elif defined(__HIPCC__)
    // Warp-level primitives
    auto tmp = val;
    tmp |= __shfl_down(tmp, 32);
    tmp |= __shfl_down(tmp, 16);
    tmp |= __shfl_down(tmp, 8);
    tmp |= __shfl_down(tmp, 4);
    tmp |= __shfl_down(tmp, 2);
    tmp |= __shfl_down(tmp, 1);
    // Broadcast
    tmp = __shfl(tmp, 0);
    return tmp;
#endif
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceAnd(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    return cg::reduce(warp, val, cg::bit_and<TData>());

    /*// Warp-level primitives (keep it for now)
    auto tmp = val;
    tmp &= __shfl_down_sync(0xffffffff, tmp, 16);
    tmp &= __shfl_down_sync(0xffffffff, tmp, 8);
    tmp &= __shfl_down_sync(0xffffffff, tmp, 4);
    tmp &= __shfl_down_sync(0xffffffff, tmp, 2);
    tmp &= __shfl_down_sync(0xffffffff, tmp, 1);
    // Broadcast
    tmp = __shfl_sync(0xffffffff, tmp, 0);
    return tmp;*/
#elif defined(__HIPCC__)
    // Warp-level primitives
    auto tmp = val;
    tmp &= __shfl_down(tmp, 32);
    tmp &= __shfl_down(tmp, 16);
    tmp &= __shfl_down(tmp, 8);
    tmp &= __shfl_down(tmp, 4);
    tmp &= __shfl_down(tmp, 2);
    tmp &= __shfl_down(tmp, 1);
    // Broadcast
    tmp = __shfl(tmp, 0);
    return tmp;
#endif
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock,
    TData *red)
{
    auto tmp = warpReduceSum(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_add<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock,
    TData *red)
{
    auto tmp = warpReduceMax(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_max<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock,
    TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceOr(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock,
    TData *red)
{
    auto tmp = warpReduceOr(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_or<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceAnd(
    const TData val, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock,
    TData *red)
{
    auto tmp = warpReduceAnd(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_and<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAll(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    return __all_sync(0xffffffff, predictate);
#elif defined(__HIPCC__)
    return __all(predictate);
#endif
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAny(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
#if defined(__CUDACC__)
    return __any_sync(0xffffffff, predictate);
#elif defined(__HIPCC__)
    return __any(predictate);
#endif
}

#if defined(__CUDACC__)
template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int warpBallot(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    return __ballot_sync(0xffffffff, predictate);
}
#elif defined(__HIPCC__)
template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned long long warpBallot(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    return __ballot(predictate);
}
#endif

template <unsigned int ndim>
NEK_DEVICE_INLINE void localBarrier(
    [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    __syncthreads();
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_and(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    return __syncthreads_and(predictate);
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_or(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    return __syncthreads_or(predictate);
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_count(
    int predictate, [[maybe_unused]] const hipcudaBlock<ndim> &threadBlock)
{
    return __syncthreads_count(predictate);
}

} // namespace Nektar
#endif
