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
#include <utility>

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

#if defined(__CUDACC__) || defined(__NEK_HIPCC__) ||                           \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
#define DEVICE_COMPILE_ONLY
#endif

#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#include <hip/hip_cooperative_groups.h>
#endif

using default_fp_type = double;

template <bool B, typename TData> struct data_type_if
{
    typedef TData type;
};

template <typename TData> struct data_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

template <bool B, typename TData> struct simd_type_if
{
    typedef tinysimd::scalarT<TData> type;
};

template <typename TData> struct simd_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

namespace NektarSpaces
{

// Memory space.
// Used to refer to any data in host memory.
struct HostSpace
{
};
#if defined(NEKTAR_ENABLE_DEVICE)
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
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 32u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 64u;
#elif defined(SYCL_ENABLE_INTEL)
    static constexpr unsigned int defaultBlockSize = 128u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 32u;
#elif defined(SYCL_ENABLE_CPU)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 1u;
#else
    static constexpr unsigned int defaultBlockSize = 1u;
    static constexpr unsigned int maximumBlockSize = 1u;
    static constexpr unsigned int warpSize         = 1u;
#endif
};

// Host memory alignment
#if defined(NEKTAR_ENABLE_SIMD)
static constexpr size_t host_memory_alignment =
    tinysimd::simd<double>::alignment;
#else
static constexpr size_t host_memory_alignment =
    __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#endif

// Vector width
template <typename ExecSpace, typename TData> struct vector_width
{
};

template <typename TData> struct vector_width<Serial, TData>
{
    static constexpr unsigned int value = 1u;
};

template <typename TData> struct vector_width<AVX, TData>
{
    static constexpr unsigned int value = tinysimd::simd<TData>::width;
};

template <typename TData> struct vector_width<Device, TData>
{
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA) ||                \
    defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP) ||                  \
    defined(SYCL_ENABLE_INTEL)
    static_assert(
        Device::warpSize % tinysimd::simd<TData>::width == 0,
        "AVX/Device back-ends interoperability requires device vector width "
        "(warpsize) to be integer multiple of SIMD vector width");
#endif
    static constexpr unsigned int value = Device::warpSize;
};

template <typename TData> struct max_vector_width
{
    // Use maximum vector width for back-ends interoperability.
    static constexpr unsigned int value = std::max(
        vector_width<AVX, TData>::value, vector_width<Device, TData>::value);
};

template <typename TData>
static unsigned int GetVectorWidth(const std::string &execName)
{
    if (execName == "Serial")
    {
        return NektarSpaces::vector_width<NektarSpaces::Serial, TData>::value;
    }
    else if (execName == "AVX")
    {
        return NektarSpaces::vector_width<NektarSpaces::AVX, TData>::value;
    }
    else if (execName == "Device")
    {
        return NektarSpaces::vector_width<NektarSpaces::Device, TData>::value;
    }
    else
    {
        return 0;
    }
}

// NEK_DEVICE_INLINE
// Used to define a device function (e.g. a function launched
// from a kernel function and executing on the device). All
// device functions must be prefixed by the NEK_DEVICE_INLINE
// decorator.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#else
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#endif

// NEK_KERNEL_KERNEL
// Used to define a kernel function (e.g. a function launched
// from the host and executing on the device). All kernel
// functions must be prefixed by the NEK_DEVICE_KERNEL decorator.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_KERNEL __global__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_KERNEL __global__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_DEVICE_KERNEL NEK_FORCE_INLINE
#else
#define NEK_DEVICE_KERNEL NEK_FORCE_INLINE
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

#if defined(NEKTAR_ENABLE_DEVICE)
extern void *internalMemoryBuffer;
extern void *internalDeviceBuffer;
extern void *internalHostBuffer;
extern unsigned int internalMaxDataSizeByte;
#endif

[[maybe_unused]] static inline void nekSetDevice(
    [[maybe_unused]] unsigned int device_rank)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
#elif defined(NEKTAR_ENABLE_SYCL)
    internalSYCLDeviceId                           = device_rank;
#else
    // Do nothing
#endif
}

[[maybe_unused]] static inline unsigned int nekGetDevice()
{
#if defined(NEKTAR_ENABLE_CUDA)
    int device_rank = 0;
    CHECK_HIPCUDA_ERROR(cudaGetDevice(&device_rank));
    return device_rank;
#elif defined(NEKTAR_ENABLE_HIP)
    int device_rank = 0;
    CHECK_HIPCUDA_ERROR(hipGetDevice(&device_rank));
    return device_rank;
#elif defined(NEKTAR_ENABLE_SYCL)
    return internalSYCLDeviceId;
#else
    return 0;
#endif
}

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

template <typename Tstream>
[[maybe_unused]] static inline void nekStreamSynchronize(
    [[maybe_unused]] Tstream &stream)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().wait();
#endif
}

template <unsigned int ndim> class hipcudaBlock
{
};

#if (defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)) ||           \
    (defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY))

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

#elif defined(NEKTAR_ENABLE_SYCL)

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(SHMEMSIZE), cgh);                               \
            cgh.parallel_for(                                                  \
                sycl::nd_range<1>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<1> item_ct1) {                               \
                    auto shmemptr = &shmem[0];                                 \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(SHMEMSIZE), cgh);                               \
            cgh.parallel_for(                                                  \
                sycl::nd_range<2>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<2> item_ct1) {                               \
                    auto shmemptr = &shmem[0];                                 \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(SHMEMSIZE), cgh);                               \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<3> item_ct1) {                               \
                    auto shmemptr = &shmem[0];                                 \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<1>.
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            cgh.parallel_for(                                                  \
                sycl::nd_range<1>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<1> item_ct1) {                               \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<2>.
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            cgh.parallel_for(                                                  \
                sycl::nd_range<2>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<2> item_ct1) {                               \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<3>.
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance();                             \
        auto args      = std::make_tuple(__VA_ARGS__);                         \
        Q.submit([=](sycl::handler &cgh) {                                     \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<3> item_ct1) {                               \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
    }

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_local_id(0);
}

template <int dim>
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

template <int dim>
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

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_local_range(0);
}

template <int dim>
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
        return 0;
    }
}

template <int dim>
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
        return 0;
    }
}

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_global_id(0);
}

template <int dim>
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

template <int dim>
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

NEK_DEVICE_INLINE static size_t getGlobalRange(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_global_range(0);
}

template <int dim>
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
        return 0;
    }
}

template <int dim>
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
        return 0;
    }
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_group(0);
}

template <int dim>
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

template <int dim>
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

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    const sycl::nd_item<1> &threadBlock)
{
    return threadBlock.get_group_range(0);
}

template <int dim>
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
        return 0;
    }
}

template <int dim>
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
        return 0;
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
                         sycl::access::address_space::local_space>(*dest)
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
                         sycl::access::address_space::local_space>(*dest)
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
        atomic_add<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceMax(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_max<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceMin(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_min<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceOr(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceOr(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_or<NektarSpaces::GlobalScope>(red, tmp);
    }
}

template <int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceAnd(
    const TData val, const sycl::nd_item<ndim> &threadBlock, TData *red)
{
    auto tmp = warpReduceAnd(val, threadBlock);
    if (getLaneIdx(threadBlock) == 0)
    {
        atomic_and<NektarSpaces::GlobalScope>(red, tmp);
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
    threadBlock.barrier(sycl::access::fence_space::local_space);
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

#elif defined(NEKTAR_ENABLE_DEVICEONHOST)
template <unsigned int ndim> class deviceOnHostBlock
{
};

template <typename TData> static void nektar_unused([[maybe_unused]] TData x)
{
    return;
}

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        nektar_unused(GRIDSIZE);                                               \
        nektar_unused(BLOCKSIZE);                                              \
        std::vector<unsigned char> shmem(SHMEMSIZE);                           \
        KERNEL(__VA_ARGS__, shmem.data(), deviceOnHostBlock<1>());             \
    }

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, STREAM,     \
                                      ...)                                     \
    nektar_unused(GRIDSIZE);                                                   \
    nektar_unused(BLOCKSIZE);                                                  \
    std::vector<unsigned char> shmem(SHMEMSIZE);                               \
    KERNEL(__VA_ARGS__, shmem.data(), deviceOnHostBlock<2>());

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, STREAM,     \
                                      ...)                                     \
    nektar_unused(GRIDSIZE);                                                   \
    nektar_unused(BLOCKSIZE);                                                  \
    std::vector<unsigned char> shmem(SHMEMSIZE);                               \
    KERNEL(__VA_ARGS__, shmem.data(), deviceOnHostBlock<3>());

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<1>.
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    nektar_unused(GRIDSIZE);                                                   \
    nektar_unused(BLOCKSIZE);                                                  \
    KERNEL(__VA_ARGS__, deviceOnHostBlock<1>());

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<2>.
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    nektar_unused(GRIDSIZE);                                                   \
    nektar_unused(BLOCKSIZE);                                                  \
    KERNEL(__VA_ARGS__, deviceOnHostBlock<2>());

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<3>.
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    nektar_unused(GRIDSIZE);                                                   \
    nektar_unused(BLOCKSIZE);                                                  \
    KERNEL(__VA_ARGS__, deviceOnHostBlock<3>());

NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 1;
}

NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 1;
}

NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 0;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 0;
}

NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    return 1;
}

template <unsigned int dim>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    return 1;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return getGlobalIdx(threadBlock);
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
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

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_or(TData *const dest, const TData val)
{
    *dest |= val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_and(TData *const dest, const TData val)
{
    *dest &= val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceOr(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceAnd(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red += val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red = std::max(*red, val);
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red = std::min(*red, val);
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceOr(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red |= val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceAnd(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red &= val;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAll(
    int predictate, [[maybe_unused]] deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAny(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpBallot(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE void localBarrier(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_and(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_or(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_count(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

#endif

} // namespace Nektar
