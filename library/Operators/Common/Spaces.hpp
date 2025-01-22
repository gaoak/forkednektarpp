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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <type_traits>

#include <float.h>
#include <limits.h>

#if defined(NEKTAR_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#endif

#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__) ||                  \
    defined(__SYCL_DEVICE_ONLY__)
#define DEVICE_COMPILE_ONLY
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) ||      \
    defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_USING_GPU
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
#elif defined(KOKKOS_ENABLE_CUDA)
struct vector_width
{
    static constexpr unsigned int value = Kokkos::Impl::CudaTraits::WarpSize;
};
#elif defined(KOKKOS_ENABLE_HIP)
struct vector_width
{
    static constexpr unsigned int value = Kokkos::Impl::HIPTraits::WarpSize;
};
#elif defined(KOKKOS_ENABLE_SYCL)
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
    defined(SYCL_ENABLE_CUDA) || defined(KOKKOS_USING_GPU)
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
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t alignment = tinysimd::simd<double>::alignment;
};

struct AVX
{
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t alignment = tinysimd::simd<double>::alignment;
};

struct CUDA
{
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
};

struct HIP
{
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 2048u;
};

struct SYCL
{
    using memory_space = NektarSpaces::DeviceSpace;
#if defined(SYCL_ENABLE_CUDA)
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

struct KOKKOS
{
    using memory_space = NektarSpaces::DeviceSpace;
#if defined(KOKKOS_ENABLE_CUDA)
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#elif defined(KOKKOS_ENABLE_HIP)
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 2048u;
#elif defined(KOKKOS_ENABLE_SYCL)
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
#else
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    static constexpr unsigned int defaultBlockSize = 1u;
#endif
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
#elif defined(NEKTAR_ENABLE_KOKKOS)
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::KOKKOS
#endif

// These are used for LoopExecution.hpp
#if (defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)) &&             \
    defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEKTAR_LAMBDA [=]
#elif defined(NEKTAR_ENABLE_KOKKOS) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA KOKKOS_LAMBDA
#else
#define NEKTAR_LAMBDA [&]
#endif

// Memory scope for atomic.
struct GlobalScope
{
};

struct LocalScope
{
};

} // namespace NektarSpaces
