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
#endif // defined(NEKTAR_ENABLE_KOKKOS)

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

using vec_t = tinysimd::simd<double>;

namespace NektarSpaces
{

// Used to refer to any data in host memory.
class HostSpace
{
};

class Serial
{
public:
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t width     = vec_t::width;
    static constexpr size_t alignment = vec_t::alignment;
};

class AVX
{
public:
    using memory_space                = NektarSpaces::HostSpace;
    static constexpr size_t width     = vec_t::width;
    static constexpr size_t alignment = vec_t::alignment;
};

using DefaultHostExecutionSpace = Serial; // Default Host implementation.

// This macro is used in the CMakeLists.txt for generating the factory
// *.cpp files. There is also a device tag.
#define NEKTAR_DEFAULT_HOST_TAG NektarSpaces::DefaultHostExecutionSpace

// Native pure GPU execution
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(KOKKOS_USING_GPU)

// Used to refer to any data in device memory.
class DeviceSpace
{
};

#else
// No specific GPU so the device is the host.
using DeviceSpace = HostSpace;

#endif

// Native pure CUDA execution
class CUDA
{
public:
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t width     = 32;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};

// Native pure HIP execution
class HIP
{
public:
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t width     = 64;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};

// Native pure SYCL execution
class SYCL
{
public:
    using memory_space                = NektarSpaces::DeviceSpace;
    static constexpr size_t width     = 64;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};

// Native pure Kokkos execution
class KOKKOS
{
public:
    using memory_space = NektarSpaces::DeviceSpace;
#if defined(KOKKOS_ENABLE_CUDA)
    static constexpr size_t width     = Kokkos::Impl::CudaTraits::WarpSize;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#elif defined(KOKKOS_ENABLE_HIP)
    static constexpr size_t width     = Kokkos::Impl::HIPTraits::WarpSize;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#elif defined(KOKKOS_ENABLE_SYCL)
    static constexpr size_t width     = 64;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
    static constexpr size_t width     = vec_t::width;
    static constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#endif
};

// Specific pure GPU execution
#if defined(NEKTAR_ENABLE_CUDA)
using DefaultExecutionSpace = CUDA;
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::CUDA
#elif defined(NEKTAR_ENABLE_HIP)
using DefaultExecutionSpace = HIP;
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::HIP
#elif defined(NEKTAR_ENABLE_SYCL)
using DefaultExecutionSpace = SYCL;
#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::SYCL
#elif defined(NEKTAR_ENABLE_KOKKOS)
using DefaultExecutionSpace = KOKKOS;
#define KOKKOS_DEFAULT_DEVICE_TAG NektarSpaces::KOKKOS
#else
using DefaultExecutionSpace = DefaultHostExecutionSpace;
#endif // GPU specific

// These are used for LoopExecution.hpp
#if (defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)) &&             \
    defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_SYCL) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=]
#elif defined(NEKTAR_ENABLE_KOKKOS) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA KOKKOS_LAMBDA
#else
#define NEKTAR_LAMBDA [&]
#endif

} // namespace NektarSpaces
