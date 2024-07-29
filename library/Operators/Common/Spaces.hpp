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

#include <type_traits>

#include <float.h>
#include <limits.h>

#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__) ||                  \
    defined(__SYCL_DEVICE_ONLY__)
#define DEVICE_COMPILE_ONLY
#endif

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

namespace NektarSpaces
{

class HostSpace
{
    // Used to refer to any data in host memory.
};

class Serial
{
public:
    using memory_space = NektarSpaces::HostSpace;
};

class AVX
{
public:
    using memory_space = NektarSpaces::HostSpace;
};

using DefaultHostExecutionSpace = Serial; // Default Host implementation.

// This macro is used in the CMakeLists.txt for generating the factory
// *.cpp files. There is also a device tag.
#define NEKTAR_DEFAULT_HOST_TAG NektarSpaces::DefaultHostExecutionSpace

// Native pure GPU execution
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL)

class DeviceSpace
{
    // Used to refer to any data in device memory.
};

#else
// No specific GPU so the device is the host.
using DeviceSpace           = HostSpace;

#endif

// Native pure CUDA execution
class CUDA
{
public:
    using memory_space = NektarSpaces::DeviceSpace;
};

// Native pure HIP execution
class HIP
{
public:
    using memory_space = NektarSpaces::DeviceSpace;
};

// Native pure SYCL execution
class SYCL
{
public:
    using memory_space = NektarSpaces::DeviceSpace;
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

#else
using DefaultExecutionSpace = DefaultHostExecutionSpace;

#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::DefaultHostExecutionSpace

#endif // GPU specific

} // namespace NektarSpaces

#if defined(NEKTAR_ENABLE_KOKKOS)

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

// GPUs
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) ||      \
    defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_USING_GPU
#endif

#define KOKKOS_DEFAULT_HOST_TAG Kokkos::DefaultHostExecutionSpace
#define KOKKOS_DEFAULT_DEVICE_TAG Kokkos::DefaultExecutionSpace

#else // !defined(NEKTAR_ENABLE_KOKKOS)

namespace Kokkos
{

class HostSpace
{
};

class DefaultExecutionSpace
{
public:
    using memory_space = NektarSpaces::HostSpace;
};

} // namespace Kokkos

#endif // !defined(NEKTAR_ENABLE_KOKKOS)

// These are used for LoopExecution.hpp
#if (defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||              \
     defined(NEKTAR_ENABLE_SYCL)) &&                                           \
    defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_KOKKOS) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA KOKKOS_LAMBDA
#else
#define NEKTAR_LAMBDA [&]
#endif

namespace NektarSpaces
{

#if defined(NEKTAR_ENABLE_KOKKOS)
template <typename TData> using ReduceSum = Kokkos::Sum<TData>;
template <typename TData> using ReduceMin = Kokkos::Min<TData>;
template <typename TData> using ReduceMax = Kokkos::Max<TData>;
#else  // !defined(NEKTAR_ENABLE_KOKKOS)
template <typename TData> class ReduceSum
{
public:
    typedef typename std::remove_cv<TData>::type value_type;
};
template <typename TData> class ReduceMin
{
public:
    typedef typename std::remove_cv<TData>::type value_type;
};
template <typename TData> class ReduceMax
{
public:
    typedef typename std::remove_cv<TData>::type value_type;
};
#endif // !defined(NEKTAR_ENABLE_KOKKOS)

} // namespace NektarSpaces
