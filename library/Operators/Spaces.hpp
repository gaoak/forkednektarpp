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

namespace Nektar::MemoryLayout
{

// Core memory layout types
class Consecutive
{
};

class Interlaced
{
};

} // namespace Nektar::MemoryLayout

// Create some data types for non-Kokkos runs. The NektarSpaces
// namespace basically duplicates concepts that are part of the Kokkos
// namespace.
namespace NektarSpaces
{

class HostSpace
{
    // Used to refer to any data in host memory.
};

class Serial
{
    // Used for legacy Nektar Serial tasks (e.g. no Kokkos)};
public:
    using memory_space  = NektarSpaces::HostSpace;
    using memory_layout = Nektar::MemoryLayout::Consecutive;
};

class AVX
{
    // Used for legacy Nektar AVX tasks (e.g. no Kokkos)};
public:
    using memory_space  = NektarSpaces::HostSpace;
    using memory_layout = Nektar::MemoryLayout::Interlaced;
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

#define NEKTAR_USING_GPU

#endif

// Native pure CUDA execution
#if defined(NEKTAR_ENABLE_CUDA)

class CUDA
{
    // Used for legacy Nektar CUDA tasks (e.g. no Kokkos)
public:
    using memory_space  = NektarSpaces::DeviceSpace;
    using memory_layout = Nektar::MemoryLayout::Interlaced;
};

using DefaultExecutionSpace = CUDA;

#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::CUDA

// Native pure HIP execution
#elif defined(NEKTAR_ENABLE_HIP)

class HIP
{
    // Used for legacy Nektar HIP tasks (e.g. no Kokkos)
public:
    using memory_space  = NektarSpaces::DeviceSpace;
    using memory_layout = Nektar::MemoryLayout::Interlaced;
};

using DefaultExecutionSpace = HIP;

#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::HIP

// Native pure SYCL execution
#elif defined(NEKTAR_ENABLE_SYCL)

class SYCL
{
    // Used for legacy Nektar SYCL tasks (e.g. no Kokkos)
public:
    using memory_space  = NektarSpaces::DeviceSpace;
    using memory_layout = Nektar::MemoryLayout::Interlaced;
};

using DefaultExecutionSpace = SYCL;

#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::SYCL

// No specific GPU so the device is the host.
#else

using DeviceSpace           = HostSpace;
using DefaultExecutionSpace = DefaultHostExecutionSpace;

#define NEKTAR_DEFAULT_DEVICE_TAG NektarSpaces::DefaultHostExecutionSpace

#endif // GPU specific

} // namespace NektarSpaces

// Example of Kokkos options following Kokkos internal order:
//     https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_Core_fwd.hpp
//     https://github.com/kokkos/kokkos/wiki/Initialization

// Execution Space                            Memory Space
// -------------------------------------      --------------
// Kokkos::DefaultHostExecutionSpace
//                              Kokkos::DefaultHostExecutionSpace::memory_space
// Kokkos::DefaultExecutionSpace
//                              Kokkos::DefaultExecutionSpace::memory_space
//
// Kokkos chooses the two spaces using the following list:
//
// 1. Kokkos::Cuda                            Kokkos::CudaSpace
// 2. Kokkos::Experimental::OpenMPTarget Kokkos::Experimental::OpenMPTargetSpace
// 3. Kokkos::Experimental::OpenACC           Kokkos::Experimental::OpenACC
// 4. Kokkos::Experimental::HIP               Kokkos::Experimental::HIPSpace
// 5. Kokkos::Experimental::SYCL Kokkos::Experimental::SYCLDeviceUSMSpace
// 6. Kokkos::OpenMP                          Kokkos::HostSpace
// 7. Kokkos::Threads                         Kokkos::HostSpace
// 8. Kokkos::Experimental::HPX               Kokkos::HostSpace
// 9. Kokkos::Serial                          Kokkos::HostSpace
//

// The highest execution space in the list which is enabled is Kokkos'
// default execution space, and the highest enabled host execution
// space is Kokkos' default host execution space.

#if defined(NEKTAR_ENABLE_KOKKOS)

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

// These are used for LoopExecution.hpp reductions. There are
// equivalent for when Kokkos is not defined.
namespace NektarSpaces
{
template <typename TData> using ReduceSum = Kokkos::Sum<TData>;
template <typename TData> using ReduceMin = Kokkos::Min<TData>;
template <typename TData> using ReduceMax = Kokkos::Max<TData>;
} // namespace NektarSpaces

// For decaring functions.  There are equivalent for when Kokkoks is
// not defined.
#define GPU_FUNCTION KOKKOS_FUNCTION
#define GPU_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
#define GPU_FORCEINLINE_FUNCTION KOKKOS_FORCEINLINE_FUNCTION

// GPUs
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) ||      \
    defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_USING_GPU
#endif

#define KOKKOS_DEFAULT_HOST_TAG Kokkos::DefaultHostExecutionSpace
#define KOKKOS_DEFAULT_DEVICE_TAG Kokkos::DefaultExecutionSpace

#else // !defined(NEKTAR_ENABLE_KOKKOS)

// For decaring functions when Kokkoks is not defined.
#define GPU_FUNCTION
#define GPU_INLINE_FUNCTION inline
#define GPU_FORCEINLINE_FUNCTION inline

#define KOKKOS_LAMBDA [&]

// Kokkos is not included in this build. Create some stub types so
// these types at least exist and so #ifdef are not needed.
namespace Kokkos
{

class HostSpace
{
};

class DefaultExecutionSpace
{
public:
    using memory_space  = NektarSpaces::HostSpace;
    using memory_layout = Nektar::MemoryLayout::Consecutive;
};

// These functions duplicate the basic Kokkos atomic functions. These
// are only used for serial execution.
template <class T>
GPU_INLINE_FUNCTION void atomic_add(T *const dest, const T val)
{
    *dest += val;
}

#define KOKKOS_DEFAULT_HOST_TAG Kokkos::DefaultExecutionSpace
#define KOKKOS_DEFAULT_DEVICE_TAG Kokkos::DefaultExecutionSpace

} // namespace Kokkos

namespace NektarSpaces
{
// These classes duplicate the basic Kokkos reducers
template <class Scalar> class Sum
{
public:
    typedef typename std::remove_cv<Scalar>::type value_type;

    GPU_INLINE_FUNCTION
    Sum(value_type &value_) : m_value(value_)
    {
    }

    GPU_INLINE_FUNCTION
    void join(Scalar &dest, const Scalar &src) const
    {
        dest += src;
    }

    GPU_INLINE_FUNCTION
    void init(Scalar &dest) const
    {
        dest = (Scalar)0;
    }

    GPU_INLINE_FUNCTION
    value_type &reference() const
    {
        return m_value;
    }

private:
    value_type m_value;
};

template <class Scalar> class Min
{
public:
    typedef typename std::remove_cv<Scalar>::type value_type;

    GPU_INLINE_FUNCTION
    Min(value_type &value_) : m_value(value_)
    {
    }

    GPU_INLINE_FUNCTION
    void join(Scalar &dest, const Scalar &src) const
    {
        if (dest > src)
        {
            dest = src;
        }
    }

    GPU_INLINE_FUNCTION
    void init(Scalar &dest) const
    {
        if constexpr (std::is_same<Scalar, int>::value)
        {
            dest = INT_MAX;
        }
        else if constexpr (std::is_same<Scalar, unsigned int>::value)
        {
            dest = UINT_MAX;
        }
        else if constexpr (std::is_same<Scalar, float>::value)
        {
            dest = FLT_MAX;
        }
        else if constexpr (std::is_same<Scalar, double>::value)
        {
            dest = DBL_MAX;
        }
    }

    GPU_INLINE_FUNCTION
    value_type &reference() const
    {
        return m_value;
    }

private:
    value_type m_value;
};

template <class Scalar> class Max
{
public:
    typedef typename std::remove_cv<Scalar>::type value_type;

    GPU_INLINE_FUNCTION
    Max(value_type &value_) : m_value(value_)
    {
    }

    GPU_INLINE_FUNCTION
    void join(Scalar &dest, const Scalar &src) const
    {
        if (dest < src)
        {
            dest = src;
        }
    }

    GPU_INLINE_FUNCTION
    void init(Scalar &dest) const
    {
        if constexpr (std::is_same<Scalar, int>::value)
        {
            dest = -INT_MAX;
        }
        else if constexpr (std::is_same<Scalar, unsigned int>::value)
        {
            dest = 0;
        }
        else if constexpr (std::is_same<Scalar, float>::value)
        {
            dest = -FLT_MAX;
        }
        else if constexpr (std::is_same<Scalar, double>::value)
        {
            dest = -DBL_MAX;
        }
    }

    GPU_INLINE_FUNCTION
    value_type &reference() const
    {
        return m_value;
    }

private:
    value_type m_value;
};

template <typename TData> using ReduceSum = NektarSpaces::Sum<TData>;
template <typename TData> using ReduceMin = NektarSpaces::Min<TData>;
template <typename TData> using ReduceMax = NektarSpaces::Max<TData>;

} // namespace NektarSpaces

#endif // !defined(NEKTAR_ENABLE_KOKKOS)

// #pragma message "The value of NEKTAR_DEFAULT_HOST_TAG:
// "STRVX(NEKTAR_DEFAULT_HOST_TAG)

// #pragma message "The value of NEKTAR_DEFAULT_DEVICE_TAG: "
// STRVX(NEKTAR_DEFAULT_DEVICE_TAG)

// #pragma message "The value of KOKKOS_DEFAULT_HOST_TAG:
// "STRVX(KOKKOS_DEFAULT_HOST_TAG)

// #pragma message "The value of KOKKOS_DEFAULT_DEVICE_TAG: "
// STRVX(KOKKOS_DEFAULT_DEVICE_TAG)
