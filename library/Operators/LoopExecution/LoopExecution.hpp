///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecution.hpp
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
// Description: Provide support for Nektar::parallel_for and
// Nektar::paralle_reduce portablity functions. Nektar::parallel_for and
// Nektar::paralle_reduce should only be used within an Operator class and not
// at the solver level. Nektar::parallel_for does NOT have provision for shared
// memory and in-kernel memory synchronization.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <limits>

#include "Operators/Common/Memory/MemoryAlloc.hpp"
#include "Operators/Common/OperatorsDeclspec.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar
{

extern OPERATORS_EXPORT unsigned int internalLoopExecutionStreamID;

[[maybe_unused]] static void LoopExecutionSetStreamID(
    const unsigned int streamID)
{
    internalLoopExecutionStreamID = streamID;
}

// NEKTAR_LAMBDA
// Use in Nektar::parallel_for:
//   Nektar::parallel_for<ExecSpace>(...,
//       NEKTAR_LAMBDA(const size_t idx) { ... });
// and Nektar::parallel_reduce:
//   Nektar::parallel_reduce<ExecSpace, Reduce>(
//       0, size, NEKTAR_LAMBDA(size_t idx) { ... }, tmp);
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEKTAR_LAMBDA [=] __device__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEKTAR_LAMBDA [=]
#else
#define NEKTAR_LAMBDA [&]
#endif

// Reduction
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

// Atomics.
template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    *dest += val;
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    Nektar::atomic_add<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    *dest -= val;
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    Nektar::atomic_sub<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    *dest = std::max(*dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    Nektar::atomic_max<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    *dest = std::min(*dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    Nektar::atomic_min<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_or(TData *const dest, const TData val)
{
    *dest |= val;
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_or(TData *const dest, const TData val)
{
    Nektar::atomic_or<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
static inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    atomic_and(TData *const dest, const TData val)
{
    *dest &= val;
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_and(TData *const dest, const TData val)
{
    Nektar::atomic_and<Scope>(dest, val);
}

} // namespace Nektar

#include "Operators/LoopExecution/LoopExecutionDeviceOnHost.hpp"
#include "Operators/LoopExecution/LoopExecutionHIPCUDA.hpp"
#include "Operators/LoopExecution/LoopExecutionSYCL.hpp"
#include "Operators/LoopExecution/LoopExecutionSerialAVX.hpp"
