///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionKokkos.hpp
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

#if defined(NEKTAR_ENABLE_KOKKOS)

#include <LibUtilities/BasicUtils/MiscUtils.hpp>

namespace Nektar
{

template <typename ExecSpace, typename Scope, typename TData>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
atomic_add(TData *const dest, const TData val)
{
    Kokkos::atomic_add(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
atomic_sub(TData *const dest, const TData val)
{
    Kokkos::atomic_sub(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
atomic_max(TData *const dest, const TData val)
{
    Kokkos::atomic_max(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
atomic_min(TData *const dest, const TData val)
{
    Kokkos::atomic_min(dest, val);
}

template <typename ExecSpace, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_for(const int begin, const int end, const Functor &functor)
{
    const unsigned int blockSize = NektarSpaces::KOKKOS::defaultBlockSize;

    std::string name = Nektar::demangleTypeName(typeid(Functor));

    Kokkos::RangePolicy<> rangePolicy(begin, end, Kokkos::ChunkSize(blockSize));

    Kokkos::parallel_for(name, rangePolicy, functor);
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_reduce(const int begin, const int end, const Functor &functor,
                typename Reduction::value_type &red)
{
    const unsigned int blockSize = NektarSpaces::KOKKOS::defaultBlockSize;

    using TData = typename Reduction::value_type;

    std::string name = Nektar::demangleTypeName(typeid(Functor));

    Kokkos::RangePolicy<> rangePolicy(begin, end, Kokkos::ChunkSize(blockSize));

    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        Kokkos::parallel_reduce(name, rangePolicy, functor,
                                Kokkos::Sum<TData>(red));
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        Kokkos::parallel_reduce(name, rangePolicy, functor,
                                Kokkos::Max<TData>(red));
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        Kokkos::parallel_reduce(name, rangePolicy, functor,
                                Kokkos::Min<TData>(red));
    }
}

} // namespace Nektar

#endif
