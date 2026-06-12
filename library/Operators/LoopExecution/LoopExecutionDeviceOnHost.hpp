///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionDeviceOnHost.hpp
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

#if defined(NEKTAR_ENABLE_DEVICEONHOST)

namespace Nektar
{

// Parallel for launchers.
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const size_t begin, const size_t end, const Functor &functor)
{
    for (size_t i = begin; i < end; ++i)
    {
        functor(i);
    }
}

// Parallel reduction launchers without device-to-host copy.
template <typename ExecSpace, bool init, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type *red)
{
    using TData = typename Reduction::value_type;

    if (init)
    {
        if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
        {
            *red = 0.0;
        }
        else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
        {
            *red = std::numeric_limits<TData>::lowest();
        }
        else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
        {
            *red = std::numeric_limits<TData>::max();
        }
    }

    for (size_t i = begin; i < end; ++i)
    {
        if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
        {
            *red += functor(i);
        }
        else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
        {
            *red = std::max(*red, functor(i));
        }
        else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
        {
            *red = std::min(*red, functor(i));
        }
    }
}

// Parallel reduction launchers with device-to-host copy.
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type &red)
{
    parallel_reduce<ExecSpace, true, Reduction>(begin, end, functor, &red);
}

} // namespace Nektar

#endif
