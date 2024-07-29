///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionSerialAVX.hpp
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

namespace Nektar
{

// CPU serial 1D range parallel_for
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
parallel_for(const int begin, const int end, const Functor &functor)
{
    for (int i = begin; i < end; ++i)
    {
        functor(i);
    }
}

// CPU serial 1D range parallel_reduce
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
parallel_reduce(const int begin, const int end, const Functor &functor,
                typename Reduction::value_type &red)
{
    using TData = typename Reduction::value_type;

    if constexpr (std::is_same<Reduction,
                               NektarSpaces::ReduceSum<TData>>::value)
    {
        red = 0.0;
    }
    else if constexpr (std::is_same<Reduction,
                                    NektarSpaces::ReduceMax<TData>>::value)
    {
        red = std::numeric_limits<TData>::min();
    }
    else if constexpr (std::is_same<Reduction,
                                    NektarSpaces::ReduceMin<TData>>::value)
    {
        red = std::numeric_limits<TData>::max();
    }

    for (int i = begin; i < end; ++i)
    {
        functor(i, red);
    }
}

// CPU serial block range parallel_for
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
parallel_for(BlockRange const &r, const Functor &functor)
{
    const int rbegin0 = r.begin(0);
    const int rbegin1 = r.begin(1);
    const int rbegin2 = r.begin(2);

    const int rend0 = r.end(0);
    const int rend1 = r.end(1);
    const int rend2 = r.end(2);

    for (int k = rbegin2; k < rend2; ++k)
    {
        for (int j = rbegin1; j < rend1; ++j)
        {
            for (int i = rbegin0; i < rend0; ++i)
            {
                functor(i, j, k);
            }
        }
    }
}

// CPU serial block range parallel_reduce
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
parallel_reduce(BlockRange const &r, const Functor &functor,
                typename Reduction::value_type &red)
{
    using TData = typename Reduction::value_type;

    if constexpr (std::is_same<Reduction,
                               NektarSpaces::ReduceSum<TData>>::value)
    {
        red = 0.0;
    }
    else if constexpr (std::is_same<Reduction,
                                    NektarSpaces::ReduceMax<TData>>::value)
    {
        red = std::numeric_limits<TData>::min();
    }
    else if constexpr (std::is_same<Reduction,
                                    NektarSpaces::ReduceMin<TData>>::value)
    {
        red = std::numeric_limits<TData>::max();
    }

    const int rbegin0 = r.begin(0);
    const int rbegin1 = r.begin(1);
    const int rbegin2 = r.begin(2);

    const int rend0 = r.end(0);
    const int rend1 = r.end(1);
    const int rend2 = r.end(2);

    for (int k = rbegin2; k < rend2; ++k)
    {
        for (int j = rbegin1; j < rend1; ++j)
        {
            for (int i = rbegin0; i < rend0; ++i)
            {
                functor(i, j, k, red);
            }
        }
    }
}

} // namespace Nektar
