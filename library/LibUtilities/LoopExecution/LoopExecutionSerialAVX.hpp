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

// Parallel for launchers.
template <typename ExecSpace, typename Functor,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
inline void parallel_for(const size_t begin, const size_t end,
                         const Functor &functor)
{
    for (size_t i = begin; i < end; ++i)
    {
        functor(i);
    }
}

// Parallel reduction launchers.
template <typename ExecSpace, bool init, typename Reduction, typename Functor,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type *red)
{
    if (init)
    {
        *red = reduce_identity<Reduction>();
    }

    for (size_t i = begin; i < end; ++i)
    {
        *red = reduce_join<Reduction>(*red, functor(i));
    }
}

// Parallel reduction launchers.
template <typename ExecSpace, typename Reduction, typename Functor,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type &red)
{
    parallel_reduce<ExecSpace, true, Reduction>(begin, end, functor, &red);
}

// Parallel reduction of nred values: functor(i, vals) writes those of index i
// to vals[0, nred), which Reduction combines into red[0, nred), together with
// the values already there unless init. At most 4 values, the partial results
// the reduction scratch holds.
template <typename ExecSpace, bool init, typename Reduction, unsigned int nred,
          typename Functor,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type *red)
{
    using TData = typename Reduction::value_type;

    static_assert(nred <= 4, "parallel_reduce: at most 4 values.");

    TData acc[nred];
    for (unsigned int s = 0; s < nred; ++s)
    {
        acc[s] = reduce_identity<Reduction>();
    }

    for (size_t i = begin; i < end; ++i)
    {
        TData vals[nred];
        functor(i, vals);
        for (unsigned int s = 0; s < nred; ++s)
        {
            acc[s] = reduce_join<Reduction>(acc[s], vals[s]);
        }
    }

    for (unsigned int s = 0; s < nred; ++s)
    {
        red[s] = init ? acc[s] : reduce_join<Reduction>(red[s], acc[s]);
    }
}

} // namespace Nektar
