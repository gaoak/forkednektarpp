///////////////////////////////////////////////////////////////////////////////
//
// File: RungeKuttaSSPKernelLaunchers.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

class RKSSPscheme;

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetRungeKuttaSSPStageCoefficients(void)
{
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // clang-format off
        return std::array<TData, 1>
                {  1.0 };
        // clang-format on
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 3>
                {  1.0,
                   1.0/4.0,   1.0/4.0 };
        // clang-format on
    }
}

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetRungeKuttaSSPCoefficients(void)
{
    // 1st order
    if constexpr (IntOrder == 1)
    {
        return std::array<TData, 1>{1.0};
    }
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        return std::array<TData, 2>{1.0 / 2.0, 1.0 / 2.0};
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        return std::array<TData, 3>{1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0};
    }
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, RKSSPscheme>, void>::type
    UpdateStageKernelImpl(const size_t idx, TData *__restrict inout,
                          const TData *__restrict solution,
                          std::integer_sequence<unsigned int, ind...>,
                          const TDatas *__restrict... explicits)
{
    constexpr unsigned int stage    = sizeof...(explicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    constexpr auto coeff = GetRungeKuttaSSPStageCoefficients<IntOrder, TData>();

    inout[idx] =
        solution[idx] + ((explicits[idx] * coeff[indStart + ind]) + ...);
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, RKSSPscheme>, void>::type
    UpdateSolutionKernelImpl(const size_t idx, TData *__restrict inout,
                             const TData *__restrict solution,
                             std::integer_sequence<unsigned int, ind...>,
                             const TDatas *__restrict... explicits)
{
    constexpr auto coeff = GetRungeKuttaSSPCoefficients<IntOrder, TData>();

    inout[idx] = solution[idx] + ((explicits[idx] * coeff[ind]) + ...);
}

} // namespace Nektar::Operators::detail

#include "Operators/TimeOps/TimeOpHelper.hpp"
