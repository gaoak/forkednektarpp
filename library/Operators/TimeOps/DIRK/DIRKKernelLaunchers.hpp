///////////////////////////////////////////////////////////////////////////////
//
// File: DIRKKernelLaunchers.hpp
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

class DIRKscheme;

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetDIRKCoefficients(void)
{
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData lambda     = (2.0 - ConstSqrt2) / 2.0;

        // clang-format off
        return std::array<TData, 1>
                { 1.0 - lambda };
        // clang-format on
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        constexpr TData lambda = 0.4358665215;

        // clang-format off
        return std::array<TData, 3> 
                {  // 1st stage
                   0.5 * (1.0 - lambda),
                   // 2nd stage
                   0.25 * (-6.0 * lambda * lambda + 16.0 * lambda - 1.0), 
                   0.25 * (6.0 * lambda * lambda - 20.0 * lambda + 5.0)
                };
        // clang-format on
    }
}

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetDIRKCoefficients2(void)
{
    // 1st order
    if constexpr (IntOrder == 1)
    {
        return std::array<TData, 1>{1.0};
    }
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData lambda     = (2.0 - ConstSqrt2) / 2.0;

        return std::array<TData, 2>{1.0 - lambda, lambda};
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        constexpr TData lambda = 0.4358665215;

        return std::array<TData, 3>{
            0.25 * (-6.0 * lambda * lambda + 16.0 * lambda - 1.0),
            0.25 * (6.0 * lambda * lambda - 20.0 * lambda + 5.0), lambda};
    }
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, DIRKscheme>, void>::type
    UpdateStageKernelImpl(const size_t idx, TData *__restrict inout,
                          const TData *__restrict solution,
                          std::integer_sequence<unsigned int, ind...>,
                          const TDatas *__restrict... implicits)
{
    constexpr unsigned int stage    = sizeof...(implicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    constexpr auto coeff = GetDIRKCoefficients<IntOrder, TData>();

    inout[idx] =
        solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, DIRKscheme>, void>::type
    UpdateSolutionKernelImpl(const size_t idx, TData *__restrict inout,
                             const TData *__restrict solution,
                             std::integer_sequence<unsigned int, ind...>,
                             const TDatas *__restrict... implicits)
{
    constexpr auto coeff = GetDIRKCoefficients2<IntOrder, TData>();

    inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
}

} // namespace Nektar::Operators::detail

#include "Operators/TimeOps/TimeOpHelper.hpp"
