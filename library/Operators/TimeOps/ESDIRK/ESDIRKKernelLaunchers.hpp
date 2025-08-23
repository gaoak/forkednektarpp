///////////////////////////////////////////////////////////////////////////////
//
// File: ESDIRKKernelLaunchers.hpp
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

class ESDIRKscheme;

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetESDIRKCoefficients(void)
{
    constexpr TData ConstSqrt2 = 1.414213562373095;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.

        constexpr TData lambda = (2.0 - ConstSqrt2) / 2.0;

        // clang-format off
        return std::array<TData, 3>
                {   // 1st stage
                    lambda,
                    // 2nd stage
                    (-1.0 + 6.0 * lambda - 4 * lambda * lambda) / (4 * lambda),
                    ( 1.0 - 2.0 * lambda) / (4 * lambda)
                };
        // clang-format on
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.

        // clang-format off
        return std::array<TData, 10> 
                {   // 1st stage
                    9.0 / 40.0,
                    // 2n stage
                    9.0 * (1.0 + ConstSqrt2) / 80.0,
                    9.0 * (1.0 + ConstSqrt2) / 80.0,
                    // 3rd stage
                    (22.0 + 15.0 * ConstSqrt2) / (80.0 * (1.0 + ConstSqrt2)),
                    (22.0 + 15.0 * ConstSqrt2) / (80.0 * (1.0 + ConstSqrt2)),
                    -7.0 / (40.0 * (1.0 + ConstSqrt2)),
                    // 4th stage
                    (2398.0 + 1205.0 * ConstSqrt2) / (2835.0 * (4.0 + 3.0 *
    ConstSqrt2)), (2398.0 + 1205.0 * ConstSqrt2) / (2835.0 * (4.0 + 3.0 *
    ConstSqrt2)), -2374.0 * (1.0 + 2.0 * ConstSqrt2) / (2835.0 * (5.0 + 3.0 *
    ConstSqrt2)), 5827.0 / 7560.0
                };
        // clang-format on
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        return std::array<TData, 15> 
                {  // 1st stage
                   0.25,
                   // 2nd stage
                   (1.0 - ConstSqrt2) / 8.0, (1.0 - ConstSqrt2) / 8.0,
                   // 3rd stage
                   (5.0 - 7.0 * ConstSqrt2) / 64.0,
                   (5.0 - 7.0 * ConstSqrt2) / 64.0,
                   7.0 * (1.0 + ConstSqrt2) / 32.0,
                   // 4th stage
                   (-13796.0 - 54539.0 * ConstSqrt2) / 125000.0,
                   (-13796.0 - 54539.0 * ConstSqrt2) / 125000.0,
                   (506605.0 + 132109.0 * ConstSqrt2) / 437500.0,
                   166.0 * (-97.0 + 376.0 * ConstSqrt2) / 109375.0,
                   // 5th stage
                   (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
                   (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
                   47.0 * (-267.0 + 1783.0 * ConstSqrt2) / 273343.0,
                  -16.0 * (-22922.0 + 3525.0 * ConstSqrt2) / 571953.0,
                  -15625.0 * (97.0 + 376.0 * ConstSqrt2) / 90749876.0};
        // clang-format on
    }
}

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetESDIRKCoefficients2(void)
{
    constexpr TData ConstSqrt2 = 1.414213562373095;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        constexpr TData lambda = (2.0 - ConstSqrt2) / 2.0;

        return std::array<TData, 3>{
                    (-1.0 + 6.0 * lambda - 4 * lambda * lambda) / (4 * lambda),
                    ( 1.0 - 2.0 * lambda) / (4 * lambda), lambda};
    }
    else if constexpr (IntOrder == 3)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        return std::array<TData, 5>{(2398.0 + 1205.0 * ConstSqrt2) /
                                       (2835.0 * (4.0 + 3.0 * ConstSqrt2)),
                                   (2398.0 + 1205.0 * ConstSqrt2) /
                                       (2835.0 * (4.0 + 3.0 * ConstSqrt2)),
                                   -2374.0 * (1.0 + 2.0 * ConstSqrt2) /
                                       (2835.0 * (5.0 + 3.0 * ConstSqrt2)),
                                   5827.0 / 7560.0, 9.0 / 40.0};
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        // See: Kennedy, Christopher A., and Mark H. Carpenter.
        // Diagonally implicit Runge-Kutta methods for ordinary
        // differential equations. A review. No. NF1676L-19716. 2016.
        // clang-format off
        return std::array<TData, 6>{
            (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
            (1181.0 - 987.0 * ConstSqrt2) / 13782.0,
            47.0 * (-267.0 + 1783.0 * ConstSqrt2) / 273343.0,
            -16.0 * (-22922.0 + 3525.0 * ConstSqrt2) / 571953.0,
            -15625.0 * (97.0 + 376.0 * ConstSqrt2) / 90749876.0,
            0.25};
    }
}

template <typename Scheme, unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, ESDIRKscheme>, void>::type
UpdateStageKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr unsigned int stage    = sizeof...(implicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    constexpr auto coeff = GetESDIRKCoefficients<IntOrder, TData>();

    inout[idx] =
        solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, ESDIRKscheme>, void>::type
    UpdateSolutionKernelImpl(const size_t idx, TData *__restrict inout,
                             const TData *__restrict solution,
                             std::integer_sequence<unsigned int, ind...>,
                             const TDatas *__restrict... implicits)
{
    constexpr auto coeff = GetESDIRKCoefficients2<IntOrder, TData>();

    inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
}

} // namespace Nektar::Operators::detail

#include "Operators/TimeOps/TimeOpHelper.hpp"
