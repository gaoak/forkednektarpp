///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXdirkKernelLaunchers.hpp
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

namespace Nektar::Operators::detail
{

class IMEXdirkScheme;

template <unsigned int ImpStage, unsigned int ExpStage, unsigned int IntOrder,
          typename TData>
static constexpr auto GetIMEXdirkTimeCoefficients(void)
{
    // IMEX Dirk 1 1 1 : Forward - Backward Euler IMEX
    if constexpr (ImpStage == 1 && ExpStage == 1 && IntOrder == 1)
    {
        return std::array<TData, 2>{0.0, 1.0};
    }
    // IMEX Dirk 1 2 1 : Forward - Backward Euler IMEX w/B implicit == B
    // explicit
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 1)
    {
        return std::array<TData, 2>{0.0, 1.0};
    }
    // IMEX Dirk 1 2 2 : Implict-Explicit Midpoint IMEX
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 2)
    {
        return std::array<TData, 2>{0.0, 0.5};
    }
    // IMEX Dirk 2 2 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 2 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = 1.0 - ConstSqrt2 / 2.0;

        return std::array<TData, 3>{0.0, gamma, 1.0};
    }
    // IMEX Dirk 2 3 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = (2.0 - ConstSqrt2) / 2.0;

        return std::array<TData, 3>{0.0, gamma, 1.0};
    }
    // IMEX Dirk 2 3 3 : L Stable, two stage, third order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 3)
    {
        constexpr TData ConstSqrt3 = 1.732050807568877;
        constexpr TData gamma      = (3.0 + ConstSqrt3) / 6.0;

        return std::array<TData, 3>{0.0, gamma, 1.0 - gamma};
    }
    // IMEX Dirk 3 4 3 : L Stable, three stage, third order IMEX
    else if constexpr (ImpStage == 3 && ExpStage == 4 && IntOrder == 3)
    {
        return std::array<TData, 4>{0.0, 0.4358665215, 0.7179332608, 1.0};
    }
    // IMEX Dirk 4 4 3 : L Stable, four stage, third order IMEX
    else if constexpr (ImpStage == 4 && ExpStage == 4 && IntOrder == 3)
    {
        return std::array<TData, 5>{0.0, 0.5, 2.0 / 3.0, 0.5, 1.0};
    }
}

template <unsigned int ImpStage, unsigned int ExpStage, unsigned int IntOrder,
          typename TData>
static constexpr auto GetIMEXdirkLambdaCoefficients(void)
{
    // IMEX Dirk 1 1 1 : Forward - Backward Euler IMEX
    if constexpr (ImpStage == 1 && ExpStage == 1 && IntOrder == 1)
    {
        return std::array<TData, 1>{1.0};
    }
    // IMEX Dirk 1 2 1 : Forward - Backward Euler IMEX w/B implicit == B
    // explicit
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 1)
    {
        return std::array<TData, 1>{1.0};
    }
    // IMEX Dirk 1 2 2 : Implict-Explicit Midpoint IMEX
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 2)
    {
        return std::array<TData, 1>{0.5};
    }
    // IMEX Dirk 2 2 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 2 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = 1.0 - ConstSqrt2 / 2.0;

        return std::array<TData, 2>{gamma, gamma};
    }
    // IMEX Dirk 2 3 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = (2.0 - ConstSqrt2) / 2.0;

        return std::array<TData, 2>{gamma, gamma};
    }
    // IMEX Dirk 2 3 3 : L Stable, two stage, third order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 3)
    {
        constexpr TData ConstSqrt3 = 1.732050807568877;
        constexpr TData gamma      = (3.0 + ConstSqrt3) / 6.0;

        return std::array<TData, 2>{gamma, gamma};
    }
    // IMEX Dirk 3 4 3 : L Stable, three stage, third order IMEX
    else if constexpr (ImpStage == 3 && ExpStage == 4 && IntOrder == 3)
    {
        return std::array<TData, 3>{0.4358665215, 0.4358665215, 0.4358665215};
    }
    // IMEX Dirk 4 4 3 : L Stable, four stage, third order IMEX
    else if constexpr (ImpStage == 4 && ExpStage == 4 && IntOrder == 3)
    {
        return std::array<TData, 4>{0.5, 0.5, 0.5, 0.5};
    }
}

template <unsigned int ImpStage, unsigned int ExpStage, unsigned int IntOrder,
          typename TData>
NEK_DEVICE_INLINE static constexpr auto GetIMEXdirkStageCoefficients(void)
{
    // IMEX Dirk 1 1 1 : Forward - Backward Euler IMEX
    if constexpr (ImpStage == 1 && ExpStage == 1 && IntOrder == 1)
    {
        // clang-format off
        return std::array<TData, 1>
                {   
                    // 1st stage (Explicit)
                    1.0
                };
        // clang-format on
    }
    // IMEX Dirk 1 2 1 : Forward - Backward Euler IMEX w/B implicit == B
    // explicit
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 1)
    {
        // clang-format off
        return std::array<TData, 1>
                {   
                    // 1st stage (Explicit)
                    1.0
                };
        // clang-format on
    }
    // IMEX Dirk 1 2 2 : Implict-Explicit Midpoint IMEX
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 2)
    {
        // clang-format off
        return std::array<TData, 1>
                {   
                    // 1st stage (Explicit)
                    0.5
                };
        // clang-format on
    }
    // IMEX Dirk 2 2 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 2 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = 1.0 - ConstSqrt2 / 2.0;
        constexpr TData delta      = -ConstSqrt2 / 2.0;

        // clang-format off
        return std::array<TData, 4>
                {   
                    // 1st stage (Explicit)
                    gamma,
                    // 2nd stage (Implicit)
                    1.0 - gamma,
                    // 2nd stage (Explicit)
                    delta, 1.0 - delta,
                };
        // clang-format on
    }
    // IMEX Dirk 2 3 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = (2.0 - ConstSqrt2) / 2.0;
        constexpr TData delta      = -2.0 * ConstSqrt2 / 3.0;

        // clang-format off
        return std::array<TData, 4>
                {   
                    // 1st stage (Explicit)
                    gamma,
                    // 2nd stage (Implicit)
                    1.0 - gamma,
                    // 2nd stage (Explicit)
                    delta, 1.0 - delta
                };
        // clang-format on
    }
    // IMEX Dirk 2 3 3 : L Stable, two stage, third order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 3)
    {
        constexpr TData ConstSqrt3 = 1.732050807568877;
        constexpr TData gamma      = (3.0 + ConstSqrt3) / 6.0;

        // clang-format off
        return std::array<TData, 4>
                {   
                    // 1st stage (Explicit)
                    gamma,
                    // 2nd stage (Implicit)
                    1.0 - 2 * gamma,
                    // 2nd stage (Explicit)
                    gamma - 1.0, 2 * (1.0 - gamma)
                };
        // clang-format on
    }
    // IMEX Dirk 3 4 3 : L Stable, three stage, third order IMEX
    else if constexpr (ImpStage == 3 && ExpStage == 4 && IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 9> 
                {  
                   // 1st stage (Explicit)
                   0.4358665215,
                   // 2nd stage (Implicit)
                   0.28206673925,
                   // 2nd stage (Explicit)
                   0.32127888624401013, 0.3966543745059896,
                   // 3rd stage (Implicit)
                   1.208496649153235, -0.6443631706532351,
                   // 3rd stage (Explicit)
                  -0.1058582958, 0.5529291479, 0.5529291479
                };
        // clang-format on
    }
    // IMEX Dirk 4 4 3 : L Stable, four stage, third order IMEX
    else if constexpr (ImpStage == 4 && ExpStage == 4 && IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 16>
                {   
                    // 1st stage (Explicit)
                    0.5,
                    // 2nd stage (Implicit)
                    1.0 / 6.0,
                    // 2nd stage (Explicit)
                    11.0 / 18.0, 1.0 / 18.0,
                    // 3rd stage (Implicit)
                    -0.5, 0.5,
                    // 3rd stage (Explicit)
                    5.0 / 6.0, -5.0 / 6.0, 1.0 / 2.0,
                    // 4th stage (Implicit)
                    1.5, - 1.5, 0.5,
                    // 4th stage (Explicit)
                    1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, - 7.0 / 4.0
                };
        // clang-format on
    }
}

template <unsigned int ImpStage, unsigned int ExpStage, unsigned int IntOrder,
          typename TData>
NEK_DEVICE_INLINE static constexpr auto GetIMEXdirkCoefficients(void)
{
    // IMEX Dirk 1 1 1 : Forward - Backward Euler IMEX
    if constexpr (ImpStage == 1 && ExpStage == 1 && IntOrder == 1)
    {
        // clang-format off
        return std::array<TData, 2>
                 {
                     // Implicit
                     1.0,
                     // Explicit
                     1.0
                 };
        // clang-format on
    }
    // IMEX Dirk 1 2 1 : Forward - Backward Euler IMEX w/B implicit == B
    // explicit
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 1)
    {
        // clang-format off
        return std::array<TData, 3>
                 {
                     // Implicit
                     1.0,
                     // Explicit
                     0.0, 1.0
                 };
        // clang-format on
    }
    // IMEX Dirk 1 2 2 : Implict-Explicit Midpoint IMEX
    else if constexpr (ImpStage == 1 && ExpStage == 2 && IntOrder == 2)
    {
        // clang-format off
        return std::array<TData, 3>
                 {
                     // Implicit
                     1.0,
                     // Explicit
                     0.0, 1.0
                 };
        // clang-format on
    }
    // IMEX Dirk 2 2 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 2 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = 1.0 - ConstSqrt2 / 2.0;
        constexpr TData delta      = -ConstSqrt2 / 2.0;

        // clang-format off
        return std::array<TData, 4>
                 {
                     // Implicit
                     1.0 - gamma, gamma,
                     // Explicit
                     delta, 1.0 - delta
                 };
        // clang-format on
    }
    // IMEX Dirk 2 3 2 : L Stable, two stage, second order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData gamma      = (2.0 - ConstSqrt2) / 2.0;

        // clang-format off
        return std::array<TData, 5>
                 {
                     // Implicit
                     1.0 - gamma, gamma,
                     // Explicit
                     0.0, 1.0 - gamma, gamma
                 };
        // clang-format on
    }
    // IMEX Dirk 2 3 3 : L Stable, two stage, third order IMEX
    else if constexpr (ImpStage == 2 && ExpStage == 3 && IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 5>
                 {
                     // Implicit
                     0.5, 0.5,
                     // Explicit
                     0.0, 0.5, 0.5
                 };
        // clang-format on
    }
    // IMEX Dirk 3 4 3 : L Stable, three stage, third order IMEX
    else if constexpr (ImpStage == 3 && ExpStage == 4 && IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 7>
                 {
                     // Implicit
                     1.208496649153235, -0.6443631706532351, 0.4358665215,
                     // Explicit
                     0.0, 1.208496649153235, -0.6443631706532351, 0.4358665215
                 };
        // clang-format on
    }
    // IMEX Dirk 4 4 3 : L Stable, four stage, third order IMEX
    else if constexpr (ImpStage == 4 && ExpStage == 4 && IntOrder == 3)
    {
        // clang-format off
        return std::array<TData, 8>
                 {
                     // Implicit
                     3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0,
                     // Explicit
                     1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0
                 };
        // clang-format on
    }
}

template <typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
          unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, IMEXdirkScheme>, void>::type
    UpdateStageKernelImpl(const size_t idx, TData *__restrict out,
                          const TData *__restrict solution,
                          std::integer_sequence<unsigned int, ind...>,
                          const TDatas *__restrict... residuals)
{
    constexpr unsigned int stage = (sizeof...(residuals) + 1) / 2;
    constexpr unsigned int indStart =
        stage * (stage - 1) / 2 + (stage - 1) * (stage - 2) / 2;

    constexpr auto coeff =
        GetIMEXdirkStageCoefficients<ImpStage, ExpStage, IntOrder, TData>();

    out[idx] = solution[idx] + ((residuals[idx] * coeff[indStart + ind]) + ...);
}

template <typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
          unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static
    typename std::enable_if<std::is_same_v<Scheme, IMEXdirkScheme>, void>::type
    UpdateSolutionKernelImpl(const size_t idx, TData *__restrict out,
                             const TData *__restrict solution,
                             std::integer_sequence<unsigned int, ind...>,
                             const TDatas *__restrict... residuals)
{
    constexpr auto coeff =
        GetIMEXdirkCoefficients<ImpStage, ExpStage, IntOrder, TData>();

    out[idx] = solution[idx] + ((residuals[idx] * coeff[ind]) + ...);
}

} // namespace Nektar::Operators::detail

#include "Operators/TimeOps/TimeOpHelper.hpp"
