///////////////////////////////////////////////////////////////////////////////
//
// File: AdamsMoultonKernelLaunchers.hpp
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

#include "SolverCore/TimeOps/TimeOpKernelHelper.hpp"

namespace Nektar::SolverCore::detail
{

class AdamsMoultonScheme;

template <unsigned int IntOrder, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetAdamsMoultonCoefficients(void)
{
    // 2nd order
    if constexpr (IntOrder == 2)
    {
        return std::array<TData, 1>{1.0 / 2.0};
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        return std::array<TData, 2>{8.0 / 12.0, -1.0 / 12.0};
    }
    // 4th order
    else if constexpr (IntOrder == 4)
    {
        return std::array<TData, 3>{19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0};
    }
}

template <typename Scheme, typename TData, unsigned int... ind,
          typename... TDatas,
          std::enable_if_t<std::is_same_v<Scheme, AdamsMoultonScheme>, bool>
              Enable = true>
NEK_DEVICE_INLINE static void UpdateSolutionKernelImpl(
    const size_t idx, TData *__restrict inout,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr unsigned int IntOrder = sizeof...(implicits) + 1;

    constexpr auto coeff = GetAdamsMoultonCoefficients<IntOrder, TData>();

    inout[idx] +=
        (ScaledTerm<coeff[ind] == TData(0)>(implicits, idx, coeff[ind]) + ...);
}

} // namespace Nektar::SolverCore::detail

#include "SolverCore/TimeOps/TimeOpHelper.hpp"
