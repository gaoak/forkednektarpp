///////////////////////////////////////////////////////////////////////////////
//
// File: CNABKernelLaunchers.hpp
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

namespace Nektar::SolverCore::detail
{

class CNABScheme;
class CNABModifiedScheme;

template <typename Scheme, typename TData>
NEK_DEVICE_INLINE static constexpr auto GetCNABCoefficients(void)
{
    if constexpr (std::is_same_v<Scheme, CNABScheme>)
    {
        return std::array<TData, 4>{1.0 / 2.0, 3.0 / 2.0, -1.0 / 2.0, 1.0};
    }
    else if constexpr (std::is_same_v<Scheme, CNABModifiedScheme>)
    {
        return std::array<TData, 5>{3.0 / 8.0, 1.0 / 16.0, 3.0 / 2.0,
                                    -1.0 / 2.0, 1.0};
    }
}

template <typename Scheme, typename TData, unsigned int... ind,
          typename... TDatas,
          std::enable_if_t<std::is_same_v<Scheme, CNABScheme> ||
                               std::is_same_v<Scheme, CNABModifiedScheme>,
                           bool>
              Enable = true>
NEK_DEVICE_INLINE static void UpdateSolutionKernelImpl(
    const size_t idx, TData *__restrict inout,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... solutions)
{
    constexpr unsigned int nSolution = sizeof...(solutions);

    constexpr auto coeff = GetCNABCoefficients<Scheme, TData>();

    TData tmp = ((solutions[idx] * coeff[ind]) + ...);
    tmp += inout[idx] * coeff[nSolution];
    inout[idx] = tmp;
}

} // namespace Nektar::SolverCore::detail

#include "SolverCore/TimeOps/TimeOpHelper.hpp"
