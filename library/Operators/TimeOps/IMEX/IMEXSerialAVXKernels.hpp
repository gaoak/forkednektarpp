///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXSerialAVXKernels.hpp
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

/*
 * Compute extrapolation of previous time steps and explicit parts.
 * extrapolation = \sum_q=0^{J-1} \alpha_q / \gamma u^{n-q}
 *               + \sum_q=0^{J-1} \beta_q f(u^{n-q})
 *               where J = IntOrder
 *  Note that the \alpha coefficients are divided by \gamma while the \beta
 *  coefficients are not.
 */
template <typename ExecSpace, typename TData, unsigned int IntOrder>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    ExtrapolateIMEXKernel(const size_t nsize, const TData dt,
                          const TData *const *solutions,
                          const TData *const *explicits, TData *inoutPtr)
{
    // use tmp to avoid multiple read-write to memory
    TData tmp;

    for (size_t i = 0; i < nsize; i++)
    {
        // Based on Karniadakis, Israeli and Orszag 1991, table IV
        if constexpr (IntOrder == 2)
        {
            tmp = 2.0 * solutions[0][i];
            tmp += -1.0 / 2.0 * inoutPtr[i];
            tmp /= dt;

            tmp += -1.0 * explicits[0][i];
            tmp += 2.0 * explicits[1][i];
        }
        // Based on Karniadakis, Israeli and Orszag 1991, table IV
        if constexpr (IntOrder == 3)
        {
            tmp = 3.0 * solutions[0][i];
            tmp += -3.0 / 2.0 * solutions[1][i];
            tmp += 1.0 / 3.0 * inoutPtr[i];
            tmp /= dt;

            tmp += 1.0 * explicits[0][i];
            tmp += 3.0 * explicits[1][i];
            tmp += -3.0 * explicits[2][i];
        }
        // Based on Asher, Ruuth and Wetton 1995, equation (33)
        if constexpr (IntOrder == 4)
        {
            tmp = 4.0 * solutions[0][i];
            tmp += -3.0 * solutions[1][i];
            tmp += 4.0 / 3.0 * solutions[2][i];
            tmp += -1.0 / 4.0 * inoutPtr[i];
            tmp /= dt;

            tmp += -1.0 * explicits[0][i];
            tmp += 4.0 * explicits[1][i];
            tmp += -6.0 * explicits[2][i];
            tmp += 4.0 * explicits[3][i];
        }
        inoutPtr[i] = tmp; // Write once to global memory
    }
}

} // namespace Nektar::Operators::detail
