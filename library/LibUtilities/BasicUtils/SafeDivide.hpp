///////////////////////////////////////////////////////////////////////////////
//
// File: SafeDivide.hpp
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
// Description: Division that gives zero where the divisor is zero.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends.hpp>

#include <type_traits>

namespace Nektar
{

/**
 * @brief `num / den`, and zero where @p den is zero.
 *
 * The zero padding of a block is where a pointwise kernel meets a zero
 * divisor: a Jacobian, a density. Taking this in place of the division keeps
 * the result zero there. For a SIMD type the whole vector is divided, and the
 * lanes are fixed up only in the rare vector that holds a zero.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE TData SafeDivide(const TData &num, const TData &den)
{
    const TData zero(0);
    if constexpr (std::is_arithmetic_v<TData>)
    {
        return (den == zero) ? zero : num / den;
    }
    else
    {
        using scalar_type = typename TData::scalarType;
        TData out         = num / den;
        if (!((abs(den) > zero) && true))
        {
            for (unsigned int l = 0; l < TData::width; ++l)
            {
                if (den[l] == scalar_type(0))
                {
                    out[l] = scalar_type(0);
                }
            }
        }
        return out;
    }
}

} // namespace Nektar
