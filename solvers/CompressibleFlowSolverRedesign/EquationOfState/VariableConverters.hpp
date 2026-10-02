///////////////////////////////////////////////////////////////////////////////
//
// File: VariableConverters.hpp
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
// Description: Variable Converters related to Equations of State
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace Nektar::detail
{

/**
 * @brief Compute the specific internal energy
 *        \f$ e = (E - rho*V^2/2)/rho \f$.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE TData GetInternalEnergy(unsigned ndim, const TData &rho,
                                              const TData *m, const TData &E)
{
    TData sum    = TData(0);
    TData invrho = TData(1) / rho;

    // tmp = (rho * u_i)^2
    for (unsigned i = 0; i < ndim; ++i)
    {
        sum += m[i] * m[i];
    }

    /// rho*V^2
    sum *= invrho;

    // Calculate  = (E - rho*V^2/2)/rho
    return (E - TData(0.5) * sum) * invrho;
}

// Constant-viscosity or Sutherland-law viscosity model.
template <typename TData>
NEK_HOSTDEVICE_INLINE TData GetDynamicViscosity(const TData &temperature,
                                                const TData &muRef,
                                                bool isVariable,
                                                const TData &oneOverTstar,
                                                const TData &tRatioSutherland)
{
    using std::sqrt;

    TData mu_star = muRef;

    if (isVariable) // define using sutherland's law
    {
        const TData onePlusC = TData(1.0) + tRatioSutherland;
        const TData ratio    = temperature * oneOverTstar;

        return mu_star * ratio * sqrt(ratio) * onePlusC /
               (ratio + tRatioSutherland);
    }
    else
    {
        return mu_star;
    }
}

} // namespace Nektar::detail