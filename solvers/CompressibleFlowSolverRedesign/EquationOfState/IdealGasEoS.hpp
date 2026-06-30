///////////////////////////////////////////////////////////////////////////////
//
// File: IdealGasEoS.hpp
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
// Description: Ideal gas equation of state
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

// definition of standard conversions
#include "EquationOfState/VariableConverters.hpp"
// definition of parameters held in struct
#include "EquationOfState/IdealGasEoSParams.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Ideal gas equation of state Helper function
 *       p = rho * R * T
 */
template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE TData GetPressure(const IdealGasEoS<TScalar> &EoS,
                                        const TData &rho, const TData &e)
{
    // Ideal gas law: P = (gamma - 1) * rho * e
    return (TData(EoS.gamma()) - TData(1.0)) * rho * e;
}

template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE TData GetSoundSpeed(const IdealGasEoS<TScalar> &EoS,
                                          const TData &rho, const TData &e)
{
    using std::sqrt;

    // Ideal gas law: P = (gamma - 1) * rho * e
    TData p = GetPressure(EoS, rho, e);
    return sqrt(TData(EoS.gamma()) * p / rho);
}

template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE TData GetRoeSoundSpeed(
    const IdealGasEoS<TScalar> &EoS, [[maybe_unused]] const TData &rhoL,
    [[maybe_unused]] const TData &pL, [[maybe_unused]] const TData &eL,
    [[maybe_unused]] const TData &HL, [[maybe_unused]] const TData &srL,
    [[maybe_unused]] const TData &rhoR, [[maybe_unused]] const TData &pR,
    [[maybe_unused]] const TData &eR, [[maybe_unused]] const TData &HR,
    [[maybe_unused]] const TData &srR, const TData &HRoe, const TData &URoe2,
    [[maybe_unused]] const TData &srLR)
{
    using std::sqrt;
    // Calculate sound speed using ideal gas relation
    return sqrt((TData(EoS.gamma()) - TData(1.0)) *
                (HRoe - TData(0.5) * URoe2));
}

template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE TData GetTemperature(const IdealGasEoS<TScalar> &EoS,
                                           [[maybe_unused]] const TData &rho,
                                           const TData &e)
{
    return e * TData(EoS.gammaMoneOgasConst());
}

} // namespace Nektar::Operators::detail
