///////////////////////////////////////////////////////////////////////////////
//
// File: IdealGasEoSParams.hpp
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

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <boost/algorithm/string/predicate.hpp>

namespace Nektar::detail
{

/**
 * @brief Ideal gas equation of state:
 *       p = rho * R * T
 */
template <typename TData> struct IdealGasEoS
{
    // Specialised constructor for updating

    IdealGasEoS(TData gamma, TData gasConst)
        : m_gamma(gamma), m_gasConst(gasConst)
    {
        m_gammaMoneOgasConst = (m_gamma - TData(1.0)) / m_gasConst;
    }
    // default constructor -> values will casuse simualations to fail
    IdealGasEoS(void) : m_gamma(0), m_gasConst(0), m_gammaMoneOgasConst(0)
    {
    }

    NEK_HOSTDEVICE_INLINE TData gamma(void) const
    {
        return m_gamma;
    }

    NEK_HOSTDEVICE_INLINE TData gasConst(void) const
    {
        return m_gasConst;
    }

    NEK_HOSTDEVICE_INLINE TData gammaMoneOgasConst(void) const
    {
        return m_gammaMoneOgasConst;
    }

    // define name used in EquationOfState Solver Info in upper case
    // for registration in factory pattern
    static inline const std::string name = "IDEALGAS";

private:
    TData m_gamma;
    TData m_gasConst;
    TData m_gammaMoneOgasConst;
};

// Each Equation of State needs an initalisation method of this form
template <typename TData>
inline void SetUpEquationOfState(LibUtilities::SessionReaderSharedPtr session,
                                 IdealGasEoS<TData> &EoS)
{
    TData gamma    = 0.0;
    TData gasConst = 0.0;

    if (session->DefinesEquationOfState())
    {
        LibUtilities::EquationOfStateScheme EoS = session->GetEquationOfState();

        if (boost::iequals(EoS.type, IdealGasEoS<TData>::name))
        {
            if (EoS.params.count("GAMMA"))
            {
                gamma = EoS.params["GAMMA"];
            }
            else
            {
                NEKERROR(ErrorUtil::efatal, "Need to specify Gamma in params "
                                            "of EquationOfState definition");
            }

            if (EoS.params.count("GASCONSTANT"))
            {
                gasConst = EoS.params["GASCONSTANT"];
            }
            else
            {
                NEKERROR(ErrorUtil::efatal,
                         "Need to specify GasConstant in params "
                         "of EquationOfState definition");
            }
        }
    }
    else
    {
        NEKERROR(ErrorUtil::efatal,
                 "No EquationOfState section defined in session file");
    }

    EoS = IdealGasEoS<TData>(gamma, gasConst);
}
} // namespace Nektar::detail