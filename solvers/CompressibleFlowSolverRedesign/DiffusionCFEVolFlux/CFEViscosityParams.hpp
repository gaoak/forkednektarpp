///////////////////////////////////////////////////////////////////////////////
//
// File: CFEViscosityParams.hpp
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
// Description: The viscosity parameters of the compressible Navier-Stokes
// equations, read off the session.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <string>

namespace Nektar::detail
{

/**
 * @brief The viscosity law and the Prandtl number of the compressible
 * Navier-Stokes equations, as every viscous flux of the solver reads them:
 * the reference viscosity, Sutherland's law when SOLVERINFO ViscosityType is
 * Variable, and the Prandtl number, given as such or through the thermal
 * conductivity, not both.
 */
template <typename TData> struct CFEViscosityParams
{
    TData muRef            = 0.0;
    TData prandtl          = 0.0;
    TData oneOverTstar     = 0.0;
    TData tRatioSutherland = 0.0;
    bool isMuVariable      = false;

    /// The parameters for the equation of state @p EoS.
    template <typename EqnOfSParams>
    static CFEViscosityParams Read(
        const LibUtilities::SessionReaderSharedPtr &session,
        const EqnOfSParams &EoS)
    {
        CFEViscosityParams v;
        v.Load(session, EoS);
        return v;
    }

private:
    template <typename EqnOfSParams>
    void Load(const LibUtilities::SessionReaderSharedPtr &session,
              const EqnOfSParams &EoS)
    {
        TData rhoInf, pInf;
        session->LoadReferenceValue("viscosity", muRef, 1.78e-05);
        session->LoadReferenceValue("density", rhoInf, 1.225);
        session->LoadReferenceValue("pressure", pInf, 101325);
        oneOverTstar = (rhoInf * EoS.gasConst()) / pInf;

        std::string viscosityType;
        session->LoadSolverInfo("ViscosityType", viscosityType, "Constant");
        isMuVariable = StringUtils::IEquals(viscosityType, "Variable");
        if (isMuVariable)
        {
            TData Tref, Tsuth;
            session->LoadReferenceValue("Temperature", Tref, 288.15);
            session->LoadReferenceValue("Tsutherland", Tsuth, 110.0);
            tRatioSutherland = Tsuth / Tref;
        }

        const TData Cp = EoS.gamma() / (EoS.gamma() - 1.0) * EoS.gasConst();
        if (session->DefinesReferenceValue("thermalConductivity"))
        {
            ASSERTL0(!session->DefinesReferenceValue("Prandtl"),
                     "Cannot define both Prandtl and thermalConductivity.");
            TData thermalConductivityRef;
            session->LoadReferenceValue("thermalConductivity",
                                        thermalConductivityRef, 1.0);
            prandtl = Cp * muRef / thermalConductivityRef;
        }
        else
        {
            session->LoadReferenceValue("Prandtl", prandtl, 0.72);
        }
    }
};

} // namespace Nektar::detail
