///////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessMHDCurrentDensity.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Computes quasi-static MHD electric current density.
//
///////////////////////////////////////////////////////////////////////////////

#include <array>
#include <string>

#include <boost/algorithm/string/predicate.hpp>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "ProcessMHDCurrentDensity.h"

namespace Nektar::FieldUtils
{

ModuleKey ProcessMHDCurrentDensity::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "MHDCurrentDensity"),
        ProcessMHDCurrentDensity::create,
        "Computes quasi-static MHD electric current density.");

ProcessMHDCurrentDensity::ProcessMHDCurrentDensity(FieldSharedPtr f)
    : ProcessModule(f)
{
    m_config["function"] = ConfigOption(
        false, "NotSet",
        "Override the electric/magnetic field function name (optional)");
}

ProcessMHDCurrentDensity::~ProcessMHDCurrentDensity()
{
}

int ProcessMHDCurrentDensity::FindField(const std::string &name) const
{
    for (int i = 0; i < m_f->m_variables.size(); ++i)
    {
        if (boost::iequals(m_f->m_variables[i], name))
        {
            return i;
        }
    }
    return -1;
}

std::string ProcessMHDCurrentDensity::GetElectromagneticFieldFunction() const
{
    if (!boost::iequals(m_config.at("function").as<std::string>(), "NotSet"))
    {
        return m_config.at("function").as<std::string>();
    }

    if (m_f->m_session->DefinesElement("Nektar/Forcing"))
    {
        TiXmlElement *forcing = m_f->m_session->GetElement("Nektar/Forcing");
        TiXmlElement *force   = forcing->FirstChildElement("FORCE");
        while (force)
        {
            const char *type = force->Attribute("TYPE");
            if (type && boost::iequals(type, "ForcingLorentz"))
            {
                TiXmlElement *fields =
                    force->FirstChildElement("ELECTRICMAGNETICFIELDS");
                ASSERTL0(fields && fields->GetText(),
                         "ForcingLorentz requires an ELECTRICMAGNETICFIELDS "
                         "function for MHDCurrentDensity.");
                return fields->GetText();
            }
            force = force->NextSiblingElement("FORCE");
        }
    }

    ASSERTL0(m_f->m_session->DefinesFunction("ElectricMagneticFields"),
             "MHDCurrentDensity could not find a ForcingLorentz "
             "ELECTRICMAGNETICFIELDS function. Specify one with "
             "function=<name>.");
    return "ElectricMagneticFields";
}

void ProcessMHDCurrentDensity::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    int spacedim = m_f->m_graph->GetMeshDimension() + m_f->m_numHomogeneousDir;
    ASSERTL0(spacedim == 2 || spacedim == 3,
             "MHDCurrentDensity only supports two- and three-dimensional "
             "fields.");

    std::array<int, 3> velocity = {FindField("u"), FindField("v"),
                                   FindField("w")};
    int potential               = FindField("phi");
    ASSERTL0(velocity[0] >= 0 && velocity[1] >= 0,
             "MHDCurrentDensity requires velocity fields u and v.");
    ASSERTL0(spacedim == 2 || velocity[2] >= 0,
             "MHDCurrentDensity requires velocity field w in 3D.");
    ASSERTL0(potential >= 0,
             "MHDCurrentDensity requires electric potential field phi.");

    NekDouble sigma = -1.0;
    m_f->m_session->LoadParameter("ElectricConductivity", sigma, -1.0);
    ASSERTL0(sigma > 0.0,
             "ElectricConductivity must be defined and positive for "
             "MHDCurrentDensity.");

    std::string functionName = GetElectromagneticFieldFunction();
    ASSERTL0(m_f->m_session->DefinesFunction(functionName),
             "Function '" + functionName + "' is not defined in the session.");

    std::array<NekDouble, 3> electric             = {0.0, 0.0, 0.0};
    std::array<NekDouble, 3> magnetic             = {0.0, 0.0, 0.0};
    const std::array<std::string, 3> electricVars = {"Ex", "Ey", "Ez"};
    const std::array<std::string, 3> magneticVars = {"Bx", "By", "Bz"};
    for (int i = 0; i < 3; ++i)
    {
        if (m_f->m_session->DefinesFunction(functionName, electricVars[i]))
        {
            electric[i] =
                m_f->m_session->GetFunction(functionName, electricVars[i])
                    ->Evaluate(0.0, 0.0, 0.0, 0.0);
        }
        if (m_f->m_session->DefinesFunction(functionName, magneticVars[i]))
        {
            magnetic[i] =
                m_f->m_session->GetFunction(functionName, magneticVars[i])
                    ->Evaluate(0.0, 0.0, 0.0, 0.0);
        }
    }

    int nfields = m_f->m_variables.size();
    m_f->m_variables.insert(m_f->m_variables.end(), {"Jx", "Jy", "Jz"});
    m_f->m_exp.resize(nfields + 3);

    // An empty partition still needs the variable list above, but has no
    // expansions on which to calculate the current.
    if (m_f->m_exp[0]->GetNumElmts() == 0)
    {
        return;
    }

    int npoints = m_f->m_exp[0]->GetNpoints();
    for (int i = 0; i < spacedim; ++i)
    {
        ASSERTL0(m_f->m_exp[velocity[i]]->GetNpoints() == npoints,
                 "MHDCurrentDensity requires velocity fields on the same "
                 "points.");
    }
    ASSERTL0(m_f->m_exp[potential]->GetNpoints() == npoints,
             "MHDCurrentDensity requires phi on the same points as the "
             "velocity fields.");

    Array<OneD, Array<OneD, NekDouble>> gradPhi(3);
    Array<OneD, Array<OneD, NekDouble>> current(3);
    for (int i = 0; i < 3; ++i)
    {
        gradPhi[i] = Array<OneD, NekDouble>(npoints, 0.0);
        current[i] = Array<OneD, NekDouble>(npoints, 0.0);
    }
    for (int i = 0; i < spacedim; ++i)
    {
        m_f->m_exp[potential]->PhysDeriv(MultiRegions::DirCartesianMap[i],
                                         m_f->m_exp[potential]->GetPhys(),
                                         gradPhi[i]);
    }

    const Array<OneD, const NekDouble> u = m_f->m_exp[velocity[0]]->GetPhys();
    const Array<OneD, const NekDouble> v = m_f->m_exp[velocity[1]]->GetPhys();
    Array<OneD, NekDouble> zero(npoints, 0.0);
    Array<OneD, const NekDouble> w = zero;
    if (spacedim == 3)
    {
        w = m_f->m_exp[velocity[2]]->GetPhys();
    }

    for (int i = 0; i < npoints; ++i)
    {
        current[0][i] = sigma * (electric[0] + v[i] * magnetic[2] -
                                 w[i] * magnetic[1] - gradPhi[0][i]);
        current[1][i] = sigma * (electric[1] + w[i] * magnetic[0] -
                                 u[i] * magnetic[2] - gradPhi[1][i]);
        current[2][i] = sigma * (electric[2] + u[i] * magnetic[1] -
                                 v[i] * magnetic[0] - gradPhi[2][i]);
    }

    for (int i = 0; i < 3; ++i)
    {
        m_f->m_exp[nfields + i] = m_f->AppendExpList(m_f->m_numHomogeneousDir);
        Vmath::Vcopy(npoints, current[i], 1,
                     m_f->m_exp[nfields + i]->UpdatePhys(), 1);
        m_f->m_exp[nfields + i]->FwdTransLocalElmt(
            current[i], m_f->m_exp[nfields + i]->UpdateCoeffs());
    }
}

} // namespace Nektar::FieldUtils
