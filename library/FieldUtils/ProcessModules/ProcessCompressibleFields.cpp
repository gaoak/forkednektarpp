////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessCompressibleFields.cpp
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
//  Description: Computes compressible flow fields such as velocity, pressure,
//  temperature, entropy, sound speed and Mach number.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <string>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <LibUtilities/BasicUtils/Vmath.hpp>

#include "ProcessCompressibleFields.h"

namespace Nektar::FieldUtils
{

ModuleKey ProcessCompressibleFields::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "compressiblefields"),
        ProcessCompressibleFields::create,
        "Computes velocity, pressure, temperature, entropy, sound speed and "
        "Mach number from compressible flow fields.");

ProcessCompressibleFields::ProcessCompressibleFields(FieldSharedPtr f)
    : ProcessModule(f)
{
}

ProcessCompressibleFields::~ProcessCompressibleFields()
{
}

void ProcessCompressibleFields::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    ASSERTL0(m_f->m_session && m_f->m_graph,
             "ProcessCompressibleFields requires a session file.");

    std::string eosType;
    m_f->m_session->LoadSolverInfo("EquationOfState", eosType, "IdealGas");
    ASSERTL0(boost::iequals(eosType, "IdealGas"),
             "ProcessCompressibleFields only supports the IdealGas equation "
             "of state.");

    int nfields  = m_f->m_variables.size();
    int spacedim = m_f->m_graph->GetMeshDimension() + m_f->m_numHomogeneousDir;

    // Locate the conservative variables by name, so that the input may contain
    // other fields in any order
    auto findField = [&](const std::string &name) {
        auto it =
            std::find(m_f->m_variables.begin(), m_f->m_variables.end(), name);
        ASSERTL0(it != m_f->m_variables.end(),
                 "ProcessCompressibleFields requires the field '" + name +
                     "' (conservative variables of a compressible flow).");
        return static_cast<int>(it - m_f->m_variables.begin());
    };

    const std::string momNames[3] = {"rhou", "rhov", "rhow"};
    const std::string velNames[3] = {"u", "v", "w"};

    int rhoId = findField("rho");
    int EId   = findField("E");
    std::vector<int> momId(spacedim);
    for (int i = 0; i < spacedim; ++i)
    {
        momId[i] = findField(momNames[i]);
    }

    // New fields: velocity components, p, T, s, a and Mach
    std::vector<std::string> newNames(velNames, velNames + spacedim);
    newNames.insert(newNames.end(), {"p", "T", "s", "a", "Mach"});
    int addfields = newNames.size();

    for (auto &name : newNames)
    {
        ASSERTL0(std::find(m_f->m_variables.begin(), m_f->m_variables.end(),
                           name) == m_f->m_variables.end(),
                 "Field '" + name +
                     "' already exists (e.g. the file was written with "
                     "OutputExtraFields); remove it first with the "
                     "removefield module.");
    }

    // Append field names
    m_f->m_variables.insert(m_f->m_variables.end(), newNames.begin(),
                            newNames.end());

    // Skip in case of empty partition
    if (m_f->m_exp[0]->GetNumElmts() == 0)
    {
        return;
    }

    int nstrips;
    m_f->m_session->LoadParameter("Strip_Z", nstrips, 1);

    // Add in new fields
    for (int s = 0; s < nstrips; ++s)
    {
        for (int i = 0; i < addfields; ++i)
        {
            MultiRegions::ExpListSharedPtr Exp =
                m_f->AppendExpList(m_f->m_numHomogeneousDir);
            m_f->m_exp.insert(m_f->m_exp.begin() + s * (nfields + addfields) +
                                  nfields + i,
                              Exp);
        }
    }

    NekDouble gamma, gasConstant;
    m_f->m_session->LoadParameter("Gamma", gamma, 1.4);
    m_f->m_session->LoadParameter("GasConstant", gasConstant, 287.058);
    NekDouble cv = gasConstant / (gamma - 1.0);

    int npoints    = m_f->m_exp[0]->GetNpoints();
    bool waveSpace = m_f->m_exp[0]->GetWaveSpace();

    Array<OneD, NekDouble> rho(npoints), e(npoints), vel2(npoints);
    Array<OneD, NekDouble> tmp(npoints);
    Array<OneD, Array<OneD, NekDouble>> outfield(addfields);
    for (int i = 0; i < addfields; ++i)
    {
        outfield[i] = Array<OneD, NekDouble>(npoints);
    }
    Array<OneD, NekDouble> &pressure    = outfield[spacedim];
    Array<OneD, NekDouble> &temperature = outfield[spacedim + 1];
    Array<OneD, NekDouble> &entropy     = outfield[spacedim + 2];
    Array<OneD, NekDouble> &soundspeed  = outfield[spacedim + 3];
    Array<OneD, NekDouble> &mach        = outfield[spacedim + 4];

    // The fields are nonlinear functions of the conservative variables, so
    // they have to be computed in physical space
    auto getPhys = [&](int fid, Array<OneD, NekDouble> &out) {
        if (waveSpace)
        {
            m_f->m_exp[fid]->HomogeneousBwdTrans(
                npoints, m_f->m_exp[fid]->GetPhys(), out);
        }
        else
        {
            Vmath::Vcopy(npoints, m_f->m_exp[fid]->GetPhys(), 1, out, 1);
        }
    };

    for (int s = 0; s < nstrips; ++s)
    {
        int offset = s * (nfields + addfields);

        getPhys(offset + rhoId, rho);

        // Velocity u_i = (rho u_i) / rho and its squared magnitude |u|^2
        Vmath::Zero(npoints, vel2, 1);
        for (int i = 0; i < spacedim; ++i)
        {
            getPhys(offset + momId[i], outfield[i]);
            Vmath::Vdiv(npoints, outfield[i], 1, rho, 1, outfield[i], 1);
            Vmath::Vvtvp(npoints, outfield[i], 1, outfield[i], 1, vel2, 1, vel2,
                         1);
        }

        // Internal energy e = E / rho - 0.5 |u|^2
        getPhys(offset + EId, e);
        Vmath::Vdiv(npoints, e, 1, rho, 1, e, 1);
        Vmath::Svtvp(npoints, -0.5, vel2, 1, e, 1, e, 1);

        // Pressure p = (gamma - 1) rho e
        Vmath::Vmul(npoints, rho, 1, e, 1, pressure, 1);
        Vmath::Smul(npoints, gamma - 1.0, pressure, 1, pressure, 1);

        // Temperature T = e / cv
        Vmath::Smul(npoints, 1.0 / cv, e, 1, temperature, 1);

        // Entropy s = cv log(T) - R log(rho)
        Vmath::Vlog(npoints, temperature, 1, entropy, 1);
        Vmath::Smul(npoints, cv, entropy, 1, entropy, 1);
        Vmath::Vlog(npoints, rho, 1, tmp, 1);
        Vmath::Svtvp(npoints, -gasConstant, tmp, 1, entropy, 1, entropy, 1);

        // Sound speed a = sqrt(gamma R T)
        Vmath::Smul(npoints, gamma * gasConstant, temperature, 1, soundspeed,
                    1);
        Vmath::Vsqrt(npoints, soundspeed, 1, soundspeed, 1);

        // Mach number M = |u| / a
        Vmath::Vsqrt(npoints, vel2, 1, mach, 1);
        Vmath::Vdiv(npoints, mach, 1, soundspeed, 1, mach, 1);

        for (int i = 0; i < addfields; ++i)
        {
            int fid = offset + nfields + i;
            if (waveSpace)
            {
                m_f->m_exp[fid]->HomogeneousFwdTrans(npoints, outfield[i],
                                                     outfield[i]);
            }
            Vmath::Vcopy(npoints, outfield[i], 1, m_f->m_exp[fid]->UpdatePhys(),
                         1);
            m_f->m_exp[fid]->FwdTransLocalElmt(outfield[i],
                                               m_f->m_exp[fid]->UpdateCoeffs());
        }
    }
}

} // namespace Nektar::FieldUtils
