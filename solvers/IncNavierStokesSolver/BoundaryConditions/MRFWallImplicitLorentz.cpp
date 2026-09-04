///////////////////////////////////////////////////////////////////////////////
//
// File: MRFWallImplicitLorentz.cpp
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
// Description: MRF wall pressure condition for implicit Lorentz damping.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/BoundaryConditions/MRFWallImplicitLorentz.h>

namespace Nektar
{

std::string MRFWallImplicitLorentz::className =
    GetIncBCFactory().RegisterCreatorFunction(
        "MRFWallImplicitLorentz", MRFWallImplicitLorentz::create,
        "MRF wall boundary condition with implicit Lorentz damping");

MRFWallImplicitLorentz::MRFWallImplicitLorentz(
    const LibUtilities::SessionReaderSharedPtr pSession,
    Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
    Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
    Array<OneD, MultiRegions::ExpListSharedPtr> exp, int nbnd, int spacedim,
    int bnddim)
    : MRFWall(pSession, pFields, cond, exp, nbnd, spacedim, bnddim,
              "MRFWallImplicitLorentz")
{
}

void MRFWallImplicitLorentz::AddLorentzDamping(
    Array<OneD, Array<OneD, NekDouble>> &N,
    std::map<std::string, NekDouble> &params, int npts0)
{
    auto damping = params.find("ImplicitLorentzDamping");
    ASSERTL0(damping != params.end() && damping->second > 0.0,
             "MRFWallImplicitLorentz requires ForcingLorentz with "
             "IMPLICITDAMPING enabled and a non-zero magnetic field.");

    const char *vars[] = {"U", "V", "W"};
    for (int i = 0; i < m_bnddim; ++i)
    {
        auto velocity = params.find(vars[i]);
        if (velocity != params.end())
        {
            Vmath::Sadd(npts0, -damping->second * velocity->second, N[i], 1,
                        N[i], 1);
        }
    }
}

} // namespace Nektar
