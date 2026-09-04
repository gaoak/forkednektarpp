///////////////////////////////////////////////////////////////////////////////
//
// File: MRFWallImplicitLorentz.h
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

#ifndef NEKTAR_SOLVERS_MRFWALLIMPLICITLORENTZ_H
#define NEKTAR_SOLVERS_MRFWALLIMPLICITLORENTZ_H

#include <IncNavierStokesSolver/BoundaryConditions/MRFWall.h>

namespace Nektar
{

class MRFWallImplicitLorentz : public MRFWall
{
public:
    friend class MemoryManager<MRFWallImplicitLorentz>;

    static IncBaseConditionSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr pSession,
        Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
        Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
        Array<OneD, MultiRegions::ExpListSharedPtr> exp, int nbnd, int spacedim,
        int bnddim)
    {
        IncBaseConditionSharedPtr p =
            MemoryManager<MRFWallImplicitLorentz>::AllocateSharedPtr(
                pSession, pFields, cond, exp, nbnd, spacedim, bnddim);
        p->Initialise(pSession);
        return p;
    }

    static std::string className;
    ~MRFWallImplicitLorentz() override = default;

protected:
    void AddLorentzDamping(Array<OneD, Array<OneD, NekDouble>> &N,
                           std::map<std::string, NekDouble> &params,
                           int npts0) override;

private:
    MRFWallImplicitLorentz(
        const LibUtilities::SessionReaderSharedPtr pSession,
        Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
        Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
        Array<OneD, MultiRegions::ExpListSharedPtr> exp, int nbnd, int spacedim,
        int bnddim);
};

} // namespace Nektar

#endif
