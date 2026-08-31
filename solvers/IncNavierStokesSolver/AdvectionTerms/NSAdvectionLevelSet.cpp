///////////////////////////////////////////////////////////////////////////////
//
// File: NSAdvectionLevelSet.cpp
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
// Description: Evaluation of the Navier Stokes advective term for VCSLevelSet
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/AdvectionTerms/NSAdvectionLevelSet.h>

namespace Nektar
{
std::string NSAdvectionLevelSet::className =
    SolverUtils::GetAdvectionFactory().RegisterCreatorFunction(
        "NSLevelSet", NSAdvectionLevelSet::create, "NSLevelSet");

/**
 *
 */
NSAdvectionLevelSet::NSAdvectionLevelSet() : Advection()

{
}

/**
 *
 */
void NSAdvectionLevelSet::v_InitObject(
    LibUtilities::SessionReaderSharedPtr pSession,
    Array<OneD, MultiRegions::ExpListSharedPtr> pFields)
{
    Advection::v_InitObject(pSession, pFields);
}

/**
 *
 */
void NSAdvectionLevelSet::v_Advect(
    const int nConvectiveFields,
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, Array<OneD, NekDouble>> &advVel,
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble &time,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &pFwd,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &pBwd)
{
    int nqtot = fields[0]->GetTotPoints();
    ASSERTL1(nConvectiveFields == inarray.size(),
             "Number of convective fields and Inarray are not compatible");

    // use dimension of Velocity vector to dictate dimension of operation
    int ndim = advVel.size();
    Array<OneD, Array<OneD, NekDouble>> AdvVel(advVel.size());

    Array<OneD, Array<OneD, NekDouble>> velocity(ndim);

    // LibUtilities::Timer timer;
    for (int i = 0; i < ndim; ++i)
    {
        velocity[i] = advVel[i];
    }

    int nPointsTot = fields[0]->GetNpoints();
    Array<OneD, NekDouble> grad0, grad1, grad2, wkSp;

    for (int i = 0; i < ndim; ++i)
    {
        AdvVel[i] = velocity[i];
    }

    wkSp = Array<OneD, NekDouble>(nPointsTot);

    switch (ndim)
    {
        case 1:
        {
            grad0 = Array<OneD, NekDouble>(fields[0]->GetNpoints());
            for (int n = 0; n < nConvectiveFields; ++n)
            {
                fields[0]->PhysDeriv(inarray[n], grad0);
                Vmath::Vmul(nPointsTot, grad0, 1, AdvVel[0], 1, outarray[n], 1);
            }
            Vmath::Zero(nqtot, outarray[ndim + 1], 1);
            Vmath::Zero(nqtot, outarray[ndim + 2], 1);
            break;
        }
        case 2:
        {
            grad0 = Array<OneD, NekDouble>(nqtot);
            grad1 = Array<OneD, NekDouble>(nqtot);
            for (int n = 0; n < nConvectiveFields; ++n)
            {
                fields[0]->PhysDeriv(inarray[n], grad0, grad1);
                Vmath::Vmul(nPointsTot, grad0, 1, AdvVel[0], 1, outarray[n], 1);
                Vmath::Vvtvp(nPointsTot, grad1, 1, AdvVel[1], 1, outarray[n], 1,
                             outarray[n], 1);
            }
            Vmath::Zero(nqtot, outarray[ndim + 1], 1);
            Vmath::Zero(nqtot, outarray[ndim + 2], 1);
            break;
        }
        case 3:
        {
            grad0                      = Array<OneD, NekDouble>(nqtot);
            grad1                      = Array<OneD, NekDouble>(nqtot);
            grad2                      = Array<OneD, NekDouble>(nqtot);
            Array<OneD, NekDouble> tmp = grad2;
            for (int n = 0; n < nConvectiveFields; ++n)
            {
                fields[0]->PhysDeriv(inarray[n], grad0, grad1, grad2);
                Vmath::Vmul(nPointsTot, grad0, 1, AdvVel[0], 1, outarray[n], 1);
                Vmath::Vvtvp(nPointsTot, grad1, 1, AdvVel[1], 1, outarray[n], 1,
                             outarray[n], 1);
                Vmath::Vvtvp(nPointsTot, grad2, 1, AdvVel[2], 1, outarray[n], 1,
                             outarray[n], 1);
            }
            Vmath::Zero(nqtot, outarray[ndim + 1], 1);
            Vmath::Zero(nqtot, outarray[ndim + 2], 1);
            break;
        }
        default:
            ASSERTL0(false, "dimension unknown");
    }

    for (int n = 0; n < nConvectiveFields; ++n)
    {
        Vmath::Neg(nqtot, outarray[n], 1);
    }
}

} // namespace Nektar
