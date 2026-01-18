///////////////////////////////////////////////////////////////////////////////
//
// File: StdExpansion1D.cpp
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
// Description: Daughter of StdExpansion. This class contains routine
// which are common to 1d expansion. Typically this inolves physiocal
// space operations.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <StdRegions/StdExpansion1D.h>

#include <LibUtilities/BasicUtils/NekInline.hpp>

namespace Nektar::StdRegions
{
// Declaration of scalar routine
using vec_t = tinysimd::scalarT<double>;

StdExpansion1D::StdExpansion1D(
    [[maybe_unused]] int numcoeffs,
    [[maybe_unused]] const LibUtilities::BasisKey &Ba)
{
}

//----------------------------
// Differentiation Methods
//-----------------------------
void StdExpansion1D::v_PhysDeriv([[maybe_unused]] const int dir,
                                 const Array<OneD, const NekDouble> &inarray,
                                 Array<OneD, NekDouble> &outarray)
{
    ASSERTL1(dir == 0, "input dir is out of range");
    v_PhysDeriv(inarray, outarray, NullNekDouble1DArray, NullNekDouble1DArray);
}

NekDouble StdExpansion1D::v_PhysEvaluate(
    const Array<OneD, const NekDouble> &Lcoord,
    const Array<OneD, const NekDouble> &physvals)
{
    ASSERTL2(Lcoord[0] >= -1 - NekConstants::kNekZeroTol, "Lcoord[0] < -1");
    ASSERTL2(Lcoord[0] <= 1 + NekConstants::kNekZeroTol, "Lcoord[0] >  1");

    return StdExpansion::BaryEvaluate<0>(Lcoord[0], &physvals[0]);
}

void StdExpansion1D::IProductWRTBaseKernel(
    const Array<OneD, const NekDouble> &base0,
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray, const Array<OneD, const NekDouble> &jac,
    const bool Deformed)
{
    v_IProductWRTBaseKernel(base0, inarray, outarray, jac, Deformed);
}

void StdExpansion1D::v_PhysInterp(std::shared_ptr<StdExpansion> fromExp,
                                  const Array<OneD, const NekDouble> &fromData,
                                  Array<OneD, NekDouble> &toData)
{
    LibUtilities::Interp1D(fromExp->GetBasis(0)->GetPointsKey(), fromData,
                           m_base[0]->GetPointsKey(), toData);
}

} // namespace Nektar::StdRegions
