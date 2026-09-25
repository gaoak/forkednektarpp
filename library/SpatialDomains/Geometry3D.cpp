////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry3D.cpp
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
//  Description: 3D geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>

#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/Geometry3D.h>
#include <SpatialDomains/PointGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <iomanip>

namespace Nektar::SpatialDomains
{

Geometry3D::Geometry3D()
{
}

Geometry3D::Geometry3D(const int coordim, Curve *curve)
    : Geometry(coordim), m_curve(curve)
{
    ASSERTL0(m_coordim > 2,
             "Coordinate dimension should be at least 3 for a 3D geometry.");
}

//---------------------------------------
// Helper functions
//---------------------------------------

//---------------------------------------
// 3D Geometry Methods
//---------------------------------------
/**
 * @brief Given local collapsed coordinate Lcoord return the value of
 * physical coordinate in direction i.
 */
NekDouble Geometry3D::v_GetCoord(const int i,
                                 const Array<OneD, const NekDouble> &Lcoord)
{
    if (m_state != ePtsFilled)
    {
        NEKERROR(ErrorUtil::ewarning, "Geometry is not in physical space");
    }

    Array<OneD, NekDouble> tmp(m_xmap->GetTotPoints());
    m_xmap->BwdTrans(m_coeffs[i], tmp);

    return m_xmap->PhysEvaluate(Lcoord, tmp);
}

//---------------------------------------
// Helper functions
//---------------------------------------

int Geometry3D::v_GetShapeDim() const
{
    return 3;
}

} // namespace Nektar::SpatialDomains
