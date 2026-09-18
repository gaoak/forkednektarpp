////////////////////////////////////////////////////////////////////////////////
//
//  File: CADVertOCE.cpp
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
//  Description: cad object methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/CADSystem/OCE/CADSystemOCE.h>
#include <SpatialDomains/CADSystem/OCE/CADVertOCE.h>

using namespace std;

namespace Nektar::SpatialDomains
{

std::string CADVertOCE::key = GetCADVertFactory().RegisterCreatorFunction(
    "oce", CADVertOCE::create, "CAD vert oce");

void CADVertOCE::Initialise(int i, TopoDS_Shape in)
{
    m_id      = i;
    m_occVert = BRep_Tool::Pnt(TopoDS::Vertex(in));

    m_loc = {m_occVert.X() / 1000.0, m_occVert.Y() / 1000.0,
             m_occVert.Z() / 1000.0};

    degen = false;
}

} // namespace Nektar::SpatialDomains
