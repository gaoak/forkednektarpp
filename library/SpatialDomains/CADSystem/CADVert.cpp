////////////////////////////////////////////////////////////////////////////////
//
//  File: CADVert.cpp
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
#include "CADVert.h"

using namespace std;

namespace Nektar::SpatialDomains
{

void CADVert::SetDegen([[maybe_unused]] int s,
                       [[maybe_unused]] CADSurfSharedPtr su, NekDouble u,
                       NekDouble v)
{
    degen     = true;
    degensurf = s;
    // locuv is ill-conditioned at a collapsed point, so the parametric
    // location is recorded here for the mesher to use instead of projecting.
    // The old NekMesh CADVert pushed this straight onto its mesh node; the
    // vertex no longer owns one, so it is stored and fetched via GetDegenUV.
    m_degenUV = {u, v};
}

std::array<NekDouble, 3> CADVert::GetLoc()
{
    return m_loc;
}

NekDouble CADVert::DistanceTo(std::array<NekDouble, 3> xyz)
{
    return sqrt((m_loc[0] - xyz[0]) * (m_loc[0] - xyz[0]) +
                (m_loc[1] - xyz[1]) * (m_loc[1] - xyz[1]) +
                (m_loc[2] - xyz[2]) * (m_loc[2] - xyz[2]));
}

} // namespace Nektar::SpatialDomains
