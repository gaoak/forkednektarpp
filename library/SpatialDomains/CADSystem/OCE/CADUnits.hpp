////////////////////////////////////////////////////////////////////////////////
//
//  File: CADUnits.hpp
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
//  Description: Length units at the OpenCASCADE boundary.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_CADSYSTEM_OCE_CADUNITS_HPP
#define NEKTAR_SPATIALDOMAINS_CADSYSTEM_OCE_CADUNITS_HPP

#include <array>

#include <LibUtilities/BasicUtils/NekFactory.hpp>

#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

/**
 * @brief Converting lengths across the OpenCASCADE boundary.
 *
 * Two different units meet here, and every quantity crossing between them has
 * to be converted:
 *
 * - **Inside OpenCASCADE**, everything is in millimetres. A STEP file carries
 *   its own unit in its header, and the readers in CADSystemOCE normalise
 *   whatever they find to millimetres as they transfer the model, so this
 *   holds whatever the file was authored in. Geometry built directly rather
 *   than read -- the NACA profiles and the `.geo` parser -- is constructed in
 *   millimetres to match.
 *
 * - **Outside**, in the mesh and everything the user writes in a `.mcf`, the
 *   unit is the metre. MinDelta, MaxDelta, boundary layer thicknesses and the
 *   coordinates of the mesh produced are all in metres.
 *
 * So a model authored in millimetres and one authored in metres both come out
 * as the same mesh, in metres, and the same `.mcf` drives either.
 *
 * Use these functions at every crossing rather than writing the factor out.
 * It is not only positions that carry it: a curvature is one over a length and
 * so converts the other way, which is easy to get backwards when the number is
 * written inline.
 */
namespace Nektar::SpatialDomains::CADUnits
{

/// CAD units in one mesh unit: millimetres in a metre.
constexpr NekDouble PerMeshUnit = 1000.0;

/// A length on its way into OpenCASCADE.
inline NekDouble ToCAD(NekDouble length)
{
    return length * PerMeshUnit;
}

/// A length on its way out of OpenCASCADE.
inline NekDouble FromCAD(NekDouble length)
{
    return length / PerMeshUnit;
}

/// A curvature, one over a length, on its way out of OpenCASCADE.
inline NekDouble CurvatureFromCAD(NekDouble curvature)
{
    return curvature * PerMeshUnit;
}

/// A point on its way into OpenCASCADE.
inline gp_Pnt PntToCAD(const std::array<NekDouble, 3> &p)
{
    return gp_Pnt(ToCAD(p[0]), ToCAD(p[1]), ToCAD(p[2]));
}

/// A point on its way out of OpenCASCADE.
inline std::array<NekDouble, 3> PntFromCAD(const gp_Pnt &p)
{
    return {FromCAD(p.X()), FromCAD(p.Y()), FromCAD(p.Z())};
}

/// A vector on its way out of OpenCASCADE.
inline std::array<NekDouble, 3> VecFromCAD(const gp_Vec &v)
{
    return {FromCAD(v.X()), FromCAD(v.Y()), FromCAD(v.Z())};
}

} // namespace Nektar::SpatialDomains::CADUnits

#endif
