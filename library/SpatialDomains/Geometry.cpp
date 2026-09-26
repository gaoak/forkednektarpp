////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry.cpp
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
//  Description:  This file contains the base class implementation for the
//                Geometry class.
//
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/Geometry2D.h>

namespace Nektar::SpatialDomains
{

/**
 * @brief Default constructor.
 */
Geometry::Geometry()
    : m_coordim(0), m_state(eNotFilled), m_setupState(false),
      m_shapeType(LibUtilities::eNoShapeType), m_globalID(-1),
      m_geomType(eNoGeomType)
{
}

/**
 * @brief Constructor when supplied a coordinate dimension.
 */
Geometry::Geometry(const int coordim)
    : m_coordim(coordim), m_state(eNotFilled), m_setupState(false),
      m_shapeType(LibUtilities::eNoShapeType), m_globalID(-1),
      m_geomType(eNoGeomType)
{
}

Geometry::Geometry(const Geometry &that)
    : m_coordim(that.m_coordim), m_xmap(that.m_xmap), m_state(that.m_state),
      m_setupState(that.m_setupState), m_shapeType(that.m_shapeType),
      m_globalID(that.m_globalID), m_coeffs(that.m_coeffs),
      m_geomType(that.m_geomType)
{
}

Geometry &Geometry::operator=(const Geometry &that)
{
    if (this != &that)
    {
        m_coordim    = that.m_coordim;
        m_xmap       = that.m_xmap;
        m_state      = that.m_state;
        m_setupState = that.m_setupState;
        m_shapeType  = that.m_shapeType;
        m_globalID   = that.m_globalID;
        m_coeffs     = that.m_coeffs;
        m_geomType   = that.m_geomType;
    }
    return *this;
}

/**
 * @brief A straight-sided element is one whose every \f$\chi\f$ basis is
 * linear.
 *
 * Derived from m_xmap, so it must be established on demand rather than in
 * Setup(), which does not re-run when Reset() rebuilds the mapping.
 */
bool Geometry::HasLinearXmap()
{
    if (!m_xmap || GetShapeDim() < 2)
    {
        return false;
    }
    for (int i = 0; i < GetShapeDim(); ++i)
    {
        if (m_xmap->GetBasisNumModes(i) != 2)
        {
            return false;
        }
    }
    return true;
}

/**
 * @copydoc Geometry::CalcGeomType()
 *
 * Also hands back the isoparametric coefficients settled on the way, which is
 * what a locator needs in order to invert the mapping.
 */
GeomType Geometry::CalcGeomType(IsoParam &iso)
{
    if (!m_setupState)
    {
        Setup();
    }
    m_geomType = v_CalcGeomType(iso);
    return m_geomType;
}

/**
 * A geometric shape is considered regular if it has constant geometric
 * information, and deformed if this information changes throughout the
 * shape.
 *
 * Deciding this *is* testing whether the nonlinear isoparametric coefficients
 * vanish, so the answer arrives together with those coefficients. When the
 * caller only wants the classification -- as MeshGraph does for every element
 * it reads -- they are built in a stack-resident temporary and discarded,
 * leaving the geometry without a locator and allocating nothing.
 *
 * @returns             The type of geometry.
 * @see GeomType
 */
GeomType Geometry::CalcGeomType()
{
    if (m_geomType != eNoGeomType)
    {
        return m_geomType;
    }

    if (!m_setupState)
    {
        Setup();
    }

    IsoParam scratch;
    m_geomType = v_CalcGeomType(scratch);

    return m_geomType;
}

bool SortByGlobalId(const Geometry *&lhs, const Geometry *&rhs)
{
    return lhs->GetGlobalID() < rhs->GetGlobalID();
}

bool GlobalIdEquality(const Geometry *&lhs, const Geometry *&rhs)
{
    return lhs->GetGlobalID() == rhs->GetGlobalID();
}

/**
 * @brief Get the ID of vertex @p i of this object.
 */
int Geometry::v_GetVid(int i) const
{
    return GetVertex(i)->GetGlobalID();
}

/**
 * @brief Get the ID of edge @p i of this object.
 */
int Geometry::GetEid(int i) const
{
    return GetEdge(i)->GetGlobalID();
}

/**
 * @brief Get the ID of face @p i of this object.
 */
int Geometry::GetFid(int i) const
{
    return GetFace(i)->GetGlobalID();
}

/**
 * @copydoc Geometry::GetVertex()
 */
PointGeom *Geometry::v_GetVertex([[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetEdge()
 */
Geometry1D *Geometry::v_GetEdge([[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetFace()
 */
Geometry2D *Geometry::v_GetFace([[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetNumVerts()
 */
int Geometry::v_GetNumVerts() const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return 0;
}

/**
 * @copydoc Geometry::GetEorient()
 */
StdRegions::Orientation Geometry::v_GetEorient(
    [[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is not valid for this geometry.");
    return StdRegions::eForwards;
}

/**
 * @copydoc Geometry::GetForient()
 */
StdRegions::Orientation Geometry::v_GetForient(
    [[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is not valid for this geometry.");
    return StdRegions::eFwd;
}

/**
 * @copydoc Geometry::GetNumEdges()
 */
int Geometry::v_GetNumEdges() const
{
    return 0;
}

/**
 * @copydoc Geometry::GetNumFaces()
 */
int Geometry::v_GetNumFaces() const
{
    return 0;
}

/**
 * @copydoc Geometry::GetNumFacets()
 */
int Geometry::v_GetNumFacets() const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return 0;
}

/**
 * @copydoc Geometry::GetFacet()
 */
Geometry *Geometry::v_GetFacet([[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetShapeDim()
 */
int Geometry::v_GetShapeDim() const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return 0;
}

/**
 * Calculates the GeomType (deformed, regular etc).
 */
GeomType Geometry::v_CalcGeomType([[maybe_unused]] IsoParam &iso)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return eNoGeomType;
}

/**
 * @copydoc Geometry::GenGeomFactors()
 */
GeomFactorsUniquePtr Geometry::v_GenGeomFactors(
    [[maybe_unused]] LibUtilities::PointsKeyVector &keyTgt)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return GeomFactorsUniquePtr();
}

/**
 * @copydoc Geometry::GetXmap()
 */
StdRegions::StdExpansionSharedPtr Geometry::v_GetXmap() const
{
    return m_xmap;
}

/**
 * @copydoc Geometry::GetVertexEdgeMap()
 */
int Geometry::v_GetVertexEdgeMap([[maybe_unused]] const int i,
                                 [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetVertexFaceMap()
 */
int Geometry::v_GetVertexFaceMap([[maybe_unused]] const int i,
                                 [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetEdgeFaceMap()
 */
int Geometry::v_GetEdgeFaceMap([[maybe_unused]] const int i,
                               [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetEdgeNormalToFaceVert()
 */
int Geometry::v_GetEdgeNormalToFaceVert([[maybe_unused]] const int i,
                                        [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetDir()
 */
int Geometry::v_GetDir([[maybe_unused]] const int i,
                       [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetCoord()
 */
NekDouble Geometry::v_GetCoord(
    [[maybe_unused]] const int i,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Lcoord)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
    return 0.0;
}

/**
 * @copydoc Geometry::FillGeom()
 */
void Geometry::v_FillGeom()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
}

/**
 * @brief Change curvature order for this Geometry
 *
 * @see v_MakeOrder()
 */
std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> Geometry::MakeOrder(
    int order, LibUtilities::PointsType pType)
{
    return v_MakeOrder(order, pType);
}

/**
 * @copydoc Geometry::MakeOrder()
 */
std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> Geometry::
    v_MakeOrder([[maybe_unused]] int order,
                [[maybe_unused]] const LibUtilities::PointsType pType)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");

    return std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>>{};
}

/**
 * @copydoc Geometry::Reset()
 */
void Geometry::v_Reset([[maybe_unused]] CurveMap &curvedEdges,
                       [[maybe_unused]] CurveMap &curvedFaces)
{
    // Reset state
    m_state = eNotFilled;
}

/**
 * @copydoc Geometry::ResetLite()
 */
void Geometry::v_ResetLite()
{
    // Reset state
    m_state = eNotFilled;
}

void Geometry::v_Setup()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
}

} // namespace Nektar::SpatialDomains
