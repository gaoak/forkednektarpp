////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry.h
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
//  Description:  This file contains the base class specification for the
//                Geometry class.
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_GEOMETRY_H
#define NEKTAR_SPATIALDOMAINS_GEOMETRY_H

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/Memory/ObjectPool.hpp>
#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/GeometryLocator.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

#include <array>
#include <map>
#include <memory>
#include <unordered_map>

namespace Nektar::SpatialDomains
{

class Geometry; // Forward declaration for typedef.
typedef unique_ptr_objpool<Geometry> GeometryUniquePtr;
typedef unique_ptr_objpool<GeomFactors> GeomFactorsUniquePtr;

class Geometry1D;
class Geometry2D;

class PointGeom;
typedef unique_ptr_objpool<PointGeom> PointGeomUniquePtr;

struct Curve;
typedef unique_ptr_objpool<Curve> CurveUniquePtr;
typedef std::map<int, CurveUniquePtr> CurveMap;
// static CurveMap NullCurveMap;

/// \brief Less than operator to sort Geometry objects by global id when sorting
/// STL containers.
SPATIAL_DOMAINS_EXPORT bool SortByGlobalId(const Geometry *&lhs,
                                           const Geometry *&rhs);

SPATIAL_DOMAINS_EXPORT bool GlobalIdEquality(const Geometry *&lhs,
                                             const Geometry *&rhs);

/// Base class for shape geometry information
class Geometry
{
public:
    SPATIAL_DOMAINS_EXPORT Geometry();
    SPATIAL_DOMAINS_EXPORT Geometry(int coordim);
    SPATIAL_DOMAINS_EXPORT virtual ~Geometry() = default;

    /// The point-location cache is not copied; the copy rebuilds it on demand.
    SPATIAL_DOMAINS_EXPORT Geometry(const Geometry &that);
    SPATIAL_DOMAINS_EXPORT Geometry &operator=(const Geometry &that);

    //---------------------------------------
    // Helper functions
    //---------------------------------------

    SPATIAL_DOMAINS_EXPORT inline int GetCoordim() const;
    SPATIAL_DOMAINS_EXPORT inline void SetCoordim(int coordim);
    SPATIAL_DOMAINS_EXPORT GeomFactorsUniquePtr
    GenGeomFactors(LibUtilities::PointsKeyVector &keyTgt);
    SPATIAL_DOMAINS_EXPORT LibUtilities::ShapeType GetShapeType(void);

    //---------------------------------------
    // Set and get ID
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetGlobalID(void) const;
    SPATIAL_DOMAINS_EXPORT inline void SetGlobalID(int globalid);

    //---------------------------------------
    // Vertex, edge and face access
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetVid(int i) const;
    SPATIAL_DOMAINS_EXPORT int GetEid(int i) const;
    SPATIAL_DOMAINS_EXPORT int GetFid(int i) const;
    SPATIAL_DOMAINS_EXPORT inline int GetTid(int i) const;
    SPATIAL_DOMAINS_EXPORT inline PointGeom *GetVertex(int i) const;
    SPATIAL_DOMAINS_EXPORT inline Geometry1D *GetEdge(int i) const;
    SPATIAL_DOMAINS_EXPORT inline Geometry2D *GetFace(int i) const;
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetEorient(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetForient(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumVerts() const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumEdges() const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumFaces() const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumFacets() const;
    SPATIAL_DOMAINS_EXPORT inline Geometry *GetFacet(int i) const;
    SPATIAL_DOMAINS_EXPORT inline int GetShapeDim() const;
    SPATIAL_DOMAINS_EXPORT virtual Curve *GetCurve()
    {
        ASSERTL0(false, "GetCurve() not valid for this geometry type");
        return nullptr;
    }

    //---------------------------------------
    // \chi mapping access
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline StdRegions::StdExpansionSharedPtr GetXmap()
        const;
    SPATIAL_DOMAINS_EXPORT inline const Array<OneD, const NekDouble> &GetCoeffs(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline void FillGeom();
    /// Evaluate the mapping: physical coordinate in direction @p i at the
    /// local collapsed coordinate @p Lcoord.
    SPATIAL_DOMAINS_EXPORT inline NekDouble GetCoord(
        const int i, const Array<OneD, const NekDouble> &Lcoord);
    SPATIAL_DOMAINS_EXPORT std::pair<CurveUniquePtr,
                                     std::vector<PointGeomUniquePtr>>
    MakeOrder(int order, const LibUtilities::PointsType pType);

    //---------------------------------------
    // Misc. helper functions
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetVertexEdgeMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetVertexFaceMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeFaceMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeNormalToFaceVert(int i,
                                                              int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetDir(const int i,
                                             const int j = 0) const;
    SPATIAL_DOMAINS_EXPORT GeomType CalcGeomType();
    /// Classify, and hand back the isoparametric coefficients settled on the
    /// way. Used by GeometryLocator, which needs both.
    SPATIAL_DOMAINS_EXPORT GeomType CalcGeomType(IsoParam &iso);
    /// True if every \f$\chi\f$ basis is linear.
    SPATIAL_DOMAINS_EXPORT bool HasLinearXmap();

    SPATIAL_DOMAINS_EXPORT inline void Reset(CurveMap &curvedEdges,
                                             CurveMap &curvedFaces);
    SPATIAL_DOMAINS_EXPORT inline void ResetLite();
    SPATIAL_DOMAINS_EXPORT inline void ResetNonRecursive(CurveMap &curvedEdges,
                                                         CurveMap &curvedFaces);

    SPATIAL_DOMAINS_EXPORT inline void Setup();

protected:
    /// Coordinate dimension of this geometry object.
    int m_coordim;
    /// \f$\chi\f$ mapping containing isoparametric transformation.
    StdRegions::StdExpansionSharedPtr m_xmap;
    /// Enumeration to dictate whether coefficients are filled.
    GeomState m_state;
    /// Wether or not the setup routines have been run
    bool m_setupState;
    /// Type of shape.
    LibUtilities::ShapeType m_shapeType;
    /// Global ID
    int m_globalID;
    /// Array containing expansion coefficients of @p m_xmap
    std::vector<Array<OneD, NekDouble>> m_coeffs;
    /// Cached regular/deformed classification; eNoGeomType until computed.
    /// Cheap enough to keep on every geometry.
    GeomType m_geomType;

    //---------------------------------------
    // Helper functions
    //---------------------------------------
    virtual int v_GetVid(int i) const;
    virtual PointGeom *v_GetVertex(const int i) const;
    virtual Geometry1D *v_GetEdge(const int i) const;
    virtual Geometry2D *v_GetFace(const int i) const;
    virtual StdRegions::Orientation v_GetEorient(const int i) const;
    virtual StdRegions::Orientation v_GetForient(const int i) const;
    virtual int v_GetNumVerts() const;
    virtual int v_GetNumEdges() const;
    virtual int v_GetNumFaces() const;
    virtual int v_GetNumFacets() const;
    virtual Geometry *v_GetFacet(const int i) const;
    virtual int v_GetShapeDim() const;

    virtual GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt);
    virtual StdRegions::StdExpansionSharedPtr v_GetXmap() const;
    virtual void v_FillGeom();
    virtual std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> v_MakeOrder(
        int order, const LibUtilities::PointsType pType);

    virtual NekDouble v_GetCoord(const int i,
                                 const Array<OneD, const NekDouble> &Lcoord);

    virtual int v_GetVertexEdgeMap(int i, int j) const;
    virtual int v_GetVertexFaceMap(int i, int j) const;
    virtual int v_GetEdgeFaceMap(int i, int j) const;
    virtual int v_GetEdgeNormalToFaceVert(const int i, const int j) const;
    virtual int v_GetDir(const int faceidx, const int facedir) const;

    /// Classify this geometry as regular or deformed, filling @p loc with the
    /// isoparametric data derived along the way.
    ///
    /// Classification and inverse-mapping setup share this one implementation
    /// because deciding regularity *is* testing whether the nonlinear
    /// isoparametric coefficients vanish. Callers that only want the
    /// classification pass scratch storage and discard it; callers that intend
    /// to invert the mapping pass the geometry's own locator.
    virtual GeomType v_CalcGeomType(IsoParam &iso);
    virtual void v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces);
    virtual void v_ResetLite();
    virtual void v_Setup();

    inline void SetUpCoeffs(const int nCoeffs);
}; // class Geometry

/**
 * @brief Unary function that constructs a hash of a Geometry object, based on
 * the vertex IDs.
 */
struct GeometryHash
{
    std::size_t operator()(GeometryUniquePtr const &p) const
    {
        int i;
        size_t seed = 0;
        int nVert   = p->GetNumVerts();
        std::vector<unsigned int> ids(nVert);

        for (i = 0; i < nVert; ++i)
        {
            ids[i] = p->GetVid(i);
        }
        std::sort(ids.begin(), ids.end());
        hash_range(seed, ids.begin(), ids.end());

        return seed;
    }
};

/**
 * @brief Return the coordinate dimension of this object (i.e. the dimension of
 * the space in which this object is embedded).
 */
inline int Geometry::GetCoordim() const
{
    return m_coordim;
}

/**
 * @brief Sets the coordinate dimension of this object (i.e. the dimension of
 * the space in which this object is embedded).
 */
inline void Geometry::SetCoordim(int dim)
{
    m_coordim = dim;
}

/**
 * @brief Get the geometric shape type of this object.
 */
inline LibUtilities::ShapeType Geometry::GetShapeType()
{
    return m_shapeType;
}

/**
 * @brief Get the ID of this object.
 */
inline int Geometry::GetGlobalID(void) const
{
    return m_globalID;
}

/**
 * @brief Set the ID of this object.
 */
inline void Geometry::SetGlobalID(int globalid)
{
    m_globalID = globalid;
}

/**
 * @brief Get the ID of trace @p i of this object.
 *
 * The trace element is the facet one dimension lower than the object; for
 * example, a quadrilateral has four trace segments forming its boundary.
 */
inline int Geometry::GetTid(int i) const
{
    const int nDim = GetShapeDim();
    return nDim == 1   ? GetVid(i)
           : nDim == 2 ? GetEid(i)
           : nDim == 3 ? GetFid(i)
                       : 0;
}

/**
 * @brief Returns global id of vertex @p i of this object.
 */
inline int Geometry::GetVid(int i) const
{
    return v_GetVid(i);
}

/**
 * @brief Returns vertex @p i of this object.
 */
inline PointGeom *Geometry::GetVertex(int i) const
{
    return v_GetVertex(i);
}

/**
 * @brief Returns edge @p i of this object.
 */
inline Geometry1D *Geometry::GetEdge(int i) const
{
    return v_GetEdge(i);
}

/**
 * @brief Returns face @p i of this object.
 */
inline Geometry2D *Geometry::GetFace(int i) const
{
    return v_GetFace(i);
}

/**
 * @brief Returns the orientation of edge @p i with respect to the ordering of
 * edges in the standard element.
 */
inline StdRegions::Orientation Geometry::GetEorient(const int i) const
{
    return v_GetEorient(i);
}

/**
 * @brief Returns the orientation of face @p i with respect to the ordering of
 * faces in the standard element.
 */
inline StdRegions::Orientation Geometry::GetForient(const int i) const
{
    return v_GetForient(i);
}

/**
 * @brief Get the number of vertices of this object.
 */
inline int Geometry::GetNumVerts() const
{
    return v_GetNumVerts();
}

/**
 * @brief Get the number of edges of this object.
 */
inline int Geometry::GetNumEdges() const
{
    return v_GetNumEdges();
}

/**
 * @brief Get the number of faces of this object.
 */
inline int Geometry::GetNumFaces() const
{
    return v_GetNumFaces();
}

/**
 * @brief Get the number of facets of this object.
 *
 * A vertex for a 1D geometry, an edge for a 2D geometry, and a face for a 3D
 * geometry.
 */
inline int Geometry::GetNumFacets() const
{
    return v_GetNumFacets();
}

/**
 * @brief Returns facet @p i of this object.
 */
inline Geometry *Geometry::GetFacet(int i) const
{
    return v_GetFacet(i);
}

/**
 * @brief Get the object's shape dimension.
 *
 * For example, a segment is one dimensional and quadrilateral is two
 * dimensional.
 */
inline int Geometry::GetShapeDim() const
{
    return v_GetShapeDim();
}

/**
 * @brief Used by Expansion to generate associated GeomFactors.
 */
inline GeomFactorsUniquePtr Geometry::GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    return v_GenGeomFactors(keyTgt);
}

/**
 * @brief Return the mapping object Geometry::m_xmap that represents the
 * coordinate transformation from standard element to physical element.
 */
inline StdRegions::StdExpansionSharedPtr Geometry::GetXmap() const
{
    return v_GetXmap();
}

/**
 * @brief Return the coefficients of the transformation Geometry::m_xmap in
 * coordinate direction @p i.
 */
inline const Array<OneD, const NekDouble> &Geometry::GetCoeffs(
    const int i) const
{
    return m_coeffs[i];
}

/**
 * @brief Populate the coordinate mapping Geometry::m_coeffs information from
 * any children geometry elements.
 *
 * @see v_FillGeom()
 */
inline void Geometry::FillGeom()
{
    // If the element is built with ElementLite, we don't have m_coeff, hence we
    // need Setup.
    if (!m_setupState)
    {
        v_Setup();
    }

    v_FillGeom();
}

/**
 * @brief Given local collapsed coordinate @p Lcoord, return the value of
 * physical coordinate in direction @p i.
 */
inline NekDouble Geometry::GetCoord(const int i,
                                    const Array<OneD, const NekDouble> &Lcoord)
{
    return v_GetCoord(i, Lcoord);
}

/**
 * @brief Returns the standard element edge IDs that are connected to a given
 * vertex.
 *
 * For example, on a prism, vertex 0 is connnected to edges 0, 3, and 4;
 * `GetVertexEdgeMap(0,j)` would therefore return the values 0, 1 and 4
 * respectively. We assume that @p j runs between 0 and 2 inclusive, which is
 * true for every 3D element asides from the pyramid.
 *
 * This function is used in the construction of the low-energy preconditioner.
 *
 * @param i  The vertex to query connectivity for.
 * @param j  The local edge index between 0 and 2 connected to this element.
 *
 * @todo Expand to work with pyramid elements.
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetVertexEdgeMap(int i, int j) const
{
    return v_GetVertexEdgeMap(i, j);
}

/**
 * @brief Returns the standard element face IDs that are connected to a given
 * vertex.
 *
 * For example, on a hexahedron, vertex 0 is connnected to faces 0, 1, and 4;
 * `GetVertexFaceMap(0,j)` would therefore return the values 0, 1 and 4
 * respectively. We assume that @p j runs between 0 and 2 inclusive, which is
 * true for every 3D element asides from the pyramid.
 *
 * This is used in the construction of the low-energy preconditioner.
 *
 * @param i  The vertex to query connectivity for.
 * @param j  The local face index between 0 and 2 connected to this element.
 *
 * @todo Expand to work with pyramid elements.
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetVertexFaceMap(int i, int j) const
{
    return v_GetVertexFaceMap(i, j);
}

/**
 * @brief Returns the standard element edge IDs that are connected to a given
 * face.
 *
 * For example, on a prism, edge 0 is connnected to faces 0 and 1;
 * `GetEdgeFaceMap(0,j)` would therefore return the values 0 and 1
 * respectively. We assume that @p j runs between 0 and 1 inclusive, since every
 * face is connected to precisely two faces for all 3D elements.
 *
 * This function is used in the construction of the low-energy preconditioner.
 *
 * @param i  The edge to query connectivity for.
 * @param j  The local face index between 0 and 1 connected to this element.
 *
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetEdgeFaceMap(int i, int j) const
{
    return v_GetEdgeFaceMap(i, j);
}

/**
 * @brief Returns the standard lement edge IDs that are normal to a given face
 * vertex.
 *
 * For example, on a hexahedron, on face 0 at vertices 0,1,2,3 the
 * edges normal to that face are 4,5,6,7, ; so
 * `GetEdgeNormalToFaceVert(0,j)` would therefore return the values 4,
 * 5, 6 and 7 respectively. We assume that @p j runs between 0 and 3
 * inclusive on a quadrilateral face and between 0 and 2 inclusive on
 * a triangular face.
 *
 * This is used to help set up a length scale normal to an face
 *
 * @param i  The face to query for the normal edge
 * @param j  The local vertex index between 0 and nverts on this face
 *
 */
inline int Geometry::GetEdgeNormalToFaceVert(int i, int j) const
{
    return v_GetEdgeNormalToFaceVert(i, j);
}

/**
 * @brief Returns the element coordinate direction corresponding to a given face
 * coordinate direction
 */
inline int Geometry::GetDir(const int faceidx, const int facedir) const
{
    return v_GetDir(faceidx, facedir);
}

/**
 * @brief Reset this geometry object: unset the current state, zero
 * Geometry::m_coeffs and remove allocated GeomFactors.
 */
inline void Geometry::Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    m_geomType = eNoGeomType;
    v_Reset(curvedEdges, curvedFaces);
}

/**
 * @brief Reset this geometry object without rebuilding mapping data.
 *
 * The fill state is cleared here rather than in v_ResetLite(), so that a shape
 * overriding it to recompute edge and face orientation cannot forget to. A
 * geometry that kept its state would also keep the coefficients FillGeom()
 * produced for the arrangement it had before, and FillGeom() returns
 * immediately once a geometry is filled.
 */
inline void Geometry::ResetLite()
{
    m_state    = eNotFilled;
    m_geomType = eNoGeomType;
    v_ResetLite();
}

/**
 * @brief Reset this geometry object non-recursively: unset the current state,
 * zero Geometry::m_coeffs and remove allocated GeomFactors.
 */
inline void Geometry::ResetNonRecursive(CurveMap &curvedEdges,
                                        CurveMap &curvedFaces)
{
    m_geomType = eNoGeomType;
    Geometry::v_Reset(curvedEdges, curvedFaces);
}

inline void Geometry::Setup()
{
    v_Setup();
}

/**
 * @brief Initialise the Geometry::m_coeffs array.
 */
inline void Geometry::SetUpCoeffs(const int nCoeffs)
{
    m_coeffs = std::vector<Array<OneD, NekDouble>>(m_coordim);

    for (int i = 0; i < m_coordim; ++i)
    {
        m_coeffs[i] = Array<OneD, NekDouble>(nCoeffs, 0.0);
    }
}

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_GEOMETRY_H
