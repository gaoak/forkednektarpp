///////////////////////////////////////////////////////////////////////////////
//
// File: Geometry.h
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
// Description: Base class specification for the Geometry class.
//
///////////////////////////////////////////////////////////////////////////////

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

/**
 * @brief Base class for the geometry of an element and of its boundary.
 *
 * A Geometry is a shape: its topology, given by the vertices, edges and faces
 * it is built from, and its position in space, given by the mapping
 * \f$\chi:\xi\mapsto x\f$ from the reference element to the physical one.
 * The mapping is held as an expansion, #GetXmap(), together with the
 * coefficients #GetCoeffs() that express each physical coordinate in it. The
 * coefficients come from the boundary -- a face takes them from its edges, an
 * element from its faces -- so a Geometry only describes itself completely
 * once #FillGeom() has gathered them, which happens on first use rather than
 * at construction.
 *
 * Shapes derive from Geometry through a per-dimension intermediate
 * (#Geometry1D, #Geometry2D, #Geometry3D), which is where the entities of one
 * dimension lower are held. The lowest dimension, #PointGeom, is the only one
 * that carries coordinates directly; everything above it refers to the same
 * shared points, so moving a point moves every entity built on it.
 *
 * Deciding whether the mapping is affine (#eRegular) or not (#eDeformed) is
 * #CalcGeomType(); the metric terms derived from it are #GeomFactors. Point
 * location -- inverting \f$\chi\f$ to ask which element contains a physical
 * coordinate -- is not part of a Geometry: it is an algorithm applied to one,
 * and lives in #GeometryLocator.
 */
class Geometry
{
public:
    /// Construct with no coordinate dimension set.
    SPATIAL_DOMAINS_EXPORT Geometry();
    /// Construct for a space of dimension @p coordim.
    SPATIAL_DOMAINS_EXPORT Geometry(int coordim);
    SPATIAL_DOMAINS_EXPORT virtual ~Geometry() = default;

    /// Copy: the point-location cache is not copied, and is rebuilt on demand.
    SPATIAL_DOMAINS_EXPORT Geometry(const Geometry &that);
    /// Assign: the point-location cache is dropped, and is rebuilt on demand.
    SPATIAL_DOMAINS_EXPORT Geometry &operator=(const Geometry &that);

    //---------------------------------------
    // Helper functions
    //---------------------------------------

    /// Dimension of the space this object is embedded in.
    SPATIAL_DOMAINS_EXPORT inline int GetCoordim() const;
    /// Set the dimension of the space this object is embedded in.
    SPATIAL_DOMAINS_EXPORT inline void SetCoordim(int coordim);
    /// Build the #GeomFactors for this object at the quadrature points
    /// @p keyTgt.
    SPATIAL_DOMAINS_EXPORT GeomFactorsUniquePtr
    GenGeomFactors(LibUtilities::PointsKeyVector &keyTgt);
    /// The shape this object is, as a LibUtilities::ShapeType.
    SPATIAL_DOMAINS_EXPORT LibUtilities::ShapeType GetShapeType(void);

    //---------------------------------------
    // Set and get ID
    //---------------------------------------

    /// Get the ID of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetGlobalID(void) const;
    /// Set the ID of this object.
    SPATIAL_DOMAINS_EXPORT inline void SetGlobalID(int globalid);

    //---------------------------------------
    // Vertex, edge and face access
    //---------------------------------------

    /// Get the ID of vertex @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetVid(int i) const;
    /// Get the ID of edge @p i of this object.
    SPATIAL_DOMAINS_EXPORT int GetEid(int i) const;
    /// Get the ID of face @p i of this object.
    SPATIAL_DOMAINS_EXPORT int GetFid(int i) const;
    /// Get the ID of trace @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetTid(int i) const;
    /// Get vertex @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline PointGeom *GetVertex(int i) const;
    /// Get edge @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline Geometry1D *GetEdge(int i) const;
    /// Get face @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline Geometry2D *GetFace(int i) const;
    /// How edge @p i sits against the standard element's ordering of edges.
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetEorient(
        const int i) const;
    /// How face @p i sits against the standard element's ordering of faces.
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetForient(
        const int i) const;
    /// Get the number of vertices of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetNumVerts() const;
    /// Get the number of edges of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetNumEdges() const;
    /// Get the number of faces of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetNumFaces() const;
    /// Get the number of facets of this object.
    SPATIAL_DOMAINS_EXPORT inline int GetNumFacets() const;
    /// Get facet @p i of this object.
    SPATIAL_DOMAINS_EXPORT inline Geometry *GetFacet(int i) const;
    /// Dimension of the shape itself, which may be less than #GetCoordim().
    SPATIAL_DOMAINS_EXPORT inline int GetShapeDim() const;
    /// The curvature attached to this object, or nullptr if it is
    /// straight-sided. Only valid for shapes that can carry one.
    SPATIAL_DOMAINS_EXPORT virtual Curve *GetCurve()
    {
        ASSERTL0(false, "GetCurve() not valid for this geometry type");
        return nullptr;
    }

    //---------------------------------------
    // \chi mapping access
    //---------------------------------------

    /// The expansion holding the mapping \f$\chi\f$ from the standard element
    /// to this one. Shared between geometries and not owned.
    SPATIAL_DOMAINS_EXPORT inline StdRegions::StdExpansion *GetXmap() const;
    /// The coefficients of \f$\chi\f$ in coordinate direction @p i, filling
    /// them first if that has not happened yet.
    SPATIAL_DOMAINS_EXPORT inline const Array<OneD, const NekDouble> &GetCoeffs(
        const int i);
    /// Gather the coefficients of \f$\chi\f$ from this object's boundary,
    /// unless that has already been done.
    SPATIAL_DOMAINS_EXPORT inline void FillGeom();
    /// Evaluate the mapping: physical coordinate in direction @p i at the local
    /// collapsed coordinate @p Lcoord.
    SPATIAL_DOMAINS_EXPORT inline NekDouble GetCoord(
        const int i, const Array<OneD, const NekDouble> &Lcoord);
    /// Re-express this object's curvature at polynomial order @p order,
    /// returning the curve and the points generated for it.
    SPATIAL_DOMAINS_EXPORT std::pair<CurveUniquePtr,
                                     std::vector<PointGeomUniquePtr>>
    MakeOrder(int order, const LibUtilities::PointsType pType);

    //---------------------------------------
    // Misc. helper functions
    //---------------------------------------

    /// The @p j'th standard element edge connected to vertex @p i.
    SPATIAL_DOMAINS_EXPORT inline int GetVertexEdgeMap(int i, int j) const;
    /// The @p j'th standard element face connected to vertex @p i.
    SPATIAL_DOMAINS_EXPORT inline int GetVertexFaceMap(int i, int j) const;
    /// The @p j'th standard element face connected to edge @p i.
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeFaceMap(int i, int j) const;
    /// The @p j'th standard element edge normal to face @p i.
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeNormalToFaceVert(int i,
                                                              int j) const;
    /// The element coordinate direction that face direction @p j of face @p i
    /// runs along.
    SPATIAL_DOMAINS_EXPORT inline int GetDir(const int i,
                                             const int j = 0) const;
    /// Classify whether this shape is regular or deformed.
    SPATIAL_DOMAINS_EXPORT GeomType CalcGeomType();
    /// Classify whether this shape is regular or deformed, and hand back the
    /// isoparametric coefficients computed as part of this decision. Used by
    /// #GeometryLocator.
    SPATIAL_DOMAINS_EXPORT GeomType CalcGeomType(IsoParam &iso);
    /// True if every \f$\chi\f$ basis is linear.
    SPATIAL_DOMAINS_EXPORT bool HasLinearXmap();

    /// Re-read this object's curvature and rebuild everything derived from
    /// it, recursing into the entities it is built from.
    SPATIAL_DOMAINS_EXPORT inline void Reset(CurveMap &curvedEdges,
                                             CurveMap &curvedFaces);
    /// Re-derive only the orientations, leaving the mapping alone.
    SPATIAL_DOMAINS_EXPORT inline void ResetLite();
    /// As #Reset(), but without recursing into this object's boundary.
    SPATIAL_DOMAINS_EXPORT inline void ResetNonRecursive(CurveMap &curvedEdges,
                                                         CurveMap &curvedFaces);

    /// Build the mapping \f$\chi\f$, unless it has already been built.
    SPATIAL_DOMAINS_EXPORT inline void Setup();

protected:
    // Note that we use a bitfield approach to compress the first five items
    // below, since these are small enough to share one word between them. This
    // provides considerable memory compression for meshes containing large
    // numbers of geometry objects.

    /// Coordinate dimension of this geometry object.
    unsigned int m_coordim : 3;
    /// Type of shape.
    LibUtilities::ShapeType m_shapeType : 6;
    /// Enumeration to dictate whether coefficients are filled.
    GeomState m_state : 3;
    /// Cached regular/deformed classification; eNoGeomType until computed.
    GeomType m_geomType : 3;
    /// Whether or not the setup routines have been run
    bool m_setupState : 1;
    /// Global ID
    int m_globalID;
    /// \f$\chi\f$ mapping containing isoparametric transformation.
    ///
    /// This is not owned by the Geometry object: every map comes from a cache
    /// that lives for the length of the process generated by #XmapFactory.
    StdRegions::StdExpansion *m_xmap = nullptr;
    /// Array containing expansion coefficients of @p m_xmap
    std::vector<Array<OneD, NekDouble>> m_coeffs;

    //---------------------------------------
    // Virtual functions (see above)
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
    virtual StdRegions::StdExpansion *v_GetXmap() const;

    /// Fill Geometry::m_coeffs from this shape's vertices, edges and faces.
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

inline int Geometry::GetCoordim() const
{
    return m_coordim;
}

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

inline int Geometry::GetGlobalID(void) const
{
    return m_globalID;
}

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

inline int Geometry::GetVid(int i) const
{
    return v_GetVid(i);
}

inline PointGeom *Geometry::GetVertex(int i) const
{
    return v_GetVertex(i);
}

inline Geometry1D *Geometry::GetEdge(int i) const
{
    return v_GetEdge(i);
}

inline Geometry2D *Geometry::GetFace(int i) const
{
    return v_GetFace(i);
}

inline StdRegions::Orientation Geometry::GetEorient(const int i) const
{
    return v_GetEorient(i);
}

inline StdRegions::Orientation Geometry::GetForient(const int i) const
{
    return v_GetForient(i);
}

inline int Geometry::GetNumVerts() const
{
    return v_GetNumVerts();
}

inline int Geometry::GetNumEdges() const
{
    return v_GetNumEdges();
}

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

inline StdRegions::StdExpansion *Geometry::GetXmap() const
{
    return v_GetXmap();
}

inline const Array<OneD, const NekDouble> &Geometry::GetCoeffs(const int i)
{
    // The coefficients are allocated and filled on demand, so asking for them
    // is what triggers it.
    FillGeom();

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
    // Every v_FillGeom() returns immediately when the geometry is already
    // filled, and being filled implies having been set up, so answer that case
    // here rather than paying a virtual call for it.
    if (m_state == ePtsFilled)
    {
        return;
    }

    // Everything a fill needs beyond the shape-specific part is done here, so
    // that v_FillGeom() is only ever the fill itself. A geometry built through
    // ElementLite has not been set up, and the coefficients are allocated on
    // first fill rather than at setup, so both happen on the way in.
    if (!m_setupState)
    {
        v_Setup();
    }

    SetUpCoeffs(m_xmap->GetNcoeffs());
    v_FillGeom();
    m_state = ePtsFilled;
}

inline NekDouble Geometry::GetCoord(const int i,
                                    const Array<OneD, const NekDouble> &Lcoord)
{
    return v_GetCoord(i, Lcoord);
}

/**
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
 * For example, on a hexahedron, on face 0 at vertices 0,1,2,3 the edges normal
 * to that face are 4,5,6,7; so `GetEdgeNormalToFaceVert(0,j)` would therefore
 * return the values 4, 5, 6 and 7 respectively. We assume that @p j runs
 * between 0 and 3 inclusive on a quadrilateral face and between 0 and 2
 * inclusive on a triangular face.
 *
 * This is used to help set up a length scale normal to an face
 *
 * @param i  The face to query for the normal edge
 * @param j  The local vertex index between 0 and nverts on this face
 */
inline int Geometry::GetEdgeNormalToFaceVert(int i, int j) const
{
    return v_GetEdgeNormalToFaceVert(i, j);
}

inline int Geometry::GetDir(const int faceidx, const int facedir) const
{
    return v_GetDir(faceidx, facedir);
}

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
