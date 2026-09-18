////////////////////////////////////////////////////////////////////////////////
//
//  File: CADAssociation.h
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
//  Description: Association between mesh entities and the CAD they lie on.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_CADSYSTEM_CADASSOCIATION
#define NEKTAR_SPATIALDOMAINS_CADSYSTEM_CADASSOCIATION

#include <array>
#include <unordered_map>
#include <vector>

#include <SpatialDomains/CADSystem/CADSystem.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

namespace Nektar::SpatialDomains
{

class Geometry;

/**
 * @brief Where a single mesh entity sits on the CAD.
 *
 * Holds the CAD object (curve, surface or vertex) together with the parametric
 * coordinates of the mesh entity on that object: @f$ t @f$ in param[0] for a
 * CAD curve, @f$ (u,v) @f$ in param[0..1] for a CAD surface. The parameters are
 * unused for a CAD vertex.
 */
struct CADLink
{
    CADLink() = default;

    CADLink(CADObjectSharedPtr pObj,
            std::array<NekDouble, 2> pParam = {0.0, 0.0})
        : obj(std::move(pObj)), param(pParam)
    {
    }

    /// The CAD curve, surface or vertex this entity lies on.
    CADObjectSharedPtr obj;
    /// Parametric position on #obj: t in [0], or (u,v) in [0],[1].
    std::array<NekDouble, 2> param{0.0, 0.0};

    CADType::cadType Type() const
    {
        return obj->GetType();
    }

    int Id() const
    {
        return obj->GetId();
    }

    /// Compares both the CAD object and the parametric position.
    bool operator==(const CADLink &rhs) const
    {
        return obj == rhs.obj && param == rhs.param;
    }

    bool operator!=(const CADLink &rhs) const
    {
        return !(*this == rhs);
    }
};

/**
 * @brief The mesh entity @f$ \leftrightarrow @f$ CAD relation for a MeshGraph.
 *
 * Mesh entities are associated with the CAD they lie on through this class,
 * rather than by holding CAD data on the geometries themselves: the great
 * majority of entities in a mesh are not connected to any CAD, so a per-entity
 * field would be almost entirely wasted. Only entities that actually carry an
 * association appear here.
 *
 * Entities are keyed by pointer, not by global ID, because high-order curvature
 * nodes do not have a meaningful global ID -- every one is created with ID 0 --
 * and pointer identity is the only thing that distinguishes them.
 * Global IDs are in any case not stable for the lifetime of a mesh: several
 * modules renumber every vertex from scratch. The consequence is that entries
 * must be removed when the entity they refer to is destroyed, or a recycled
 * address will silently appear to carry the dead entity's association; see
 * Remove().
 *
 * A single entity may hold several links at once — a vertex sitting on two CAD
 * surfaces and the CAD curve between them, for instance — and at most one link
 * per distinct CAD object. The typed accessors (GetSurf(), GetCurve(),
 * GetVert()) filter by CAD type.
 */
class CADAssociation
{
public:
    SPATIAL_DOMAINS_EXPORT CADAssociation()  = default;
    SPATIAL_DOMAINS_EXPORT ~CADAssociation() = default;

    /// The CAD system these associations refer into.
    SPATIAL_DOMAINS_EXPORT CADSystemSharedPtr &GetCAD()
    {
        return m_cad;
    }

    SPATIAL_DOMAINS_EXPORT const CADSystemSharedPtr &GetCAD() const
    {
        return m_cad;
    }

    SPATIAL_DOMAINS_EXPORT void SetCAD(CADSystemSharedPtr cad)
    {
        m_cad = std::move(cad);
    }

    // ------------------------------------------------------------------
    // Queries. None of these insert.
    // ------------------------------------------------------------------

    /// True if @p geom carries any CAD association.
    SPATIAL_DOMAINS_EXPORT bool Has(const Geometry *geom) const
    {
        return !GetLinks(geom).empty();
    }

    /// True if @p geom is associated with at least one CAD object of type @p t.
    SPATIAL_DOMAINS_EXPORT bool Has(const Geometry *geom,
                                    CADType::cadType t) const
    {
        return Count(geom, t) > 0;
    }

    /// Number of CAD objects of type @p t that @p geom is associated with.
    SPATIAL_DOMAINS_EXPORT size_t Count(const Geometry *geom,
                                        CADType::cadType t) const;

    /**
     * @brief All links held by @p geom, in insertion order.
     *
     * Returns a reference to a shared empty vector when @p geom has no
     * association, so that callers may iterate unconditionally.
     */
    SPATIAL_DOMAINS_EXPORT const std::vector<CADLink> &GetLinks(
        const Geometry *geom) const;

    /// The links held by @p geom whose CAD object is of type @p t.
    SPATIAL_DOMAINS_EXPORT std::vector<CADLink> GetLinks(
        const Geometry *geom, CADType::cadType t) const;

    /**
     * @brief The link to the CAD object of type @p t and ID @p cadId, or
     * nullptr if @p geom is not associated with it.
     *
     * The type is part of the key because CAD curves, surfaces and vertices are
     * numbered independently: curve 3 and surface 3 both exist, and a mesh
     * vertex commonly holds a link to each.
     */
    SPATIAL_DOMAINS_EXPORT const CADLink *GetLink(const Geometry *geom,
                                                  CADType::cadType t,
                                                  int cadId) const;

    /// The first link of type @p t held by @p geom, or nullptr if there is
    /// none.
    SPATIAL_DOMAINS_EXPORT const CADLink *GetLink(const Geometry *geom,
                                                  CADType::cadType t) const;

    /// The first CAD surface @p geom lies on, or nullptr.
    SPATIAL_DOMAINS_EXPORT CADSurfSharedPtr GetSurf(const Geometry *geom) const;

    /// The first CAD curve @p geom lies on, or nullptr.
    SPATIAL_DOMAINS_EXPORT CADCurveSharedPtr
    GetCurve(const Geometry *geom) const;

    /// The CAD vertex @p geom coincides with, or nullptr.
    SPATIAL_DOMAINS_EXPORT CADVertSharedPtr GetVert(const Geometry *geom) const;

    /// All CAD surfaces @p geom lies on.
    SPATIAL_DOMAINS_EXPORT std::vector<CADSurfSharedPtr> GetSurfs(
        const Geometry *geom) const;

    /// All CAD curves @p geom lies on.
    SPATIAL_DOMAINS_EXPORT std::vector<CADCurveSharedPtr> GetCurves(
        const Geometry *geom) const;

    /**
     * @brief The @f$ (u,v) @f$ of @p geom on the CAD surface of ID @p cadId.
     *
     * It is a fatal error for @p geom not to be associated with that surface.
     */
    SPATIAL_DOMAINS_EXPORT std::array<NekDouble, 2> GetSurfUV(
        const Geometry *geom, int cadId) const;

    /**
     * @brief The @f$ t @f$ of @p geom on the CAD curve of ID @p cadId.
     *
     * It is a fatal error for @p geom not to be associated with that curve.
     */
    SPATIAL_DOMAINS_EXPORT NekDouble GetCurveT(const Geometry *geom,
                                               int cadId) const;

    // ------------------------------------------------------------------
    // Mutation. Add() is the only way an entry is created.
    // ------------------------------------------------------------------

    /**
     * @brief Associate @p geom with the CAD object in @p link.
     *
     * At most one link is held per distinct CAD object: adding a link to a CAD
     * object already present overwrites the parametric position rather than
     * appending a second entry. Links to other CAD objects are kept, so this is
     * the right call where an entity may legitimately sit on several CAD
     * objects of one type — a mesh vertex on two CAD surfaces, say.
     */
    SPATIAL_DOMAINS_EXPORT void Add(const Geometry *geom, CADLink link);

    /**
     * @brief Associate @p geom with @p link, replacing any link it already
     * holds of the same CAD type.
     *
     * For relations that are single-valued by construction: a mesh face lies on
     * exactly one CAD surface, a mesh edge on one CAD curve, and a mesh vertex
     * coincides with at most one CAD vertex. Use Add() otherwise.
     */
    SPATIAL_DOMAINS_EXPORT void Set(const Geometry *geom, CADLink link);

    /**
     * @brief Drop every association held by @p geom.
     *
     * Also drops the associations of @p geom's curvature nodes, since those die
     * with the curve that owns them. Must be called whenever a geometry is
     * destroyed: the pool allocator hands freed addresses back out to the next
     * allocation of the same type, so a stale entry would otherwise reappear
     * attached to an unrelated entity.
     */
    SPATIAL_DOMAINS_EXPORT void Remove(Geometry *geom);

    /// Drop links of type @p t from zero-dimensional entities (vertices and
    /// curvature nodes), leaving edge and face associations in place.
    SPATIAL_DOMAINS_EXPORT void ClearVertLinks(CADType::cadType t);

    /// Drop every association, but keep the CAD system itself.
    SPATIAL_DOMAINS_EXPORT void Clear()
    {
        m_links.clear();
    }

    // ------------------------------------------------------------------
    // Iteration, for I/O.
    // ------------------------------------------------------------------

    SPATIAL_DOMAINS_EXPORT auto begin() const
    {
        return m_links.begin();
    }

    SPATIAL_DOMAINS_EXPORT auto end() const
    {
        return m_links.end();
    }

    SPATIAL_DOMAINS_EXPORT size_t size() const
    {
        return m_links.size();
    }

private:
    CADSystemSharedPtr m_cad;
    std::unordered_map<const Geometry *, std::vector<CADLink>> m_links;
};

typedef std::shared_ptr<CADAssociation> CADAssociationSharedPtr;

} // namespace Nektar::SpatialDomains

#endif
