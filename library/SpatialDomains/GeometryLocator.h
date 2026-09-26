////////////////////////////////////////////////////////////////////////////////
//
//  File: GeometryLocator.h
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
//  Description: Derived state used only for inverse mapping (x -> xi) and
//               geometry classification.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_GEOMETRYLOCATOR_H
#define NEKTAR_SPATIALDOMAINS_GEOMETRYLOCATOR_H

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <SpatialDomains/SpatialDomains.hpp>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

#include <array>
#include <limits>
#include <memory>

namespace Nektar::SpatialDomains
{

class Geometry;

/// Extents bounded across every supported shape: three coordinate directions,
/// eight isoparametric coefficients (the trilinear hexahedron) and four
/// vertices (a quadrilateral). Nothing here needs a runtime-sized container.
constexpr int kMaxGeomDim      = 3;
constexpr int kMaxGeomIsoCoeff = 8;
constexpr int kMaxGeomVerts    = 4;

/**
 * @brief The isoparametric description of a shape's \f$\chi\f$ mapping.
 *
 * This is precisely what a Geometry knows how to compute about itself, and
 * everything the inverse mapping needs from it. Keeping it separate from
 * GeometryLocator means classification -- which is exactly the test of whether
 * the nonlinear coefficients vanish -- can be answered with a stack-resident
 * temporary and no dependency on the locator at all.
 *
 * @see Geometry::CalcGeomType
 */
struct IsoParam
{
    /// Coefficients per coordinate direction. Only the leading @p m_nCoeff
    /// entries of each row are meaningful, and that count identifies the
    /// shape: 3 triangle, 4 quadrilateral or tetrahedron, 5 pyramid, 6 prism,
    /// 8 hexahedron.
    int m_nCoeff = 0;
    std::array<std::array<NekDouble, kMaxGeomIsoCoeff>, kMaxGeomDim> m_coeff{};

    /// 2D only: coordinate directions spanning the manifold.
    std::array<int, kMaxGeomDim> m_manifold{};
};

/**
 * @brief Inverts the mapping \f$\chi:\xi\mapsto x\f$ for one geometry.
 *
 * Point location -- deciding whether an element contains a Cartesian
 * coordinate, recovering the local coordinate that maps to it, or measuring
 * the distance to it -- is an algorithm applied to a geometry, not a property
 * of one. For anything but an affine element it is a nonlinear solve, and it
 * needs a cache of derived quantities to be quick. Neither the algorithm nor
 * the cache belongs on Geometry, so both live here and are owned by whoever
 * does the searching.
 *
 * Build one with GeometryLocator::Create() and keep it for as long as the
 * geometry is unchanged; it must not outlive the geometry it refers to. A
 * locator is invalidated by anything that moves or re-curves the geometry,
 * which is to say by any of the Geometry Reset entry points.
 *
 * The concrete algorithms differ by the dimension of the shape rather than by
 * the shape itself, which is why the subclasses are per-dimension.
 */
class GeometryLocator
{
public:
    SPATIAL_DOMAINS_EXPORT explicit GeometryLocator(Geometry *geom);
    SPATIAL_DOMAINS_EXPORT virtual ~GeometryLocator() = default;

    /// Build the locator appropriate to @p geom's dimension, with its cache
    /// already populated.
    SPATIAL_DOMAINS_EXPORT static std::unique_ptr<GeometryLocator> Create(
        Geometry *geom);

    /// Populate the cache. Called by Create() once the concrete type exists,
    /// since it dispatches virtually.
    SPATIAL_DOMAINS_EXPORT void Build();

    SPATIAL_DOMAINS_EXPORT Geometry *GetGeometry() const
    {
        return m_geom;
    }

    //---------------------------------------
    // Point lookups
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT std::array<NekDouble, 6> GetBoundingBox();
    SPATIAL_DOMAINS_EXPORT void ClearBoundingBox();

    SPATIAL_DOMAINS_EXPORT bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord, NekDouble tol = 0.0);
    SPATIAL_DOMAINS_EXPORT bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord,
        Array<OneD, NekDouble> &locCoord, NekDouble tol);
    SPATIAL_DOMAINS_EXPORT bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord,
        Array<OneD, NekDouble> &locCoord, NekDouble tol, NekDouble &dist);

    SPATIAL_DOMAINS_EXPORT NekDouble
    GetLocCoords(const Array<OneD, const NekDouble> &coords,
                 Array<OneD, NekDouble> &Lcoords);
    SPATIAL_DOMAINS_EXPORT NekDouble FindDistance(
        const Array<OneD, const NekDouble> &xs, Array<OneD, NekDouble> &xi);

    SPATIAL_DOMAINS_EXPORT int PreliminaryCheck(
        const Array<OneD, const NekDouble> &gloCoord);
    SPATIAL_DOMAINS_EXPORT bool MinMaxCheck(
        const Array<OneD, const NekDouble> &gloCoord);
    /// Clamp local coordinates into the reference element
    /// \f$[-1-tol,1+tol]^{dim}\f$, returning whether anything was clamped.
    /// Depends only on the reference element, so it needs no locator.
    SPATIAL_DOMAINS_EXPORT static bool ClampLocCoords(
        Array<OneD, NekDouble> &locCoord, int dim,
        NekDouble tol = std::numeric_limits<NekDouble>::epsilon());

protected:
    /// The geometry being searched. Not owned.
    Geometry *m_geom;

    /// The shape's isoparametric coefficients. The straight-edged 2D solve
    /// rewrites them into a factored form, so the locator keeps its own copy
    /// rather than sharing the geometry's.
    IsoParam m_iso;

    /// Non-zero if every \f$\chi\f$ basis is linear. Bits 1 and 2 are used
    /// by the 2D straight-edged solve to record which axes were swapped.
    int m_straightEdge = 0;

    /// Inverse of the linear part of @p m_iso, for regular elements.
    std::array<std::array<NekDouble, kMaxGeomDim>, kMaxGeomDim> m_invIsoParam{};

    /// Element bounding box as (xmin, ymin, zmin, xmax, ymax, zmax).
    bool m_hasBoundingBox = false;
    std::array<NekDouble, 6> m_boundingBox{};

    /// 2D only: inward edge normals used to reject points outside
    /// straight-sided elements. A curved edge gets no normal, so
    /// @p m_hasEdgeNormal records which entries are usable.
    bool m_edgeNormalBuilt = false;
    std::array<bool, kMaxGeomVerts> m_hasEdgeNormal{};
    std::array<std::array<NekDouble, 2>, kMaxGeomVerts> m_edgeNormal{};

    virtual NekDouble v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                                     Array<OneD, NekDouble> &Lcoords);
    virtual NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                                     Array<OneD, NekDouble> &xi);
    virtual int v_AllLeftCheck(const Array<OneD, const NekDouble> &gloCoord);
    virtual void v_CalculateInverseIsoParam();
    virtual void v_PreSolveStraightEdge();
};

typedef std::unique_ptr<GeometryLocator> GeometryLocatorUniquePtr;

/// Inverse mapping for a segment.
class GeometryLocator1D final : public GeometryLocator
{
public:
    using GeometryLocator::GeometryLocator;

protected:
    NekDouble v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                             Array<OneD, NekDouble> &Lcoords) override;
    NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                             Array<OneD, NekDouble> &xi) override;
};

/// Inverse mapping for a triangle or quadrilateral, including the closed-form
/// solve available when every edge is straight.
class GeometryLocator2D final : public GeometryLocator
{
public:
    using GeometryLocator::GeometryLocator;

protected:
    NekDouble v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                             Array<OneD, NekDouble> &Lcoords) override;
    NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                             Array<OneD, NekDouble> &xi) override;
    int v_AllLeftCheck(const Array<OneD, const NekDouble> &gloCoord) override;
    void v_CalculateInverseIsoParam() override;
    void v_PreSolveStraightEdge() override;

private:
    void SolveStraightEdgeQuad(const Array<OneD, const NekDouble> &coords,
                               Array<OneD, NekDouble> &Lcoords);
    void NewtonIterationForLocCoord(const Array<OneD, const NekDouble> &coords,
                                    const Array<OneD, const NekDouble> &ptsx,
                                    const Array<OneD, const NekDouble> &ptsy,
                                    Array<OneD, NekDouble> &Lcoords,
                                    NekDouble &dist);
};

/// Inverse mapping for a tetrahedron, pyramid, prism or hexahedron.
class GeometryLocator3D final : public GeometryLocator
{
public:
    using GeometryLocator::GeometryLocator;

protected:
    NekDouble v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                             Array<OneD, NekDouble> &Lcoords) override;
    void v_CalculateInverseIsoParam() override;

private:
    void NewtonIterationForLocCoord(const Array<OneD, const NekDouble> &coords,
                                    Array<OneD, NekDouble> &Lcoords);
    void NewtonIterationForLocCoord(const Array<OneD, const NekDouble> &coords,
                                    const Array<OneD, const NekDouble> &ptsx,
                                    const Array<OneD, const NekDouble> &ptsy,
                                    const Array<OneD, const NekDouble> &ptsz,
                                    Array<OneD, NekDouble> &Lcoords,
                                    NekDouble &dist);
};

} // namespace Nektar::SpatialDomains

#endif
