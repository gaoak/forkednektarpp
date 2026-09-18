////////////////////////////////////////////////////////////////////////////////
//
//  File: Mesh.cpp
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
//  Description: Mesh object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/Mesh.h>
#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <SpatialDomains/CADSystem/CADCurve.h>
#include <SpatialDomains/CADSystem/CADSurf.h>
#include <iomanip>
#include <limits>
#include <map>

using namespace std;

namespace Nektar::NekMesh
{

namespace
{
/**
 * @brief Does this entity store curvature, on itself or any edge or face?
 *
 * This is a question about the mesh representation, and deliberately not
 * @c Geometry::CalcGeomType(), which derives regular versus deformed from the
 * geometric factors. The two legitimately disagree: a curved element whose
 * curve happens to be affine is @c eRegular, and a straight-sided but
 * non-affine quadrilateral is @c eDeformed.
 */
bool CarriesCurvature(SpatialDomains::Geometry *geom)
{
    // A point has no curvature to carry, and no GetCurve() to ask.
    if (geom->GetShapeDim() == 0)
    {
        return false;
    }

    if (geom->GetCurve() != nullptr)
    {
        return true;
    }

    for (int i = 0; i < geom->GetNumEdges(); ++i)
    {
        if (geom->GetEdge(i)->GetCurve() != nullptr)
        {
            return true;
        }
    }

    for (int i = 0; i < geom->GetNumFaces(); ++i)
    {
        if (geom->GetFace(i)->GetCurve() != nullptr)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief Move generated curvature nodes on to the CAD they belong to.
 *
 * MakeOrder places a new node by evaluating the polynomial map of the geometry
 * it started from, which for a straight-sided or lower-order element does not
 * follow the true boundary: on a curved surface the node lands on the chord
 * rather than the arc. Where the entity carrying the curve is itself
 * associated with a CAD curve or surface, the node is projected back on to it.
 *
 * Recording the association matters as much as moving the node. ProcessVarOpti
 * constrains a node to the CAD entity it lies on, so a boundary node with no
 * association is treated as a free interior node and is at liberty to be
 * optimised off the geometry.
 */
std::pair<size_t, NekDouble> ProjectCurveNodesToCAD(
    SpatialDomains::MeshGraphSharedPtr &graph)
{
    auto &assoc     = graph->GetCADAssociation();
    size_t nMoved   = 0;
    NekDouble maxRy = 0.0;

    auto record = [&](SpatialDomains::PointGeom *p,
                      const std::array<NekDouble, 3> &from,
                      const std::array<NekDouble, 3> &to) {
        NekDouble d = sqrt((to[0] - from[0]) * (to[0] - from[0]) +
                           (to[1] - from[1]) * (to[1] - from[1]) +
                           (to[2] - from[2]) * (to[2] - from[2]));
        maxRy       = std::max(maxRy, d);
        p->UpdatePosition(to[0], to[1], to[2]);
        ++nMoved;
    };

    auto onCurve = [&](SpatialDomains::PointGeom *p,
                       const SpatialDomains::CADCurveSharedPtr &c) {
        std::array<NekDouble, 3> loc = {(*p)(0), (*p)(1), (*p)(2)};
        NekDouble t                  = 0.0;
        c->loct(loc, t);
        record(p, loc, c->P(t));
        assoc->Add(p, SpatialDomains::CADLink(c, {t, 0.0}));
    };

    auto onSurf = [&](SpatialDomains::PointGeom *p,
                      const SpatialDomains::CADSurfSharedPtr &s) {
        std::array<NekDouble, 3> loc = {(*p)(0), (*p)(1), (*p)(2)};
        auto uv                      = s->locuv(loc);
        record(p, loc, s->P(uv));
        assoc->Add(p, SpatialDomains::CADLink(s, {uv[0], uv[1]}));
    };

    // A node that already carries an association is either a mesh vertex the
    // surface mesher placed, or one an earlier entity here has dealt with --
    // an edge's nodes appear again in the curve of every face using that edge,
    // and a face's in the curve of every element using that face.
    auto project = [&](SpatialDomains::Geometry *geom) {
        SpatialDomains::Curve *curve = geom->GetCurve();
        if (curve == nullptr)
        {
            return;
        }

        // An edge lying along a CAD curve is pinned to it; one merely interior
        // to a surface is free along that surface.
        auto cadCurve = assoc->GetCurve(geom);
        auto cadSurf  = cadCurve ? nullptr : assoc->GetSurf(geom);

        if (!cadCurve && !cadSurf)
        {
            return;
        }

        for (auto *p : curve->m_points)
        {
            if (assoc->Has(p))
            {
                continue;
            }

            cadCurve ? onCurve(p, cadCurve) : onSurf(p, cadSurf);
        }
    };

    for (auto [id, geom] : graph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        project(geom);
    }
    for (auto [id, geom] : graph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        project(geom);
    }
    for (auto [id, geom] : graph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        project(geom);
    }

    return {nMoved, maxRy};
}
} // namespace

/**
 * @brief Return the number of elements of the expansion dimension.
 */
unsigned int Mesh::GetNumElements()
{
    unsigned int i = m_meshGraph->GetMeshDimension();
    return m_elementTags[i].size();
}

/**
 * @brief Return the number of boundary elements (i.e. one below the
 * expansion dimension).
 */
unsigned int Mesh::GetNumBndryElements()
{
    unsigned int i = m_meshGraph->GetMeshDimension();
    return m_elementTags[i - 1].size();
}

/**
 * @brief Return the total number of entities in the mesh (i.e. all
 * elements, regardless of dimension).
 */
unsigned int Mesh::GetNumTaggedEntities()
{
    unsigned int nEnt = 0;
    for (unsigned int d = 0; d <= m_meshGraph->GetMeshDimension(); ++d)
    {
        nEnt += m_elementTags[d].size();
    }
    return nEnt;
}

/**
 * @brief Convert this mesh into a mesh of uniform polynomial order @p order
 * with a curve point distribution @p distType.
 *
 * This routine adds curvature points into a mesh so that the resulting elements
 * are all of a uniform order @p order and all high-order vertices are
 * consistently ordered.
 */
void Mesh::MakeOrder(int order, LibUtilities::PointsType distType, Logger &log)
{
    // Decide on distribution of points to use for each shape type based on the
    // input we've been supplied.
    std::map<LibUtilities::ShapeType, LibUtilities::PointsType> pTypes;
    if (distType == LibUtilities::ePolyEvenlySpaced)
    {
        pTypes[LibUtilities::eSegment]  = LibUtilities::ePolyEvenlySpaced;
        pTypes[LibUtilities::eTriangle] = LibUtilities::eNodalTriEvenlySpaced;
        pTypes[LibUtilities::eQuadrilateral] = LibUtilities::ePolyEvenlySpaced;
        pTypes[LibUtilities::eTetrahedron] =
            LibUtilities::eNodalTetEvenlySpaced;
        pTypes[LibUtilities::ePrism]   = LibUtilities::eNodalPrismEvenlySpaced;
        pTypes[LibUtilities::ePyramid] = LibUtilities::eNodalPyrEvenlySpaced;
        pTypes[LibUtilities::eHexahedron] = LibUtilities::ePolyEvenlySpaced;
    }
    else if (distType == LibUtilities::eGaussLobattoLegendre)
    {
        pTypes[LibUtilities::eSegment]  = LibUtilities::eGaussLobattoLegendre;
        pTypes[LibUtilities::eTriangle] = LibUtilities::eNodalTriElec;
        pTypes[LibUtilities::eQuadrilateral] =
            LibUtilities::eGaussLobattoLegendre;
        pTypes[LibUtilities::ePrism]       = LibUtilities::eNodalPrismElec;
        pTypes[LibUtilities::eTetrahedron] = LibUtilities::eNodalTetElec;
        pTypes[LibUtilities::eHexahedron] = LibUtilities::eGaussLobattoLegendre;

        // The distributions above are trace-compatible: restricted to a face
        // or an edge they reduce to the ones used there, so the boundary
        // points an element takes from its shared face and edge curves agree
        // with its own distribution. There is no electrostatic pyramid
        // distribution and the evenly spaced one is not trace-compatible with
        // GLL edges, so a pyramid would get a curve labelled
        // eNodalPyrEvenlySpaced whose boundary points are not evenly spaced.
        if (m_meshGraph->HasGeoms<SpatialDomains::PyrGeom>())
        {
            log(FATAL) << "Cannot raise the order of a mesh containing "
                       << "pyramids using a Gauss-Lobatto-Legendre "
                       << "distribution: no compatible nodal pyramid "
                       << "distribution exists." << std::endl;
        }
    }
    else
    {
        ASSERTL1(false, "Mesh::MakeOrder does not support this points type.");
    }

    // Pass 0: drop each geometry's cached state. FillGeom() returns
    // immediately once a geometry is filled, so on a second call to MakeOrder
    // -- a second varopti module in the same pipeline, say -- the fill below
    // would be a no-op and the curves generated afterwards would be evaluated
    // against coefficients describing the mesh as it was before the previous
    // module moved its nodes. Reset also rebuilds the xmap and coefficient
    // storage, which the curves of the preceding order have resized.
    auto &curvedEdges = m_meshGraph->GetCurvedEdges();
    auto &curvedFaces = m_meshGraph->GetCurvedFaces();
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>())
    {
        geom->Reset(curvedEdges, curvedFaces);
    }

    // Pass 1: FillGeom everything so all m_coeffs are set from the original
    // geometry before any curve is modified by MakeOrder
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>())
    {
        geom->FillGeom();
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>())
    {
        geom->FillGeom();
    }

    // Pass 2: MakeOrder bottom-up, edges first so face curves can read from
    // them, then faces, then top-level elements
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        // Insert generated curves and nodes into MeshGraph
        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedEdge(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }

    const int nFaces = m_meshGraph->GetNumGeoms<SpatialDomains::TriGeom>() +
                       m_meshGraph->GetNumGeoms<SpatialDomains::QuadGeom>();
    int i = 0;
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        log(VERBOSE).Progress(i++, nFaces, "MakeOrder: Elements");

        // Insert generated curves and nodes into MeshGraph
        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedFace(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        log(VERBOSE).Progress(i++, nFaces, "MakeOrder: Elements");

        // Insert generated curves and nodes into MeshGraph
        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedFace(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }

    // Finally the volumes. A 3D element's interior nodes belong to it alone,
    // but its boundary nodes are read from the face curves generated above,
    // reoriented into this element's view of each face, so the faces have to
    // be done first.
    const int nVolumes = m_meshGraph->GetNumGeoms<SpatialDomains::TetGeom>() +
                         m_meshGraph->GetNumGeoms<SpatialDomains::PrismGeom>() +
                         m_meshGraph->GetNumGeoms<SpatialDomains::PyrGeom>() +
                         m_meshGraph->GetNumGeoms<SpatialDomains::HexGeom>();
    int j = 0;
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>())
    {
        log(VERBOSE).Progress(j++, nVolumes, "MakeOrder: Volumes");

        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedVolume(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>())
    {
        log(VERBOSE).Progress(j++, nVolumes, "MakeOrder: Volumes");

        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedVolume(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>())
    {
        log(VERBOSE).Progress(j++, nVolumes, "MakeOrder: Volumes");

        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedVolume(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }
    for (auto [id, geom] : m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>())
    {
        log(VERBOSE).Progress(j++, nVolumes, "MakeOrder: Volumes");

        auto curveData = geom->MakeOrder(order, pTypes[geom->GetShapeType()]);
        m_meshGraph->AddCurvedVolume(std::move(curveData.first));
        m_meshGraph->AddCurveNodes(curveData.second);
    }

    // The nodes generated above sit where the old geometry's polynomial map
    // put them, which on a curved boundary is not on the boundary. Put the
    // ones belonging to a CAD entity back on it.
    if (m_meshGraph->HasCAD())
    {
        auto [nMoved, maxMove] = ProjectCurveNodesToCAD(m_meshGraph);
        log(VERBOSE) << "  - Projected " << nMoved
                     << " generated nodes onto the CAD (max move " << maxMove
                     << ")" << std::endl;

        // Moving them invalidates the coefficients filled above.
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
        for (auto [id, geom] :
             m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>())
        {
            geom->Reset(curvedEdges, curvedFaces);
        }
    }
}

/**
 * @brief Print out basic statistics of this mesh.
 */
void Mesh::PrintStats(Logger &log)
{
    log << "Mesh statistics:" << std::endl;

    log << "  - Mesh dimension       : " << m_meshGraph->GetSpaceDimension()
        << std::endl
        << "  - Element dimension    : " << m_meshGraph->GetMeshDimension()
        << std::endl
        << "  - Has CAD attached     : "
        << (m_meshGraph->HasCAD() ? "yes" : "no") << std::endl
        << "  - Node count           : " << m_meshGraph->GetNvertices()
        << std::endl;

    size_t nEdges = m_meshGraph->GetNumGeoms<SpatialDomains::SegGeom>();
    if (m_meshGraph->GetMeshDimension() > 1)
    {
        log << "  - Edge count           : " << nEdges << std::endl;
    }

    size_t nFaces = m_meshGraph->GetNumGeoms<SpatialDomains::TriGeom>() +
                    m_meshGraph->GetNumGeoms<SpatialDomains::QuadGeom>();
    if (m_meshGraph->GetMeshDimension() > 2)
    {
        log << "  - Face count           : " << nFaces << std::endl;
    }

    log << "  - Elements             : "
        << m_elementTags[m_meshGraph->GetMeshDimension()].size() << std::endl;
    log << "  - Bnd elements         : "
        << m_elementTags[m_meshGraph->GetMeshDimension() - 1].size()
        << std::endl;

    // Print out number of composites
    log << "  - Number of composites : " << m_meshGraph->GetComposites().size()
        << std::endl;

    // Calculate domain extent
    double inf     = std::numeric_limits<double>::max();
    double lower_x = inf, lower_y = inf, lower_z = inf;
    double upper_x = -inf, upper_y = -inf, upper_z = -inf;
    for (auto vertex : m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        lower_x = std::min(lower_x, (*vertex.second)[0]);
        lower_y = std::min(lower_y, (*vertex.second)[1]);
        lower_z = std::min(lower_z, (*vertex.second)[2]);
        upper_x = std::max(upper_x, (*vertex.second)[0]);
        upper_y = std::max(upper_y, (*vertex.second)[1]);
        upper_z = std::max(upper_z, (*vertex.second)[2]);
    }
    for (auto &vertex : m_meshGraph->GetAllCurveNodes())
    {
        lower_x = std::min(lower_x, (*vertex)[0]);
        lower_y = std::min(lower_y, (*vertex)[1]);
        lower_z = std::min(lower_z, (*vertex)[2]);
        upper_x = std::max(upper_x, (*vertex)[0]);
        upper_y = std::max(upper_y, (*vertex)[1]);
        upper_z = std::max(upper_z, (*vertex)[2]);
    }

    log << "  - Lower mesh extent    : " << lower_x << " " << lower_y << " "
        << lower_z << std::endl
        << "  - Upper mesh extent    : " << upper_x << " " << upper_y << " "
        << upper_z << std::endl;

    std::map<LibUtilities::ShapeType, std::pair<int, int>> elmtCounts;

    for (int i = 1; i < LibUtilities::SIZE_ShapeType; ++i)
    {
        elmtCounts[(LibUtilities::ShapeType)i] = std::make_pair(0, 0);
    }

    for (int dim = 0; dim <= 3; ++dim)
    {
        for (auto &entry : m_elementTags[dim])
        {
            auto *elmt   = entry.first;
            auto &counts = elmtCounts[elmt->GetShapeType()];

            if (CarriesCurvature(elmt))
            {
                counts.second++;
            }
            else
            {
                counts.first++;
            }
        }
    }

    log << "Element counts (regular/deformed/total):" << std::endl;
    for (int i = 1; i < LibUtilities::SIZE_ShapeType; ++i)
    {
        auto shapeType = (LibUtilities::ShapeType)i;
        auto counts    = elmtCounts[shapeType];

        if (counts.first + counts.second == 0)
        {
            continue;
        }

        log << "  - " << std::setw(14) << std::left
            << LibUtilities::ShapeTypeMap[(LibUtilities::ShapeType)i] << ": "
            << setw(12) << counts.first << "  " << setw(12) << counts.second
            << "  " << setw(12) << counts.first + counts.second << std::endl;
    }
}

} // namespace Nektar::NekMesh
