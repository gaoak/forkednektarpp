////////////////////////////////////////////////////////////////////////////////
//
//  File: Element.h
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
//  Description: Mesh element.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MESHELEMENTS_ELEMENT
#define NEKMESH_MESHELEMENTS_ELEMENT

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/Foundations/PointsType.h>

#include <NekMesh/MeshElements/ElementConfig.h>
#include <NekMesh/NekMeshDeclspec.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/SegGeom.h>

namespace Nektar::NekMesh
{

/**
 * @brief Helper function to sort 4 numbers using sorting network.
 */
template <typename T>
inline static std::array<T, 4> sort4(std::array<T, 4> const &in)
{
    std::array<T, 4> out = in;

#define MIN(x, y) (x < y ? x : y)
#define MAX(x, y) (x < y ? y : x)
#define SWAP(x, y)                                                             \
    {                                                                          \
        const T a = MIN(out[x], out[y]);                                       \
        const T b = MAX(out[x], out[y]);                                       \
        out[x]    = a;                                                         \
        out[y]    = b;                                                         \
    }

    SWAP(0, 1);
    SWAP(2, 3);
    SWAP(0, 2);
    SWAP(1, 3);
    SWAP(1, 2);

    return out;

#undef SWAP
#undef MIN
#undef MAX
}

/**
 * @brief Defines a hash function for edges.
 *
 * The hash of an edge is defined using the IDs of the two nodes which
 * define it. First the minimum ID is hashed, then the maximum
 * ID, which takes the two possible orientations into account.
 */
struct EdgeHash
{
    std::size_t operator()(std::pair<int, int> const &p) const
    {
        // We can create a collision-free, branch-free hash for two integ
        const uint64_t a = static_cast<uint64_t>(p.first);
        const uint64_t b = static_cast<uint64_t>(p.second);

        const uint64_t h0 = (b << 32) | a, h1 = (a << 32) | b;

        return (p.first < p.second) ? h0 : h1;
    }
};

struct EdgeKey
{
    bool operator()(std::pair<int, int> const &p1,
                    std::pair<int, int> const &p2) const
    {
        const unsigned int id1 = p1.first, id2 = p1.second;
        const unsigned int id3 = p2.first, id4 = p2.second;
        return (id1 == id3 && id2 == id4) || (id1 == id4 && id2 == id3);
    }
};

typedef std::unordered_map<std::pair<int, int>, SpatialDomains::SegGeom *,
                           EdgeHash, EdgeKey>
    EdgeMap;

struct FaceHash
{
    inline static size_t xorrer(size_t const &l, size_t const &r)
    {
        static constexpr size_t mask[4] = {
            6275646296298018756UL, 15434735078582858566UL,
            17364106835760292997UL, 7426250665610396718UL};

        return l ^ mask[r];
    }

    std::size_t operator()(std::array<int, 4> const &p) const
    {
        std::hash<int> hasher;
        return hasher(p[0]) + hasher(p[1]) + hasher(p[2]) + hasher(p[3]);
    }
};

struct FaceKey
{
    bool operator()(std::array<int, 4> const &p1,
                    std::array<int, 4> const &p2) const
    {
        return sort4(p1) == sort4(p2);
    }
};

typedef std::unordered_map<std::array<int, 4>, SpatialDomains::Geometry2D *,
                           FaceHash, FaceKey>
    FaceMap;

/// Explicit ids for a new element and its sub-components. Pass a null
/// pointer instead to let the graph assign ids automatically.
struct ElmtIds
{
    std::vector<int> edges; ///< edge ids, indexed by local edge
    std::vector<int> faces; ///< face ids, indexed by local face
    int elmt;               ///< element id
};

/// Element factory definition.
typedef LibUtilities::NekFactoryBase<
    LibUtilities::ShapeType, SpatialDomains::Geometry,
    SpatialDomains::Geometry *, std::vector<SpatialDomains::PointGeom *> &,
    SpatialDomains::MeshGraphSharedPtr &, EdgeMap &, FaceMap &, ElmtConfig &,
    std::set<int> *, std::unordered_set<int> *, std::unordered_set<int> *,
    ElmtIds *>
    ElementFactory;

NEKMESH_EXPORT ElementFactory &GetElementFactory();

/**
 * @brief Find unused edge id
 * With the redesign operations erase them, so the size of the maps is changing.
 */
NEKMESH_EXPORT inline int NextEdgeId(
    SpatialDomains::MeshGraphSharedPtr &meshGraph)
{
    auto &segs = meshGraph->GetGeomMap<SpatialDomains::SegGeom>();
    if (segs.size() == 0)
    {
        return 0;
    }
    return segs.rbegin()->first + 1;
}

/**
 * @brief Find unused face id from quads and tri geoms.
 * Triags and Quads have same same face-ids,so unique over both maps.
 */
NEKMESH_EXPORT inline int NextFaceId(
    SpatialDomains::MeshGraphSharedPtr &meshGraph)
{
    int id = 0;

    auto &tris = meshGraph->GetGeomMap<SpatialDomains::TriGeom>();
    if (tris.size() > 0)
    {
        id = tris.rbegin()->first + 1;
    }

    auto &quads = meshGraph->GetGeomMap<SpatialDomains::QuadGeom>();
    if (quads.size() > 0 && quads.rbegin()->first + 1 > id)
    {
        id = quads.rbegin()->first + 1;
    }

    return id;
}

/**
 * @brief Find unused vertex id
 * In the future h-adapt this will be necessary as the point list will be
 * dynamic size. TBD new logic for parallel insertion.
 */
NEKMESH_EXPORT inline int NextPointId(
    SpatialDomains::MeshGraphSharedPtr &meshGraph)
{
    auto &pts = meshGraph->GetGeomMap<SpatialDomains::PointGeom>();
    if (pts.size() == 0)
    {
        return 0;
    }
    return pts.rbegin()->first + 1;
}

/**
 * @brief Set the coordinate dimension of the first @p id points.
 *
 * A reader that infers the space dimension from the coordinates it has seen so
 * far only discovers the true dimension part way through the point list, so
 * the points created before that point need their coordim corrected.
 */
NEKMESH_EXPORT inline void UpdateCoordim(
    SpatialDomains::MeshGraphSharedPtr &meshGraph, int id, int coordim)
{
    for (int i = 0; i < id; ++i)
    {
        meshGraph->GetPointGeom(i)->SetCoordim(coordim);
    }
}

/**
 * @brief Create a straight-sided element of @p type from its vertices.
 *
 * The common case: a linear element, no face or volume nodes, no reorientation
 * and none of the ID bookkeeping the full factory signature offers. The
 * element, and any edge or face it needs that does not already exist, is
 * written into @p meshGraph; @p edgeMap and @p faceMap are what let a
 * neighbour share those rather than duplicate them.
 */
NEKMESH_EXPORT inline SpatialDomains::Geometry *CreateElementLite(
    LibUtilities::ShapeType type,
    std::vector<SpatialDomains::PointGeom *> &nodeList,
    SpatialDomains::MeshGraphSharedPtr &meshGraph, EdgeMap &edgeMap,
    FaceMap &faceMap)
{
    ElmtConfig conf(type, 1, false, false, false);
    return GetElementFactory().CreateInstance(type, nodeList, meshGraph,
                                              edgeMap, faceMap, conf, nullptr,
                                              nullptr, nullptr, nullptr);
}

/**
 * @brief Assemble the full node list of a curved triangle or quadrilateral.
 *
 * Returns the vertices, then the interior nodes of each edge taken from the
 * edge curves, then @p faceNodes, in the nodal ordering for
 * @p conf_faceCurveType. Edge nodes are reversed where the edge is stored
 * against the face's sense, so the result is in the face's own orientation.
 */
NEKMESH_EXPORT std::vector<SpatialDomains::PointGeom *> GetCurvedNodesTri(
    LibUtilities::PointsType conf_faceCurveType,
    std::array<SpatialDomains::PointGeom *, 3> &vertexList,
    std::array<SpatialDomains::SegGeom *, 3> &edgeList,
    std::vector<SpatialDomains::PointGeom *> &faceNodes);
/// @copydoc GetCurvedNodesTri
NEKMESH_EXPORT std::vector<SpatialDomains::PointGeom *> GetCurvedNodesQuad(
    LibUtilities::PointsType conf_faceCurveType,
    std::array<SpatialDomains::PointGeom *, 4> &vertexList,
    std::array<SpatialDomains::SegGeom *, 4> &edgeList,
    std::vector<SpatialDomains::PointGeom *> &faceNodes);

/**
 * @brief Return every node of @p geom, in the nodal ordering.
 *
 * This is the inverse of the node list an element is built from: the curve a
 * geometry carries holds its vertices, edge nodes, face nodes and, where the
 * shape supports it, its volume nodes, in the nodal ordering described in the
 * developer guide. Call Mesh::MakeOrder first, which is what puts that curve
 * in place.
 */
NEKMESH_EXPORT std::vector<SpatialDomains::PointGeom *> GetCurvedNodes(
    SpatialDomains::Geometry *geom);

inline int GetNumNodes(LibUtilities::ShapeType st, const ElmtConfig &pConf)
{
    switch (st)
    {
        case LibUtilities::eSegment:
        {
            return pConf.m_order + 1;
        }

        case LibUtilities::eQuadrilateral:
        {
            int n = pConf.m_order;
            if (!pConf.m_faceNodes)
            {
                return 4 * n;
            }
            else
            {
                return (n + 1) * (n + 1);
            }

            break;
        }

        case LibUtilities::eTriangle:
        {
            int n = pConf.m_order;
            if (!pConf.m_faceNodes)
            {
                return (n + 1) + 2 * (n - 1) + 1;
            }
            else
            {
                return (n + 1) * (n + 2) / 2;
            }

            break;
        }

        case LibUtilities::eHexahedron:
        {
            int n = pConf.m_order;
            if (pConf.m_faceNodes && pConf.m_volumeNodes)
            {
                return (n + 1) * (n + 1) * (n + 1);
            }
            else if (pConf.m_faceNodes && !pConf.m_volumeNodes)
            {
                return 6 * (n + 1) * (n + 1) - 12 * (n + 1) + 8;
            }
            else
            {
                return 12 * (n + 1) - 16;
            }

            break;
        }

        case LibUtilities::ePrism:
        {
            int n = pConf.m_order;
            if (pConf.m_faceNodes && pConf.m_volumeNodes)
            {
                return (n + 1) * (n + 1) * (n + 2) / 2;
            }
            else if (pConf.m_faceNodes && !pConf.m_volumeNodes)
            {
                return 3 * (n + 1) * (n + 1) + 2 * (n + 1) * (n + 2) / 2 -
                       9 * (n + 1) + 6;
            }
            else
            {
                return 9 * (n + 1) - 12;
            }
        }

        case LibUtilities::ePyramid:
        {
            int n = pConf.m_order;

            // valid for any order pyramid
            return (5             // corners
                    + 8 * (n - 1) // mid-edge
                    + pConf.m_faceNodes *
                          ((n - 1) * (n - 1) +
                           4 * (n - 1) * (n - 2) /
                               2) // square base + 4xtriangle-number faces
                    + pConf.m_volumeNodes * (n - 2) * (n - 1) * (2 * n - 3) /
                          6 // square pyramidal numbers
            );
        }

        case LibUtilities::eTetrahedron:
        {
            int n = pConf.m_order;
            if (pConf.m_volumeNodes && pConf.m_faceNodes)
            {
                return (n + 1) * (n + 2) * (n + 3) / 6;
            }
            else if (!pConf.m_volumeNodes && pConf.m_faceNodes)
            {
                return 4 * (n + 1) * (n + 2) / 2 - 6 * (n + 1) + 4;
            }
            else
            {
                return 6 * (n + 1) - 8;
            }
        }

        default:
            // Returning a node count of -1 here would be read as a real
            // count by every caller, so refuse instead.
            NEKERROR(ErrorUtil::efatal,
                     "GetNumNodes: no node count for shape type " +
                         std::string(LibUtilities::ShapeTypeMap[st]));
            return -1;
    }
}

} // namespace Nektar::NekMesh

#endif
