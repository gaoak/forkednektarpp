////////////////////////////////////////////////////////////////////////////////
//
//  File: Tetrahedron.cpp
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
//  Description: Tetrahedral elements
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/TetGeom.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/HOAlignment.h>
#include <SpatialDomains/TriGeom.h>

namespace Nektar::NekMesh
{

struct tetrahedronHelper
{
    /**
     * @brief Helper function to sort 3 numbers using sorting network.
     */
    template <typename K> static void sort3(K &x, K &y, K &z)
    {
#define SWAP(a, b)                                                             \
    if (a > b)                                                                 \
        std::swap(a, b);
        SWAP(y, z);
        SWAP(x, z);
        SWAP(x, y);
#undef SWAP
    }

    /// Local vertices that make up each tetrahedral edge.
    constexpr static int edgeVertMap[6][2] = {{0, 1}, {1, 2}, {0, 2},
                                              {0, 3}, {1, 3}, {2, 3}};

    /// Local edges that make up each tetrahedral face.
    constexpr static int faceEdgeMap[4][3] = {
        {0, 1, 2}, {0, 4, 3}, {1, 5, 4}, {2, 5, 3}};

    /// Local vertices that make up each tetrahedral face.
    constexpr static int faceVertMap[4][3] = {
        {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};

    /**
     * @brief Create a tetrahedron element.
     */
    static SpatialDomains::Geometry *create(
        std::vector<SpatialDomains::PointGeom *> &nodeList,
        SpatialDomains::MeshGraphSharedPtr &meshGraph, EdgeMap &edgeMap,
        FaceMap &faceMap, ElmtConfig &conf, std::set<int> *vertIDs,
        std::unordered_set<int> *curveNodeIDs,
        std::unordered_set<int> *naiveTriIDs, ElmtIds *forceIDs)
    {
        int n = conf.m_order - 1;

        auto &curvedEdges = meshGraph->GetCurvedEdges();
        auto &curvedFaces = meshGraph->GetCurvedFaces();

        // Add vertices
        std::array<SpatialDomains::PointGeom *, 4> vertex;
        for (int i = 0; i < 4; ++i)
        {
            vertex[i] = nodeList[i];
        }

        // Reorient the tet to ensure collapsed coordinates align between
        // adjacent elements.
        std::array<int, 4> orientationMap{};
        std::array<int, 4> origVertMap{};
        if (conf.m_reorient)
        {
            std::tie(orientationMap, origVertMap) = OrientTet(vertex);
        }
        else
        {
            // If we didn't need to orient the tet then set up the
            // orientation map as the identity mapping.
            for (int i = 0; i < 4; ++i)
            {
                orientationMap[i] = origVertMap[i] = i;
            }
        }

        // Create edges (with corresponding set of edge points). Apply
        // orientation logic to get the right interior points for each edge.
        std::array<SpatialDomains::SegGeom *, 6> edges{};
        for (int e = 0; e < 6; ++e)
        {
            int origEdge = -1;
            bool rev     = false;
            for (int j = 0; j < 6; ++j)
            {
                if (edgeVertMap[e][0] == origVertMap[edgeVertMap[j][0]] &&
                    edgeVertMap[e][1] == origVertMap[edgeVertMap[j][1]])
                {
                    origEdge = j;
                    break;
                }
                else if (edgeVertMap[e][0] == origVertMap[edgeVertMap[j][1]] &&
                         edgeVertMap[e][1] == origVertMap[edgeVertMap[j][0]])
                {
                    origEdge = j;
                    rev      = true;
                    break;
                }
            }
            std::array<SpatialDomains::PointGeom *, 2> edgeVerts;
            if (rev)
            {
                edgeVerts[0] = vertex[edgeVertMap[e][1]];
                edgeVerts[1] = vertex[edgeVertMap[e][0]];
            }
            else
            {
                edgeVerts[0] = vertex[edgeVertMap[e][0]];
                edgeVerts[1] = vertex[edgeVertMap[e][1]];
            }

            // Check if edge already exists in edgeSet using edgeVert ids, if it
            // does we can just reuse that...
            auto idPair = std::make_pair(edgeVerts[0]->GetGlobalID(),
                                         edgeVerts[1]->GetGlobalID());
            auto it     = edgeMap.find(idPair);

            if (it == edgeMap.end())
            {
                // Create edge.
                int id = NextEdgeId(meshGraph);
                if (forceIDs != nullptr)
                {
                    id = forceIDs->edges[e];
                }
                SpatialDomains::Curve *curvePtr = nullptr;

                if (conf.m_order > 1)
                {
                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    tmpNodeList.emplace_back(edgeVerts[0]);
                    for (int j = 0; j < n; ++j)
                    {
                        tmpNodeList.emplace_back(
                            nodeList[4 + origEdge * n + j]);
                    }
                    tmpNodeList.emplace_back(edgeVerts[1]);

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(id, conf.m_edgeCurveType);
                    for (auto &node : tmpNodeList)
                    {
                        curveNodeIDs->insert(node->GetGlobalID());
                        curve->m_points.emplace_back(node);
                    }
                    curvePtr        = curve.get();
                    curvedEdges[id] = std::move(curve);
                }

                auto seg =
                    ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                        id, nodeList[0]->GetCoordim(), edgeVerts, curvePtr);

                edges[e]        = seg.get();
                edgeMap[idPair] = seg.get();
                meshGraph->AddGeom<SpatialDomains::SegGeom>(id, std::move(seg));
            }
            else
            {
                edges[e] = it->second;

                // Create edge curvature if it's needed and we haven't already
                if (conf.m_order > 1 && edges[e]->GetCurve() == nullptr)
                {
                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    tmpNodeList.emplace_back(edgeVerts[0]);
                    for (int j = 0; j < n; ++j)
                    {
                        tmpNodeList.emplace_back(
                            nodeList[4 + origEdge * n + j]);
                    }
                    tmpNodeList.emplace_back(edgeVerts[1]);

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(edges[e]->GetGlobalID(),
                                          conf.m_edgeCurveType);
                    for (auto &node : tmpNodeList)
                    {
                        curveNodeIDs->insert(node->GetGlobalID());
                        curve->m_points.emplace_back(node);
                    }
                    edges[e]->SetCurve(curve.get());
                    curvedEdges[edges[e]->GetGlobalID()] = std::move(curve);
                }
            }
        }

        // Create faces
        std::array<SpatialDomains::TriGeom *, 4> faces{};
        std::array<SpatialDomains::PointGeom *, 3> faceVertices;
        std::array<SpatialDomains::SegGeom *, 3> faceEdges;

        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                faceVertices[k] = vertex[faceVertMap[j][k]];
            }

            // Check if tri already exists in faceSet using vertex IDs
            std::array<int, 4> vids = {faceVertices[0]->GetGlobalID(),
                                       faceVertices[1]->GetGlobalID(),
                                       faceVertices[2]->GetGlobalID(), -1};

            auto it = faceMap.find(vids);

            if (it == faceMap.end())
            {
                int id = NextFaceId(meshGraph);
                if (forceIDs != nullptr)
                {
                    id = forceIDs->faces[j];
                }

                for (int k = 0; k < 3; ++k)
                {
                    faceEdges[k] = edges[faceEdgeMap[j][k]];
                }

                // When face curvature is supplied, it may have been the case
                // that our tetrahedron was reoriented. In this case, faces have
                // different vertex IDs and so we have to rotate the face
                // curvature so that the two align appropriately.
                SpatialDomains::Curve *curvePtr = nullptr;
                if (conf.m_faceNodes)
                {
                    const int nFaceNodes = n * (n - 1) / 2;
                    // Get the vertex IDs of whatever face we are processing.
                    std::vector<int> faceVertIds(3);
                    faceVertIds[0] = faceVertices[0]->GetGlobalID();
                    faceVertIds[1] = faceVertices[1]->GetGlobalID();
                    faceVertIds[2] = faceVertices[2]->GetGlobalID();

                    // Find out the original face number as we were given it in
                    // the constructor using the orientation map.
                    int origFace = -1;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (orientationMap[i] == j)
                        {
                            origFace = i;
                            break;
                        }
                    }
                    ASSERTL0(origFace >= 0, "Couldn't find face");

                    // Now get the face nodes for the original face.
                    std::vector<SpatialDomains::PointGeom *> faceNodes;
                    int N = 4 + 6 * n + origFace * nFaceNodes;
                    for (int i = 0; i < nFaceNodes; ++i)
                    {
                        faceNodes.push_back(nodeList[N + i]);
                        curveNodeIDs->insert(nodeList[N + i]->GetGlobalID());
                    }

                    // Find the original face vertex IDs.
                    std::vector<int> origFaceIds(3);
                    origFaceIds[0] =
                        nodeList[faceVertMap[origFace][0]]->GetGlobalID();
                    origFaceIds[1] =
                        nodeList[faceVertMap[origFace][1]]->GetGlobalID();
                    origFaceIds[2] =
                        nodeList[faceVertMap[origFace][2]]->GetGlobalID();

                    // Construct a HOTriangle object which performs the
                    // orientation magically for us.
                    HOTriangle<SpatialDomains::PointGeom *> hoTri(origFaceIds,
                                                                  faceNodes);
                    hoTri.Align(faceVertIds);

                    // Copy the face nodes back again.
                    faceNodes = hoTri.surfVerts;

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(id, conf.m_faceCurveType);
                    curve->m_points.emplace_back(faceVertices[0]);
                    curve->m_points.emplace_back(faceVertices[1]);
                    curve->m_points.emplace_back(faceVertices[2]);

                    // Check faceEdges[l] m_points[0] is equal to
                    // faceVertices[l] and if not then reverse...
                    for (int l = 0; l < 3; ++l)
                    {
                        std::vector<SpatialDomains::PointGeom *> tmp(
                            faceEdges[l]->GetCurve()->m_points.begin() + 1,
                            faceEdges[l]->GetCurve()->m_points.end() - 1);
                        if (faceEdges[l]->GetCurve()->m_points[0] !=
                            faceVertices[l])
                        {
                            std::reverse(tmp.begin(), tmp.end());
                        }

                        curve->m_points.insert(curve->m_points.end(),
                                               tmp.begin(), tmp.end());
                    }

                    curve->m_points.insert(curve->m_points.end(),
                                           faceNodes.begin(), faceNodes.end());

                    curvePtr        = curve.get();
                    curvedFaces[id] = std::move(curve);
                }

                auto triGeom =
                    ObjPoolManager<SpatialDomains::TriGeom>::AllocateUniquePtr(
                        id, faceEdges, faceVertices, false, curvePtr);
                auto triPtr = triGeom.get();
                meshGraph->AddGeom<SpatialDomains::TriGeom>(id,
                                                            std::move(triGeom));

                faces[j]      = triPtr;
                faceMap[vids] = triPtr;
            }
            else
            {
                faces[j] = static_cast<SpatialDomains::TriGeom *>(it->second);

                bool naiveTri = false;
                if (naiveTriIDs != nullptr)
                {
                    naiveTri = naiveTriIDs->find(faces[j]->GetGlobalID()) !=
                               naiveTriIDs->end();
                }
                // Reset adopted face alignment if naively created by
                // Triangle.cpp
                //@TODO is this okay
                if (naiveTri)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        faces[j]->SetEdge(k, edges[faceEdgeMap[j][k]]);
                        faces[j]->SetVertex(k, vertex[faceVertMap[j][k]]);
                    }
                }

                // Create face curvature if it's needed, and we haven't already
                // or it was naively created by Triangle.cpp with incorrect
                // alignment
                if (conf.m_faceNodes &&
                    (faces[j]->GetCurve() == nullptr || naiveTri))
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        faceEdges[k] = edges[faceEdgeMap[j][k]];
                    }
                    const int nFaceNodes = n * (n - 1) / 2;
                    // Get the vertex IDs of whatever face we are processing.
                    std::vector<int> faceVertIds(3);
                    faceVertIds[0] = faceVertices[0]->GetGlobalID();
                    faceVertIds[1] = faceVertices[1]->GetGlobalID();
                    faceVertIds[2] = faceVertices[2]->GetGlobalID();

                    // Find out the original face number as we were given it in
                    // the constructor using the orientation map.
                    int origFace = -1;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (orientationMap[i] == j)
                        {
                            origFace = i;
                            break;
                        }
                    }
                    ASSERTL0(origFace >= 0, "Couldn't find face");

                    // Now get the face nodes for the original face.
                    std::vector<SpatialDomains::PointGeom *> faceNodes;
                    int N = 4 + 6 * n + origFace * nFaceNodes;
                    for (int i = 0; i < nFaceNodes; ++i)
                    {
                        faceNodes.push_back(nodeList[N + i]);
                        curveNodeIDs->insert(nodeList[N + i]->GetGlobalID());
                    }

                    // Find the original face vertex IDs.
                    std::vector<int> origFaceIds(3);
                    origFaceIds[0] =
                        nodeList[faceVertMap[origFace][0]]->GetGlobalID();
                    origFaceIds[1] =
                        nodeList[faceVertMap[origFace][1]]->GetGlobalID();
                    origFaceIds[2] =
                        nodeList[faceVertMap[origFace][2]]->GetGlobalID();

                    // Construct a HOTriangle object which performs the
                    // orientation magically for us.
                    HOTriangle<SpatialDomains::PointGeom *> hoTri(origFaceIds,
                                                                  faceNodes);
                    hoTri.Align(faceVertIds);

                    // Copy the face nodes back again.
                    faceNodes = hoTri.surfVerts;

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(faces[j]->GetGlobalID(),
                                          conf.m_faceCurveType);
                    curve->m_points.emplace_back(faceVertices[0]);
                    curve->m_points.emplace_back(faceVertices[1]);
                    curve->m_points.emplace_back(faceVertices[2]);

                    // Check faceEdges[l] m_points[0] is equal to
                    // faceVertices[l] and if not then reverse...
                    for (int l = 0; l < 3; ++l)
                    {
                        std::vector<SpatialDomains::PointGeom *> tmp(
                            faceEdges[l]->GetCurve()->m_points.begin() + 1,
                            faceEdges[l]->GetCurve()->m_points.end() - 1);
                        if (faceEdges[l]->GetCurve()->m_points[0] !=
                            faceVertices[l])
                        {
                            std::reverse(tmp.begin(), tmp.end());
                        }

                        curve->m_points.insert(curve->m_points.end(),
                                               tmp.begin(), tmp.end());
                    }

                    curve->m_points.insert(curve->m_points.end(),
                                           faceNodes.begin(), faceNodes.end());

                    faces[j]->SetCurve(curve.get());
                    curvedFaces[faces[j]->GetGlobalID()] = std::move(curve);
                }
            }
        }

        if (vertIDs != nullptr)
        {
            for (auto vert : vertex)
            {
                vertIDs->insert(vert->GetGlobalID());
            }
        }

        int id =
            forceIDs != nullptr ? forceIDs->elmt : meshGraph->GetNumElements(3);
        auto tetGeom =
            ObjPoolManager<SpatialDomains::TetGeom>::AllocateUniquePtr(
                id, faces, edges, vertex, true);
        auto tetPtr = tetGeom.get();
        meshGraph->AddGeom<SpatialDomains::TetGeom>(id, std::move(tetGeom));

        // Interior nodes, kept as a curve on the element. Edges and faces are
        // shared and so hold their own curvature; the inside of an element
        // belongs to nothing else, and is lost unless it is held here.
        //
        // The nodes arrive in the ordering a nodal tetrahedron puts them in,
        // but numbered against the vertices as the caller listed them, and
        // this element may have reordered its vertices above. Aligning is a
        // relabelling of the four corners, which moves every node to where its
        // barycentric position says it now belongs; nothing is evaluated, so
        // each node keeps the position it was given.
        const int nPoints = conf.m_order + 1;
        const int nTetPts = nPoints * (nPoints + 1) * (nPoints + 2) / 6;
        const int nInt    = (nPoints - 4) * (nPoints - 3) * (nPoints - 2) / 6;

        if (conf.m_volumeNodes && nInt > 0 && (int)nodeList.size() == nTetPts)
        {
            std::vector<int> listIds(4), elmtIds(4);
            for (int i = 0; i < 4; ++i)
            {
                listIds[i] = nodeList[i]->GetGlobalID();
                elmtIds[i] = tetPtr->GetVertex(i)->GetGlobalID();
            }

            // The nodal distribution runs local edge 2 from vertex 2 to
            // vertex 0, against the direction the caller lists it in.
            const int nEdgeNodes                           = nPoints - 2;
            std::vector<SpatialDomains::PointGeom *> nodal = nodeList;
            std::reverse(nodal.begin() + 4 + 2 * nEdgeNodes,
                         nodal.begin() + 4 + 3 * nEdgeNodes);

            SpatialDomains::HOTetrahedron<SpatialDomains::PointGeom *> hoTet(
                listIds, nodal, nPoints);
            hoTet.Align(elmtIds);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    id, LibUtilities::eNodalTetEvenlySpaced);
            curve->m_points = hoTet.volVerts;

            tetPtr->SetCurve(curve.get());
            meshGraph->AddCurvedVolume(std::move(curve));
        }

        return tetPtr;
    }

    /**
     * @brief Orient tetrahedron to align degenerate vertices.
     *
     * Orientation of tetrahedral elements is required so that the
     * singular vertices of triangular faces (which occur as a part of the
     * collapsed co-ordinate system) align. The algorithm is based on that
     * used in T. Warburton's thesis and in the original Nektar source.
     *
     * First the vertices are ordered with the highest global vertex at
     * the top degenerate point, and the base degenerate point has second
     * lowest ID. These vertices are swapped if the element is incorrectly
     * oriented.
     */
    /**
     * @brief Put the vertices in a canonical order and report how they moved.
     *
     * The degeneracy checks compare squared quantities. |dist| / area is
     * 2|c.n| / |n|^2 with n the unnormalised cross product, so neither the
     * square root nor the three divisions the normalisation used to do are
     * needed on the common path -- only inside the warning, which does not
     * fire. Two of the three checks are diagnostic only; just the first
     * decides anything, and only through the sign of the triple product.
     */
    static std::pair<std::array<int, 4>, std::array<int, 4>> OrientTet(
        std::array<SpatialDomains::PointGeom *, 4> &vertex)
    {
        // Create a copy of the original vertex ordering. This is used to
        // construct a mapping, #orientationMap, which maps the original
        // face ordering to the new face ordering.
        int orig_faces[4][3];
        for (int i = 0; i < 4; ++i)
        {
            int v0id = vertex[faceVertMap[i][0]]->GetGlobalID();
            int v1id = vertex[faceVertMap[i][1]]->GetGlobalID();
            int v2id = vertex[faceVertMap[i][2]]->GetGlobalID();
            sort3(v0id, v1id, v2id);
            orig_faces[i][0] = v0id;
            orig_faces[i][1] = v1id;
            orig_faces[i][2] = v2id;
        }

        // Store a copy of the original vertex ordering so we can create a
        // permutation map later.
        std::array<SpatialDomains::PointGeom *, 4> origVert;
        std::copy(std::begin(vertex), std::end(vertex), std::begin(origVert));

        // Order vertices with highest global vertex at top degenerate
        // point. Place second highest global vertex at base degenerate
        // point.
        std::sort(
            vertex.begin(), vertex.end(),
            [](SpatialDomains::PointGeom *t1, SpatialDomains::PointGeom *t2) {
                return t1->GetGlobalID() < t2->GetGlobalID();
            });

        const NekDouble ax = (*vertex[1])(0) - (*vertex[0])(0);
        const NekDouble ay = (*vertex[1])(1) - (*vertex[0])(1);
        const NekDouble az = (*vertex[1])(2) - (*vertex[0])(2);
        const NekDouble bx = (*vertex[2])(0) - (*vertex[0])(0);
        const NekDouble by = (*vertex[2])(1) - (*vertex[0])(1);
        const NekDouble bz = (*vertex[2])(2) - (*vertex[0])(2);
        const NekDouble cx = (*vertex[3])(0) - (*vertex[0])(0);
        const NekDouble cy = (*vertex[3])(1) - (*vertex[0])(1);
        const NekDouble cz = (*vertex[3])(2) - (*vertex[0])(2);

        auto check = [](NekDouble nx, NekDouble ny, NekDouble nz, NekDouble dx,
                        NekDouble dy, NekDouble dz, const char *which) {
            const NekDouble nsq  = nx * nx + ny * ny + nz * nz;
            const NekDouble trip = dx * nx + dy * ny + dz * nz;
            if (2.0 * fabs(trip) <= 1e-4 * nsq)
            {
                std::cerr << "Warning: degenerate tetrahedron, " << which
                          << " vertex is = " << trip / sqrt(nsq) << " from face"
                          << std::endl;
            }
            return trip;
        };

        // Calculate a.(b x c); if negative, reverse the order of the
        // non-degenerate points to orientate the tet correctly.
        const NekDouble trip = check(ay * bz - az * by, az * bx - ax * bz,
                                     ax * by - ay * bx, cx, cy, cz, "3rd");
        if (trip < 0)
        {
            std::swap(vertex[0], vertex[1]);
        }

        check(ay * cz - az * cy, az * cx - ax * cz, ax * cy - ay * cx, bx, by,
              bz, "2nd");
        check(by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx, ax, ay,
              az, "1st");

        // Search for the face in the original set of face nodes. Then use
        // this to construct the #orientationMap.
        std::array<int, 4> orientationMap{};
        std::array<int, 4> origVertMap{};
        for (int i = 0; i < 4; ++i)
        {
            int v0id = vertex[faceVertMap[i][0]]->GetGlobalID();
            int v1id = vertex[faceVertMap[i][1]]->GetGlobalID();
            int v2id = vertex[faceVertMap[i][2]]->GetGlobalID();
            sort3(v0id, v1id, v2id);

            for (int j = 0; j < 4; ++j)
            {
                if (v0id == orig_faces[j][0] && v1id == orig_faces[j][1] &&
                    v2id == orig_faces[j][2])
                {
                    orientationMap[j] = i;
                    break;
                }
            }

            for (int j = 0; j < 4; ++j)
            {
                if (vertex[i]->GetGlobalID() == origVert[j]->GetGlobalID())
                {
                    origVertMap[j] = i;
                    break;
                }
            }
        }

        return std::make_pair(orientationMap, origVertMap);
    }
};

LibUtilities::ShapeType tetType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::eTetrahedron, tetrahedronHelper::create, "Tetrahedron");

} // namespace Nektar::NekMesh
