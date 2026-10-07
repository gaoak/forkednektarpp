////////////////////////////////////////////////////////////////////////////////
//
//  File: Prism.cpp
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
//  Description: Mesh prism object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/PrismExp.h>
#include <StdRegions/StdNodalPrismExp.h>

#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/PrismGeom.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/HOAlignment.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/TriGeom.h>

namespace Nektar::NekMesh
{

struct prismHelper
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

    /**
     * @brief Create a prism element.
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

        int edgeVertsInds[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4},
                                   {1, 4}, {2, 5}, {3, 5}, {4, 5}};

        int faceIds[5][4] = {{0, 1, 2, 3},
                             {0, 1, 4, -1},
                             {1, 2, 5, 4},
                             {3, 2, 5, -1},
                             {0, 3, 5, 4}};

        // Add vertices
        std::array<SpatialDomains::PointGeom *, 6> vertex;
        for (int i = 0; i < 6; ++i)
        {
            vertex[i] = nodeList[i];
        }

        // Create edges (with corresponding set of edge points)
        std::array<SpatialDomains::SegGeom *, 9> edges{};
        for (int e = 0; e < 9; ++e)
        {
            std::array<SpatialDomains::PointGeom *, 2> edgeVerts = {
                vertex[edgeVertsInds[e][0]], vertex[edgeVertsInds[e][1]]};

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
                    for (int j = 7 + e * n; j < 7 + e * n + n; ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j - 1]);
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
                    for (int j = 7 + e * n; j < 7 + e * n + n; ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j - 1]);
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

        /**
         * Orientation of prism; unchanged = 0; clockwise = 1;
         * counter-clockwise = 2. This is set by OrientPrism.
         */
        unsigned int orientation;
        if (conf.m_reorient)
        {
            orientation = OrientPrism(vertex);
        }
        else
        {
            orientation = 0;
        }

        // Create faces
        int face_edge_ids[5][4];

        int face_offset[5];
        face_offset[0] = 6 + 9 * n;
        for (int j = 0; j < 4; ++j)
        {
            int facenodeID     = j % 2 == 0 ? n * n : n * (n - 1) / 2;
            face_offset[j + 1] = face_offset[j] + facenodeID;
        }

        std::array<SpatialDomains::Geometry2D *, 5> faces{};
        for (int j = 0; j < 5; ++j)
        {
            int nEdge = 3 - (j % 2 - 1);
            Array<OneD, SpatialDomains::PointGeom *> faceVertices(nEdge);
            Array<OneD, SpatialDomains::SegGeom *> faceEdges(nEdge);

            for (int k = 0; k < nEdge; ++k)
            {
                faceVertices[k]              = vertex[faceIds[j][k]];
                SpatialDomains::PointGeom *a = vertex[faceIds[j][k]];
                SpatialDomains::PointGeom *b =
                    vertex[faceIds[j][(k + 1) % nEdge]];
                unsigned int i;
                bool broke = false;
                for (i = 0; i < edges.size(); ++i)
                {
                    if ((edges[i]->GetVid(0) == a->GetGlobalID() &&
                         edges[i]->GetVid(1) == b->GetGlobalID()) ||
                        (edges[i]->GetVid(0) == b->GetGlobalID() &&
                         edges[i]->GetVid(1) == a->GetGlobalID()))
                    {
                        faceEdges[k]        = edges[i];
                        face_edge_ids[j][k] = i;
                        broke               = true;
                        break;
                    }
                }

                if (!broke)
                {
                    face_edge_ids[j][k] = -1;
                }
            }

            // Check if face already exists in faceSet using vertex IDs
            int fourthID = (nEdge == 3) ? -1 : faceVertices[3]->GetGlobalID();
            std::array<int, 4> vids = {
                faceVertices[0]->GetGlobalID(), faceVertices[1]->GetGlobalID(),
                faceVertices[2]->GetGlobalID(), fourthID};

            auto it = faceMap.find(vids);

            if (it == faceMap.end())
            {
                int id = NextFaceId(meshGraph);
                if (forceIDs != nullptr)
                {
                    id = forceIDs->faces[j];
                }

                std::array<SpatialDomains::PointGeom *, 3> faceVertArrayTri;
                std::array<SpatialDomains::SegGeom *, 3> faceEdgeArrayTri;
                std::array<SpatialDomains::PointGeom *, 4> faceVertArrayQuad;
                std::array<SpatialDomains::SegGeom *, 4> faceEdgeArrayQuad;
                if (nEdge == 3)
                {
                    faceVertArrayTri = {faceVertices[0], faceVertices[1],
                                        faceVertices[2]};
                    faceEdgeArrayTri = {faceEdges[0], faceEdges[1],
                                        faceEdges[2]};
                }
                else
                {
                    faceVertArrayQuad = {faceVertices[0], faceVertices[1],
                                         faceVertices[2], faceVertices[3]};
                    faceEdgeArrayQuad = {faceEdges[0], faceEdges[1],
                                         faceEdges[2], faceEdges[3]};
                }

                SpatialDomains::Curve *curvePtr = nullptr;
                // if n == 1 triangles will not have face nodes
                if (conf.m_faceNodes && (nEdge == 4 || n > 1))
                {
                    int face = j, facenodes;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;

                    if (j % 2 == 0)
                    {
                        facenodes = n * n;
                        if (orientation == 1)
                        {
                            face = (face + 4) % 6;
                        }
                        else if (orientation == 2)
                        {
                            face = (face + 2) % 6;
                        }

                        for (int i = 0; i < facenodes; ++i)
                        {
                            faceNodes.push_back(
                                nodeList[face_offset[face] + i]);
                        }

                        // Find the original face vertex IDs.
                        std::vector<int> origFaceIds(4);
                        origFaceIds[0] =
                            nodeList[faceIds[face][0]]->GetGlobalID();
                        origFaceIds[1] =
                            nodeList[faceIds[face][1]]->GetGlobalID();
                        origFaceIds[2] =
                            nodeList[faceIds[face][2]]->GetGlobalID();
                        origFaceIds[3] =
                            nodeList[faceIds[face][3]]->GetGlobalID();

                        // Construct a HOQuadrilateral object which performs the
                        // orientation magically for us.
                        HOQuadrilateral<SpatialDomains::PointGeom *> hoq(
                            origFaceIds, faceNodes);
                        std::vector<int> faceVertIds(4);
                        faceVertIds[0] = vertex[faceIds[j][0]]->GetGlobalID();
                        faceVertIds[1] = vertex[faceIds[j][1]]->GetGlobalID();
                        faceVertIds[2] = vertex[faceIds[j][2]]->GetGlobalID();
                        faceVertIds[3] = vertex[faceIds[j][3]]->GetGlobalID();
                        hoq.Align(faceVertIds);

                        // Copy the face nodes back again.
                        faceNodes = hoq.surfVerts;
                    }
                    else
                    {
                        facenodes = n * (n - 1) / 2;

                        for (int i = 0; i < facenodes; ++i)
                        {
                            faceNodes.push_back(
                                nodeList[face_offset[face] + i]);
                        }

                        // Find the original face vertex IDs.
                        std::vector<int> origFaceIds(3);
                        origFaceIds[0] =
                            nodeList[faceIds[face][0]]->GetGlobalID();
                        origFaceIds[1] =
                            nodeList[faceIds[face][1]]->GetGlobalID();
                        origFaceIds[2] =
                            nodeList[faceIds[face][2]]->GetGlobalID();

                        // Construct a HOTriangle object which performs the
                        // orientation magically for us.
                        HOTriangle<SpatialDomains::PointGeom *> hoTri(
                            origFaceIds, faceNodes);
                        std::vector<int> faceVertIds(3);
                        faceVertIds[0] = vertex[faceIds[j][0]]->GetGlobalID();
                        faceVertIds[1] = vertex[faceIds[j][1]]->GetGlobalID();
                        faceVertIds[2] = vertex[faceIds[j][2]]->GetGlobalID();
                        hoTri.Align(faceVertIds);

                        // Copy the face nodes back again.
                        faceNodes = hoTri.surfVerts;
                    }

                    // Try to translate between common face curve types
                    LibUtilities::PointsType pType = conf.m_faceCurveType;
                    if (pType == LibUtilities::ePolyEvenlySpaced && nEdge == 3)
                    {
                        pType = LibUtilities::eNodalTriEvenlySpaced;
                    }

                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    if (nEdge == 3)
                    {
                        tmpNodeList =
                            GetCurvedNodesTri(pType, faceVertArrayTri,
                                              faceEdgeArrayTri, faceNodes);
                    }
                    else
                    {
                        tmpNodeList =
                            GetCurvedNodesQuad(pType, faceVertArrayQuad,
                                               faceEdgeArrayQuad, faceNodes);
                    }

                    auto curve = ObjPoolManager<
                        SpatialDomains::Curve>::AllocateUniquePtr(id, pType);
                    for (auto &node : tmpNodeList)
                    {
                        curveNodeIDs->insert(node->GetGlobalID());
                        curve->m_points.emplace_back(node);
                    }
                    curvePtr        = curve.get();
                    curvedFaces[id] = std::move(curve);
                }

                SpatialDomains::Geometry2D *facePtr = nullptr;
                if (nEdge == 3)
                {
                    auto triGeom = ObjPoolManager<SpatialDomains::TriGeom>::
                        AllocateUniquePtr(id, faceEdgeArrayTri,
                                          faceVertArrayTri, false, curvePtr);
                    facePtr = triGeom.get();
                    meshGraph->AddGeom<SpatialDomains::TriGeom>(
                        id, std::move(triGeom));
                }
                else
                {
                    auto quadGeom = ObjPoolManager<SpatialDomains::QuadGeom>::
                        AllocateUniquePtr(id, faceEdgeArrayQuad,
                                          faceVertArrayQuad, false, curvePtr);
                    facePtr = quadGeom.get();
                    meshGraph->AddGeom<SpatialDomains::QuadGeom>(
                        id, std::move(quadGeom));
                }
                faces[j]      = facePtr;
                faceMap[vids] = facePtr;
            }
            else
            {
                faces[j] = it->second;

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
                        SpatialDomains::TriGeom *triPtr =
                            static_cast<SpatialDomains::TriGeom *>(faces[j]);
                        triPtr->SetEdge(k, edges[face_edge_ids[j][k]]);
                        triPtr->SetVertex(k, vertex[faceIds[j][k]]);
                    }
                }

                // Create face curvature if it's needed and we haven't already
                if ((conf.m_faceNodes && (nEdge == 4 || n > 1)) &&
                    (faces[j]->GetCurve() == nullptr || naiveTri))
                {
                    int face = j, facenodes;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;

                    if (j % 2 == 0)
                    {
                        facenodes = n * n;
                        if (orientation == 1)
                        {
                            face = (face + 4) % 6;
                        }
                        else if (orientation == 2)
                        {
                            face = (face + 2) % 6;
                        }

                        for (int i = 0; i < facenodes; ++i)
                        {
                            faceNodes.push_back(
                                nodeList[face_offset[face] + i]);
                        }

                        // Find the original face vertex IDs.
                        std::vector<int> origFaceIds(4);
                        origFaceIds[0] =
                            nodeList[faceIds[face][0]]->GetGlobalID();
                        origFaceIds[1] =
                            nodeList[faceIds[face][1]]->GetGlobalID();
                        origFaceIds[2] =
                            nodeList[faceIds[face][2]]->GetGlobalID();
                        origFaceIds[3] =
                            nodeList[faceIds[face][3]]->GetGlobalID();

                        // Construct a HOQuadrilateral object which performs the
                        // orientation magically for us.
                        HOQuadrilateral<SpatialDomains::PointGeom *> hoq(
                            origFaceIds, faceNodes);
                        std::vector<int> faceVertIds(4);
                        faceVertIds[0] = vertex[faceIds[j][0]]->GetGlobalID();
                        faceVertIds[1] = vertex[faceIds[j][1]]->GetGlobalID();
                        faceVertIds[2] = vertex[faceIds[j][2]]->GetGlobalID();
                        faceVertIds[3] = vertex[faceIds[j][3]]->GetGlobalID();
                        hoq.Align(faceVertIds);

                        // Copy the face nodes back again.
                        faceNodes = hoq.surfVerts;
                    }
                    else
                    {
                        facenodes = n * (n - 1) / 2;

                        for (int i = 0; i < facenodes; ++i)
                        {
                            faceNodes.push_back(
                                nodeList[face_offset[face] + i]);
                        }

                        // Find the original face vertex IDs.
                        std::vector<int> origFaceIds(3);
                        origFaceIds[0] =
                            nodeList[faceIds[face][0]]->GetGlobalID();
                        origFaceIds[1] =
                            nodeList[faceIds[face][1]]->GetGlobalID();
                        origFaceIds[2] =
                            nodeList[faceIds[face][2]]->GetGlobalID();

                        // Construct a HOTriangle object which performs the
                        // orientation magically for us.
                        HOTriangle<SpatialDomains::PointGeom *> hoTri(
                            origFaceIds, faceNodes);
                        std::vector<int> faceVertIds(3);
                        faceVertIds[0] = vertex[faceIds[j][0]]->GetGlobalID();
                        faceVertIds[1] = vertex[faceIds[j][1]]->GetGlobalID();
                        faceVertIds[2] = vertex[faceIds[j][2]]->GetGlobalID();
                        hoTri.Align(faceVertIds);

                        // Copy the face nodes back again.
                        faceNodes = hoTri.surfVerts;
                    }

                    // Try to translate between common face curve types
                    LibUtilities::PointsType pType = conf.m_faceCurveType;
                    if (pType == LibUtilities::ePolyEvenlySpaced && nEdge == 3)
                    {
                        pType = LibUtilities::eNodalTriEvenlySpaced;
                    }

                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    if (nEdge == 3)
                    {
                        std::array<SpatialDomains::PointGeom *, 3>
                            faceVertArrayTri = {faceVertices[0],
                                                faceVertices[1],
                                                faceVertices[2]};
                        std::array<SpatialDomains::SegGeom *, 3>
                            faceEdgeArrayTri = {faceEdges[0], faceEdges[1],
                                                faceEdges[2]};
                        tmpNodeList =
                            GetCurvedNodesTri(pType, faceVertArrayTri,
                                              faceEdgeArrayTri, faceNodes);
                    }
                    else
                    {
                        std::array<SpatialDomains::PointGeom *, 4>
                            faceVertArrayQuad = {
                                faceVertices[0], faceVertices[1],
                                faceVertices[2], faceVertices[3]};
                        std::array<SpatialDomains::SegGeom *, 4>
                            faceEdgeArrayQuad = {faceEdges[0], faceEdges[1],
                                                 faceEdges[2], faceEdges[3]};
                        tmpNodeList =
                            GetCurvedNodesQuad(pType, faceVertArrayQuad,
                                               faceEdgeArrayQuad, faceNodes);
                    }

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(faces[j]->GetGlobalID(), pType);
                    for (auto &node : tmpNodeList)
                    {
                        curveNodeIDs->insert(node->GetGlobalID());
                        curve->m_points.emplace_back(node);
                    }
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

        // Re-order edge array to be consistent with Nektar++ ordering.
        std::array<SpatialDomains::SegGeom *, 9> tmp{};
        ASSERTL1(face_edge_ids[0][0] != -1, "face_edge_ids[0][0] == -1");
        tmp[0] = edges[face_edge_ids[0][0]];
        ASSERTL1(face_edge_ids[0][1] != -1, "face_edge_ids[0][1] == -1");
        tmp[1] = edges[face_edge_ids[0][1]];
        ASSERTL1(face_edge_ids[0][2] != -1, "face_edge_ids[0][2] == -1");
        tmp[2] = edges[face_edge_ids[0][2]];
        ASSERTL1(face_edge_ids[0][3] != -1, "face_edge_ids[0][3] == -1");
        tmp[3] = edges[face_edge_ids[0][3]];
        ASSERTL1(face_edge_ids[1][2] != -1, "face_edge_ids[1][2] == -1");
        tmp[4] = edges[face_edge_ids[1][2]];
        ASSERTL1(face_edge_ids[1][1] != -1, "face_edge_ids[1][1] == -1");
        tmp[5] = edges[face_edge_ids[1][1]];
        ASSERTL1(face_edge_ids[2][1] != -1, "face_edge_ids[2][1] == -1");
        tmp[6] = edges[face_edge_ids[2][1]];
        ASSERTL1(face_edge_ids[3][2] != -1, "face_edge_ids[3][2] == -1");
        tmp[7] = edges[face_edge_ids[3][2]];
        ASSERTL1(face_edge_ids[4][2] != -1, "face_edge_ids[4][2] == -1");
        tmp[8] = edges[face_edge_ids[4][2]];
        edges  = tmp;

        // Use the supplied element id if given, else the running 3D count.
        int id =
            forceIDs != nullptr ? forceIDs->elmt : meshGraph->GetNumElements(3);
        auto prismGeom =
            ObjPoolManager<SpatialDomains::PrismGeom>::AllocateUniquePtr(
                id, faces, edges, vertex, true);
        auto prismPtr = prismGeom.get();
        meshGraph->AddGeom<SpatialDomains::PrismGeom>(id, std::move(prismGeom));

        // Interior nodes, kept as a curve on the element: edges and faces are
        // shared and hold their own curvature, but the inside of an element
        // belongs to nothing else and is lost unless it is held here.
        const int nPoints    = conf.m_order + 1;
        const int nPrismPts  = nPoints * (nPoints + 1) / 2 * nPoints;
        const int nEdgeNodes = nPoints - 2;
        const int nInt       = nPrismPts - 6 - 9 * nEdgeNodes -
                         3 * nEdgeNodes * nEdgeNodes -
                         2 * nEdgeNodes * (nPoints - 3) / 2;

        if (conf.m_volumeNodes && nInt > 0 && (int)nodeList.size() == nPrismPts)
        {
            std::vector<SpatialDomains::PointGeom *> &nodal = nodeList;

            std::vector<int> listIds(6), elmtIds(6);
            for (int i = 0; i < 6; ++i)
            {
                listIds[i] = nodal[i]->GetGlobalID();
                elmtIds[i] = prismPtr->GetVertex(i)->GetGlobalID();
            }

            // OrientPrism may have turned the element since the caller listed
            // its vertices. Turning the nodes to match is a rotation of the
            // triangular cross-section; nothing is evaluated.
            SpatialDomains::HOPrism<SpatialDomains::PointGeom *> hoPrism(
                listIds, nodal, nPoints);
            hoPrism.Align(elmtIds);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    id, LibUtilities::eNodalPrismEvenlySpaced);
            curve->m_points = hoPrism.volVerts;

            prismPtr->SetCurve(curve.get());
            meshGraph->AddCurvedVolume(std::move(curve));
        }

        return prismPtr;
    }

    /**
     * @brief Orient prism to align degenerate vertices.
     *
     * Orientation of prismatric elements is required so that the singular
     * vertices of triangular faces (which occur as a part of the
     * collapsed co-ordinate system) align. The algorithm is based on that
     * used in T. Warburton's thesis and in the original Nektar source.
     *
     * First the points are re-ordered so that the highest global IDs
     * represent the two singular points of the prism. Then, if necessary,
     * the nodes are rotated either clockwise or counter-clockwise (w.r.t
     * to the p-r plane) to correctly align the prism. The #orientation
     * variable is set to:
     *
     * - 0 if the prism is not rotated;
     * - 1 if the prism is rotated clockwise;
     * - 2 if the prism is rotated counter-clockwise.
     *
     * This is necessary for some input modules (e.g. #InputNek) which add
     * high-order information or bounary conditions to faces.
     */
    static int OrientPrism(std::array<SpatialDomains::PointGeom *, 6> &vertex)
    {
        int lid[6], gid[6];

        // Re-order vertices.
        for (int i = 0; i < 6; ++i)
        {
            lid[i] = i;
            gid[i] = vertex[i]->GetGlobalID();
        }

        gid[0] = gid[3] = std::max(gid[0], gid[3]);
        gid[1] = gid[2] = std::max(gid[1], gid[2]);
        gid[4] = gid[5] = std::max(gid[4], gid[5]);

        for (int i = 1; i < 6; ++i)
        {
            if (gid[0] < gid[i])
            {
                std::swap(gid[i], gid[0]);
                std::swap(lid[i], lid[0]);
            }
        }

        if (lid[0] == 4 || lid[0] == 5)
        {
            return 0;
        }
        else if (lid[0] == 1 || lid[0] == 2)
        {
            // Rotate prism clockwise in p-r plane
            std::array<SpatialDomains::PointGeom *, 6> vertexmap;
            vertexmap[0] = vertex[4];
            vertexmap[1] = vertex[0];
            vertexmap[2] = vertex[3];
            vertexmap[3] = vertex[5];
            vertexmap[4] = vertex[1];
            vertexmap[5] = vertex[2];
            vertex       = vertexmap;
            return 1;
        }
        else if (lid[0] == 0 || lid[0] == 3)
        {
            // Rotate prism counter-clockwise in p-r plane
            std::array<SpatialDomains::PointGeom *, 6> vertexmap;
            vertexmap[0] = vertex[1];
            vertexmap[1] = vertex[4];
            vertexmap[2] = vertex[5];
            vertexmap[3] = vertex[2];
            vertexmap[4] = vertex[0];
            vertexmap[5] = vertex[3];
            vertex       = vertexmap;
            return 2;
        }
        else
        {
            std::cerr << "Warning: possible prism orientation problem."
                      << std::endl;
        }

        // No rotation was applied in the anomalous case above, so report the
        // prism as unrotated. Falling off the end here returned a garbage
        // orientation, which the caller uses to pick which block of the node
        // list belongs to which face.
        return 0;
    }
};

LibUtilities::ShapeType prismType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::ePrism, prismHelper::create, "Prism");

} // namespace Nektar::NekMesh
