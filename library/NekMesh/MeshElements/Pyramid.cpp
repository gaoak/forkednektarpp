////////////////////////////////////////////////////////////////////////////////
//
//  File: Pyramid.cpp
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
//  Description: Mesh pyramid object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/PrismExp.h>
#include <StdRegions/StdPyrExp.h>

#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/PyrGeom.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/HOAlignment.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/TriGeom.h>

using namespace std;

namespace Nektar::NekMesh
{

struct pyramidHelper
{
    /// Map from local edges to local vert - MeshGraph ordering.
    constexpr static int edgeVertsInds[8][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3},
                                                {0, 4}, {1, 4}, {2, 4}, {3, 4}};

    /// Map Vertex IDs to pyramid faces. -1 means is empty placeholder
    constexpr static int faceIds[5][4] = {{0, 1, 2, 3},
                                          {0, 1, 4, -1},
                                          {1, 2, 4, -1},
                                          {3, 2, 4, -1},
                                          {0, 3, 4, -1}};

    /**
     * @brief Create a pyramid.
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

        // Vertix list
        std::array<SpatialDomains::PointGeom *, 5> vertices;
        for (int i = 0; i < 5; ++i)
        {
            vertices[i] = nodeList[i];
        }

        // Create edges
        std::array<SpatialDomains::SegGeom *, 8> edges{};
        for (int e = 0; e < 8; ++e)
        {
            std::array<SpatialDomains::PointGeom *, 2> edgeVerts = {
                vertices[edgeVertsInds[e][0]], vertices[edgeVertsInds[e][1]]};

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
                    for (int j = 6 + e * n; j < 6 + e * n + n; ++j)
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

                // Create edge nodes and curvature
                if (conf.m_order > 1 && edges[e]->GetCurve() == nullptr)
                {
                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    tmpNodeList.emplace_back(edgeVerts[0]);
                    for (int j = 6 + e * n; j < 6 + e * n + n; ++j)
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

        // Create faces
        int face_edge_ids[5][4];

        int face_offset[5];
        face_offset[0] = 5 + 8 * n;
        for (int j = 0; j < 4; ++j)
        {
            int facenodeID     = j == 0 ? n * n : n * (n - 1) / 2;
            face_offset[j + 1] = face_offset[j] + facenodeID;
        }

        std::array<SpatialDomains::Geometry2D *, 5> faces{};
        for (int j = 0; j < 5; ++j)
        {
            int nEdge = j > 0 ? 3 : 4; // if j>0 -> nEdge=3 ,:else nEdge=4
            Array<OneD, SpatialDomains::PointGeom *> faceVertices(nEdge);
            Array<OneD, SpatialDomains::SegGeom *> faceEdges(nEdge);

            for (int k = 0; k < nEdge; ++k)
            {
                faceVertices[k]              = vertices[faceIds[j][k]];
                SpatialDomains::PointGeom *a = vertices[faceIds[j][k]];
                SpatialDomains::PointGeom *b =
                    vertices[faceIds[j][(k + 1) % nEdge]];
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

            // Check if face already exists in faceSet using vertices IDs
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
                    int facenodes = nEdge == 4 ? n * n : n * (n - 1) / 2;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;

                    for (int i = 0; i < facenodes; ++i)
                    {
                        faceNodes.push_back(nodeList[face_offset[j] + i]);
                    }

                    // Find the original face vertex IDs.
                    std::vector<int> origFaceIds(nEdge);
                    std::vector<int> faceVertIds(nEdge);
                    for (int k = 0; k < nEdge; ++k)
                    {
                        origFaceIds[k] = nodeList[faceIds[j][k]]->GetGlobalID();
                        faceVertIds[k] = vertices[faceIds[j][k]]->GetGlobalID();
                    }

                    // Construct a HOTriangle/HOQuadrilateral object which
                    // performs the orientation magically for us.
                    if (nEdge == 4)
                    {
                        HOQuadrilateral<SpatialDomains::PointGeom *> hoq(
                            origFaceIds, faceNodes);
                        hoq.Align(faceVertIds);
                        faceNodes = hoq.surfVerts;
                    }
                    else
                    {
                        HOTriangle<SpatialDomains::PointGeom *> hoTri(
                            origFaceIds, faceNodes);
                        hoTri.Align(faceVertIds);
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
                if (naiveTri)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        SpatialDomains::TriGeom *triPtr =
                            static_cast<SpatialDomains::TriGeom *>(faces[j]);
                        triPtr->SetEdge(k, edges[face_edge_ids[j][k]]);
                        triPtr->SetVertex(k, vertices[faceIds[j][k]]);
                    }
                }

                // Create face curvature if it's needed and we haven't already
                if ((conf.m_faceNodes && (nEdge == 4 || n > 1)) &&
                    (faces[j]->GetCurve() == nullptr || naiveTri))
                {
                    int facenodes = nEdge == 4 ? n * n : n * (n - 1) / 2;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;

                    for (int i = 0; i < facenodes; ++i)
                    {
                        faceNodes.push_back(nodeList[face_offset[j] + i]);
                    }

                    // Find the original face vertex IDs.
                    std::vector<int> origFaceIds(nEdge);
                    std::vector<int> faceVertIds(nEdge);
                    for (int k = 0; k < nEdge; ++k)
                    {
                        origFaceIds[k] = nodeList[faceIds[j][k]]->GetGlobalID();
                        faceVertIds[k] = vertices[faceIds[j][k]]->GetGlobalID();
                    }

                    if (nEdge == 4)
                    {
                        HOQuadrilateral<SpatialDomains::PointGeom *> hoq(
                            origFaceIds, faceNodes);
                        hoq.Align(faceVertIds);
                        faceNodes = hoq.surfVerts;
                    }
                    else
                    {
                        HOTriangle<SpatialDomains::PointGeom *> hoTri(
                            origFaceIds, faceNodes);
                        hoTri.Align(faceVertIds);
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
            for (auto vert : vertices)
            {
                vertIDs->insert(vert->GetGlobalID());
            }
        }

        // Use the supplied element id if given, else the running 3D count.
        int id =
            forceIDs != nullptr ? forceIDs->elmt : meshGraph->GetNumElements(3);
        // PyrGeom needs onlt faces, edges and vertices are alocated from them
        auto pyrGeom =
            ObjPoolManager<SpatialDomains::PyrGeom>::AllocateUniquePtr(id,
                                                                       faces);
        auto pyrPtr = pyrGeom.get();
        meshGraph->AddGeom<SpatialDomains::PyrGeom>(id, std::move(pyrGeom));

        // Interior nodes, kept as a curve on the element: edges and faces are
        // shared and hold their own curvature, but the inside of an element
        // belongs to nothing else and is lost unless it is held here. A
        // pyramid keeps the vertices it was given, so unlike the prism there
        // is nothing to turn back.
        const int nPoints = conf.m_order + 1;
        const int nPyrPts = nPoints * (nPoints + 1) * (2 * nPoints + 1) / 6;
        const int nInt = (nPoints - 3) * (nPoints - 2) * (2 * nPoints - 5) / 6;

        if (conf.m_volumeNodes && nInt > 0 && (int)nodeList.size() == nPyrPts)
        {
            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    id, LibUtilities::eNodalPyrEvenlySpaced);
            curve->m_points = nodeList;

            pyrPtr->SetCurve(curve.get());
            meshGraph->AddCurvedVolume(std::move(curve));
        }

        return pyrPtr;
    }
};

LibUtilities::ShapeType pyrType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::ePyramid, pyramidHelper::create, "Pyramid");

} // namespace Nektar::NekMesh
