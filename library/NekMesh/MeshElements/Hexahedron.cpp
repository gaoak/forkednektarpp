////////////////////////////////////////////////////////////////////////////////
//
//  File: Hexahedron.cpp
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
//  Description: Hexahedral elements
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/HexGeom.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/QuadGeom.h>

namespace Nektar::NekMesh
{

struct hexahedronHelper
{
    /**
     * @brief Create a hexahedron element.
     */
    static SpatialDomains::Geometry *create(
        std::vector<SpatialDomains::PointGeom *> &nodeList,
        SpatialDomains::MeshGraphSharedPtr &meshGraph, EdgeMap &edgeMap,
        FaceMap &faceMap, ElmtConfig &conf, std::set<int> *vertIDs,
        std::unordered_set<int> *curveNodeIDs,
        [[maybe_unused]] std::unordered_set<int> *naiveTriIDs,
        ElmtIds *forceIDs)
    {

        auto &curvedEdges = meshGraph->GetCurvedEdges();
        auto &curvedFaces = meshGraph->GetCurvedFaces();

        // Vertex IDs that make up the hexahedron edges
        const unsigned int edgeIds[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                             {0, 4}, {1, 5}, {2, 6}, {3, 7},
                                             {4, 5}, {5, 6}, {6, 7}, {7, 4}};

        // Vertex IDs that make up hexahedron faces.
        const unsigned int faceIds[6][4] = {{0, 1, 2, 3}, {0, 1, 5, 4},
                                            {1, 2, 6, 5}, {3, 2, 6, 7},
                                            {0, 3, 7, 4}, {4, 5, 6, 7}};

        // Edges that make up hexahedron faces
        const unsigned int faceEdgeIds[6][4] = {{0, 1, 2, 3},  {0, 5, 8, 4},
                                                {1, 6, 9, 5},  {2, 6, 10, 7},
                                                {3, 7, 11, 4}, {8, 9, 10, 11}};

        // Add vertices
        std::array<SpatialDomains::PointGeom *, 8> vertex;
        for (int i = 0; i < 8; ++i)
        {
            vertex[i] = nodeList[i];
        }

        // Create edges (with corresponding set of edge points)
        std::array<SpatialDomains::SegGeom *, 12> edges{};
        for (int e = 0; e < 12; ++e)
        {
            std::array<SpatialDomains::PointGeom *, 2> edgeVerts = {
                nodeList[edgeIds[e][0]], nodeList[edgeIds[e][1]]};

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
                    int n = conf.m_order - 1;
                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    tmpNodeList.emplace_back(edgeVerts[0]);
                    for (int j = 8 + n * e; j < 8 + n * (e + 1); ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j]);
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
                    int n = conf.m_order - 1;
                    std::vector<SpatialDomains::PointGeom *> tmpNodeList;
                    tmpNodeList.emplace_back(edgeVerts[0]);
                    for (int j = 8 + n * e; j < 8 + n * (e + 1); ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j]);
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
        std::array<SpatialDomains::QuadGeom *, 6> faces{};
        std::array<SpatialDomains::PointGeom *, 4> faceVertices;
        std::array<SpatialDomains::SegGeom *, 4> faceEdges;

        for (int j = 0; j < 6; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                faceVertices[k] = vertex[faceIds[j][k]];
            }

            // Check if quad already exists in faceSet using vertex IDs
            std::array<int, 4> vids = {
                faceVertices[0]->GetGlobalID(), faceVertices[1]->GetGlobalID(),
                faceVertices[2]->GetGlobalID(), faceVertices[3]->GetGlobalID()};

            auto it = faceMap.find(vids);

            if (it == faceMap.end())
            {
                int id = NextFaceId(meshGraph);
                if (forceIDs != nullptr)
                {
                    id = forceIDs->faces[j];
                }

                for (int k = 0; k < 4; ++k)
                {
                    faceEdges[k] = edges[faceEdgeIds[j][k]];
                }

                SpatialDomains::Curve *curvePtr = nullptr;
                if (conf.m_faceNodes)
                {
                    int n = conf.m_order - 1;
                    int N = 8 + 12 * n + j * n * n;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;
                    for (int i = 0; i < n * n; ++i)
                    {
                        faceNodes.emplace_back(nodeList[N + i]);
                    }

                    auto tmpNodeList =
                        GetCurvedNodesQuad(conf.m_faceCurveType, faceVertices,
                                           faceEdges, faceNodes);

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(id, conf.m_faceCurveType);
                    for (auto &node : tmpNodeList)
                    {
                        curveNodeIDs->insert(node->GetGlobalID());
                        curve->m_points.emplace_back(node);
                    }
                    curvePtr        = curve.get();
                    curvedFaces[id] = std::move(curve);
                }

                auto quadGeom =
                    ObjPoolManager<SpatialDomains::QuadGeom>::AllocateUniquePtr(
                        id, faceEdges, faceVertices, false, curvePtr);
                auto quadPtr = quadGeom.get();
                meshGraph->AddGeom<SpatialDomains::QuadGeom>(
                    id, std::move(quadGeom));

                faces[j]      = quadPtr;
                faceMap[vids] = quadPtr;
            }
            else
            {
                faces[j] = static_cast<SpatialDomains::QuadGeom *>(it->second);

                // Create face curvature if it's needed and we haven't already
                if (conf.m_faceNodes && faces[j]->GetCurve() == nullptr)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        faceEdges[k] = edges[faceEdgeIds[j][k]];
                    }
                    int n = conf.m_order - 1;
                    int N = 8 + 12 * n + j * n * n;
                    std::vector<SpatialDomains::PointGeom *> faceNodes;
                    for (int i = 0; i < n * n; ++i)
                    {
                        faceNodes.emplace_back(nodeList[N + i]);
                    }
                    auto tmpNodeList =
                        GetCurvedNodesQuad(conf.m_faceCurveType, faceVertices,
                                           faceEdges, faceNodes);

                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(faces[j]->GetGlobalID(),
                                          conf.m_faceCurveType);
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

        int id =
            forceIDs != nullptr ? forceIDs->elmt : meshGraph->GetNumElements(3);
        auto hexGeom =
            ObjPoolManager<SpatialDomains::HexGeom>::AllocateUniquePtr(
                id, faces, edges, vertex, true);
        auto hexPtr = hexGeom.get();
        meshGraph->AddGeom<SpatialDomains::HexGeom>(id, std::move(hexGeom));

        return hexPtr;
    }
};

LibUtilities::ShapeType hexType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::eHexahedron, hexahedronHelper::create, "Hexahedron");

} // namespace Nektar::NekMesh
