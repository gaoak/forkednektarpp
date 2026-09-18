////////////////////////////////////////////////////////////////////////////////
//
//  File: Quadrilateral.cpp
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
//  Description: Mesh quad object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/QuadExp.h>
#include <NekMesh/MeshElements/Element.h>

#include <LibUtilities/Foundations/ManagerAccess.h>

namespace Nektar::NekMesh
{

struct quadrilateralHelper
{
    /**
     * @brief Create a quadrilateral element during input.
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

        // Vertex IDs that make up the quadrilateral edges
        const unsigned int edgeIds[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

        // Add vertices. This logic will determine (in 2D) whether the
        // element is clockwise (sum > 0) or counter-clockwise (sum < 0).
        std::array<SpatialDomains::PointGeom *, 4> vertex = {
            nodeList[0], nodeList[1], nodeList[2], nodeList[3]};

        // Bugfix - face can already exist, so let's check to avoid creating a
        // duplicate
        std::array<int, 4> vIDs = {
            vertex[0]->GetGlobalID(), vertex[1]->GetGlobalID(),
            vertex[2]->GetGlobalID(), vertex[3]->GetGlobalID()};

        auto found = faceMap.find(vIDs);
        if (found != faceMap.end() && !conf.m_faceNodes)
        {
            return found->second;
        }

        // Create edges (with corresponding set of edge points)
        std::array<SpatialDomains::SegGeom *, 4> edges;
        for (int e = 0; e < 4; ++e)
        {
            std::array<SpatialDomains::PointGeom *, 2> edgeVerts = {
                nodeList[edgeIds[e][0]], nodeList[edgeIds[e][1]]};

            // Check if edge already exists in edgeSet using edgeVert IDs, if it
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
                    for (int j = 4 + n * e; j < 4 + n * (e + 1); ++j)
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
                    for (int j = 4 + n * e; j < 4 + n * (e + 1); ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j]);
                    }
                    tmpNodeList.emplace_back(edgeVerts[1]);

                    // The edge was created by a neighbour, which may have run
                    // along it the other way; a curve is stored in the
                    // SegGeom's own direction. Triangle::create has always
                    // done this.
                    if (edgeVerts[1]->GetGlobalID() ==
                        edges[e]->GetVertex(0)->GetGlobalID())
                    {
                        std::reverse(tmpNodeList.begin(), tmpNodeList.end());
                    }

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

        int id = NextFaceId(meshGraph);
        if (forceIDs != nullptr)
        {
            id = forceIDs->elmt;
        }
        SpatialDomains::Curve *curvePtr = nullptr;
        if (conf.m_faceNodes)
        {
            int n = conf.m_order - 1;
            int N = 4 + 4 * n;
            std::vector<SpatialDomains::PointGeom *> faceNodes;
            for (int i = 0; i < n * n; ++i)
            {
                faceNodes.emplace_back(nodeList[N + i]);
            }

            auto tmpNodeList = GetCurvedNodesQuad(conf.m_faceCurveType, vertex,
                                                  edges, faceNodes);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    id, conf.m_faceCurveType);
            for (auto &node : tmpNodeList)
            {
                curveNodeIDs->insert(node->GetGlobalID());
                curve->m_points.emplace_back(node);
            }
            curvePtr        = curve.get();
            curvedFaces[id] = std::move(curve);
        }

        if (conf.m_reorient)
        {
            NekDouble sum = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                int o = (i + 1) % 4;
                sum += (nodeList[o]->x() - nodeList[i]->x()) *
                       (nodeList[o]->y() + nodeList[i]->y());
            }
            if (sum > 0.0)
            {
                std::reverse(edges.begin(), edges.end());
                std::swap(vertex[1], vertex[3]);
            }

            // @TODO: need to reorient face curvature here?
        }

        std::array<int, 4> vids = {
            vertex[0]->GetGlobalID(), vertex[1]->GetGlobalID(),
            vertex[2]->GetGlobalID(), vertex[3]->GetGlobalID()};

        if (vertIDs != nullptr)
        {
            for (auto vert : vertex)
            {
                vertIDs->insert(vert->GetGlobalID());
            }
        }

        auto quadGeom =
            ObjPoolManager<SpatialDomains::QuadGeom>::AllocateUniquePtr(
                id, edges, vertex, true, curvePtr);
        auto quadPtr = quadGeom.get();

        faceMap[vids] = quadPtr;
        meshGraph->AddGeom<SpatialDomains::QuadGeom>(id, std::move(quadGeom));
        return quadPtr;
    };
};

LibUtilities::ShapeType quadType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::eQuadrilateral, quadrilateralHelper::create, "Quadrilateral");

} // namespace Nektar::NekMesh
