////////////////////////////////////////////////////////////////////////////////
//
//  File: Triangle.cpp
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
//  Description: Mesh triangle object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/TriExp.h>
#include <NekMesh/MeshElements/Element.h>
#include <StdRegions/StdNodalTriExp.h>

#include <LibUtilities/Foundations/ManagerAccess.h>

namespace Nektar::NekMesh
{

struct triangleHelper
{
    /**
     * @brief Create a triangle element.
     */
    static SpatialDomains::Geometry *create(
        std::vector<SpatialDomains::PointGeom *> &nodeList,
        SpatialDomains::MeshGraphSharedPtr &meshGraph, EdgeMap &edgeMap,
        FaceMap &faceMap, ElmtConfig &conf, std::set<int> *vertIDs,
        std::unordered_set<int> *curveNodeIDs,
        std::unordered_set<int> *naiveTriIDs, ElmtIds *forceIDs)
    {
        auto &curvedEdges = meshGraph->GetCurvedEdges();
        auto &curvedFaces = meshGraph->GetCurvedFaces();

        // Vertex IDs that make up the triangle edges
        const unsigned int edgeIds[3][2] = {{0, 1}, {1, 2}, {2, 0}};

        // Add vertices. This logic will determine (in 2D) whether the
        // element is clockwise (sum > 0) or counter-clockwise (sum < 0).
        std::array<SpatialDomains::PointGeom *, 3> vertex = {
            nodeList[0], nodeList[1], nodeList[2]};

        // Bugfix - face can already exist, so let's check to avoid creating a
        // duplicate
        std::array<int, 4> vIDs = {vertex[0]->GetGlobalID(),
                                   vertex[1]->GetGlobalID(),
                                   vertex[2]->GetGlobalID(), -1};

        auto found = faceMap.find(vIDs);
        if (found != faceMap.end() && !conf.m_faceNodes)
        {
            return found->second;
        }

        // Create edges (with corresponding set of edge points)
        std::array<SpatialDomains::SegGeom *, 3> edges;
        for (int e = 0; e < 3; ++e)
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
                    for (int j = 3 + n * e; j < 3 + n * (e + 1); ++j)
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
                    for (int j = 3 + n * e; j < 3 + n * (e + 1); ++j)
                    {
                        tmpNodeList.emplace_back(nodeList[j]);
                    }
                    tmpNodeList.emplace_back(edgeVerts[1]);
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

        if (conf.m_reorient)
        {
            NekDouble sum = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                int o = (i + 1) % 3;
                sum += (nodeList[o]->x() - nodeList[i]->x()) *
                       (nodeList[o]->y() + nodeList[i]->y());
            }
            if (sum > 0.0)
            {
                std::reverse(edges.begin(), edges.end());
                std::swap(vertex[1], vertex[2]);
            }
        }

        // The idea is to take the NextFaceID instead of .size(),
        // as the set shrinks and expands
        int id = NextFaceId(meshGraph);
        if (forceIDs != nullptr)
        {
            id = forceIDs->elmt;
        }
        SpatialDomains::Curve *curvePtr = nullptr;
        if (conf.m_faceNodes)
        {
            int n = conf.m_order - 1;
            int N = 3 + 3 * n;
            std::vector<SpatialDomains::PointGeom *> faceNodes;
            for (int i = 0; i < n * (n - 1) / 2; ++i)
            {
                faceNodes.emplace_back(nodeList[N + i]);
            }

            auto tmpNodeList = GetCurvedNodesTri(conf.m_faceCurveType, vertex,
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

        // Dummy index in last position to make use of quads' faceMap
        std::array<int, 4> vids = {vertex[0]->GetGlobalID(),
                                   vertex[1]->GetGlobalID(),
                                   vertex[2]->GetGlobalID(), -1};

        if (vertIDs != nullptr)
        {
            for (auto vert : vertex)
            {
                vertIDs->insert(vert->GetGlobalID());
            }
        }

        auto triGeom =
            ObjPoolManager<SpatialDomains::TriGeom>::AllocateUniquePtr(
                id, edges, vertex, true, curvePtr);
        auto triPtr = triGeom.get();

        faceMap[vids] = triPtr;
        meshGraph->AddGeom<SpatialDomains::TriGeom>(id, std::move(triGeom));
        if (naiveTriIDs != nullptr)
        {
            naiveTriIDs->insert(id);
        }
        return triPtr;
    };
};

LibUtilities::ShapeType triType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::eTriangle, triangleHelper::create, "Triangle");

} // namespace Nektar::NekMesh
