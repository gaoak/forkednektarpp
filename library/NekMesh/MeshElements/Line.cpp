////////////////////////////////////////////////////////////////////////////////
//
//  File: Line.cpp
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
//  Description: Mesh line object.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/Curve.hpp>

using namespace std;

namespace Nektar::NekMesh
{

struct lineHelper
{
    static SpatialDomains::Geometry *create(
        std::vector<SpatialDomains::PointGeom *> &nodeList,
        SpatialDomains::MeshGraphSharedPtr &meshGraph, EdgeMap &edgeMap,
        [[maybe_unused]] FaceMap &faceMap, ElmtConfig &conf,
        std::set<int> *vertIDs, std::unordered_set<int> *curveNodeIDs,
        [[maybe_unused]] std::unordered_set<int> *naiveTriIDs,
        ElmtIds *forceIDs)
    {
        auto &curvedEdges = meshGraph->GetCurvedEdges();
        // auto &pointGeoms  = meshGraph->GetGeomMap<PointGeom>();
        // auto &curveNodes  = meshGraph->GetAllCurveNodes();

        int id = NextEdgeId(meshGraph);
        if (forceIDs != nullptr)
        {
            id = forceIDs->elmt;
        }

        // Create seg curvature
        SpatialDomains::Curve *curvePtr = nullptr;
        if (conf.m_order > 1)
        {
            int n = conf.m_order - 1;
            std::vector<SpatialDomains::PointGeom *> tmpNodeList;
            tmpNodeList.emplace_back(nodeList[0]);
            for (int j = 2; j < 2 + n; ++j)
            {
                tmpNodeList.emplace_back(nodeList[j]);
            }
            tmpNodeList.emplace_back(nodeList[1]);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    id, conf.m_edgeCurveType);
            for (auto &node : tmpNodeList)
            {
                curveNodeIDs->insert(node->GetGlobalID());
                curve->m_points.emplace_back(node);
            }
            curvePtr        = curve.get();
            curvedEdges[id] = std::move(curve);
        }

        if (vertIDs != nullptr)
        {
            for (int i = 0; i < 2; i++)
            {
                vertIDs->insert(nodeList[i]->GetGlobalID());
            }
        }

        std::array<SpatialDomains::PointGeom *, 2> verts = {nodeList[0],
                                                            nodeList[1]};
        auto segGeom =
            ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                id, nodeList[0]->GetCoordim(), verts, curvePtr);
        auto segPtr = segGeom.get();
        meshGraph->AddGeom<SpatialDomains::SegGeom>(id, std::move(segGeom));

        edgeMap[std::make_pair(nodeList[0]->GetGlobalID(),
                               nodeList[1]->GetGlobalID())] = segPtr;

        return segPtr;
    }
};

LibUtilities::ShapeType segType = GetElementFactory().RegisterCreatorFunction(
    LibUtilities::eSegment, lineHelper::create, "Line");

} // namespace Nektar::NekMesh
