////////////////////////////////////////////////////////////////////////////////
//
//  File: Element.cpp
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

#include <NekMesh/MeshElements/Element.h>

using namespace std;

namespace Nektar::NekMesh
{

ElementFactory &GetElementFactory()
{
    static ElementFactory instance;
    return instance;
}

std::vector<SpatialDomains::PointGeom *> GetCurvedNodesTri(
    LibUtilities::PointsType conf_faceCurveType,
    std::array<SpatialDomains::PointGeom *, 3> &vertexList,
    std::array<SpatialDomains::SegGeom *, 3> &edgeList,
    std::vector<SpatialDomains::PointGeom *> &faceNodes)
{
    std::vector<SpatialDomains::PointGeom *> nodeList;

    // Treat 2D point distributions differently to 3D.
    ASSERTL0(conf_faceCurveType == LibUtilities::eNodalTriFekete ||
                 conf_faceCurveType == LibUtilities::eNodalTriEvenlySpaced ||
                 conf_faceCurveType == LibUtilities::eNodalTriElec,
             "Incorrect conf_faceCurveType for GetCurvedNodesTri")

    int n  = edgeList[0]->GetCurve()->m_points.size();
    int n2 = edgeList[0]->GetCurve()->m_points.size();
    int n3 = edgeList[0]->GetCurve()->m_points.size();

    bool same = (n == n2 ? (n2 == n3) : false);
    ASSERTL0(same, "Edges are not consistent");

    nodeList.insert(nodeList.end(), vertexList.begin(), vertexList.end());
    for (int k = 0; k < edgeList.size(); ++k)
    {
        for (auto it = std::next(edgeList[k]->GetCurve()->m_points.begin());
             it != std::prev(edgeList[k]->GetCurve()->m_points.end()); ++it)
        {
            nodeList.emplace_back(*it);
        }

        if (edgeList[k]->GetVertex(0) != vertexList[k])
        {
            // If edge orientation is reversed relative to node
            // ordering, we need to reverse order of nodes.
            std::reverse(nodeList.begin() + 3 + k * (n - 2),
                         nodeList.begin() + 3 + (k + 1) * (n - 2));
        }
    }
    nodeList.insert(nodeList.end(), faceNodes.begin(), faceNodes.end());

    return nodeList;
}

std::vector<SpatialDomains::PointGeom *> GetCurvedNodesQuad(
    [[maybe_unused]] LibUtilities::PointsType conf_faceCurveType,
    std::array<SpatialDomains::PointGeom *, 4> &vertexList,
    std::array<SpatialDomains::SegGeom *, 4> &edgeList,
    std::vector<SpatialDomains::PointGeom *> &faceNodes)
{
    std::vector<SpatialDomains::PointGeom *> nodeList;

    // Write out in 2D tensor product order.

    int n1 = faceNodes.size();
    for (auto &i : edgeList)
    {
        n1 += i->GetCurve()->m_points.size();
    }
    n1 -= vertexList.size();

    int n = (int)sqrt((NekDouble)n1);
    nodeList.resize(n * n);

    ASSERTL0(n * n == n1, "Wrong number of modes?");

    // Write vertices
    nodeList[0]           = vertexList[0];
    nodeList[n - 1]       = vertexList[1];
    nodeList[n * n - 1]   = vertexList[2];
    nodeList[n * (n - 1)] = vertexList[3];

    // Write edge-interior
    int skips[4][2] = {{0, 1}, {n - 1, n}, {n * n - 1, -1}, {n * (n - 1), -n}};
    for (int i = 0; i < 4; ++i)
    {
        bool reverseEdge = edgeList[i]->GetVertex(0) == vertexList[i];

        if (!reverseEdge)
        {
            for (int j = 1; j < n - 1; ++j)
            {
                nodeList[skips[i][0] + j * skips[i][1]] =
                    edgeList[i]->GetCurve()->m_points[n - 1 - j];
            }
        }
        else
        {
            for (int j = 1; j < n - 1; ++j)
            {
                nodeList[skips[i][0] + j * skips[i][1]] =
                    edgeList[i]->GetCurve()->m_points[j];
            }
        }
    }

    // Write interior
    for (int i = 1; i < n - 1; ++i)
    {
        for (int j = 1; j < n - 1; ++j)
        {
            nodeList[i * n + j] = faceNodes[(i - 1) * (n - 2) + (j - 1)];
        }
    }

    return nodeList;
}

std::vector<SpatialDomains::PointGeom *> GetCurvedNodes(
    SpatialDomains::Geometry *geom)
{
    SpatialDomains::Curve *curve = geom->GetCurve();

    ASSERTL0(curve != nullptr,
             "Element " + std::to_string(geom->GetGlobalID()) +
                 " carries no curve, so its nodes cannot be gathered.");

    return curve->m_points;
}

} // namespace Nektar::NekMesh
