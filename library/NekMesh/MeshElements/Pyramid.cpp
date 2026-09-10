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

#include <SpatialDomains/PyrGeom.h>

#include <NekMesh/MeshElements/HOAlignment.h>
#include <NekMesh/MeshElements/Pyramid.h>

#include <LibUtilities/Foundations/ManagerAccess.h>

using namespace std;

namespace Nektar::NekMesh
{

LibUtilities::ShapeType Pyramid::type =
    GetElementFactory().RegisterCreatorFunction(LibUtilities::ePyramid,
                                                Pyramid::create, "Pyramid");

/// Vertex IDs that make up pyramid faces.
int Pyramid::m_faceIds[5][4] = {
    {0, 1, 2, 3}, {0, 1, 4, -1}, {1, 2, 4, -1}, {3, 2, 4, -1}, {0, 3, 4, -1}};

/// Vertex IDs that make up pyramid edges, in the order and direction used by
/// the standard element: the four base edges followed by the four edges
/// running up to the apex. Note that edges 2 and 3 are directed 3->2 and 0->3
/// respectively, matching the ordering that NodalPyrEvenlySpaced emits.
int Pyramid::m_edgeVerts[8][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3},
                                  {0, 4}, {1, 4}, {2, 4}, {3, 4}};

/**
 * @brief Create a pyramidic element.
 */
Pyramid::Pyramid(ElmtConfig pConf, vector<NodeSharedPtr> pNodeList,
                 vector<int> pTagList)
    : Element(pConf, GetNumNodes(pConf), pNodeList.size())
{
    m_tag     = "P";
    m_dim     = 3;
    m_taglist = pTagList;
    int n     = m_conf.m_order - 1;

    // This edge-node map is based on Nektar++ ordering.
    map<pair<int, int>, int> edgeNodeMap;
    map<pair<int, int>, int>::iterator it;
    edgeNodeMap[pair<int, int>(1, 2)] = 6;
    edgeNodeMap[pair<int, int>(2, 3)] = 6 + n;
    edgeNodeMap[pair<int, int>(4, 3)] = 6 + 2 * n;
    edgeNodeMap[pair<int, int>(1, 4)] = 6 + 3 * n;
    edgeNodeMap[pair<int, int>(1, 5)] = 6 + 4 * n;
    edgeNodeMap[pair<int, int>(2, 5)] = 6 + 5 * n;
    edgeNodeMap[pair<int, int>(3, 5)] = 6 + 6 * n;
    edgeNodeMap[pair<int, int>(4, 5)] = 6 + 7 * n;

    // Add vertices
    for (int i = 0; i < 5; ++i)
    {
        m_vertex.push_back(pNodeList[i]);
    }

    // Create edges (with corresponding set of edge points)
    int eid = 0;
    for (it = edgeNodeMap.begin(); it != edgeNodeMap.end(); ++it)
    {
        vector<NodeSharedPtr> edgeNodes;
        if (m_conf.m_order > 1)
        {
            for (int j = it->second; j < it->second + n; ++j)
            {
                edgeNodes.push_back(pNodeList[j - 1]);
            }
        }
        m_edge.push_back(EdgeSharedPtr(new Edge(
            pNodeList[it->first.first - 1], pNodeList[it->first.second - 1],
            edgeNodes, m_conf.m_edgeCurveType)));
        m_edge.back()->m_id = eid++;
    }

    // Create faces
    int face_edges[5][4];
    int faceoffset = 5 + 8 * n;
    for (int j = 0; j < 5; ++j)
    {
        vector<NodeSharedPtr> faceVertices;
        vector<EdgeSharedPtr> faceEdges;
        vector<NodeSharedPtr> faceNodes;
        int nEdge = j > 0 ? 3 : 4;

        for (int k = 0; k < nEdge; ++k)
        {
            faceVertices.push_back(m_vertex[m_faceIds[j][k]]);
            NodeSharedPtr a = m_vertex[m_faceIds[j][k]];
            NodeSharedPtr b = m_vertex[m_faceIds[j][(k + 1) % nEdge]];
            for (unsigned int i = 0; i < m_edge.size(); ++i)
            {
                if ((m_edge[i]->m_n1 == a && m_edge[i]->m_n2 == b) ||
                    (m_edge[i]->m_n1 == b && m_edge[i]->m_n2 == a))
                {
                    faceEdges.push_back(m_edge[i]);
                    face_edges[j][k] = i;
                    break;
                }
            }
        }

        if (m_conf.m_faceNodes)
        {
            int facenodes = j == 0 ? n * n : n * (n - 1) / 2;
            for (int i = 0; i < facenodes; ++i)
            {
                faceNodes.push_back(pNodeList[faceoffset + i]);
            }
            faceoffset += facenodes;
        }

        // Try to translate between common face curve types
        LibUtilities::PointsType pType = m_conf.m_faceCurveType;

        if (pType == LibUtilities::ePolyEvenlySpaced && (j > 0))
        {
            pType = LibUtilities::eNodalTriEvenlySpaced;
        }

        m_face.push_back(
            FaceSharedPtr(new Face(faceVertices, faceNodes, faceEdges, pType)));
    }

    // Reorder edges to align with Nektar++ order.
    vector<EdgeSharedPtr> tmp(8);
    tmp[0] = m_edge[face_edges[0][0]];
    tmp[1] = m_edge[face_edges[0][1]];
    tmp[2] = m_edge[face_edges[0][2]];
    tmp[3] = m_edge[face_edges[0][3]];
    tmp[4] = m_edge[face_edges[1][2]];
    tmp[5] = m_edge[face_edges[1][1]];
    tmp[6] = m_edge[face_edges[3][1]];
    tmp[7] = m_edge[face_edges[3][2]];
    m_edge = tmp;
}

SpatialDomains::Geometry *Pyramid::GetGeom(int coordDim,
                                           SpatialDomains::EntityHolder &holder)
{
    std::array<SpatialDomains::Geometry2D *, 5> faces;

    for (int i = 0; i < 5; ++i)
    {
        faces[i] = m_face[i]->GetGeom(coordDim, holder);
    }

    SpatialDomains::PyrGeomUniquePtr pyr =
        ObjPoolManager<SpatialDomains::PyrGeom>::AllocateUniquePtr(m_id, faces);
    auto ret = dynamic_cast<SpatialDomains::Geometry *>(pyr.get());
    holder.m_pyrVec.push_back(std::move(pyr));

    ret->Setup();
    return ret;
}

/**
 * @brief Return the number of nodes defining a pyramid.
 */
unsigned int Pyramid::GetNumNodes(ElmtConfig pConf)
{
    int n = pConf.m_order;

    // valid for any order pyramid
    return (5             // corners
            + 8 * (n - 1) // mid-edge
            +
            pConf.m_faceNodes * ((n - 1) * (n - 1) +
                                 4 * (n - 1) * (n - 2) /
                                     2) // square base + 4xtriangle-number faces
            + pConf.m_volumeNodes * (n - 2) * (n - 1) * (2 * n - 3) /
                  6 // square pyramidal numbers
    );
}

StdRegions::Orientation Pyramid::GetEdgeOrient(int edgeId, EdgeSharedPtr edge)
{
    if (edge->m_n1 == m_vertex[m_edgeVerts[edgeId][0]])
    {
        return StdRegions::eForwards;
    }
    else if (edge->m_n1 == m_vertex[m_edgeVerts[edgeId][1]])
    {
        return StdRegions::eBackwards;
    }
    else
    {
        ASSERTL1(false, "Edge is not connected to this pyramid.");
    }

    return StdRegions::eNoOrientation;
}

void Pyramid::MakeOrder(int order, SpatialDomains::Geometry *geom,
                        LibUtilities::PointsType pType, int coordDim, int &id,
                        bool justConfig)
{
    m_conf.m_order = order;
    m_curveType    = pType;
    m_volumeNodes.clear();

    if (order == 1)
    {
        m_conf.m_volumeNodes = m_conf.m_faceNodes = false;
        return;
    }
    else if (order == 2)
    {
        m_conf.m_faceNodes   = true;
        m_conf.m_volumeNodes = false;
        return;
    }

    m_conf.m_faceNodes   = true;
    m_conf.m_volumeNodes = true;

    if (justConfig)
    {
        return;
    }

    int nPoints                            = order + 1;
    StdRegions::StdExpansionSharedPtr xmap = geom->GetXmap();

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsKey pKey(nPoints, pType);
    ASSERTL1(pKey.GetPointsDim() == 3, "Points distribution must be 3D");
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    Array<OneD, Array<OneD, NekDouble>> phys(coordDim);

    for (int i = 0; i < coordDim; ++i)
    {
        phys[i] = Array<OneD, NekDouble>(xmap->GetTotPoints());
        xmap->BwdTrans(geom->GetCoeffs(i), phys[i]);
    }

    // The nodal distribution stacks an (nPoints - k) by (nPoints - k) layer at
    // each height k, so the total is the square pyramidal number P(nPoints).
    // Stripping the boundary leaves the same lattice three sizes down, i.e.
    // P(nPoints - 3), and those points come last in the nodal ordering.
    const int nPyrPts = nPoints * (nPoints + 1) * (2 * nPoints + 1) / 6;
    const int nPyrIntPts =
        (nPoints - 3) * (nPoints - 2) * (2 * nPoints - 5) / 6;
    m_volumeNodes.resize(nPyrIntPts);

    for (int i = nPyrPts - nPyrIntPts, cnt = 0; i < nPyrPts; ++i, ++cnt)
    {
        Array<OneD, NekDouble> xp(3);
        xp[0] = px[i];
        xp[1] = py[i];
        xp[2] = pz[i];

        Array<OneD, NekDouble> x(3, 0.0);
        for (int j = 0; j < coordDim; ++j)
        {
            x[j] = xmap->PhysEvaluate(xp, phys[j]);
        }

        m_volumeNodes[cnt] =
            std::shared_ptr<Node>(new Node(id++, x[0], x[1], x[2]));
    }
}

void Pyramid::GetCurvedNodes(std::vector<NodeSharedPtr> &nodeList) const
{
    int n = m_edge[0]->GetNodeCount();
    nodeList.resize(n * (n + 1) * (2 * n + 1) / 6);

    for (int i = 0; i < 5; ++i)
    {
        nodeList[i] = m_vertex[i];
    }
    int k = 5;

    for (int i = 0; i < 8; i++)
    {
        bool reverseEdge = m_edge[i]->m_n1 == m_vertex[m_edgeVerts[i][0]];
        if (reverseEdge)
        {
            for (int j = 0; j < n - 2; j++)
            {
                nodeList[k++] = m_edge[i]->m_edgeNodes[j];
            }
        }
        else
        {
            for (int j = n - 3; j >= 0; j--)
            {
                nodeList[k++] = m_edge[i]->m_edgeNodes[j];
            }
        }
    }

    // Target vertex ordering of each face, taken from the standard element.
    vector<vector<int>> ts;
    for (int i = 0; i < 5; ++i)
    {
        const int nFaceVert = i == 0 ? 4 : 3;
        vector<int> t(nFaceVert);
        for (int j = 0; j < nFaceVert; ++j)
        {
            t[j] = m_vertex[m_faceIds[i][j]]->m_id;
        }
        ts.push_back(t);
    }

    for (int i = 0; i < ts.size(); i++)
    {
        if (ts[i].size() == 3)
        {
            vector<int> fcid;
            fcid.push_back(m_face[i]->m_vertexList[0]->m_id);
            fcid.push_back(m_face[i]->m_vertexList[1]->m_id);
            fcid.push_back(m_face[i]->m_vertexList[2]->m_id);

            HOTriangle<NodeSharedPtr> hot(fcid, m_face[i]->m_faceNodes);

            hot.Align(ts[i]);

            std::copy(hot.surfVerts.begin(), hot.surfVerts.end(),
                      nodeList.begin() + k);
            k += hot.surfVerts.size();
        }
        else
        {
            vector<int> fcid;
            fcid.push_back(m_face[i]->m_vertexList[0]->m_id);
            fcid.push_back(m_face[i]->m_vertexList[1]->m_id);
            fcid.push_back(m_face[i]->m_vertexList[2]->m_id);
            fcid.push_back(m_face[i]->m_vertexList[3]->m_id);

            HOQuadrilateral<NodeSharedPtr> hoq(fcid, m_face[i]->m_faceNodes);

            hoq.Align(ts[i]);

            std::copy(hoq.surfVerts.begin(), hoq.surfVerts.end(),
                      nodeList.begin() + k);
            k += hoq.surfVerts.size();
        }
    }

    std::copy(m_volumeNodes.begin(), m_volumeNodes.end(), nodeList.begin() + k);
}
} // namespace Nektar::NekMesh
