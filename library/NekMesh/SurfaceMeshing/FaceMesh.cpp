////////////////////////////////////////////////////////////////////////////////
//
//  File: FaceMesh.cpp
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/ExtLibInterface/TriangleInterface.h>
#include <NekMesh/Octree/Octree.h>
#include <NekMesh/SurfaceMeshing/FaceMesh.h>
#include <limits>

using namespace std;
namespace Nektar::NekMesh
{

bool FaceMesh::ValidateCurves()
{
    vector<int> curvesInSurface;
    for (int i = 0; i < m_edgeloops.size(); i++)
    {
        for (int j = 0; j < m_edgeloops[i]->edges.size(); j++)
        {
            curvesInSurface.push_back(m_edgeloops[i]->edges[j]->GetId());
        }
    }

    bool error  = false;
    auto &graph = m_mesh->m_meshGraph;

    // Note: a curve mesh point's stored (u,v) on this surface is only as
    // accurate as the projection that produced it, and does not map back
    // exactly onto the point: a curve's end points come from CADVert::GetLoc()
    // while their surface parameters come from CADSurf::locuv, and the two
    // disagree by around 1e-5 on the test geometries. That is not an error,
    // and checking for it here is what used to make surface meshing fail.

    for (int i = 0; i < curvesInSurface.size(); i++)
    {
        vector<SpatialDomains::SegGeom *> es =
            m_curvemeshes[curvesInSurface[i]]->GetMeshEdges();

        for (int j = i; j < curvesInSurface.size(); j++)
        {
            if (i == j)
            {
                continue;
            }

            vector<SpatialDomains::SegGeom *> es2 =
                m_curvemeshes[curvesInSurface[j]]->GetMeshEdges();

            for (int l = 0; l < es.size(); l++)
            {
                auto P1 = graph->GetCADAssociation()->GetSurfUV(
                    es[l]->GetVertex(0), m_id);
                auto P2 = graph->GetCADAssociation()->GetSurfUV(
                    es[l]->GetVertex(1), m_id);
                for (int k = 0; k < es2.size(); k++)
                {
                    if (es[l]->GetVertex(0) == es2[k]->GetVertex(0) ||
                        es[l]->GetVertex(0) == es2[k]->GetVertex(1) ||
                        es[l]->GetVertex(1) == es2[k]->GetVertex(0) ||
                        es[l]->GetVertex(1) == es2[k]->GetVertex(1))
                    {
                        continue;
                    }

                    auto P3 = graph->GetCADAssociation()->GetSurfUV(
                        es2[k]->GetVertex(0), m_id);
                    auto P4 = graph->GetCADAssociation()->GetSurfUV(
                        es2[k]->GetVertex(1), m_id);

                    NekDouble den = (P4[0] - P3[0]) * (P2[1] - P1[1]) -
                                    (P2[0] - P1[0]) * (P4[1] - P3[1]);

                    if (fabs(den) < 1e-8)
                    {
                        continue;
                    }

                    NekDouble t = ((P1[0] - P3[0]) * (P4[1] - P3[1]) -
                                   (P4[0] - P3[0]) * (P1[1] - P3[1])) /
                                  den;
                    NekDouble u =
                        (P1[0] - P3[0] + t * (P2[0] - P1[0])) / (P4[0] - P3[0]);

                    if (t < 1.0 && t > 0.0 && u < 1.0 && u > 0.0)
                    {
                        std::array<NekDouble, 2> uv = {
                            P1[0] + t * (P2[0] - P1[0]),
                            P1[1] + t * (P2[1] - P1[1])};
                        auto loc = m_cadsurf->P(uv);

                        m_log(VERBOSE).Newline();
                        m_log(WARNING)
                            << "Curve mesh error at " << loc[0] << " " << loc[1]
                            << " " << loc[2] << " on face " << m_id
                            << ": curve " << curvesInSurface[i] << " edge " << l
                            << " (uv " << P1[0] << "," << P1[1] << " -> "
                            << P2[0] << "," << P2[1] << ") crosses curve "
                            << curvesInSurface[j] << " edge " << k << " (uv "
                            << P3[0] << "," << P3[1] << " -> " << P4[0] << ","
                            << P4[1] << ")" << endl;
                        error = true;
                    }
                }
            }
        }
    }
    return error;
}

void FaceMesh::ValidateLoops()
{
    OrientateCurves();

    for (int i = 0; i < orderedLoops.size(); i++)
    {
        int numPoints = orderedLoops[i].size();
        if (numPoints == 2)
        {
            // force a remesh of the curves
            for (int j = 0; j < m_edgeloops[i]->edges.size(); j++)
            {
                int cid = m_edgeloops[i]->edges[j]->GetId();
                m_curvemeshes[cid]->ReMesh();
            }
        }
    }
}

void FaceMesh::Mesh()
{
    Stretching();
    OrientateCurves();

    int numPoints = 0;
    for (int i = 0; i < orderedLoops.size(); i++)
    {
        numPoints += orderedLoops[i].size();
        for (int j = 0; j < orderedLoops[i].size(); j++)
        {
            m_inBoundary.insert(orderedLoops[i][j]);
        }
    }

    stringstream ss;
    ss << "3 points required for triangulation, " << numPoints << " in loop"
       << endl;
    ss << "curves: ";
    for (int i = 0; i < m_edgeloops.size(); i++)
    {
        for (int j = 0; j < m_edgeloops[i]->edges.size(); j++)
        {
            ss << m_edgeloops[i]->edges[j]->GetId() << " ";
        }
    }

    ASSERTL0(numPoints > 2, "number of verts in face is less than 3");

    // create interface to triangle thirdparty library
    TriangleInterfaceSharedPtr pplanemesh =
        MemoryManager<TriangleInterface>::AllocateSharedPtr();

    vector<std::array<NekDouble, 2>> centers;
    for (int i = 0; i < m_edgeloops.size(); i++)
    {
        centers.push_back(m_edgeloops[i]->center);
    }

    // TriangleInterface meshes in the parametric plane but holds no CAD or
    // graph handle, so the (u,v) of every node it is given is passed in.
    auto &graph = m_mesh->m_meshGraph;

    NodeUVMap nodeUV;
    for (int i = 0; i < orderedLoops.size(); i++)
    {
        for (int j = 0; j < orderedLoops[i].size(); j++)
        {
            nodeUV[orderedLoops[i][j]] =
                graph->GetCADAssociation()->GetSurfUV(orderedLoops[i][j], m_id);
        }
    }

    pplanemesh->Assign(orderedLoops, centers, m_id, nodeUV, m_str);

    pplanemesh->Mesh();

    pplanemesh->Extract(m_connec);

    bool repeat = true;

    // continuously remesh until all triangles conform to the spacing in the
    // octree
    while (repeat)
    {
        repeat = Validate();
        if (!repeat)
        {
            break;
        }
        m_connec.clear();

        // Validate() appends to m_stienerpoints, so rebuild the uv for the
        // points added on this pass before handing them over.
        NodeUVMap stienerUV;
        for (int i = 0; i < m_stienerpoints.size(); i++)
        {
            stienerUV[m_stienerpoints[i]] =
                graph->GetCADAssociation()->GetSurfUV(m_stienerpoints[i], m_id);
        }

        pplanemesh->AssignStiener(m_stienerpoints, stienerUV);
        pplanemesh->Mesh();
        pplanemesh->Extract(m_connec);
    }

    // build a local version of the mesh (one set of triangles).  this is done
    // so edge connectivity infomration can be used for optimisation
    BuildLocalMesh();

    OptimiseLocalMesh();

    // The triangles are registered with the graph as BuildLocalMesh sowe record
    // which composite they belong
    for (int i = 0; i < m_localElements.size(); i++)
    {
        m_mesh->m_elementTags[2][m_localElements[i]] = m_compId;
    }

    m_log(VERBOSE).Overwrite();
    m_log(VERBOSE) << "    - Face " << m_id << endl;
    m_log(VERBOSE) << "        Nodes    : " << m_localNodes.size() << endl;
    m_log(VERBOSE) << "        Edges    : " << m_localEdges.size() << endl;
    m_log(VERBOSE) << "        Triangles: " << m_localElements.size() << endl;
    m_log(VERBOSE) << "        Loops    : " << m_edgeloops.size() << endl;
    m_log(VERBOSE) << "        STR      : " << scientific << m_str << endl;
}

void FaceMesh::OptimiseLocalMesh()
{
    // Each optimisation algorithm is based on the work in chapter 19.
    //
    // Two rounds do not converge. Swapping a diagonal moves the nodes that
    // smoothing then relaxes, and relaxing them makes further diagonals worth
    // swapping, so the two have to alternate until they stop changing
    // anything. Stopping early leaves the result depending on the order the
    // nodes and edges were visited in, which is a property of the containers
    // holding them rather than of the geometry: on 3d_bl_cyl the worst scaled
    // Jacobian came out at 0.0108 under one node ordering and 0.0032 under
    // another, against 0.1485 for either once converged -- the same value to
    // nine figures, which is what makes the result independent of the order.
    // The third round is within 3e-4 of that fixed point and the sixth sits on
    // it.
    //
    // Three is what is used here rather than six: from four rounds on,
    // 3d_bl_wing fails to build its boundary layer at all ("shrinking to
    // nothing"), so the smoother surface mesh is more than that case's
    // boundary layer can extrude. That is a separate weakness worth fixing on
    // its own, and until it is, this is a value chosen between two cliffs
    // rather than a converged one.
    for (int i = 0; i < 3; ++i)
    {
        DiagonalSwap();

        Smoothing();
    }
}

void FaceMesh::Smoothing()
{
    auto &graph = m_mesh->m_meshGraph;
    auto bounds = m_cadsurf->GetBounds();

    // Vertex connectivity, keyed on the vertex itself rather than its id.
    std::unordered_map<SpatialDomains::PointGeom *,
                       vector<SpatialDomains::SegGeom *>>
        connectingedges;
    std::unordered_map<SpatialDomains::PointGeom *,
                       vector<SpatialDomains::Geometry *>>
        connectingelements;

    for (auto &[key, ed] : m_localEdges)
    {
        connectingedges[ed->GetVertex(0)].push_back(ed);
        connectingedges[ed->GetVertex(1)].push_back(ed);
    }

    for (int i = 0; i < m_localElements.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            connectingelements[m_localElements[i]->GetVertex(j)].push_back(
                m_localElements[i]);
        }
    }

    // perform 4 runs of elastic relaxation based on the octree
    for (int q = 0; q < 4; q++)
    {
        for (auto *I : m_localNodes)
        {
            // node is on curve so skip
            if (m_inBoundary.find(I) != m_inBoundary.end())
            {
                continue;
            }

            vector<SpatialDomains::SegGeom *> &edges = connectingedges[I];
            vector<SpatialDomains::Geometry *> &els  = connectingelements[I];

            // The old implementation built throwaway Node objects to carry the
            // parametric location of each spoke; only the (u,v) is ever read,
            // so the locations are collected directly and no vertex is created.
            vector<std::array<NekDouble, 2>> uvsystem;

            auto ui = graph->GetCADAssociation()->GetSurfUV(I, m_id);

            for (int i = 0; i < edges.size(); i++)
            {
                vector<NekDouble> lambda;

                SpatialDomains::PointGeom *J;
                if (I == edges[i]->GetVertex(0))
                {
                    J = edges[i]->GetVertex(1);
                }
                else if (I == edges[i]->GetVertex(1))
                {
                    J = edges[i]->GetVertex(0);
                }
                else
                {
                    ASSERTL0(false, "could not find node");
                }

                auto uj = graph->GetCADAssociation()->GetSurfUV(J, m_id);

                for (int j = 0; j < els.size(); j++)
                {
                    // elememt is adjacent to J therefore no intersection on IJ
                    if (els[j]->GetVertex(0) == J ||
                        els[j]->GetVertex(1) == J || els[j]->GetVertex(2) == J)
                    {
                        continue;
                    }

                    // need to find other edge
                    SpatialDomains::Geometry1D *AtoB = nullptr;
                    bool found                       = false;
                    for (int k = 0; k < els[j]->GetNumEdges(); k++)
                    {
                        auto *e = els[j]->GetEdge(k);
                        if (!(e->GetVertex(0) == I || e->GetVertex(1) == I))
                        {
                            found = true;
                            AtoB  = e;
                            break;
                        }
                    }
                    ASSERTL0(found, "failed to find edge to test");

                    auto A = graph->GetCADAssociation()->GetSurfUV(
                        AtoB->GetVertex(0), m_id);
                    auto B = graph->GetCADAssociation()->GetSurfUV(
                        AtoB->GetVertex(1), m_id);

                    NekDouble lam = ((A[0] - uj[0]) * (B[1] - A[1]) -
                                     (A[1] - uj[1]) * (B[0] - A[0])) /
                                    ((ui[0] - uj[0]) * (B[1] - A[1]) -
                                     (ui[1] - uj[1]) * (B[0] - A[0]));

                    if (!(lam < 0) && !(lam > 1))
                    {
                        lambda.push_back(lam);
                    }
                }

                if (lambda.size() > 0)
                {
                    sort(lambda.begin(), lambda.end());
                    // parametric location of the dummy point on the spoke
                    uvsystem.push_back({uj[0] + lambda[0] * (ui[0] - uj[0]),
                                        uj[1] + lambda[0] * (ui[1] - uj[1])});
                }
                else
                {
                    uvsystem.push_back(uj);
                }
            }

            std::array<NekDouble, 2> u0 = {0.0, 0.0};

            for (int i = 0; i < uvsystem.size(); i++)
            {
                u0[0] += uvsystem[i][0] / uvsystem.size();
                u0[1] += uvsystem[i][1] / uvsystem.size();
            }

            bool inbounds = true;
            if (u0[0] < bounds[0])
            {
                inbounds = false;
            }
            else if (u0[0] > bounds[1])
            {
                inbounds = false;
            }
            else if (u0[1] < bounds[2])
            {
                inbounds = false;
            }
            else if (u0[1] > bounds[3])
            {
                inbounds = false;
            }

            if (!inbounds)
            {
                continue;
            }

            // Node::Move set both the position and the parametric location.
            auto l2 = m_cadsurf->P(u0);
            I->UpdatePosition(l2[0], l2[1], l2[2]);
            graph->GetCADAssociation()->Add(I, {m_cadsurf, {u0[0], u0[1]}});
        }
    }

    // Vertices have moved, so any cached geometric information on the elements
    // and their edges is now stale.
    for (int i = 0; i < m_localElements.size(); i++)
    {
        m_localElements[i]->ResetLite();
    }
}

void FaceMesh::DiagonalSwap()
{
    auto &graph = m_mesh->m_meshGraph;

    auto locOf = [](SpatialDomains::PointGeom *p) {
        std::array<NekDouble, 3> l;
        p->GetCoords(l[0], l[1], l[2]);
        return l;
    };

    // Interior angle at P in the triangle P-X-Y, oriented by the surface
    // normal so that the value returned is the angle inside the element.
    auto angleAt = [&](SpatialDomains::PointGeom *P,
                       SpatialDomains::PointGeom *X,
                       SpatialDomains::PointGeom *Y) {
        return P->Angle(
            locOf(X), locOf(Y),
            m_cadsurf->N(graph->GetCADAssociation()->GetSurfUV(P, m_id)));
    };

    // Ideal and actual valency, keyed on the vertex rather than its id.
    std::unordered_map<SpatialDomains::PointGeom *, int> idealConnec;
    std::unordered_map<SpatialDomains::PointGeom *, int> actualConnec;
    std::unordered_map<SpatialDomains::PointGeom *,
                       vector<SpatialDomains::SegGeom *>>
        nodetoedge;

    for (auto &[key, ed] : m_localEdges)
    {
        nodetoedge[ed->GetVertex(0)].push_back(ed);
        nodetoedge[ed->GetVertex(1)].push_back(ed);
    }

    for (auto *nd : m_localNodes)
    {
        // interior vertices want six neighbours, boundary vertices four
        idealConnec[nd] = (m_inBoundary.find(nd) == m_inBoundary.end()) ? 6 : 4;
        actualConnec[nd] = nodetoedge[nd].size();
    }

    // Position of each element in m_localElements, so a swapped pair can be
    // replaced in place.
    std::unordered_map<SpatialDomains::Geometry *, size_t> elIndex;
    for (size_t i = 0; i < m_localElements.size(); i++)
    {
        elIndex[m_localElements[i]] = i;
    }

    // Edges no longer carry element back-links (the old m_elLink), so the
    // adjacency the swap needs is built here and patched as elements are
    // replaced.
    std::unordered_map<SpatialDomains::SegGeom *,
                       vector<SpatialDomains::Geometry *>>
        edgeToEl;

    auto addToAdj = [&](SpatialDomains::Geometry *el) {
        for (int i = 0; i < el->GetNumEdges(); i++)
        {
            edgeToEl[static_cast<SpatialDomains::SegGeom *>(el->GetEdge(i))]
                .push_back(el);
        }
    };
    auto removeFromAdj = [&](SpatialDomains::Geometry *el) {
        for (int i = 0; i < el->GetNumEdges(); i++)
        {
            auto &v = edgeToEl[static_cast<SpatialDomains::SegGeom *>(
                el->GetEdge(i))];
            v.erase(std::remove(v.begin(), v.end(), el), v.end());
        }
    };

    for (auto *el : m_localElements)
    {
        addToAdj(el);
    }

    // edgeswapping fun times
    // perfrom edge swap based on node defect and then angle
    for (int q = 0; q < 4; q++)
    {
        size_t edgesStart = m_localEdges.size();
        EdgeMap edges     = m_localEdges;
        m_localEdges.clear();

        for (auto &[key, e] : edges)
        {
            SpatialDomains::PointGeom *n1 = e->GetVertex(0);
            SpatialDomains::PointGeom *n2 = e->GetVertex(1);

            if (m_inBoundary.find(n1) != m_inBoundary.end() &&
                m_inBoundary.find(n2) != m_inBoundary.end())
            {
                m_localEdges[key] = e;
                continue;
            }

            auto adjIt = edgeToEl.find(e);
            if (adjIt == edgeToEl.end() || adjIt->second.size() != 2)
            {
                // not shared by exactly two triangles, cannot swap
                m_localEdges[key] = e;
                continue;
            }

            SpatialDomains::Geometry *tri1 = adjIt->second[0];
            SpatialDomains::Geometry *tri2 = adjIt->second[1];

            // identify node a,b,c,d of the swapping
            SpatialDomains::PointGeom *A = nullptr, *B = nullptr, *C = nullptr,
                                      *D = nullptr;

            std::array<SpatialDomains::PointGeom *, 3> nt = {
                tri1->GetVertex(0), tri1->GetVertex(1), tri1->GetVertex(2)};

            if (nt[0] != n1 && nt[0] != n2)
            {
                C = nt[0];
                B = nt[1];
                A = nt[2];
            }
            else if (nt[1] != n1 && nt[1] != n2)
            {
                C = nt[1];
                B = nt[2];
                A = nt[0];
            }
            else if (nt[2] != n1 && nt[2] != n2)
            {
                C = nt[2];
                B = nt[0];
                A = nt[1];
            }
            else
            {
                ASSERTL0(false, "failed to identify verticies in tri1");
            }

            nt = {tri2->GetVertex(0), tri2->GetVertex(1), tri2->GetVertex(2)};

            if (nt[0] != n1 && nt[0] != n2)
            {
                D = nt[0];
            }
            else if (nt[1] != n1 && nt[1] != n2)
            {
                D = nt[1];
            }
            else if (nt[2] != n1 && nt[2] != n2)
            {
                D = nt[2];
            }
            else
            {
                ASSERTL0(false, "failed to identify verticies in tri2");
            }

            // determine signed area of alternate config
            auto ai = graph->GetCADAssociation()->GetSurfUV(A, m_id);
            auto bi = graph->GetCADAssociation()->GetSurfUV(B, m_id);
            auto ci = graph->GetCADAssociation()->GetSurfUV(C, m_id);
            auto di = graph->GetCADAssociation()->GetSurfUV(D, m_id);

            NekDouble CDA, CBD;

            CDA = 0.5 * (-di[0] * ci[1] + ai[0] * ci[1] + ci[0] * di[1] -
                         ai[0] * di[1] - ci[0] * ai[1] + di[0] * ai[1]);

            CBD = 0.5 * (-bi[0] * ci[1] + di[0] * ci[1] + ci[0] * bi[1] -
                         di[0] * bi[1] - ci[0] * di[1] + bi[0] * di[1]);

            // if signed area of the swapping triangles is less than zero
            // that configuration is invalid and swap cannot be performed
            if (!(CDA > 0.001 && CBD > 0.001))
            {
                m_localEdges[key] = e;
                continue;
            }

            bool swap = false; // assume do not swap

            if (q < 2)
            {
                int nodedefectbefore = 0;
                nodedefectbefore += abs(actualConnec[A] - idealConnec[A]);
                nodedefectbefore += abs(actualConnec[B] - idealConnec[B]);
                nodedefectbefore += abs(actualConnec[C] - idealConnec[C]);
                nodedefectbefore += abs(actualConnec[D] - idealConnec[D]);

                int nodedefectafter = 0;
                nodedefectafter += abs(actualConnec[A] - 1 - idealConnec[A]);
                nodedefectafter += abs(actualConnec[B] - 1 - idealConnec[B]);
                nodedefectafter += abs(actualConnec[C] + 1 - idealConnec[C]);
                nodedefectafter += abs(actualConnec[D] + 1 - idealConnec[D]);

                if (nodedefectafter < nodedefectbefore)
                {
                    swap = true;
                }
            }
            else
            {
                NekDouble minanglebefore = angleAt(C, A, B);
                minanglebefore = min(minanglebefore, angleAt(A, B, C));
                minanglebefore = min(minanglebefore, angleAt(B, A, C));
                minanglebefore = min(minanglebefore, angleAt(B, A, D));
                minanglebefore = min(minanglebefore, angleAt(A, B, D));
                minanglebefore = min(minanglebefore, angleAt(D, A, B));

                NekDouble minangleafter = angleAt(C, B, D);
                minangleafter           = min(minangleafter, angleAt(D, B, C));
                minangleafter           = min(minangleafter, angleAt(B, C, D));
                minangleafter           = min(minangleafter, angleAt(C, A, D));
                minangleafter           = min(minangleafter, angleAt(A, C, D));
                minangleafter           = min(minangleafter, angleAt(D, A, C));

                if (minangleafter > minanglebefore)
                {
                    swap = true;
                }
            }

            if (!swap)
            {
                m_localEdges[key] = e;
                continue;
            }

            actualConnec[A]--;
            actualConnec[B]--;
            actualConnec[C]++;
            actualConnec[D]++;

            // Note what has to survive the removal below: once the geometry
            // is extracted from the graph, tri1, tri2 and e are dangling.
            size_t i1     = elIndex[tri1];
            size_t i2     = elIndex[tri2];
            int id1       = tri1->GetGlobalID();
            int id2       = tri2->GetGlobalID();
            int oldEdgeId = e->GetGlobalID();

            std::array<int, 4> vids1 = {tri1->GetVertex(0)->GetGlobalID(),
                                        tri1->GetVertex(1)->GetGlobalID(),
                                        tri1->GetVertex(2)->GetGlobalID(), -1};
            std::array<int, 4> vids2 = {tri2->GetVertex(0)->GetGlobalID(),
                                        tri2->GetVertex(1)->GetGlobalID(),
                                        tri2->GetVertex(2)->GetGlobalID(), -1};

            removeFromAdj(tri1);
            removeFromAdj(tri2);
            elIndex.erase(tri1);
            elIndex.erase(tri2);
            m_mesh->m_elementTags[2].erase(tri1);
            m_mesh->m_elementTags[2].erase(tri2);
            edgeToEl.erase(e);
            m_mesh->m_edgeSet.erase(key);

            // Bug fix - need to clean the faces and the mid edge from the set
            m_localEdges.erase(key);
            m_mesh->m_faceSet.erase(vids1);
            m_mesh->m_faceSet.erase(vids2);

            // Take ownership back out of the graph and let it go; the four
            // surrounding edges are untouched and will be reused below.
            graph->ExtractGeom<SpatialDomains::TriGeom>(id1, true);
            graph->ExtractGeom<SpatialDomains::TriGeom>(id2, true);
            graph->ExtractGeom<SpatialDomains::SegGeom>(oldEdgeId, true);

            ElmtConfig conf(LibUtilities::eTriangle, 1, false, false,
                            graph->GetSpaceDimension() != 3);

            vector<SpatialDomains::PointGeom *> t1 = {B, D, C};
            vector<SpatialDomains::PointGeom *> t2 = {A, C, D};

            SpatialDomains::Geometry *ntri1 =
                GetElementFactory().CreateInstance(
                    LibUtilities::eTriangle, t1, graph, m_mesh->m_edgeSet,
                    m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr,
                    nullptr);
            SpatialDomains::Geometry *ntri2 =
                GetElementFactory().CreateInstance(
                    LibUtilities::eTriangle, t2, graph, m_mesh->m_edgeSet,
                    m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr,
                    nullptr);

            graph->GetCADAssociation()->Set(ntri1, {m_cadsurf});
            graph->GetCADAssociation()->Set(ntri2, {m_cadsurf});
            m_mesh->m_elementTags[2][ntri1] = m_compId;
            m_mesh->m_elementTags[2][ntri2] = m_compId;

            m_localElements[i1] = ntri1;
            m_localElements[i2] = ntri2;
            elIndex[ntri1]      = i1;
            elIndex[ntri2]      = i2;

            addToAdj(ntri1);
            addToAdj(ntri2);

            // Record the edges of the new pair, which picks up the new C-D
            // edge as well as re-registering the four reused ones.
            for (auto *el : {ntri1, ntri2})
            {
                for (int i = 0; i < el->GetNumEdges(); i++)
                {
                    auto *ed =
                        static_cast<SpatialDomains::SegGeom *>(el->GetEdge(i));
                    m_localEdges[{ed->GetVid(0), ed->GetVid(1)}] = ed;
                }
            }
        }

        ASSERTL0(m_localEdges.size() == edgesStart, "mismatch edge count");
    }
}

void FaceMesh::BuildLocalMesh()
{
    /*************************
    // build a local set of nodes edges and elemenets for optimstaion prior to
    putting them into m_mesh
    */

    auto &graph = m_mesh->m_meshGraph;

    for (int i = 0; i < m_connec.size(); i++)
    {
        // Triangle produces its connectivity in the parameter plane,
        // which is not the orientation the element needs. We need orient.
        ElmtConfig conf(LibUtilities::eTriangle, 1, false, false,
                        graph->GetSpaceDimension() != 3);

        // The factory uses m_edgeSet for every edge of the new triag,
        // so an edge already created by a curve mesh or a neighbouring
        // triangle is reused rather than duplicated.
        SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
            LibUtilities::eTriangle, m_connec[i], graph, m_mesh->m_edgeSet,
            m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr, nullptr);

        graph->GetCADAssociation()->Set(E, {m_cadsurf});

        for (int j = 0; j < E->GetNumVerts(); j++)
        {
            // nodes are already unique some will insert some wont
            m_localNodes.insert(E->GetVertex(j));
        }

        for (int j = 0; j < E->GetNumEdges(); j++)
        {
            auto *ed = static_cast<SpatialDomains::SegGeom *>(E->GetEdge(j));
            m_localEdges[{ed->GetVid(0), ed->GetVid(1)}] = ed;
        }

        m_localElements.push_back(E);
    }
}

void FaceMesh::Stretching()
{
    // define a sampling and calculate the aspect ratio of the paramter plane
    m_str     = 0.0;
    auto bnds = m_cadsurf->GetBounds();

    NekDouble dxu = int(bnds[1] - bnds[0] < bnds[3] - bnds[2]
                            ? 40
                            : (bnds[1] - bnds[0]) / (bnds[3] - bnds[2]) * 40);
    NekDouble dxv = int(bnds[3] - bnds[2] < bnds[1] - bnds[0]
                            ? 40
                            : (bnds[3] - bnds[2]) / (bnds[1] - bnds[0]) * 40);

    NekDouble du = (bnds[1] - bnds[0]) / dxu;
    NekDouble dv = (bnds[3] - bnds[2]) / dxv;

    int ct = 0;

    for (int i = 0; i < dxu; i++)
    {
        for (int j = 0; j < dxv; j++)
        {
            std::array<NekDouble, 2> uv = {bnds[0] + i * du, bnds[2] + j * dv};
            if (i == dxu - 1)
            {
                uv[0] = bnds[1];
            }
            if (j == dxv - 1)
            {
                uv[1] = bnds[3];
            }
            auto r = m_cadsurf->D1(uv);

            NekDouble ru = sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5]);
            NekDouble rv = sqrt(r[6] * r[6] + r[7] * r[7] + r[8] * r[8]);

            ru *= du;
            rv *= dv;

            if (rv < 1E-8)
            {
                continue;
            }

            m_str += ru / rv;
            ct++;
        }
    }

    m_str /= ct;
}

bool FaceMesh::Validate()
{
    // check all edges in the current mesh for length against the octree
    // if the octree is not conformed to add a new point inside the triangle
    Octree &octree = GetOctree(m_mesh, m_log);
    // if no new points are added meshing can stop
    int pointBefore = m_stienerpoints.size();
    auto &graph     = m_mesh->m_meshGraph;

    for (int i = 0; i < m_connec.size(); i++)
    {
        std::array<NekDouble, 3> r, a;

        vector<std::array<NekDouble, 2>> info;

        for (int j = 0; j < 3; j++)
        {
            info.push_back(
                graph->GetCADAssociation()->GetSurfUV(m_connec[i][j], m_id));
        }

        std::array<std::array<NekDouble, 3>, 3> loc;
        for (int j = 0; j < 3; j++)
        {
            m_connec[i][j]->GetCoords(loc[j][0], loc[j][1], loc[j][2]);
        }

        r[0] = m_connec[i][0]->dist(*m_connec[i][1]);
        r[1] = m_connec[i][1]->dist(*m_connec[i][2]);
        r[2] = m_connec[i][2]->dist(*m_connec[i][0]);

        a[0] = m_connec[i][0]->Angle(loc[1], loc[2], m_cadsurf->N(info[0]));
        a[1] = m_connec[i][1]->Angle(loc[2], loc[0], m_cadsurf->N(info[1]));
        a[2] = m_connec[i][2]->Angle(loc[0], loc[1], m_cadsurf->N(info[2]));

        NekDouble d1 = octree.Query(loc[0]);
        NekDouble d2 = octree.Query(loc[1]);
        NekDouble d3 = octree.Query(loc[2]);

        std::array<NekDouble, 2> uvc = {
            (info[0][0] + info[1][0] + info[2][0]) / 3.0,
            (info[0][1] + info[1][1] + info[2][1]) / 3.0};

        auto locc    = m_cadsurf->P(uvc);
        NekDouble d4 = octree.Query(locc);

        NekDouble d = (d1 + d2 + d3 + d4) / 4.0;

        vector<bool> valid(3);
        valid[0] = r[0] < d * 1.5;
        valid[1] = r[1] < d * 1.5;
        valid[2] = r[2] < d * 1.5;

        vector<bool> angValid(3);
        angValid[0] = a[0] / M_PI * 180.0 > 20.0 && a[0] / M_PI * 180.0 < 120.0;
        angValid[1] = a[1] / M_PI * 180.0 > 20.0 && a[1] / M_PI * 180.0 < 120.0;
        angValid[2] = a[2] / M_PI * 180.0 > 20.0 && a[2] / M_PI * 180.0 < 120.0;

        int numValid    = 0;
        int numAngValid = 0;
        for (int j = 0; j < 3; j++)
        {
            if (valid[j])
            {
                numValid++;
            }
            if (angValid[j])
            {
                numAngValid++;
            }
        }

        // if numvalid is zero no work to be done
        /*if (numValid != 3)
        {
            AddNewPoint(uvc);
        }*/

        if (numValid != 3 || numAngValid != 3)
        {
            // break the bad edge
            /*int a=0, b=0;
            if(!valid[0])
            {
                a = 0;
                b = 1;
            }
            else if(!valid[1])
            {
                a = 1;
                b = 2;
            }
            else if(!valid[2])
            {
                a = 2;
                b = 0;
            }

            Array<OneD, NekDouble> uvn(2);
            uvn[0] = (info[a][0] + info[b][0]) / 2.0;
            uvn[1] = (info[a][1] + info[b][1]) / 2.0;*/
            AddNewPoint(uvc);
        }
    }

    if (m_stienerpoints.size() == pointBefore)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void FaceMesh::AddNewPoint(std::array<NekDouble, 2> uv)
{
    // adds a new point but checks that there are no other points nearby first
    auto &graph = m_mesh->m_meshGraph;

    auto np           = m_cadsurf->P(uv);
    NekDouble npDelta = GetOctree(m_mesh, m_log).Query(np);

    // Candidate location, used only for the proximity tests. The vertex is
    // not created in the graph unless it is actually kept, so that a rejected
    // candidate leaves nothing behind.
    SpatialDomains::PointGeom candidate(3, -1, np[0], np[1], np[2]);

    bool add = true;

    for (int i = 0; i < orderedLoops.size(); i++)
    {
        for (int j = 0; j < orderedLoops[i].size(); j++)
        {
            NekDouble r = orderedLoops[i][j]->dist(candidate);

            if (r < npDelta / 2.0)
            {
                add = false;
                break;
            }
        }
    }

    if (add)
    {
        for (int i = 0; i < m_stienerpoints.size(); i++)
        {
            NekDouble r = m_stienerpoints[i]->dist(candidate);

            if (r < npDelta / 2.0)
            {
                add = false;
                break;
            }
        }
    }

    if (add)
    {
        int newId = NextPointId(graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, np[0], np[1], np[2]);

        SpatialDomains::PointGeom *n = pt.get();
        graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        graph->GetCADAssociation()->Add(n, {m_cadsurf, {uv[0], uv[1]}});
        m_stienerpoints.push_back(n);
    }
}

void FaceMesh::OrientateCurves()
{
    // this could be a second run on orentate so clear some info
    orderedLoops.clear();

    // create list of bounding loop nodes
    for (int i = 0; i < m_edgeloops.size(); i++)
    {
        vector<SpatialDomains::PointGeom *> cE;
        for (int j = 0; j < m_edgeloops[i]->edges.size(); j++)
        {
            int cid = m_edgeloops[i]->edges[j]->GetId();
            vector<SpatialDomains::PointGeom *> edgePoints =
                m_curvemeshes[cid]->GetMeshPoints();

            int numPoints = m_curvemeshes[cid]->GetNumPoints();

            if (m_edgeloops[i]->edgeo[j] ==
                SpatialDomains::CADOrientation::eForwards)
            {
                for (int k = 0; k < numPoints - 1; k++)
                {
                    cE.push_back(edgePoints[k]);
                }
            }
            else
            {
                for (int k = numPoints - 1; k > 0; k--)
                {
                    cE.push_back(edgePoints[k]);
                }
            }
        }
        orderedLoops.push_back(cE);
    }
}
} // namespace Nektar::NekMesh
