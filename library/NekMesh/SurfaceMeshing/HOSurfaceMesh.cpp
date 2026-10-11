////////////////////////////////////////////////////////////////////////////////
//
//  File: HOSurfaceMesh.cpp
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
//  Description: surfacemeshing object methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <list>

#include <NekMesh/Optimisation/BGFS-B.h>
#include <NekMesh/SurfaceMeshing/HOSurfaceMesh.h>
#include <NekMesh/SurfaceMeshing/OptimiseFunctions.h>
#include <SpatialDomains/CADSystem/CADCurve.h>
#include <SpatialDomains/CADSystem/CADSurf.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LocalRegions/MatrixKey.h>
#include <SpatialDomains/CADSystem/CADAssociation.h>

using namespace std;
namespace Nektar::NekMesh
{

ModuleKey HOSurfaceMesh::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "hosurface"), HOSurfaceMesh::create,
    "Generates a high-order surface mesh based on CAD");

HOSurfaceMesh::HOSurfaceMesh(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["no_opti"] =
        ConfigOption(false, "0", "Disable edge node optimisation.");
    m_config["order"] = ConfigOption(false, "4", "order for mesh curving.");
    m_config["third_party"] =
        ConfigOption(false, "0", "When third-party linear meshes are used.");
}

HOSurfaceMesh::~HOSurfaceMesh()
{
}

void HOSurfaceMesh::Process()
{
    // Order
    if (m_config["order"].beenSet)
    {
        m_order   = m_config["order"].as<int>();
        m_nummode = m_order + 1;
    }

    // The order used to be read from Mesh::m_nummode, which every caller had
    // to set beforehand. It is a module option now, so a pipeline assembled
    // by hand has to pass it: InputMCF registers it from the .mcf's Order
    // parameter. Without it m_nummode is still -1, which reaches the points
    // manager below as a request for -1 points and fails as a bad allocation.
    if (m_nummode < 2)
    {
        m_log(FATAL) << "High-order surface meshing needs an order: pass "
                     << "order=N to this module." << endl;
    }

    m_log(VERBOSE) << "High-order surface meshing order =  " << m_order << endl;
    LibUtilities::PointsKey ekey(m_nummode,
                                 LibUtilities::eGaussLobattoLegendre);
    Array<OneD, NekDouble> gll;

    LibUtilities::PointsManager()[ekey]->GetPoints(gll);

    LibUtilities::PointsKey pkey(m_nummode, LibUtilities::eNodalTriElec);
    LibUtilities::PointsKey qkey(m_nummode,
                                 LibUtilities::eGaussLobattoLegendre);

    Array<OneD, NekDouble> u, v;

    int cntBreak = 0;

    // N-quadrature points
    int nq = m_order + 1;
    // N all Triangular points
    int np = nq * (nq + 1) / 2;
    LibUtilities::PointsManager()[pkey]->GetPoints(u, v);

    // HO surface optimisation eneabled by default
    bool qOpti = !m_config["no_opti"].beenSet;

    // loop over all the faces in the surface mesh, check all three edges for
    // high order info, if nothing high-order the edge.

    EdgeMap surfaceEdges;
    EdgeMap completedEdges;

    for (auto &[elmt, tag] : m_mesh->m_elementTags[2])
    {
        for (int j = 0; j < elmt->GetNumEdges(); j++)
        {
            auto *edge =
                static_cast<SpatialDomains::SegGeom *>(elmt->GetEdge(j));
            int v0                 = edge->GetVertex(0)->GetGlobalID();
            int v1                 = edge->GetVertex(1)->GetGlobalID();
            surfaceEdges[{v0, v1}] = edge;
            if (edge->GetCurve() && edge->GetCurve()->m_points.size())
            {
                completedEdges[{v0, v1}] = edge;
            }
        }
    }
    m_log(VERBOSE) << "  - Edges already carrying curvature: "
                   << completedEdges.size() << endl;

    int i                           = 0;
    std::array<int, 4> m_quadVorder = {0, 1, 3, 2};
    for (auto &[element, tag] : m_mesh->m_elementTags[2])
    {
        auto *elem2D = static_cast<SpatialDomains::Geometry2D *>(element);

        m_log(VERBOSE).Progress(i++, m_mesh->m_elementTags[2].size(),
                                "    Surface elements");

        // Avoid the surface elements with no parentCAD
        if (!m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(element))
        {
            // no parent cad
            cntBreak++;
            continue;
        }

        SpatialDomains::CADSurfSharedPtr surf =
            m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(element);

        int ToBreak = 0;
        for (int j = 0; j < element->GetNumEdges(); j++)
        {
            SpatialDomains::SegGeom *edge =
                static_cast<SpatialDomains::SegGeom *>(element->GetEdge(j));
            // test insert the edge into completedEdges
            // if the edge already exists move on
            // if not figure out its high-order information
            SpatialDomains::PointGeom *v0 = edge->GetVertex(0);
            SpatialDomains::PointGeom *v1 = edge->GetVertex(1);

            if (completedEdges[{v0->GetGlobalID(), v1->GetGlobalID()}])
            {
                continue;
            }

            if (m_config["third_party"].beenSet &&
                !(m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge) ||
                  m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge)))
            {
                m_log(WARNING)
                    << "This edge does not have a CAD object "
                       "WARNING - CASE3 - so no edge / face projection"
                    << endl;
                ToBreak = 1;
                continue;
            }
            else
            {
                m_mesh->m_meshGraph->GetCADAssociation()->Set(edge, {surf});
            }

            // GetVertex Locations fors later assigning the edgeNodes
            NekDouble x, y, z, x1, y1, z1;
            v0->GetCoords(x, y, z);
            v1->GetCoords(x1, y1, z1);

            auto edgeCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    edge->GetGlobalID(),
                    LibUtilities::PointsManager()[ekey]->GetPointsType());

            if (m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge))
            {
                // the edge is on the CAD curve 1D optimisation(CASE2)
                SpatialDomains::CADCurveSharedPtr curveCAD =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge);
                NekDouble tb =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurveT(
                        v0, curveCAD->GetId());
                NekDouble te =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurveT(
                        v1, curveCAD->GetId());

                // distribute points along curve as inital guess
                Array<OneD, NekDouble> ti(m_nummode);
                for (int k = 0; k < m_nummode; k++)
                {
                    ti[k] =
                        tb * (1.0 - gll[k]) / 2.0 + te * (1.0 + gll[k]) / 2.0;
                }

                if (qOpti)
                {
                    std::vector<NekDouble> xi(ti.begin() + 1, ti.end() - 1);
                    std::vector<NekDouble> z(gll.begin(), gll.end());

                    auto result =
                        BGFSOptimise(CurveEdgeObjective(curveCAD, tb, te, z),
                                     xi, 1E-7, 2000);

                    // Deliberately take the result whether or not it converged;
                    // we assume this will be better than nothing, but it is
                    // something to consider changing later.
                    for (int k = 1; k < nq - 1; k++)
                    {
                        ti[k] = result.x[k - 1];
                    }

                    if (!result.converged)
                    {
                        m_log(TRACE) << "Failed to optimise on curve "
                                     << curveCAD->GetId() << ", norm "
                                     << result.norm << endl;
                    }
                }

                // KK - do we need the orientation here?
                // vector<pair<weak_ptr<CADSurf>, CADOrientation::Orientation>>
                // s =
                //     curveCAD->GetAdjSurf();
                // Each node is recorded against the CAD curve as it is made.
                // The parametric position is known exactly here, so this is
                // both cheaper and more accurate than leaving Mesh::MakeOrder
                // to recover it later with loct(). Without it the node is
                // indistinguishable from a free interior one, and
                // ProcessVarOpti would be at liberty to move it off the curve.
                auto &assoc = m_mesh->m_meshGraph->GetCADAssociation();

                auto addNode = [&](NekDouble t, NekDouble px, NekDouble py,
                                   NekDouble pz) {
                    auto *node =
                        m_mesh->m_meshGraph->CreateCurveNode(3, 0, px, py, pz);
                    assoc->Add(node,
                               SpatialDomains::CADLink(curveCAD, {t, 0.0}));
                    edgeCurve->m_points.push_back(node);
                };

                addNode(ti[0], x, y, z);
                for (int k = 1; k < m_nummode - 1; k++)
                {
                    auto loc = curveCAD->P(ti[k]);
                    addNode(ti[k], loc[0], loc[1], loc[2]);
                }
                addNode(ti[m_nummode - 1], x1, y1, z1);
            }
            else
            {
                // edge is on surface and needs 2D optimisation (CASE1)

                // To Do : Check if CADSurf is the same as surf (face CADSurf)
                // if not, then it is CASE3 (robustness !)
                std::array<NekDouble, 2> uvb, uve;

                if (m_config["third_party"].beenSet)
                {
                    // For the third-party case we re-evaluate the uv of the
                    // vertices, for robustness (cylinder bug in the past).
                    NekDouble dist0, dist1, Umin, Usup, Vmin, Vsup;
                    surf->GetBounds(Umin, Usup, Vmin, Vsup);

                    uvb = surf->locuv({x, y, z}, dist0, Umin, Usup, Vmin, Vsup);
                    uve = surf->locuv({x1, y1, z1}, dist1, Umin, Usup, Vmin,
                                      Vsup);
                    // This check is necessary if identification is not
                    // perfect, especially in periodic curves/surfaces
                    NekDouble tol = v0->dist(*v1) * 0.025;
                    if (dist0 > tol || dist1 > tol)
                    {
                        ToBreak = 1;
                        m_log(WARNING)
                            << "Edge vertices too far from CADSurf dist = "
                            << dist0 << " | " << dist1 << " tol = " << tol
                            << endl;
                        continue;
                    }
                }
                else
                {
                    uvb = m_mesh->m_meshGraph->GetCADAssociation()->GetSurfUV(
                        v0, surf->GetId());
                    uve = m_mesh->m_meshGraph->GetCADAssociation()->GetSurfUV(
                        v1, surf->GetId());
                }

                m_mesh->m_meshGraph->GetCADAssociation()->Set(edge, {surf});
                Array<OneD, std::array<NekDouble, 2>> uvi(nq);

                for (int k = 0; k < nq; k++)
                {
                    std::array<NekDouble, 2> uv;
                    uv[0] = uvb[0] * (1.0 - gll[k]) / 2.0 +
                            uve[0] * (1.0 + gll[k]) / 2.0;
                    uv[1] = uvb[1] * (1.0 - gll[k]) / 2.0 +
                            uve[1] * (1.0 + gll[k]) / 2.0;
                    uvi[k] = uv;
                }
                if (qOpti)
                {
                    std::vector<NekDouble> xi;
                    xi.reserve(2 * (nq - 2));
                    for (int k = 1; k < nq - 1; k++)
                    {
                        xi.push_back(uvi[k][0]);
                        xi.push_back(uvi[k][1]);
                    }
                    std::vector<NekDouble> z(gll.begin(), gll.end());

                    auto result = BGFSOptimise(
                        SurfaceEdgeObjective(surf, uvi[0], uvi[nq - 1], z), xi,
                        1E-8, 2000);

                    // As above: taken whether or not it converged.
                    for (int k = 1; k < nq - 1; k++)
                    {
                        uvi[k][0] = result.x[(k - 1) * 2 + 0];
                        uvi[k][1] = result.x[(k - 1) * 2 + 1];
                    }

                    if (!result.converged)
                    {
                        m_log(VERBOSE).Newline();
                        m_log(WARNING) << "  Failed to optimise on edge, "
                                       << "norm = " << result.norm << endl;
                    }
                }

                edgeCurve->m_points.push_back(
                    m_mesh->m_meshGraph->CreateCurveNode(3, 0, x, y, z));

                for (int k = 1; k < m_nummode - 1; k++)
                {
                    auto loc = surf->P(uvi[k]);
                    edgeCurve->m_points.push_back(
                        m_mesh->m_meshGraph->CreateCurveNode(3, 0, loc[0],
                                                             loc[1], loc[2]));
                }
                edgeCurve->m_points.push_back(
                    m_mesh->m_meshGraph->CreateCurveNode(3, 0, x1, y1, z1));
            }

            // Assign the curve to the EDGE ? - To Do - we shouldn't do both
            // SetCurve and std::move. Possibly move to
            // m_graph->SetEdgeCurve(edgeID , edgeCurvePtr )
            edge->SetCurve(edgeCurve.get());
            m_mesh->m_meshGraph->GetCurvedEdges()[edge->GetGlobalID()] =
                std::move(edgeCurve);

            completedEdges[{v0->GetGlobalID(), v1->GetGlobalID()}] = edge;
        }

        if (ToBreak == 1)
        {
            // if any edge is not projected or is already high-order
            // before hand
            // then likely the face is CASE3 (between 2 CAD surfaces)
            cntBreak++;
            continue;
        }

        // just add the face interior nodes through interp and project (no
        // optimization)

        element->Reset(m_mesh->m_meshGraph->GetCurvedEdges(),
                       m_mesh->m_meshGraph->GetCurvedFaces());
        element->FillGeom();
        StdRegions::StdExpansion *xmap = element->GetXmap();
        Array<OneD, NekDouble> coeffs0 = element->GetCoeffs(0);
        Array<OneD, NekDouble> coeffs1 = element->GetCoeffs(1);
        Array<OneD, NekDouble> coeffs2 = element->GetCoeffs(2);

        Array<OneD, NekDouble> xc(xmap->GetTotPoints());
        Array<OneD, NekDouble> yc(xmap->GetTotPoints());
        Array<OneD, NekDouble> zc(xmap->GetTotPoints());

        xmap->BwdTrans(coeffs0, xc);
        xmap->BwdTrans(coeffs1, yc);
        xmap->BwdTrans(coeffs2, zc);

        if (element->GetNumVerts() == 3 && m_order > 2)
        {
            // build an array of all uvs
            // KK the new version should start from 0 not np-ni
            vector<std::array<NekDouble, 2>> uvi;

            // These points all lie within one element, so each is a short step
            // from the one before and the search for its parameters can start
            // from that answer rather than from nothing. The first starts from
            // a vertex, whose parameters are already known.
            std::array<NekDouble, 2> guess =
                m_mesh->m_meshGraph->GetCADAssociation()->GetSurfUV(
                    element->GetVertex(0), surf->GetId());

            for (int j = 0; j < np; j++)
            {
                Array<OneD, NekDouble> xp(2);
                xp[0]                        = u[j];
                xp[1]                        = v[j];
                std::array<NekDouble, 3> loc = {xmap->PhysEvaluate(xp, xc),
                                                xmap->PhysEvaluate(xp, yc),
                                                xmap->PhysEvaluate(xp, zc)};

                NekDouble dist0;
                guess = surf->locuvFrom(loc, guess, dist0);
                uvi.push_back(guess);
            }

            auto faceCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    element->GetGlobalID(),
                    LibUtilities::PointsManager()[pkey]->GetPointsType());

            // Vertices (Reuse)
            for (int v = 0; v < element->GetNumVerts(); v++)
            {
                faceCurve->m_points.push_back(element->GetVertex(v));
            }

            // Edge nodes from the edges curve, orientation important
            // EdgeNodes after optimization are not corresponding to
            // FACENODES!!!
            // This means we have to reproject them and fill again the edge
            // curve
            int numEdgeNodes = 0;
            for (int e = 0; e < element->GetNumEdges(); e++)
            {
                auto ecurve   = element->GetEdge(e)->GetCurve();
                bool reversed = (element->GetEdge(e)->GetVertex(0) !=
                                 element->GetVertex(e));
                if (reversed)
                {
                    for (auto it = ecurve->m_points.rbegin() + 1;
                         it != ecurve->m_points.rend() - 1; ++it)
                    {
                        faceCurve->m_points.push_back(*it);
                    }
                }
                else
                {
                    for (auto it = ecurve->m_points.begin() + 1;
                         it != ecurve->m_points.end() - 1; ++it)
                    {
                        faceCurve->m_points.push_back(*it);
                    }
                }
                numEdgeNodes += ecurve->m_points.size() - 2;
            }

            // Continue with projections only if all edges have the correct
            // number
            if (numEdgeNodes != (nq - 2) * element->GetNumEdges())
            {
                m_log(WARNING)
                    << "Variable edges, meaning no facenodes projection = "
                    << numEdgeNodes << " " << (nq - 2) * element->GetNumEdges()
                    << endl;
                continue;
            }

            // Interior face nodes
            for (int k = 3 + numEdgeNodes; k < uvi.size(); k++)
            {
                auto locP = surf->P(uvi[k]);
                faceCurve->m_points.push_back(
                    m_mesh->m_meshGraph->CreateCurveNode(3, 0, locP[0], locP[1],
                                                         locP[2]));
            }

            elem2D->SetCurve(faceCurve.get());
            m_mesh->m_meshGraph->GetCurvedFaces()[element->GetGlobalID()] =
                std::move(faceCurve);
        }
        else if (element->GetNumVerts() == 4)
        {
            // build an array of all uvs
            vector<std::array<NekDouble, 2>> uvi;
            // If non Lobatto points are targetted the for loop should be
            // changed to avoid recreating the vertics
            for (int k = 0; k < nq; k++)
            {
                // z-alignment
                for (int j = 0; j < nq; j++)
                {
                    Array<OneD, NekDouble> xp(2);
                    xp[0]                        = gll[j];
                    xp[1]                        = gll[k];
                    std::array<NekDouble, 3> loc = {xmap->PhysEvaluate(xp, xc),
                                                    xmap->PhysEvaluate(xp, yc),
                                                    xmap->PhysEvaluate(xp, zc)};
                    uvi.push_back(surf->locuv(loc));
                }
            }

            auto faceCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    element->GetGlobalID(),
                    LibUtilities::PointsManager()[qkey]->GetPointsType());

            // Project all nodes to the CAD - not Vertices are REPROJECTED
            // again.
            // for (int j = 0; j < nq * nq; j++)
            for (int k = 0; k < nq; k++)
            {
                for (int j = 0; j < nq; j++)
                {
                    std::array<NekDouble, 3> loc;

                    if ((j == 0 || j == nq - 1) && (k == 0 || k == nq - 1))
                    {
                        // for vertex reuse the curved node
                        // int vid = (int[]){0, 1, 3,
                        //                   2}[(j == nq - 1) + 2 * (k == nq -
                        //                   1)];
                        int rawVID = (j == nq - 1) + 2 * (k == nq - 1);
                        int vid    = m_quadVorder[rawVID];
                        faceCurve->m_points.push_back(element->GetVertex(vid));
                    }
                    else
                    {

                        // for facenodes(also edge nodes) reproject them
                        // this might create a discrepency between edgenodes and
                        // facenodes on the edge
                        loc = surf->P(uvi[k * (nq) + j]);
                        faceCurve->m_points.push_back(
                            m_mesh->m_meshGraph->CreateCurveNode(
                                3, 0, loc[0], loc[1], loc[2]));
                    }
                }
            }
            elem2D->SetCurve(faceCurve.get());
            m_mesh->m_meshGraph->GetCurvedFaces()[element->GetGlobalID()] =
                std::move(faceCurve);
        }
    }
    // Curves were attached to surface edges and faces above, but only the
    // surface elements themselves were reset as they were curved. Every other
    // geometry sharing those edges -- the interior faces, and all the volume
    // elements -- is still set up for the linear mesh.
    m_mesh->m_meshGraph->ResetGeometry();

    m_log(VERBOSE) << "  - Node spreading optimisation: "
                   << (qOpti ? "on" : "off") << endl;

    // Not a warning in itself: a surface element with no CAD behind it has
    // nothing to curve onto, which is ordinary for a mesh that is only partly
    // generated from CAD. It is worth saying when it is most of them.
    if (cntBreak > 0)
    {
        auto &log = cntBreak * 2 > (int)m_mesh->m_elementTags[2].size()
                        ? m_log(WARNING)
                        : m_log(VERBOSE);

        log << "  - " << cntBreak << " of " << m_mesh->m_elementTags[2].size()
            << " surface elements were left straight, having no CAD surface "
               "to curve onto."
            << endl;
    }
}
} // namespace Nektar::NekMesh
