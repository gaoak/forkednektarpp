////////////////////////////////////////////////////////////////////////////////
//
//  File: CurveMesh.cpp
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
//  Description: curvemesh object methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/Octree/Octree.h>
#include <NekMesh/SurfaceMeshing/CurveMesh.h>

using namespace std;
namespace Nektar::NekMesh
{

SpatialDomains::PointGeom *CurveMesh::GetOrCreateCADVertPoint(
    SpatialDomains::CADVertSharedPtr cadVert)
{
    // CADVert::GetNode() is gone and MeshGraph only maps mesh->CAD, so the
    // mesh vertex sitting on a CAD vertex is tracked alongside the pipeline.
    // Curves meeting at a CAD vertex must share the vertex, otherwise the
    // curve meshes come out disconnected.
    auto &cadVertPoints = m_mesh->GetContext().Get<CADVertPoints>().points;

    auto search = cadVertPoints.find(cadVert->GetId());
    if (search != cadVertPoints.end())
    {
        return search->second;
    }

    auto &graph = m_mesh->m_meshGraph;
    auto loc    = cadVert->GetLoc();

    int newId = NextPointId(graph);
    auto pt   = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
        3, newId, loc[0], loc[1], loc[2]);

    SpatialDomains::PointGeom *ret = pt.get();
    graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));
    graph->GetCADAssociation()->Set(ret, {cadVert});
    cadVertPoints[cadVert->GetId()] = ret;

    return ret;
}

void CurveMesh::ReMesh()
{
    m_meshpoints.clear();
    m_dst.clear();
    m_ps.clear();
    meshsvalue.clear();
    for (int i = 0; i < m_meshedges.size(); i++)
    {
        // erase the edge in the edgeset
        m_mesh->m_edgeSet.erase(
            {m_meshedges[i]->GetVid(0), m_meshedges[i]->GetVid(1)});
        // erase the old geometry pointer from mesh graph
        m_mesh->m_meshGraph->ExtractGeom<SpatialDomains::SegGeom>(
            m_meshedges[i]->GetGlobalID(), true);
    }
    m_meshedges.clear();

    Mesh(true);
}

void CurveMesh::Mesh(bool forceThree)
{
    // this algorithm is mostly based on the work in chapter 19

    m_bounds      = m_cadcurve->GetBounds();
    m_curvelength = m_cadcurve->GetTotLength();
    m_numSamplePoints =
        int(m_curvelength / GetOctree(m_mesh, m_log).GetMinDelta()) + 10;
    ds = m_curvelength / (m_numSamplePoints - 1);

    // compute the offset due to adjacent BLs
    NekDouble totalOffset = 0.0;
    for (map<unsigned, NekDouble>::iterator ie = m_endoffset.begin();
         ie != m_endoffset.end(); ++ie)
    {
        totalOffset += ie->second;
    }
    ASSERTL0(m_curvelength > totalOffset,
             "Boundary layers too thick for adjacent curve");

    GetSampleFunction();

    Ae = 0.0;

    for (int i = 0; i < m_numSamplePoints - 1; i++)
    {
        Ae += ds * (1.0 / m_dst[i][0] + 1.0 / m_dst[i + 1][0]) / 2.0;
    }

    Ne = round(Ae);

    if (Ne + 1 < 2 + m_endoffset.size())
    {
        Ne = 1 + m_endoffset.size();

        meshsvalue.resize(Ne + 1);
        meshsvalue[0] = 0.0;
        meshsvalue[1] = m_curvelength;

        if (m_endoffset.count(0))
        {
            meshsvalue[1] = m_endoffset[0];
        }
        if (m_endoffset.count(1))
        {
            meshsvalue[Ne - 1] = m_curvelength - m_endoffset[1];
        }
    }
    else if (Ne + 1 == 2 && forceThree)
    {
        Ne++;
        meshsvalue.resize(Ne + 1);
        meshsvalue[0] = 0.0;
        meshsvalue[1] = m_curvelength / 2.0;
        meshsvalue[2] = m_curvelength;
    }
    else
    {

        GetPhiFunction();

        meshsvalue.resize(Ne + 1);
        meshsvalue[0]  = 0.0;
        meshsvalue[Ne] = m_curvelength;

        // force the second and/or the second to last point(s) if an offset is
        // defined
        if (m_endoffset.count(0))
        {
            meshsvalue[1] = m_endoffset[0];
        }
        if (m_endoffset.count(1))
        {
            meshsvalue[Ne - 1] = m_curvelength - m_endoffset[1];
        }

        for (int i = 1 + m_endoffset.count(0); i < Ne - m_endoffset.count(1);
             i++)
        {
            int iterationcounter = 0;
            bool iterate         = true;
            int k                = i;
            NekDouble ski        = meshsvalue[i - 1];
            NekDouble lastSki;
            while (iterate)
            {
                iterationcounter++;
                NekDouble rhs = EvaluateDS(ski) / Ae * (EvaluatePS(ski) - k);
                lastSki       = ski;
                ski           = ski - rhs;
                if (abs(lastSki - ski) < 1E-8)
                {
                    iterate = false;
                }

                ASSERTL0(iterationcounter < 1000000, "iteration failed");
            }

            meshsvalue[i] = ski;
        }
    }

    NekDouble t;
    std::array<NekDouble, 3> loc;

    auto &graph = m_mesh->m_meshGraph;

    vector<SpatialDomains::CADVertSharedPtr> verts = m_cadcurve->GetVertex();
    vector<pair<weak_ptr<SpatialDomains::CADSurf>,
                SpatialDomains::CADOrientation::Orientation>>
        s = m_cadcurve->GetAdjSurf();

    SpatialDomains::PointGeom *n = GetOrCreateCADVertPoint(verts[0]);
    t                            = m_bounds[0];
    graph->GetCADAssociation()->Add(n, {m_cadcurve, {t, 0.0}});
    n->GetCoords(loc[0], loc[1], loc[2]);
    for (int j = 0; j < s.size(); j++)
    {
        if (verts[0]->IsDegen() == s[j].first.lock()->GetId())
        {
            // locuv is ill-conditioned at a collapsed point, so take the
            // parametric location the CAD system recorded in SetDegen rather
            // than projecting.
            auto duv = verts[0]->GetDegenUV();
            graph->GetCADAssociation()->Add(
                n, {s[j].first.lock(), {duv[0], duv[1]}});
            continue;
        }

        auto uv = s[j].first.lock()->locuv(loc);
        graph->GetCADAssociation()->Add(n, {s[j].first.lock(), {uv[0], uv[1]}});
    }
    m_meshpoints.push_back(n);

    for (int i = 1; i < meshsvalue.size() - 1; i++)
    {
        t   = m_cadcurve->tAtArcLength(meshsvalue[i]);
        loc = m_cadcurve->P(t);

        int newId = NextPointId(graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, loc[0], loc[1], loc[2]);
        SpatialDomains::PointGeom *n2 = pt.get();
        graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        graph->GetCADAssociation()->Add(n2, {m_cadcurve, {t, 0.0}});
        for (int j = 0; j < s.size(); j++)
        {
            auto uv = s[j].first.lock()->locuv(loc);
            graph->GetCADAssociation()->Add(
                n2, {s[j].first.lock(), {uv[0], uv[1]}});
        }
        m_meshpoints.push_back(n2);
    }

    n = GetOrCreateCADVertPoint(verts[1]);
    t = m_bounds[1];
    graph->GetCADAssociation()->Add(n, {m_cadcurve, {t, 0.0}});
    n->GetCoords(loc[0], loc[1], loc[2]);
    for (int j = 0; j < s.size(); j++)
    {
        if (verts[1]->IsDegen() == s[j].first.lock()->GetId())
        {
            // as above: use the recorded degenerate parametric location
            auto duv = verts[1]->GetDegenUV();
            graph->GetCADAssociation()->Add(
                n, {s[j].first.lock(), {duv[0], duv[1]}});
            continue;
        }

        auto uv = s[j].first.lock()->locuv(loc);
        graph->GetCADAssociation()->Add(n, {s[j].first.lock(), {uv[0], uv[1]}});
    }
    m_meshpoints.push_back(n);

    ASSERTL0(Ne + 1 == m_meshpoints.size(),
             "incorrect number of points in curve mesh");

    // make edges and add them to the edgeset for the face mesher to use
    for (int i = 0; i < m_meshpoints.size() - 1; i++)
    {
        vector<SpatialDomains::PointGeom *> ns = {m_meshpoints[i],
                                                  m_meshpoints[i + 1]};

        auto *e = static_cast<SpatialDomains::SegGeom *>(
            CreateElementLite(LibUtilities::eSegment, ns, graph,
                              m_mesh->m_edgeSet, m_mesh->m_faceSet));

        graph->GetCADAssociation()->Set(e, {m_cadcurve});
        m_meshedges.push_back(e);
    }

    // Nuke progress bar
    m_log(VERBOSE).Overwrite();
    m_log(VERBOSE) << "    - Curve " << m_id << endl;
    m_log(VERBOSE) << "        Length       : " << scientific << m_curvelength
                   << endl;
    m_log(VERBOSE) << "        Nodes        : " << m_meshpoints.size() << endl;
    m_log(VERBOSE) << "        Sample points: " << m_numSamplePoints << endl;
}

void CurveMesh::GetPhiFunction()
{
    m_ps.resize(m_numSamplePoints);
    vector<NekDouble> newPhi;
    newPhi.resize(2);

    newPhi[0] = 0.0;
    newPhi[1] = 0.0;

    m_ps[0] = newPhi;

    NekDouble runningInt = 0.0;

    for (int i = 1; i < m_numSamplePoints; i++)
    {
        runningInt += (1.0 / m_dst[i - 1][0] + 1.0 / m_dst[i][0]) / 2.0 * ds;
        newPhi[0] = Ne / Ae * runningInt;
        newPhi[1] = m_dst[i][1];
        m_ps[i]   = newPhi;
    }
}

NekDouble CurveMesh::EvaluateDS(NekDouble s)
{
    int a = 0;
    int b = 0;

    ASSERTL1(!(s < 0) && !(s > m_curvelength), "s out of bounds");

    if (s == 0)
    {
        return m_dst[0][0];
    }
    else if (s == m_curvelength)
    {
        return m_dst[m_numSamplePoints - 1][0];
    }

    for (int i = 0; i < m_numSamplePoints - 1; i++)
    {
        if (m_dst[i][1] < s && m_dst[i + 1][1] >= s)
        {
            a = i;
            b = i + 1;
            break;
        }
    }

    NekDouble s1 = m_dst[a][1];
    NekDouble s2 = m_dst[b][1];
    NekDouble d1 = m_dst[a][0];
    NekDouble d2 = m_dst[b][0];

    NekDouble m = (d2 - d1) / (s2 - s1);
    NekDouble c = d2 - m * s2;

    ASSERTL0(m * s + c == m * s + c, "DS"); // was getting nans here

    return m * s + c;
}

NekDouble CurveMesh::EvaluatePS(NekDouble s)
{
    int a = 0;
    int b = 0;

    ASSERTL1(!(s < 0) && !(s > m_curvelength), "s out of bounds");

    if (s == 0)
    {
        return m_ps[0][0];
    }
    else if (s == m_curvelength)
    {
        return m_ps[m_numSamplePoints - 1][0];
    }

    for (int i = 0; i < m_numSamplePoints - 1; i++)
    {
        if (m_ps[i][1] < s && m_ps[i + 1][1] >= s)
        {
            a = i;
            b = i + 1;
            break;
        }
    }

    /*
    if (a == b)
    {
        exit(-1);
    }
    */

    NekDouble s1 = m_ps[a][1];
    NekDouble s2 = m_ps[b][1];
    NekDouble d1 = m_ps[a][0];
    NekDouble d2 = m_ps[b][0];

    NekDouble m = (d2 - d1) / (s2 - s1);
    NekDouble c = d2 - m * s2;

    ASSERTL0(m * s + c == m * s + c, "PS");

    return m * s + c;
}

void CurveMesh::GetSampleFunction()
{
    Octree &octree = GetOctree(m_mesh, m_log);

    m_dst.resize(m_numSamplePoints);

    vector<NekDouble> dsti;
    dsti.resize(3);

    for (int i = 0; i < m_numSamplePoints; i++)
    {
        dsti[1]     = i * ds;
        NekDouble t = m_cadcurve->tAtArcLength(dsti[1]);

        auto loc = m_cadcurve->P(t);

        bool found = false;

        // if inside the BL, dsti[0] set to the BL thickness, i.e. the offset
        if (m_endoffset.count(0))
        {
            if (dsti[1] < m_endoffset[0])
            {
                dsti[0] = m_endoffset[0];
                found   = true;
            }
        }
        if (m_endoffset.count(1) && !found)
        {
            if (dsti[1] > m_curvelength - m_endoffset[1])
            {
                dsti[0] = m_endoffset[1];
                found   = true;
            }
        }
        // else, dsti[0] is found from the octree
        if (!found)
        {
            dsti[0] = octree.Query(loc);
        }

        dsti[2] = t;

        m_dst[i] = dsti;
    }
}

void CurveMesh::PeriodicOverwrite(CurveMeshSharedPtr from)
{
    // clear current mesh points and remove edges from edgeset
    m_meshpoints.clear();
    for (int i = 0; i < m_meshedges.size(); i++)
    {
        m_mesh->m_edgeSet.erase(
            {m_meshedges[i]->GetVid(0), m_meshedges[i]->GetVid(1)});
    }
    m_meshedges.clear();

    ///////

    auto &graph = m_mesh->m_meshGraph;
    auto cad    = graph->GetCAD();

    int tid = from->GetId();
    auto T  = cad->GetPeriodicTranslationVector(tid, m_id);

    SpatialDomains::CADCurveSharedPtr c1 = cad->GetCurve(tid);

    bool reversed = c1->GetOrienationWRT(1) == m_cadcurve->GetOrienationWRT(1);

    vector<SpatialDomains::PointGeom *> nodes = from->GetMeshPoints();

    vector<pair<weak_ptr<SpatialDomains::CADSurf>,
                SpatialDomains::CADOrientation::Orientation>>
        surfs = m_cadcurve->GetAdjSurf();

    for (int i = 1; i < nodes.size() - 1; i++)
    {
        std::array<NekDouble, 3> loc;
        nodes[i]->GetCoords(loc[0], loc[1], loc[2]);

        int newId = NextPointId(graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, loc[0] + T[0], loc[1] + T[1], 0.0);
        SpatialDomains::PointGeom *nn = pt.get();
        graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        std::array<NekDouble, 3> nloc;
        nn->GetCoords(nloc[0], nloc[1], nloc[2]);

        for (int j = 0; j < surfs.size(); j++)
        {
            auto uv = surfs[j].first.lock()->locuv(nloc);
            graph->GetCADAssociation()->Add(
                nn, {surfs[j].first.lock(), {uv[0], uv[1]}});
        }

        NekDouble t;
        m_cadcurve->loct(nloc, t);
        graph->GetCADAssociation()->Add(nn, {m_cadcurve, {t, 0.0}});

        m_meshpoints.push_back(nn);
    }

    // Reverse internal nodes of the vector if necessary
    if (reversed)
    {
        reverse(m_meshpoints.begin(), m_meshpoints.end());
    }

    vector<SpatialDomains::CADVertSharedPtr> verts = m_cadcurve->GetVertex();

    m_meshpoints.insert(m_meshpoints.begin(),
                        GetOrCreateCADVertPoint(verts[0]));
    m_meshpoints.push_back(GetOrCreateCADVertPoint(verts[1]));
    // dont need to realign cad for vertices

    // make edges and add to the EdgeSet for the face mesher
    for (int i = 0; i < m_meshpoints.size() - 1; i++)
    {
        vector<SpatialDomains::PointGeom *> ns = {m_meshpoints[i],
                                                  m_meshpoints[i + 1]};

        auto *e = static_cast<SpatialDomains::SegGeom *>(
            CreateElementLite(LibUtilities::eSegment, ns, graph,
                              m_mesh->m_edgeSet, m_mesh->m_faceSet));

        graph->GetCADAssociation()->Set(e, {m_cadcurve});
        m_meshedges.push_back(e);
    }
}
} // namespace Nektar::NekMesh
