////////////////////////////////////////////////////////////////////////////////
//
//  File: TetMesh.cpp
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
//  Description: tet meshing methods
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/Octree/Octree.h>
#include <NekMesh/VolumeMeshing/TetMeshing/TetMesh.h>
#include <SpatialDomains/CADSystem/CADAssociation.h>

using namespace std;
namespace Nektar::NekMesh
{

void TetMesh::CheckManifold(
    const std::map<int, SpatialDomains::PointGeom *> &idToNode,
    const std::vector<std::array<int, 3>> &surfacetris)
{
    // TetGen needs a watertight manifold boundary: every edge of the surface
    // triangulation shared by exactly two triangles. Handed anything else it
    // aborts by throwing an int, which the Q flag it is called with suppresses
    // any explanation of, so the only thing the user sees is a core dump. The
    // arithmetic is cheap next to the meshing either side of it, so it is
    // always on.
    std::map<std::pair<int, int>, std::vector<int>> edgeUse;

    for (size_t i = 0; i < surfacetris.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int a = surfacetris[i][j], b = surfacetris[i][(j + 1) % 3];
            edgeUse[{std::min(a, b), std::max(a, b)}].push_back(i);
        }
    }

    std::vector<std::pair<std::pair<int, int>, std::vector<int>>> bad;
    for (auto &use : edgeUse)
    {
        if (use.second.size() != 2)
        {
            bad.push_back(use);
        }
    }

    if (bad.empty())
    {
        return;
    }

    auto &graph = m_mesh->m_meshGraph;

    // Report the geometry rather than the ids: the ids here are local to this
    // call and mean nothing outside it, whereas the coordinates and the CAD
    // surfaces are what the user has to go and look at.
    std::stringstream ss;
    ss << "the surface mesh is not a closed manifold: " << bad.size()
       << " of its " << edgeUse.size()
       << " edges are not shared by exactly two triangles, and the "
          "tetrahedral mesher requires a watertight manifold boundary.";

    const size_t maxReported = 10;
    for (size_t i = 0; i < std::min(bad.size(), maxReported); ++i)
    {
        std::array<NekDouble, 3> l0, l1;
        idToNode.at(bad[i].first.first)->GetCoords(l0[0], l0[1], l0[2]);
        idToNode.at(bad[i].first.second)->GetCoords(l1[0], l1[1], l1[2]);

        ss << std::endl
           << "    (" << l0[0] << ", " << l0[1] << ", " << l0[2] << ") -> ("
           << l1[0] << ", " << l1[1] << ", " << l1[2]
           << "): " << bad[i].second.size() << " triangles, on CAD surface";

        // The CAD surfaces carrying the triangles are the thing to go and
        // inspect, so name them. There is no CAD when the surface mesh was
        // read from a file rather than generated, hence the guard.
        std::set<int> surfs;
        for (int t : bad[i].second)
        {
            auto surf = graph->GetCADAssociation()->GetSurf(m_surface[t]);
            if (surf)
            {
                surfs.insert(surf->GetId());
            }
        }

        if (surfs.empty())
        {
            ss << "(es) unknown";
        }
        else
        {
            ss << (surfs.size() > 1 ? "s " : " ");
            for (auto it = surfs.begin(); it != surfs.end(); ++it)
            {
                ss << (it == surfs.begin() ? "" : ", ") << *it;
            }
        }
    }

    if (bad.size() > maxReported)
    {
        ss << std::endl
           << "    ... and " << bad.size() - maxReported << " more.";
    }

    m_log(FATAL) << ss.str() << endl;
}

void TetMesh::Mesh()
{
    auto &m_graph  = m_mesh->m_meshGraph;
    Octree &octree = GetOctree(m_mesh, m_log);

    vector<std::array<NekDouble, 3>> voidPts =
        m_graph->GetCAD()->GetVoidPoints();
    tetgen = MemoryManager<TetGenInterface>::AllocateSharedPtr(voidPts, m_log);

    map<int, SpatialDomains::PointGeom *> IdToNode;
    map<SpatialDomains::PointGeom *, int> IdToNodeRev;

    // build sequentially ordered maps of nodes that exist and there delta value
    // in the octree
    map<int, NekDouble> IdToDelta;
    vector<std::array<int, 3>> surfacetris;
    std::unordered_set<SpatialDomains::PointGeom *> alreadyInSurface;

    if (m_surface.size() == 0)
    {
        for (auto &[geom, tag] : m_mesh->m_elementTags[2])
        {
            m_surface.push_back(geom);
        }
    }

    int cnt = 0;
    for (int i = 0; i < m_surface.size(); i++)
    {
        std::array<int, 3> tri;
        for (int j = 0; j < m_surface[i]->GetNumVerts(); j++)
        {
            SpatialDomains::PointGeom *n = m_surface[i]->GetVertex(j);
            auto testIns                 = alreadyInSurface.insert(n);

            if (testIns.second)
            {
                std::array<NekDouble, 3> loc;
                n->GetCoords(loc[0], loc[1], loc[2]);

                tri[j]         = cnt;
                IdToNode[cnt]  = n;
                IdToNodeRev[n] = cnt;
                IdToDelta[cnt] = octree.Query(loc);
                cnt++;
            }
            else
            {
                tri[j] = IdToNodeRev[n];
            }
        }
        surfacetris.push_back(tri);
    }

    m_log(VERBOSE) << "  Initial node count: " << IdToNode.size() << endl;

    CheckManifold(IdToNode, surfacetris);

    tetgen->InitialMesh(IdToNode, surfacetris);

    vector<std::array<NekDouble, 3>> newp;
    int ctbefore = IdToNode.size();
    int newpb;
    do
    {
        newpb = newp.size();
        newp.clear();
        tetgen->GetNewPoints(ctbefore, newp);
        for (int i = 0; i < newp.size(); i++)
        {
            NekDouble d             = octree.Query(newp[i]);
            IdToDelta[ctbefore + i] = d;
        }
        tetgen->RefineMesh(IdToDelta);
    } while (newpb != newp.size());

    // make new map of all nodes to build tets.
    newp.clear();
    tetgen->GetNewPoints(ctbefore, newp);
    for (int i = 0; i < newp.size(); i++)
    {
        int newId = NextPointId(m_graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, newp[i][0], newp[i][1], newp[i][2]);

        SpatialDomains::PointGeom *n = pt.get();
        m_graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        IdToNode[ctbefore + i] = n;
    }

    // initial ids coming from the surface meshing
    std::unordered_set<int> naiveTriIDs;
    for (int i = 0; i < m_surface.size(); i++)
    {
        naiveTriIDs.insert(m_surface[i]->GetGlobalID());
    }

    m_tetconnect = tetgen->Extract();

    // create tets
    for (int i = 0; i < m_tetconnect.size(); i++)
    {
        vector<SpatialDomains::PointGeom *> n = {
            IdToNode[m_tetconnect[i][0]], IdToNode[m_tetconnect[i][1]],
            IdToNode[m_tetconnect[i][2]], IdToNode[m_tetconnect[i][3]]};
        ElmtConfig conf(LibUtilities::eTetrahedron, 1, false, false);

        SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
            LibUtilities::eTetrahedron, n, m_graph, m_mesh->m_edgeSet,
            m_mesh->m_faceSet, conf, nullptr, nullptr, &naiveTriIDs, nullptr);

        m_mesh->m_elementTags[3][E] = m_id;
    }

    m_log(VERBOSE) << "  Volume meshing complete: " << m_tetconnect.size()
                   << " tetrahedra generated." << endl;
}
} // namespace Nektar::NekMesh
