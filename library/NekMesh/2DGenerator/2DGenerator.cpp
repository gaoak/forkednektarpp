////////////////////////////////////////////////////////////////////////////////
//
//  File: 2DGenerator.cpp
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
//  Description: 2D generator object methods.
//
////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cmath>

#include <NekMesh/2DGenerator/2DGenerator.h>

#include <NekMesh/Octree/Octree.h>

#include <LibUtilities/BasicUtils/ParseUtils.h>

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <boost/algorithm/string.hpp>

using namespace std;
namespace Nektar::NekMesh
{

ModuleKey Generator2D::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "2dgenerator"), Generator2D::create,
    "Generates a 2D mesh");

Generator2D::Generator2D(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["blcurves"] =
        ConfigOption(false, "", "Generate parallelograms on these curves");
    m_config["blthick"] =
        ConfigOption(false, "0.0", "Parallelogram layer thickness");
    m_config["periodic"] =
        ConfigOption(false, "", "Set of pairs of periodic curves");
    m_config["bltadjust"] =
        ConfigOption(false, "2.0", "Boundary layer thickness adjustment");
    m_config["adjustblteverywhere"] =
        ConfigOption(true, "0", "Adjust thickness everywhere");
    m_config["smoothbl"] =
        ConfigOption(true, "0",
                     "Smooth the BL normal directions to avoid "
                     "(nearly) intersecting normals");
    m_config["spaceoutbl"] = ConfigOption(
        false, "0.5", "Threshold to space out BL according to Delta");
    m_config["nospaceoutsurf"] =
        ConfigOption(false, "", "Surfaces where spacing out shouldn't be used");
}

Generator2D::~Generator2D()
{
}

void Generator2D::Process()
{
    auto cad = m_mesh->m_meshGraph->GetCAD();

    // Check that cad is 2D
    auto bndBox = cad->GetBoundingBox();

    if (fabs(bndBox[5] - bndBox[4]) > 1.0e-7)
    {
        m_log(FATAL) << "Specified CAD file is not a 2D geometry." << endl;
    }

    m_log(VERBOSE) << "Beginning 2D mesh generation:" << endl;
    m_log(VERBOSE) << "  Curve meshing:" << endl;

    m_thickness_ID =
        m_thickness.DefineFunction("x y z", m_config["blthick"].as<string>());
    ParseUtils::GenerateSeqVector(m_config["blcurves"].as<string>(),
                                  m_blCurves);

    // find the ends of the BL curves
    if (m_config["blcurves"].beenSet)
    {
        FindBLEnds();
    }

    // linear mesh all curves
    for (int i = 1; i <= cad->GetNumCurve(); i++)
    {
        m_log(VERBOSE).Progress(i, cad->GetNumCurve(), "Curve progress");

        vector<unsigned int>::iterator f =
            find(m_blCurves.begin(), m_blCurves.end(), i);
        if (f == m_blCurves.end())
        {
            m_curvemeshes[i] =
                MemoryManager<CurveMesh>::AllocateSharedPtr(i, m_mesh, m_log);

            // Fheck if this curve is at an end of the BL
            // If so, define an offset for the second node, corresponding to the
            // BL thickness
            if (m_blends.count(i))
            {
                vector<SpatialDomains::CADVertSharedPtr> vertices =
                    cad->GetCurve(i)->GetVertex();
                std::array<NekDouble, 3> loc;
                NekDouble t;

                // offset needed at first node (or both)
                if (m_blends[i] == 0 || m_blends[i] == 2)
                {
                    loc = vertices[0]->GetLoc();
                    t   = m_thickness.Evaluate(m_thickness_ID, loc[0], loc[1],
                                               loc[2], 0.0);
                    m_curvemeshes[i]->SetOffset(0, t);
                }
                // offset needed at second node (or both)
                if (m_blends[i] == 1 || m_blends[i] == 2)
                {
                    loc = vertices[1]->GetLoc();
                    t   = m_thickness.Evaluate(m_thickness_ID, loc[0], loc[1],
                                               loc[2], 0.0);
                    m_curvemeshes[i]->SetOffset(1, t);
                }
            }
        }
        else
        {
            m_curvemeshes[i] = MemoryManager<CurveMesh>::AllocateSharedPtr(
                i, m_mesh, m_log, m_config["blthick"].as<string>());
        }
        m_curvemeshes[i]->Mesh();
    }

    ////////
    // consider periodic curves

    if (m_config["periodic"].beenSet)
    {
        PeriodicPrep();
        MakePeriodic();
    }

    ////////////////////////////////////////

    if (m_config["blcurves"].beenSet)
    {
        // we need to do the boundary layer generation in a face by face basis
        MakeBLPrep();
        for (int i = 1; i <= cad->GetNumSurf(); i++)
        {
            MakeBL(i);
        }

        // If the BL doesn't form closed loops, we need to remove the outside
        // nodes from the curve meshes
        for (auto &ic : m_blends)
        {
            vector<SpatialDomains::PointGeom *> nodes =
                m_curvemeshes[ic.first]->GetMeshPoints();

            if (ic.second == 0 || ic.second == 2)
            {
                nodes.erase(nodes.begin());
            }
            if (ic.second == 1 || ic.second == 2)
            {
                nodes.erase(nodes.end() - 1);
            }

            // Rebuild the curvemesh without the first node, the last node or
            // both
            m_curvemeshes[ic.first] =
                MemoryManager<CurveMesh>::AllocateSharedPtr(ic.first, m_mesh,
                                                            nodes, m_log);
        }

        m_log(VERBOSE) << "    Boundary layer meshing complete." << endl;
    }

    m_log(VERBOSE) << "  Face meshing:" << endl;

    // linear mesh all surfaces
    for (int i = 1; i <= cad->GetNumSurf(); i++)
    {
        m_log(VERBOSE).Progress(i, cad->GetNumSurf(), "Face progress");
        m_facemeshes[i] = MemoryManager<FaceMesh>::AllocateSharedPtr(
            i, m_mesh, m_curvemeshes, 99 + i, m_log);

        // Check the curve meshes bounding this surface do not cross in the
        // parameter plane before handing to a triangle, which would
        // otherwise fail later with a node index error.
        if (m_facemeshes[i]->ValidateCurves())
        {
            m_log(WARNING) << "Curve meshes intersect on CAD surface " << i
                           << endl;
        }

        m_facemeshes[i]->Mesh();
    }
    ////////////////////////////////////

    // 1D boundary elemen -SegGeom itself, aprt of m_elementTags[1] with id of
    // the CADcurve
    for (auto &[curveId, curveMesh] : m_curvemeshes)
    {
        for (auto &edge : curveMesh->GetMeshEdges())
        {
            m_mesh->m_elementTags[1][edge] = curveId;
        }
    }

    // BL curve meshes were replaced by their outer nodes and carry no edges,
    // so their boundary comes from the original edges kept in m_blEdges
    for (auto &edge : m_blEdges)
    {
        m_mesh->m_elementTags[1][edge] =
            m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge)->GetId();
    }

    ProcessElements();
    ProcessComposites();
    Report();
}

void Generator2D::FindBLEnds()
{
    // Set of CAD vertices
    // Vertices of each curve are added to the set if not found and removed from
    // the set if found
    // This leaves us with a set of vertices that are at the end of BL open
    // loops
    set<SpatialDomains::CADVertSharedPtr> cadverts;

    for (auto &it : m_blCurves)
    {
        vector<SpatialDomains::CADVertSharedPtr> vertices =
            m_mesh->m_meshGraph->GetCAD()->GetCurve(it)->GetVertex();

        for (auto &iv : vertices)
        {
            set<SpatialDomains::CADVertSharedPtr>::iterator is =
                cadverts.find(iv);

            if (is != cadverts.end())
            {
                cadverts.erase(is);
            }
            else
            {
                cadverts.insert(iv);
            }
        }
    }

    // Build m_blends based on the previously constructed set of vertices
    // m_blends is a map of curve number (the curves right outside the BL open
    // loops) to the offset node number: 0, 1 or 2 (for both)
    for (int i = 1; i <= m_mesh->m_meshGraph->GetCAD()->GetNumCurve(); ++i)
    {
        if (find(m_blCurves.begin(), m_blCurves.end(), i) != m_blCurves.end())
        {
            continue;
        }

        vector<SpatialDomains::CADVertSharedPtr> vertices =
            m_mesh->m_meshGraph->GetCAD()->GetCurve(i)->GetVertex();

        for (int j = 0; j < 2; ++j)
        {
            if (!cadverts.count(vertices[j]))
            {
                continue;
            }

            if (m_blends.count(i))
            {
                m_blends[i] = 2;
            }
            else
            {
                m_blends[i] = j;
            }
        }
    }
}

void Generator2D::MakeBLPrep()
{
    m_log(VERBOSE) << "  Boundary layer meshing:" << endl;

    // identify the nodes and edges which will become the boundary layer.
    for (auto &it : m_blCurves)
    {
        vector<SpatialDomains::SegGeom *> localedges =
            m_curvemeshes[it]->GetMeshEdges();
        for (auto &ie : localedges)
        {
            m_blEdges.push_back(ie);
            m_nodesToEdge[ie->GetVertex(0)].push_back(ie);
            m_nodesToEdge[ie->GetVertex(1)].push_back(ie);
        }
    }
}

void Generator2D::MakeBL(int faceid)
{
    auto &m_graph  = m_mesh->m_meshGraph;
    Octree &octree = GetOctree(m_mesh, m_log);

    std::unordered_map<SpatialDomains::SegGeom *, std::array<NekDouble, 2>>
        edgeNormals;

    for (auto &it : m_blCurves)
    {
        SpatialDomains::CADOrientation::Orientation edgeo =
            m_graph->GetCAD()->GetCurve(it)->GetOrienationWRT(faceid);
        vector<SpatialDomains::SegGeom *> es =
            m_curvemeshes[it]->GetMeshEdges();
        // on each !!!EDGE!!! calculate a normal
        // always to the left unless edgeo is 1
        // normal must be done in the parametric space (and then projected back)
        // because of face orientation
        for (auto &ie : es)
        {
            auto p1 = m_graph->GetCADAssociation()->GetSurfUV(ie->GetVertex(0),
                                                              faceid);
            auto p2 = m_graph->GetCADAssociation()->GetSurfUV(ie->GetVertex(1),
                                                              faceid);

            std::array<NekDouble, 2> n = {p1[1] - p2[1], p2[0] - p1[0]};
            if (edgeo == SpatialDomains::CADOrientation::eBackwards)
            {
                n[0] *= -1.0;
                n[1] *= -1.0;
            }
            NekDouble mag = sqrt(n[0] * n[0] + n[1] * n[1]);
            n[0] /= mag;
            n[1] /= mag;

            std::array<NekDouble, 2> np = {p1[0] + n[0], p1[1] + n[1]};

            std::array<NekDouble, 3> loc;
            ie->GetVertex(0)->GetCoords(loc[0], loc[1], loc[2]);
            auto locp = m_graph->GetCAD()->GetSurf(faceid)->P(np);
            n[0]      = locp[0] - loc[0];
            n[1]      = locp[1] - loc[1];
            mag       = sqrt(n[0] * n[0] + n[1] * n[1]);
            n[0] /= mag;
            n[1] /= mag;
            edgeNormals[ie] = n;
        }
    }

    bool adjust           = m_config["bltadjust"].beenSet;
    NekDouble divider     = m_config["bltadjust"].as<NekDouble>();
    bool adjustEverywhere = m_config["adjustblteverywhere"].beenSet;
    bool smoothbl         = m_config["smoothbl"].beenSet;
    bool spaceoutbl       = m_config["spaceoutbl"].beenSet;
    NekDouble spaceoutthr = m_config["spaceoutbl"].as<NekDouble>();

    if (divider < 2.0)
    {
        m_log(WARNING) << "    BndLayerAdjustment too low, corrected to 2.0"
                       << endl;
        divider = 2.0;
    }

    std::unordered_map<SpatialDomains::PointGeom *, SpatialDomains::PointGeom *>
        nodeNormals;
    std::unordered_set<SpatialDomains::PointGeom *> createdNormals;
    for (auto &it : m_nodesToEdge)
    {
        if (it.second.size() != 1 && it.second.size() != 2)
        {
            m_log(FATAL) << "    Error with identifying nodes with edges: check"
                         << " that your boundary layer surfaces are correctly"
                         << " defined." << endl;
        }

        // If node at the end of the BL open loop, the "normal node" isn't
        // constructed by computing a normal but found on the adjacent curve
        if (it.second.size() == 1)
        {
            auto curves = m_graph->GetCADAssociation()->GetLinks(
                it.first, SpatialDomains::CADType::eCurve);

            vector<SpatialDomains::SegGeom *> edges =
                m_curvemeshes[curves[0].Id()]->GetMeshEdges();
            vector<SpatialDomains::SegGeom *>::iterator ie =
                find(edges.begin(), edges.end(), it.second[0]);
            int rightCurve =
                (ie == edges.end()) ? curves[0].Id() : curves[1].Id();

            vector<SpatialDomains::PointGeom *> nodes =
                m_curvemeshes[rightCurve]->GetMeshPoints();
            nodeNormals[it.first] =
                (nodes[0] == it.first) ? nodes[1] : nodes[nodes.size() - 2];

            continue;
        }

        std::array<NekDouble, 3> n = {0.0, 0.0, 0.0};
        auto n1                    = edgeNormals[it.second[0]];
        auto n2                    = edgeNormals[it.second[1]];
        n[0]                       = (n1[0] + n2[0]) / 2.0;
        n[1]                       = (n1[1] + n2[1]) / 2.0;
        NekDouble mag              = sqrt(n[0] * n[0] + n[1] * n[1]);
        n[0] /= mag;
        n[1] /= mag;
        NekDouble t = m_thickness.Evaluate(m_thickness_ID, it.first->x(),
                                           it.first->y(), 0.0, 0.0);
        // Adjust thickness according to angle between normals
        if (adjust)
        {
            if (adjustEverywhere ||
                m_graph->GetCADAssociation()->Count(
                    it.first, SpatialDomains::CADType::eCurve) > 1)
            {
                NekDouble angle = acos(n1[0] * n2[0] + n1[1] * n2[1]);
                angle           = (angle > M_PI) ? 2 * M_PI - angle : angle;
                t /= cos(angle / divider);
            }
        }

        n[0] = n[0] * t + it.first->x();
        n[1] = n[1] * t + it.first->y();

        int newId = NextPointId(m_graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, n[0], n[1], 0.0);
        SpatialDomains::PointGeom *nn = pt.get();
        m_graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        SpatialDomains::CADSurfSharedPtr s = m_graph->GetCAD()->GetSurf(faceid);
        auto uv                            = s->locuv(n);
        m_graph->GetCADAssociation()->Add(nn, {s, {uv[0], uv[1]}});
        nodeNormals[it.first] = nn;
    }

    // Check for any intersecting boundary layer normals and smooth them if
    // needed
    if (smoothbl)
    {
        // Nodes that need normal smoothing and their unit normal. Ordered
        // by node ID, not by address: smoothing walks these and moves each
        // node, so the order decides the result.
        map<SpatialDomains::PointGeom *, vector<std::array<NekDouble, 3>>,
            GeometryPtrIdLess>
            unitNormals;
        // Nodes that need normal smoothing and their BL thickness
        map<SpatialDomains::PointGeom *, NekDouble, GeometryPtrIdLess> dist;

        int count = 0;

        do
        {
            unitNormals.clear();
            dist.clear();

            for (const auto &it : m_blEdges)
            {
                // Line intersection based on
                // https://stackoverflow.com/a/565282/7241595
                SpatialDomains::PointGeom *p = it->GetVertex(0);
                SpatialDomains::PointGeom *q = it->GetVertex(1);

                std::array<NekDouble, 3> r = {nodeNormals[p]->x() - p->x(),
                                              nodeNormals[p]->y() - p->y(),
                                              nodeNormals[p]->z() - p->z()};
                std::array<NekDouble, 3> s = {nodeNormals[q]->x() - q->x(),
                                              nodeNormals[q]->y() - q->y(),
                                              nodeNormals[q]->z() - q->z()};

                NekDouble d = r[0] * s[1] - r[1] * s[0];

                // Should probably use tolerance to check parallelism
                if (d == 0)
                {
                    continue;
                }

                std::array<NekDouble, 3> qp = {q->x() - p->x(), q->y() - p->y(),
                                               q->z() - p->z()};

                NekDouble t = (qp[0] * s[1] - qp[1] * s[0]) / d;
                NekDouble u = (qp[0] * r[1] - qp[1] * r[0]) / d;

                // Check for intersection of the infinite continuation of one
                // normal with the other. A tolerance of 0.5 times the length of
                // thenormalis used. Could maybe be decreased to a less
                // aggressive value.
                if ((-0.5 < t && t <= 1.5) || (-0.5 < u && u <= 1.5))
                {
                    dist[p] = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                    dist[q] = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);

                    std::array<NekDouble, 3> sum = {
                        r[0] / dist[p] + s[0] / dist[q],
                        r[1] / dist[p] + s[1] / dist[q],
                        r[2] / dist[p] + s[2] / dist[q]};

                    unitNormals[p].push_back(sum);
                    unitNormals[q].push_back(sum);
                }
            }

            // Smooth each normal one by one
            for (const auto &it : unitNormals)
            {
                std::array<NekDouble, 3> avg = {0.0, 0.0, 0.0};

                for (const auto &i : it.second)
                {
                    avg[0] += i[0];
                    avg[1] += i[1];
                    avg[2] += i[2];
                }

                NekDouble mag =
                    sqrt(avg[0] * avg[0] + avg[1] * avg[1] + avg[2] * avg[2]);
                avg[0] /= mag;
                avg[1] /= mag;
                avg[2] /= mag;

                // Move the BL node onto the smoothed normal
                SpatialDomains::PointGeom *nn = nodeNormals[it.first];
                std::array<NekDouble, 3> loc  = {
                    it.first->x() + avg[0] * dist[it.first],
                    it.first->y() + avg[1] * dist[it.first], 0.0};
                nn->UpdatePosition(loc[0], loc[1], loc[2]);

                SpatialDomains::CADSurfSharedPtr s =
                    m_graph->GetCADAssociation()->GetSurf(nn);
                auto uv = s->locuv(loc);
                m_graph->GetCADAssociation()->Add(nn, {s, {uv[0], uv[1]}});

                nodeNormals[it.first] = nn;
            }
        } while (unitNormals.size() && count++ < 50);

        if (count < 50)
        {
            m_log(VERBOSE) << "    Normals smoothed in " << count
                           << " iterations." << endl;
        }
        else
        {
            m_log(WARNING) << "    Normals smoothed. Algorithm didn't "
                           << "converge after " << count << " iterations."
                           << endl;
        }
    }

    // Space out the outter BL nodes to better fit the local required Delta
    if (spaceoutbl)
    {
        if (spaceoutthr < 0.0 || spaceoutthr > 1.0)
        {
            m_log(WARNING) << "  The boundary layer space out threshold should "
                           << "be between 0 and 1. It will now be adjusted to "
                           << "0.5." << endl;
            spaceoutthr = 0.5;
        }

        vector<unsigned int> nospaceoutsurf;
        ParseUtils::GenerateSeqVector(m_config["nospaceoutsurf"].as<string>(),
                                      nospaceoutsurf);

        // List of connected nodes at need spacing out
        vector<deque<SpatialDomains::PointGeom *>> nodesToMove;

        int count = 0;

        // This will supposedly spread the number of nodes to be moved until
        // sufficient space is found
        do
        {
            nodesToMove.clear();

            // Find which nodes need to be spaced out
            for (const auto &ie : m_blEdges)
            {
                auto it =
                    find(nospaceoutsurf.begin(), nospaceoutsurf.end(),
                         m_graph->GetCADAssociation()->GetCurve(ie)->GetId());
                if (it != nospaceoutsurf.end())
                {
                    continue;
                }

                SpatialDomains::PointGeom *n1 = nodeNormals[ie->GetVertex(0)];
                SpatialDomains::PointGeom *n2 = nodeNormals[ie->GetVertex(1)];

                std::array<NekDouble, 3> mid = {(n1->x() + n2->x()) / 2.0,
                                                (n1->y() + n2->y()) / 2.0,
                                                (n1->z() + n2->z()) / 2.0};

                NekDouble targetD = octree.Query(mid);
                NekDouble realD   = n1->dist(*n2);

                // Add nodes if condition fulfilled
                if (realD < spaceoutthr * targetD)
                {
                    bool connected = false;

                    for (auto &il : nodesToMove)
                    {
                        if (il.front() == n1)
                        {
                            il.push_front(n2);
                            connected = true;
                            break;
                        }
                        if (il.front() == n2)
                        {
                            il.push_front(n1);
                            connected = true;
                            break;
                        }
                        if (il.back() == n1)
                        {
                            il.push_back(n2);
                            connected = true;
                            break;
                        }
                        if (il.back() == n2)
                        {
                            il.push_back(n1);
                            connected = true;
                            break;
                        }
                    }

                    // Create new set of connected nodes if necessary
                    if (!connected)
                    {
                        deque<SpatialDomains::PointGeom *> newList;
                        newList.push_back(n1);
                        newList.push_back(n2);

                        nodesToMove.push_back(newList);
                    }
                }
            }

            for (int i = 0;; ++i)
            {
                // Reconnect sets of connected nodes together if need be. Done
                // once before ad once after expanding the set by one node to
                // find extra space.
                for (int i1 = 0; i1 < nodesToMove.size(); ++i1)
                {
                    SpatialDomains::PointGeom *n11 = nodesToMove[i1].front();
                    SpatialDomains::PointGeom *n12 = nodesToMove[i1].back();

                    for (int i2 = i1 + 1; i2 < nodesToMove.size(); ++i2)
                    {
                        SpatialDomains::PointGeom *n21 =
                            nodesToMove[i2].front();
                        SpatialDomains::PointGeom *n22 = nodesToMove[i2].back();

                        if (n11 == n21 || n11 == n22 || n12 == n21 ||
                            n12 == n22)
                        {
                            if (n11 == n21 || n12 == n22)
                            {
                                reverse(nodesToMove[i2].begin(),
                                        nodesToMove[i2].end());
                                n21 = nodesToMove[i2].front();
                                n22 = nodesToMove[i2].back();
                            }

                            if (n11 == n22)
                            {
                                nodesToMove[i1].insert(nodesToMove[i1].begin(),
                                                       nodesToMove[i2].begin(),
                                                       nodesToMove[i2].end() -
                                                           1);
                            }
                            else
                            {
                                nodesToMove[i1].insert(nodesToMove[i1].end(),
                                                       nodesToMove[i2].begin() +
                                                           1,
                                                       nodesToMove[i2].end());
                            }

                            nodesToMove.erase(nodesToMove.begin() + i2);
                            continue;
                        }
                    }
                }

                if (i >= 1)
                {
                    break;
                }

                set<SpatialDomains::SegGeom *> addedEdges;

                // Expand each set of connected nodes by one node to allow for
                // extra space
                for (auto &il : nodesToMove)
                {
                    SpatialDomains::PointGeom *n11 = *(il.begin() + 0);
                    SpatialDomains::PointGeom *n12 = *(il.begin() + 1);

                    SpatialDomains::PointGeom *n13 = *(il.rbegin() + 1);
                    SpatialDomains::PointGeom *n14 = *(il.rbegin() + 0);

                    for (const auto &ie : m_blEdges)
                    {
                        auto it =
                            find(nospaceoutsurf.begin(), nospaceoutsurf.end(),
                                 m_graph->GetCADAssociation()
                                     ->GetCurve(ie)
                                     ->GetId());
                        if (addedEdges.count(ie) || it != nospaceoutsurf.end())
                        {
                            continue;
                        }

                        SpatialDomains::PointGeom *n21 =
                            nodeNormals[ie->GetVertex(0)];
                        SpatialDomains::PointGeom *n22 =
                            nodeNormals[ie->GetVertex(1)];

                        SpatialDomains::PointGeom *frontPush = nullptr;
                        SpatialDomains::PointGeom *backPush  = nullptr;

                        if (n11)
                        {
                            if (n11 == n21 && n12 != n22)
                            {
                                frontPush = n22;
                            }
                            else if (n11 == n22 && n12 != n21)
                            {
                                frontPush = n21;
                            }

                            if (frontPush)
                            {
                                il.push_front(frontPush);
                                n11 = nullptr;
                                addedEdges.insert(ie);
                            }
                        }
                        if (n14 && !frontPush)
                        {
                            if (n14 == n21 && n13 != n22)
                            {
                                backPush = n22;
                            }
                            if (n14 == n22 && n13 != n21)
                            {
                                backPush = n21;
                            }

                            if (backPush)
                            {
                                il.push_back(backPush);
                                n14 = nullptr;
                                addedEdges.insert(ie);
                            }
                        }

                        if (!n11 && !n14)
                        {
                            break;
                        }
                    }
                }
            }

            // Actual spacing out of the nodes. Done by simple linear
            // interpolation between the 2 end nodes. Weights come from the
            // required Delta or each edge.
            for (const auto &il : nodesToMove)
            {
                SpatialDomains::PointGeom *ni = il.front();
                SpatialDomains::PointGeom *nf = il.back();

                vector<NekDouble> deltas;
                NekDouble total = 0.0;

                for (int i = 0; i < il.size() - 1; ++i)
                {
                    SpatialDomains::PointGeom *n1 = il[i];
                    SpatialDomains::PointGeom *n2 = il[i + 1];

                    std::array<NekDouble, 3> mid = {(n1->x() + n2->x()) / 2.0,
                                                    (n1->y() + n2->y()) / 2.0,
                                                    (n1->z() + n2->z()) / 2.0};

                    deltas.push_back(octree.Query(mid));
                    total += deltas.back();
                }

                for (auto &id : deltas)
                {
                    id /= total;
                }

                NekDouble runningTotal = 0.0;

                for (int i = 1; i < il.size() - 1; ++i)
                {
                    runningTotal += deltas[i - 1];
                    std::array<NekDouble, 3> loc = {
                        ni->x() * (1.0 - runningTotal) + nf->x() * runningTotal,
                        ni->y() * (1.0 - runningTotal) + nf->y() * runningTotal,
                        ni->z() * (1.0 - runningTotal) +
                            nf->z() * runningTotal};

                    SpatialDomains::CADSurfSharedPtr s =
                        m_graph->GetCAD()->GetSurf(faceid);
                    auto uv = s->locuv(loc);

                    il[i]->UpdatePosition(loc[0], loc[1], loc[2]);
                    m_graph->GetCADAssociation()->Add(il[i],
                                                      {s, {uv[0], uv[1]}});
                }
            }
        } while (nodesToMove.size() && count++ < 50);

        if (count < 50)
        {
            m_log(VERBOSE) << "    BL spaced out in " << count << " iterations."
                           << endl;
        }
        else
        {
            m_log(WARNING) << "    BL spaced out. Algorithm didn't "
                           << "converge after " << count << " iterations."
                           << endl;
        }
    }
    m_log(VERBOSE) << "   N (BLcurves) = " << m_blCurves.size() << endl;
    for (auto &it : m_blCurves)
    {
        SpatialDomains::CADOrientation::Orientation edgeo =
            m_graph->GetCAD()->GetCurve(it)->GetOrienationWRT(faceid);
        vector<SpatialDomains::PointGeom *> ns =
            m_curvemeshes[it]->GetMeshPoints();
        vector<SpatialDomains::PointGeom *> newNs;
        for (auto &in : ns)
        {
            newNs.push_back(nodeNormals[in]);
        }

        m_curvemeshes[it] = MemoryManager<CurveMesh>::AllocateSharedPtr(
            it, m_mesh, newNs, m_log);

        if (edgeo == SpatialDomains::CADOrientation::eBackwards)
        {
            reverse(ns.begin(), ns.end());
        }
        for (int i = 0; i < ns.size() - 1; ++i)
        {
            vector<SpatialDomains::PointGeom *> qns;
            qns.push_back(ns[i]);
            qns.push_back(ns[i + 1]);
            qns.push_back(nodeNormals[ns[i + 1]]);
            qns.push_back(nodeNormals[ns[i]]);

            // quad from a boundary edge and its two normals with
            // reorientations
            // KK we might be able to scale and project the CAD Curve ontop
            // so we preserve curvature on the both sides of the BLs quad
            ElmtConfig conf(LibUtilities::eQuadrilateral, 1, false, false,
                            m_graph->GetSpaceDimension() != 3);

            SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
                LibUtilities::eQuadrilateral, qns, m_graph, m_mesh->m_edgeSet,
                m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr, nullptr);

            m_graph->GetCADAssociation()->Set(
                E, {m_graph->GetCAD()->GetSurf(faceid)});
            m_mesh->m_elementTags[2][E] = 101;
        }
    }
}

void Generator2D::PeriodicPrep()
{
    m_periodicPairs.clear();
    set<unsigned> periodic;

    // Build periodic curve pairs
    string s = m_config["periodic"].as<string>();
    vector<string> lines;

    boost::split(lines, s, boost::is_any_of(":"));

    for (auto &il : lines)
    {
        vector<string> tmp;
        boost::split(tmp, il, boost::is_any_of(","));

        if (tmp.size() != 2)
        {
            m_log(FATAL) << "Unable to define periodic pairs: should be defined"
                         << " as 'a,b' where a and b are curve IDs." << endl;
        }

        vector<unsigned> data(2);
        data[0] = std::stol(tmp[0]);
        data[1] = std::stol(tmp[1]);

        for (int i = 0; i < 1; ++i)
        {
            if (periodic.count(data[i]))
            {
                m_log(FATAL) << "Curve '" << data[i] << "' is already defined "
                             << "as periodic." << endl;
            }
        }

        m_periodicPairs[data[0]] = data[1];
        periodic.insert(data[0]);
        periodic.insert(data[1]);
    }
}

void Generator2D::MakePeriodic()
{
    // Override slave curves

    for (auto &ip : m_periodicPairs)
    {
        m_curvemeshes[ip.second]->PeriodicOverwrite(m_curvemeshes[ip.first]);
    }

    m_log(VERBOSE) << "  Periodic boundary conditions:" << endl;

    for (auto &it : m_periodicPairs)
    {
        m_log(VERBOSE) << "    - Curves " << it.first << " => " << it.second
                       << endl;
    }

    m_log(VERBOSE) << endl;
}

void Generator2D::Report()
{
    auto &graph = m_mesh->m_meshGraph;

    int ns = graph->GetNumGeoms<SpatialDomains::PointGeom>();
    int es = graph->GetNumGeoms<SpatialDomains::SegGeom>();
    int ts = graph->GetNumGeoms<SpatialDomains::TriGeom>();
    int qs = graph->GetNumGeoms<SpatialDomains::QuadGeom>();
    int ep = ns - es + ts + qs;

    m_log(VERBOSE) << "Surface meshing complete. Statistics:" << endl;
    m_log(VERBOSE) << "  - Nodes         : " << ns << endl;
    m_log(VERBOSE) << "  - Edges         : " << es << endl;
    m_log(VERBOSE) << "  - Triangles     : " << ts << endl;
    m_log(VERBOSE) << "  - Quads         : " << qs << endl;
    m_log(VERBOSE) << "  - Euler-Poincaré: " << ep << endl;
}
} // namespace Nektar::NekMesh
