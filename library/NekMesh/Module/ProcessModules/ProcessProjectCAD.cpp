////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessProjectCAD.cpp
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

#include "ProcessProjectCAD.h"
#include <NekMesh/MeshElements/Element.h>

#include <LibUtilities/Foundations/ManagerAccess.h>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace Nektar::NekMesh
{

const NekDouble prismU1[6] = {-1.0, 1.0, 1.0, -1.0, -1.0, -1.0};
const NekDouble prismV1[6] = {-1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
const NekDouble prismW1[6] = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0};

const int pyramidV0[4] = {0, 1, 2, 3};
const int pyramidV1[4] = {1, 2, 3, 0};
const int pyramidV2[4] = {3, 0, 1, 2};
const int pyramidV3[4] = {4, 4, 4, 4};

ModuleKey ProcessProjectCAD::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "projectcad"), ProcessProjectCAD::create,
        "Projects mesh to CAD");

ProcessProjectCAD::ProcessProjectCAD(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["file"]  = ConfigOption(false, "", "CAD file");
    m_config["order"] = ConfigOption(false, "4", "Enforce a polynomial order");
    m_config["surfopti"] = ConfigOption(false, "1", "Run HO-Surface Module");
    m_config["varopti"] =
        ConfigOption(false, "0", "Run the Variational Optmiser");
    m_config["cLength"] =
        ConfigOption(false, "1", "Characteristic Length CAD-Reconstruction");
    m_config["tolv1"] =
        ConfigOption(false, "1e-6",
                     "(Optional) min distance of initial Vertex to CADSurface");
    m_config["tolv2"] = ConfigOption(
        false, "1e-5",
        " (Optional) max distance of initial Vertex to CADSurface");
    m_config["ho"] =
        ConfigOption(false, "", "Pass when the input is already HO-mesh.");
    m_config["extract"] = ConfigOption(false, "", "Export the CAD to cad.txt.");
}

ProcessProjectCAD::~ProcessProjectCAD()
{
}

bool ProcessProjectCAD::FindAndProject(
    bgi::rtree<boxI, bgi::quadratic<16>> &rtree, std::array<NekDouble, 3> &in,
    [[maybe_unused]] int &surf)
{

    point q(in[0], in[1], in[2]);
    vector<boxI> result;
    rtree.query(bgi::intersects(q), back_inserter(result));

    if (result.size() == 0)
    {
        // along a projecting edge the node is too far from any surface boxes
        // this is hardly surprising but rare, return false and linearise
        m_log(VERBOSE) << "FindAndProject() - Edge node not in any of the "
                          "Bounding boxes - No projection! "
                       << endl;
        return false;
    }

    int minsurf       = 0;
    NekDouble minDist = numeric_limits<double>::max();
    NekDouble dist;

    for (int j = 0; j < result.size(); j++)
    {
        m_mesh->m_meshGraph->GetCAD()
            ->GetSurf(result[j].second)
            ->locuv(in, dist);

        if (dist < minDist)
        {
            minDist = dist;
            minsurf = result[j].second;
        }
    }

    auto uv = m_mesh->m_meshGraph->GetCAD()->GetSurf(minsurf)->locuv(in, dist);

    in = m_mesh->m_meshGraph->GetCAD()->GetSurf(minsurf)->P(uv);

    return true;
}

bool ProcessProjectCAD::IsNotValid(vector<SpatialDomains::Geometry3D *> els)
{
    for (int i = 0; i < els.size(); i++)
    {
        if (els[i]->GetShapeType() == LibUtilities::ShapeType::ePrism)
        {
            vector<SpatialDomains::PointGeom *> ns(6);
            for (int k = 0; k < 6; k++)
            {
                ns[k] = els[i]->GetVertex(k);
            }

            for (int j = 0; j < 6; j++)
            {
                NekDouble a2 = 0.5 * (1 + prismU1[j]);
                NekDouble b1 = 0.5 * (1 - prismV1[j]);
                NekDouble b2 = 0.5 * (1 + prismV1[j]);
                NekDouble c2 = 0.5 * (1 + prismW1[j]);
                NekDouble d  = 0.5 * (prismU1[j] + prismW1[j]);

                std::array<NekDouble, 9> jac;
                NekDouble x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,
                    y4, z4, x5, y5, z5;
                ns[0]->GetCoords(x0, y0, z0);
                ns[1]->GetCoords(x1, y1, z1);
                ns[2]->GetCoords(x2, y2, z2);
                ns[3]->GetCoords(x3, y3, z3);
                ns[4]->GetCoords(x4, y4, z4);
                ns[5]->GetCoords(x5, y5, z5);

                jac[0] = -0.5 * b1 * x0 + 0.5 * b1 * x1 + 0.5 * b2 * x2 -
                         0.5 * b2 * x3;
                jac[1] = -0.5 * b1 * y0 + 0.5 * b1 * y1 + 0.5 * b2 * y2 -
                         0.5 * b2 * y3;
                jac[2] = -0.5 * b1 * z0 + 0.5 * b1 * z1 + 0.5 * b2 * z2 -
                         0.5 * b2 * z3;

                jac[3] = 0.5 * d * x0 - 0.5 * a2 * x1 + 0.5 * a2 * x2 -
                         0.5 * d * x3 - 0.5 * c2 * x4 + 0.5 * c2 * x5;
                jac[4] = 0.5 * d * y0 - 0.5 * a2 * y1 + 0.5 * a2 * y2 -
                         0.5 * d * y3 - 0.5 * c2 * y4 + 0.5 * c2 * y5;
                jac[5] = 0.5 * d * z0 - 0.5 * a2 * z1 + 0.5 * a2 * z2 -
                         0.5 * d * z3 - 0.5 * c2 * z4 + 0.5 * c2 * z5;

                jac[6] = -0.5 * b1 * x0 - 0.5 * b2 * x3 + 0.5 * b1 * x4 +
                         0.5 * b2 * x5;
                jac[7] = -0.5 * b1 * y0 - 0.5 * b2 * y3 + 0.5 * b1 * y4 +
                         0.5 * b2 * y5;
                jac[8] = -0.5 * b1 * z0 - 0.5 * b2 * z3 + 0.5 * b1 * z4 +
                         0.5 * b2 * z5;

                NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                               jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                               jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

                if (jc < NekConstants::kNekZeroTol)
                {
                    return true;
                }
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::ePyramid)
        {
            vector<SpatialDomains::PointGeom *> ns(5);
            for (int k = 0; k < 5; k++)
            {
                ns[k] = els[i]->GetVertex(k);
            }

            for (int j = 0; j < 4; j++)
            {
                std::array<NekDouble, 9> jac;

                NekDouble x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3;
                ns[pyramidV0[j]]->GetCoords(x0, y0, z0);
                ns[pyramidV1[j]]->GetCoords(x1, y1, z1);
                ns[pyramidV2[j]]->GetCoords(x2, y2, z2);
                ns[pyramidV3[j]]->GetCoords(x3, y3, z3);

                jac[0] = 0.5 * (x1 - x0);
                jac[1] = 0.5 * (y1 - y0);
                jac[2] = 0.5 * (z1 - z0);
                jac[3] = 0.5 * (x2 - x0);
                jac[4] = 0.5 * (y2 - y0);
                jac[5] = 0.5 * (z2 - z0);
                jac[6] = 0.5 * (x3 - x0);
                jac[7] = 0.5 * (y3 - y0);
                jac[8] = 0.5 * (z3 - z0);

                NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                               jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                               jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

                if (jc < NekConstants::kNekZeroTol)
                {
                    return true;
                }
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::eTetrahedron)
        {
            vector<SpatialDomains::PointGeom *> ns(4);
            for (int k = 0; k < 4; k++)
            {
                ns[k] = els[i]->GetVertex(k);
            }

            std::array<NekDouble, 9> jac;

            NekDouble x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3;
            ns[0]->GetCoords(x0, y0, z0);
            ns[1]->GetCoords(x1, y1, z1);
            ns[2]->GetCoords(x2, y2, z2);
            ns[3]->GetCoords(x3, y3, z3);

            jac[0] = 0.5 * (x1 - x0);
            jac[1] = 0.5 * (y1 - y0);
            jac[2] = 0.5 * (z1 - z0);
            jac[3] = 0.5 * (x2 - x0);
            jac[4] = 0.5 * (y2 - y0);
            jac[5] = 0.5 * (z2 - z0);
            jac[6] = 0.5 * (x3 - x0);
            jac[7] = 0.5 * (y3 - y0);
            jac[8] = 0.5 * (z3 - z0);

            NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                           jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                           jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

            if (jc < NekConstants::kNekZeroTol)
            {
                return true;
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::eHexahedron)
        {
            // Hexes are not checked at the moment!
            NekDouble jc = 1.0;
            if (jc < NekConstants::kNekZeroTol)
            {
                return true;
            }
        }
        else
        {
            m_log(FATAL) << "Only prisms, pyramids and tetrahedra supported."
                         << endl;
        }
    }

    return false;
}

void ProcessProjectCAD::Process()
{
    m_log(VERBOSE) << "Projecting CAD back onto linear mesh." << endl;

    m_log(WARNING) << "ProcessAssignCAD: Warning: This module is designed for "
                   << "use with Star-CCM+ meshes only; it also requires that "
                   << "the star mesh was created in a certain way." << endl;

    if (!m_config["order"].beenSet)
    {
        m_log(VERBOSE) << "Mesh order not set: will assume order 4" << endl;
    }

    // Projection Order
    m_order   = m_config["order"].as<int>();
    m_nummode = m_order + 1;

    // Tolerances for vertex association
    NekDouble tolv1, tolv2;
    tolv1 = m_config["tolv1"].as<NekDouble>();
    tolv2 = m_config["tolv2"].as<NekDouble>();

    // Characteristic Length for CAD Reconstruction (auto calculation of tolv1
    // and tolv2)
    NekDouble cLength = 1.0;
    if (m_config["cLength"].beenSet)
    {
        cLength = m_config["cLength"].as<NekDouble>();
        tolv1 *= cLength;
        tolv2 *= cLength;
    }
    m_log(WARNING) << "Tolerances = " << tolv1 << " " << tolv2 << endl;
    // 1. Load CAD instance of the CAD model
    std::string filename = m_config["file"].as<string>();
    LoadCAD(filename);

    // 2. Create Bounding boxes of the CAD surfaces into a k-d tree
    NekDouble scale = cLength;
    bgi::rtree<boxI, bgi::quadratic<16>> rtree;
    bgi::rtree<boxI, bgi::quadratic<16>> rtreeCurve;
    bgi::rtree<boxI, bgi::quadratic<16>> rtreeNode;

    CreateBoundingBoxes(rtree, rtreeCurve, rtreeNode, tolv2, scale);

    m_log(VERBOSE) << "Bounding Boxes Surf/Curv/Vertex= " << rtree.size() << " "
                   << rtreeCurve.size() << " " << rtreeNode.size() << endl;

    // 3. Auxilaries ( can be moved to Module.cpp)
    // SurfNodes , surfNodeToEl, m_minConEdge
    Auxilaries();

    // 4.  Link Surface Vertices to CAD and Project them to the closest CAD
    LinkVertexToCAD(m_mesh, true, lockedNodes, tolv1, tolv2, rtree, rtreeCurve,
                    rtreeNode);

    // 5. clear the associations with CAD surfaces
    // necessary since the projection of the edges will change the surface uv
    // and the association will be wrong to some surfaces that were closed
    // beforehand
    m_mesh->m_meshGraph->GetCADAssociation()->ClearVertLinks(
        SpatialDomains::CADType::eSurf);
    m_mesh->m_meshGraph->GetCADAssociation()->ClearVertLinks(
        SpatialDomains::CADType::eCurve);

    // 6. Update the secondary tolerances on already projected nodes and do the
    //  final Linking Vertex - CAD Surface / Curve
    tolv1 = tolv1 * 0.1, tolv2 = tolv2 * 0.1;
    LinkVertexToCAD(m_mesh, true, lockedNodes, tolv1, tolv2, rtree, rtreeCurve,
                    rtreeNode);

    // 7. Associate Edges to CAD
    LinkEdgeToCAD(surfEdges, tolv1);

    // 8. Associate Faces to CAD
    LinkFaceToCAD(tolv1);

    // Project the Edges to CAD that
    int orderCASE3 = 2;
    ProjectEdges(surfEdges, orderCASE3, rtree);

    // Curving an edge only attaches a curve to it; every face and element
    // containing that edge is still carrying the linear map it was set up
    // with when the mesh was read. Rebuild them before anything -- the
    // high-order surface module below, or a later module -- reads geometry
    // through them.
    m_mesh->m_meshGraph->ResetGeometry();

    if (m_config["ho"].beenSet)
    {
        LinkHOtoCAD(surfEdges, tolv1 * 10.0);
    }

    ////**** HOSurface ****////
    int m_surfopti = m_config["surfopti"].as<bool>();
    if (m_surfopti == 1)
    {
        ModuleSharedPtr module = GetModuleFactory().CreateInstance(
            ModuleKey(eProcessModule, "hosurface"), m_mesh);
        module->SetLogger(m_log);

        //        module->RegisterConfig("opti","");
        module->RegisterConfig("third_party", "");
        module->RegisterConfig("order", std::to_string(m_order));

        try
        {
            module->SetDefaults();
            module->Process();
        }
        catch (runtime_error &e)
        {
            m_log(WARNING)
                << "High-order surface meshing has failed with message:"
                << endl;
            m_log(WARNING) << e.what() << endl;
            m_log(WARNING) << "The mesh will be written as normal but the "
                           << "incomplete surface will remain faceted" << endl;
            m_mesh->m_meshGraph->ResetGeometry();
            return;
        }

        // As above: the surface module has curved more edges and faces.
        m_mesh->m_meshGraph->ResetGeometry();
    }

    Diagnostics();
    if (m_config["extract"].beenSet)
    {
        m_log(VERBOSE) << "Extract CAD " << endl;
        // ExtractCAD();
    }
    m_log(VERBOSE) << "HO-Surface CAD complete." << endl;
}

void ProcessProjectCAD::LoadCAD(std::string filename)
{
    m_log(VERBOSE) << "Start Loading the CAD file " << endl;
    ModuleSharedPtr module = GetModuleFactory().CreateInstance(
        ModuleKey(eProcessModule, "loadcad"), m_mesh);
    module->RegisterConfig("filename", filename);
    module->SetDefaults();
    module->Process();
    m_log(VERBOSE) << "CAD loaded succesfully!" << endl;
}

void ProcessProjectCAD::Auxilaries()
{
    // find nodes on the surface
    // find unique nodes on the surface

    for (auto &[facePtr, tag] : m_mesh->m_elementTags[2])
    {
        for (int k = 0; k < facePtr->GetNumVerts(); k++)
        {
            surfNodes[facePtr->GetVertex(k)->GetGlobalID()] =
                facePtr->GetVertex(k);
        }
    }

    // link surface nodes to their 3D element
    for (auto &[geomPtr, tag] : m_mesh->m_elementTags[3])
    {
        auto *el = static_cast<SpatialDomains::Geometry3D *>(geomPtr);
        for (int k = 0; k < el->GetNumVerts(); k++)
        {
            auto pt = el->GetVertex(k);
            if (surfNodes.count(pt->GetGlobalID()) > 0)
            {
                surfNodeToEl[pt].push_back(el);
            }
        }
    }

    // Calculate the min edge length of the elements
    // Calculate MinConEdge
    CalculateMinEdgeLength();

    // make edges of surface mesh unique
    // ClearElementLinks();
    // EdgeSet surfEdges;
    //    EdgeMap surfEdges;
    // vector<ElementSharedPtr> &elmt = m_mesh->m_element[2];
    // map<int, int> surfIdToLoc;
    for (auto &[geomPtr, tag] : m_mesh->m_elementTags[2])
    {
        for (int j = 0; j < geomPtr->GetNumEdges(); j++)
        {
            auto *seg =
                static_cast<SpatialDomains::SegGeom *>(geomPtr->GetEdge(j));
            int v0              = seg->GetVertex(0)->GetGlobalID();
            int v1              = seg->GetVertex(1)->GetGlobalID();
            surfEdges[{v0, v1}] = seg;
        }
    }
}

void ProcessProjectCAD::CalculateMinEdgeLength()
{
    for (auto &[geomPtr, tag] : m_mesh->m_elementTags[2])
    {
        for (int j = 0; j < geomPtr->GetNumEdges(); j++)
        {
            auto *seg =
                static_cast<SpatialDomains::SegGeom *>(geomPtr->GetEdge(j));
            SpatialDomains::PointGeom *v0 = seg->GetVertex(0);
            SpatialDomains::PointGeom *v1 = seg->GetVertex(1);

            NekDouble x0, y0, z0, x1, y1, z1;
            v0->GetCoords(x0, y0, z0);
            v1->GetCoords(x1, y1, z1);
            NekDouble len = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) +
                                 (z1 - z0) * (z1 - z0));

            for (auto *v : {v0, v1})
            {
                auto it = m_minConEdge.find(v);
                if (it != m_minConEdge.end())
                {

                    it->second = min(it->second, len);
                }
                else
                {
                    m_minConEdge[v] = len;
                }
            }
        }
    }
}

void ProcessProjectCAD::CreateBoundingBoxes(
    bgi::rtree<boxI, bgi::quadratic<16>> &rtreeSurf,
    bgi::rtree<boxI, bgi::quadratic<16>> &rtreeCurve,
    bgi::rtree<boxI, bgi::quadratic<16>> &rtreeNode, NekDouble tolv2,
    NekDouble scale)
{
    // // CAD Surfs
    vector<boxI> boxes;
    for (int i = 1; i <= m_mesh->m_meshGraph->GetCAD()->GetNumSurf(); i++)
    {
        // m_log(VERBOSE).Progress(i,
        // m_mesh->m_meshGraph->GetCAD()->GetNumSurf(),
        //                         "building surface bboxes", i - 1);
        auto bx = m_mesh->m_meshGraph->GetCAD()->GetSurf(i)->BoundingBox(scale);
        boxes.push_back(make_pair(
            box(point(bx[0], bx[1], bx[2]), point(bx[3], bx[4], bx[5])), i));

        m_log(VERBOSE) << " Bounding box = " << i << endl;
        m_log(VERBOSE) << bx[0] << " " << bx[1] << " " << bx[2] << endl;
        m_log(VERBOSE) << bx[3] << " " << bx[4] << " " << bx[5] << endl;
    }

    m_log(VERBOSE).Newline();
    m_log(VERBOSE) << "Building Surf admin data structures." << endl;
    rtreeSurf.insert(boxes.begin(), boxes.end());

    // CAD Curves
    boxes.clear();
    for (int i = 1; i <= m_mesh->m_meshGraph->GetCAD()->GetNumCurve(); i++)
    {
        auto bx =
            m_mesh->m_meshGraph->GetCAD()->GetCurve(i)->BoundingBox(scale);

        boxes.push_back(make_pair(
            box(point(bx[0], bx[1], bx[2]), point(bx[3], bx[4], bx[5])), i));
    }
    rtreeCurve.insert(boxes.begin(), boxes.end());

    // CAD Vertices
    boxes.clear();
    NekDouble tol = 100 * tolv2; // 1e-8 * scale
    int i         = 0;
    for (auto [id, vert] : m_mesh->m_meshGraph->GetCAD()->GetVerts())
    {
        std::array<NekDouble, 3> locT = vert->GetLoc();

        boxes.push_back(
            make_pair(box(point(locT[0] - tol, locT[1] - tol, locT[2] - tol),
                          point(locT[0] + tol, locT[1] + tol, locT[2] + tol)),
                      id));
        m_log(VERBOSE) << " Bounding Vertex box = " << i << endl;
        m_log(VERBOSE) << boxes[i].first.min_corner().get<0>() << " "
                       << boxes[i].first.min_corner().get<1>() << " "
                       << boxes[i].first.min_corner().get<2>() << endl;
        m_log(VERBOSE) << boxes[i].first.max_corner().get<0>() << " "
                       << boxes[i].first.max_corner().get<1>() << " "
                       << boxes[i].first.max_corner().get<2>() << endl;

        // m_log(VERBOSE) << " Bounding Vertex box = " << i << endl;
        // m_log(VERBOSE) << bx[0] << " " << bx[1] << " " << bx[2] << endl;
        // m_log(VERBOSE) << bx[3] << " " << bx[4] << " " << bx[5] << endl;
        i++;
    }
    rtreeNode.insert(boxes.begin(), boxes.end());
}

void ProcessProjectCAD::LinkVertexToCAD(
    NekMesh::MeshSharedPtr &m_mesh, bool CADCurveVertexProject,
    std::unordered_set<SpatialDomains::PointGeom *> &lockedNodes,
    NekDouble tolv1, NekDouble tolv2,
    bgi::rtree<boxI, bgi::quadratic<16>> &rtree,
    bgi::rtree<boxI, bgi::quadratic<16>> &rtreeCurve,
    bgi::rtree<boxI, bgi::quadratic<16>> &rtreeNode)
{
    map<int, vector<int>> finds;

    // m_log(VERBOSE) << "Searching tree." << endl;

    NekDouble maxNodeCor = 0;

    // find nodes surface and parametric location
    for (auto &[id, vertex] : surfNodes)
    {
        NekDouble x, y, z;
        vertex->GetCoords(x, y, z);

        point q(x, y, z);
        vector<boxI> result;
        rtree.query(bgi::intersects(q), back_inserter(result));

        if (result.size() == 0)
        {
            // Vertex is too far from any surface bounding boxes
            m_log(WARNING)
                << "Vertex  " << x << " " << y << " " << z
                << " is not in any boundin boxes. 1. "
                   "Make sure you use the correct STEP file. 2. The problem is "
                   " likely in the linear mesh->try to refine this region."
                << endl;
            continue;
        }

        // Vertex Tolerances to the CAD Surface - 0.5 * min edge length + [min
        // tol max tol]
        NekDouble tol = m_minConEdge[vertex] * 0.5;
        tol           = min(tol, tolv1);
        tol           = max(tol, tolv2);
        vector<int> distId;
        vector<NekDouble> distList;
        // sort the surfaces by distance to the node
        for (int j = 0; j < result.size(); j++)
        {
            NekDouble dist;
            NekDouble x, y, z; // possible issue
            vertex->GetCoords(x, y, z);
            m_mesh->m_meshGraph->GetCAD()
                ->GetSurf(result[j].second)
                ->locuv({x, y, z}, dist);
            distList.push_back(dist);
            distId.push_back(result[j].second);
        }

        bool repeat = true;
        while (repeat)
        {
            repeat = false;
            for (int j = 0; j < distId.size() - 1; j++)
            {
                if (distList[j + 1] < distList[j])
                {
                    repeat = true;
                    swap(distList[j + 1], distList[j]);
                    swap(distId[j + 1], distId[j]);
                }
            }
        }

        int pos = 0;
        for (int j = 0; j < distId.size(); j++)
        {
            if (distList[j] < tol)
            {
                pos++;
            }
        }

        distId.resize(pos);

        finds[pos].push_back(0);
        // if the node is not close to any surface lock it (no CAD given + no
        // projection for its edges)
        if (pos == 0)
        {
            lockedNodes.insert(vertex);
            m_log(WARNING) << "surface minDist = " << distList[0] << " unknown "
                           << "(tolerance: " << tol << ")   xyz = " << (vertex)
                           << endl;
        }
        else
        {
            NekDouble shift;
            bool st = false;
            for (int j = 0; j < distId.size(); j++)
            {
                if (distList[j] > tol)
                {
                    continue;
                }
                if (m_mesh->m_meshGraph->GetCAD()
                        ->GetSurf(distId[j])
                        ->IsPlanar())
                {
                    // Do we want to skip the projection on planar faces?
                    // continue;
                }

                shift          = distList[j];
                NekDouble dist = 0;
                SpatialDomains::CADSurfSharedPtr s =
                    m_mesh->m_meshGraph->GetCAD()->GetSurf(distId[j]);

                NekDouble x, y, z;
                vertex->GetCoords(x, y, z);
                std::array<NekDouble, 3> loc = {x, y, z};

                [[maybe_unused]] auto uvt = s->locuv(loc, dist);

                if (true)
                {
                    // Project vertex to the CAD
                    x = s->P(uvt)[0];
                    y = s->P(uvt)[1];
                    z = s->P(uvt)[2];

                    vertex->UpdatePosition(x, y, z);

                    // Check if valid afterwards if not restart
                    if (ProcessProjectCAD::IsNotValid(surfNodeToEl[vertex]))
                    {
                        vertex->UpdatePosition(loc[0], loc[1], loc[2]);

                        m_log(VERBOSE) << "Element not valid after vertex ";
                        m_log(VERBOSE)
                            << "projection reset it and lock the vertex  "
                            << loc[0] << " " << loc[1] << " " << loc[2] << endl;
                        break;
                    }
                }

                st = true;
                break;
            }

            if (!st)
            {
                lockedNodes.insert(vertex);
                continue;
            }

            for (int j = 0; j < distId.size(); j++)
            {
                if (distList[j] > tol)
                {
                    continue;
                }
                if (m_mesh->m_meshGraph->GetCAD()
                        ->GetSurf(distId[j])
                        ->IsPlanar())
                {
                    // continue;
                }

                SpatialDomains::CADSurfSharedPtr s =
                    m_mesh->m_meshGraph->GetCAD()->GetSurf(distId[j]);
                NekDouble dist = 0;

                NekDouble x1, y1, z1;
                vertex->GetCoords(x1, y1, z1);
                auto uv = s->locuv({x1, y1, z1}, dist);

                SpatialDomains::CADLink surfUV = {s, {uv[0], uv[1]}};
                m_mesh->m_meshGraph->GetCADAssociation()->Add(vertex, surfUV);
            }

            maxNodeCor = max(maxNodeCor, shift);
        }
    }
    // for the vertices with multiple CAD Surfaces, check CADCurves with the
    // bounding boxes Intersect with CAD Curves rtree
    bool CADCurveLook = true;
    if (CADCurveLook)
    {
        for (auto &[id, vertex] : surfNodes)
        {
            // CAD Curve
            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eSurf) > 1)
            {
                NekDouble x, y, z;
                vertex->GetCoords(x, y, z);

                point q(x, y, z);
                vector<boxI> result;
                rtreeCurve.query(bgi::intersects(q), back_inserter(result));
                if (result.size() == 1)
                {
                    // Single CAD Curve
                    SpatialDomains::CADCurveSharedPtr CADCurve_t =
                        m_mesh->m_meshGraph->GetCAD()->GetCurve(
                            result[0].second);
                    NekDouble t0, dist0;
                    NekDouble tmin, tmax;
                    CADCurve_t->GetBounds(tmin, tmax);

                    dist0 = CADCurve_t->loct({x, y, z}, t0, tmin, tmax);
                    if (dist0 < tolv1)
                    {
                        SpatialDomains::CADLink curve_t = {CADCurve_t,
                                                           {t0, 0.0}};

                        if (CADCurveVertexProject)
                        {

                            std::array<NekDouble, 3> loc = {x, y, z};

                            x = CADCurve_t->P(t0)[0]; // This gets the node far
                                                      // from CADSurf???
                            y = CADCurve_t->P(t0)[1];
                            z = CADCurve_t->P(t0)[2];

                            vertex->UpdatePosition(x, y, z);

                            if (ProcessProjectCAD::IsNotValid(
                                    surfNodeToEl[vertex]))
                            {
                                vertex->UpdatePosition(loc[0], loc[1], loc[2]);

                                m_log(VERBOSE)
                                    << "Element not valid after vertex ";
                                m_log(VERBOSE)
                                    << "projection reset lock the vertex  "
                                    << loc[0] << " " << loc[1] << " " << loc[2]
                                    << endl;
                            }
                            else
                            {
                                m_mesh->m_meshGraph->GetCADAssociation()->Add(
                                    vertex, curve_t);
                            }
                        }
                        else
                        {
                            m_mesh->m_meshGraph->GetCADAssociation()->Add(
                                vertex, curve_t);
                        }
                    }
                    else
                    {
                        m_log(TRACE)
                            << "Vertex  " << x << " " << y << " " << z
                            << " not close enough to the CAD Curve " << endl;
                    }
                }
                else if (result.size() == 0)
                {
                    m_log(WARNING)
                        << "No CAD Curve found in the bounding box "
                        << "Vertex  " << x << " " << y << " " << z << endl;

                    // continue;
                }
                else
                {
                    m_log(TRACE)
                        << "Multiple CAD Curves found for the "
                        << "Vertex  " << x << " " << y << " " << z << endl;

                    for (auto res : result)
                    {
                        SpatialDomains::CADCurveSharedPtr CADCurve_t =
                            m_mesh->m_meshGraph->GetCAD()->GetCurve(res.second);
                        NekDouble t0, dist0;
                        NekDouble tmin, tmax;
                        CADCurve_t->GetBounds(tmin, tmax);
                        dist0 = CADCurve_t->loct({x, y, z}, t0, tmin, tmax);
                        if (dist0 < tolv1)
                        {
                            SpatialDomains::CADLink curve_t = {CADCurve_t,
                                                               {t0, 0.0}};

                            if (CADCurveLook)
                            {
                                NekDouble tmpX = x;
                                NekDouble tmpY = y;
                                NekDouble tmpZ = z;

                                x = CADCurve_t->P(t0)[0]; // This gets the node
                                                          // far from CADSurf???
                                y = CADCurve_t->P(t0)[1];
                                z = CADCurve_t->P(t0)[2];
                                vertex->UpdatePosition(x, y, z);

                                if (ProcessProjectCAD::IsNotValid(
                                        surfNodeToEl[vertex]))
                                {
                                    vertex->UpdatePosition(tmpX, tmpY, tmpZ);

                                    m_log(VERBOSE)
                                        << "Element not valid after vertex";
                                    m_log(VERBOSE)
                                        << " projection reset it and"
                                           " lock the vertex "
                                        << "Vertex  " << tmpX << " " << tmpY
                                        << " " << tmpZ << endl;
                                }
                                else
                                {
                                    // BUG Discovered there is a bug here - if
                                    // the vertex projection lead to an invalid
                                    // element, the vertex still retains the cad
                                    // curve, Hence
                                    m_mesh->m_meshGraph->GetCADAssociation()
                                        ->Add(vertex, curve_t);
                                }
                            }
                        }
                        else
                        {
                            m_log(TRACE)
                                << "Vertex  " << x << " " << y << " " << z
                                << " not close enough to the CAD Curve "
                                << dist0 << " tol = " << tolv2 << endl;
                        }
                    }
                }
            }

            // CAD Vertex (for the moment no benefits )
            bool CADVertexAssociation = true;
            if (CADVertexAssociation)
            {
                NekDouble x, y, z;
                vertex->GetCoords(x, y, z);

                if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                        vertex, SpatialDomains::CADType::eSurf) > 1 ||
                    m_mesh->m_meshGraph->GetCADAssociation()->Count(
                        vertex, SpatialDomains::CADType::eCurve) > 1)
                {
                    point q(x, y, z);
                    // cout << x << "  " << y << " " << z << endl;
                    // cout << q.get<0>() << " " << q.get<1>() << " " <<
                    // q.get<2>()
                    //      << endl;
                    // FOR SOME REASON THE INTERSECTS DOES NOT WORK HERE
                    // OPTION 1 - the correct vertices do not reach here - not
                    // true all 8 vertices with 3CADs are here
                    // Option 2 - the bounding box is not correct or large
                    // enough, see boundingbox rtreeNode?

                    vector<boxI> result;
                    rtreeNode.query(bgi::intersects(q), back_inserter(result));

                    // cout << "CADSURF.size() = "
                    //      <<
                    //      m_mesh->m_meshGraph->GetCADAssociation()->Count(vertex,
                    //      SpatialDomains::CADType::eSurf)
                    //      << endl;
                    // cout << "Multiple CAD Surf ? " << result.size() << endl;
                    // cout << x << " " << y << " " << z << endl;

                    if (result.size() == 1)
                    {
                        // Single CAD Vertex ( we do not project on it for now!)
                        SpatialDomains::CADVertSharedPtr vCAD =
                            m_mesh->m_meshGraph->GetCAD()->GetVert(
                                result[0].second);

                        m_mesh->m_meshGraph->GetCADAssociation()->Set(
                            vertex, {vCAD}); // No Adj Curves
                        // Assigned to the V0 !
                    }
                    else if (result.size() > 1)
                    {
                        // Multiple CAD Vertices
                        m_log(WARNING)
                            << "Multiple CAD Vertices found for the "
                            << " vertex " << x << " " << y << "   " << z
                            << " size = " << result.size() << endl;
                    }
                }
            }
        }
    }

    m_log(VERBOSE) << "  - max surface Vertex correction " << maxNodeCor
                   << endl;
    m_log(VERBOSE) << "  - lockedNodes N= " << lockedNodes.size() << endl;
}

void ProcessProjectCAD::LinkEdgeToCAD(EdgeMap &surfEdges, NekDouble tolv1)
{
    // Every Edge needs to have only 1 CAD Object CADCurve or CADSurf
    // This is not necessary to be perfect as it will be filled by the Associate
    // Faces However it can be used as a verification for the FACE association
    // in the future It is beneficial to associate the edge to CAD Curve due to
    // optimization and projection sliding on the CAD Curve is
    // more rorbust than to the CADSurf .

    for (auto &[id, edge] : surfEdges)
    {
        SpatialDomains::PointGeom *v1 = edge->GetVertex(0);
        SpatialDomains::PointGeom *v2 = edge->GetVertex(1);

        if (lockedNodes.count(v1) || lockedNodes.count(v2))
        {
            continue;
        }

        // 1. Get CAD Curve based on the vertex CADCurves - should be 99% of
        // CADCurve edges
        if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                v1, SpatialDomains::CADType::eCurve) &&
            m_mesh->m_meshGraph->GetCADAssociation()->Count(
                v2, SpatialDomains::CADType::eCurve))
        {
            if (m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                    v1, SpatialDomains::CADType::eCurve)[0] ==
                m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                    v2, SpatialDomains::CADType::eCurve)[0])
            {
                SpatialDomains::CADCurveSharedPtr cadCurve =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(v1);
                m_mesh->m_meshGraph->GetCADAssociation()->Set(edge, {cadCurve});
                continue;
            }
            else
            {
                vector<int> cmn = IntersectCADLinks(
                    m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                        v1, SpatialDomains::CADType::eCurve),
                    m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                        v2, SpatialDomains::CADType::eCurve));
                if (cmn.size() == 1)
                {
                    m_mesh->m_meshGraph->GetCADAssociation()->Set(
                        edge,
                        {m_mesh->m_meshGraph->GetCAD()->GetCurve(cmn[0])});
                    continue;
                }
                else if (cmn.size() > 1)
                {
                    // This is often the case when you have two CAD Curvesthat
                    // are the same, but  topologically different and OCE CAD
                    // Sewing (sew_tolerance) has not merged them
                    NekDouble x, y, z, x1, y1, z1;
                    edge->GetVertex(0)->GetCoords(x, y, z);
                    edge->GetVertex(1)->GetCoords(x1, y1, z1);
                    m_log(WARNING)
                        << "Edge with different CAD Curves cmn.size=  "
                        << cmn.size() << " v1 = " << x << " " << y << " " << z
                        << " v2 = " << x1 << " " << y1 << " " << z1 << endl;
                }
            }
        }

        // 2. Try to associate the edge to CADCurves based on the vertex
        vector<int> cmn = IntersectCADLinks(
            m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                v1, SpatialDomains::CADType::eSurf),
            m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                v2, SpatialDomains::CADType::eSurf));

        if (cmn.size() == 0)
        {
            NekDouble x, y, z, x1, y1, z1;
            edge->GetVertex(0)->GetCoords(x, y, z);
            edge->GetVertex(1)->GetCoords(x1, y1, z1);

            // no CAD surface found for the edge (CASE3)
            m_log(TRACE) << "Case 3 edge association (NO-CADSurf or Curve) "
                         << " v1 = " << x << " " << y << " " << z
                         << " v2 = " << x1 << " " << y1 << " " << z1 << endl;
            continue;
        }

        if (cmn.size() == 1)
        {
            // Clearly Edge is on a single CAD surface (internal)
            m_mesh->m_meshGraph->GetCADAssociation()->Set(
                edge, {m_mesh->m_meshGraph->GetCAD()->GetSurf(cmn[0])});
        }
        else if (cmn.size() == 2)
        {
            // N=2 CAD Surfaces could be CAD-curve or CAD-surface
            // (CASE2)
            // Try to find the correct edge topologically
            // // 1.Create vi1 , vi2
            vector<int> CADCurves_uv1;
            vector<int> CADCurves_uv2;

            SpatialDomains::CADSurfSharedPtr EdgeSurf1 =
                m_mesh->m_meshGraph->GetCAD()->GetSurf(cmn[0]);
            SpatialDomains::CADSurfSharedPtr EdgeSurf2 =
                m_mesh->m_meshGraph->GetCAD()->GetSurf(cmn[1]);

            // Checking overlapping CADCurves between the surfaces
            for (auto EdgeLoop : EdgeSurf1->GetEdges())
            {
                for (auto CADCurve : EdgeLoop->edges)
                {
                    CADCurves_uv1.push_back(CADCurve->GetId());
                }
            }
            for (auto EdgeLoop : EdgeSurf2->GetEdges())
            {
                for (auto CADCurve : EdgeLoop->edges)
                {
                    CADCurves_uv2.push_back(CADCurve->GetId());
                }
            }

            sort(CADCurves_uv1.begin(), CADCurves_uv1.end());
            sort(CADCurves_uv2.begin(), CADCurves_uv2.end());

            vector<int> commonCADCurves;
            set_intersection(CADCurves_uv1.begin(), CADCurves_uv1.end(),
                             CADCurves_uv2.begin(), CADCurves_uv2.end(),
                             back_inserter(commonCADCurves));

            NekDouble tolDist = tolv1; // POSSIBLE PROBLEM !!!!
            if (commonCADCurves.size() == 1)
            {
                SpatialDomains::CADCurveSharedPtr CADCurve_t =
                    m_mesh->m_meshGraph->GetCAD()->GetCurve(commonCADCurves[0]);

                NekDouble t0, t1, dist0, dist1;
                NekDouble tmin, tmax;
                CADCurve_t->GetBounds(tmin, tmax);

                NekDouble x, y, z, x1, y1, z1;
                edge->GetVertex(0)->GetCoords(x, y, z);
                edge->GetVertex(1)->GetCoords(x1, y1, z1);

                dist0 = CADCurve_t->loct({x, y, z}, t0, tmin, tmax);
                dist1 = CADCurve_t->loct({x1, y1, z1}, t1, tmin, tmax);

                if ((dist0 < tolDist) && (dist1 < tolDist))
                {

                    SpatialDomains::CADLink curve_t = {CADCurve_t, {t0, 0.0}};
                    m_mesh->m_meshGraph->GetCADAssociation()->Add(
                        edge->GetVertex(0), curve_t);
                    curve_t = {CADCurve_t, {t1, 0.0}};
                    m_mesh->m_meshGraph->GetCADAssociation()->Add(
                        edge->GetVertex(1), curve_t);

                    m_mesh->m_meshGraph->GetCADAssociation()->Set(edge,
                                                                  {CADCurve_t});
                }
                else
                {
                    // Associate to one of the CADSurf.
                    m_log(VERBOSE)
                        << " dist > distol (cmnCADCurve.size=1) = " << dist0
                        << " " << dist1 << endl;
                }
            }
            else if (commonCADCurves.size() >= 2)
            {
                // common when the CAD is not perfect (2 CADcurves that
                // overlap
                // are two different topological objects )
                //    in this case just
                // take the curve and assign the closest one
                //    within tolv2 * 0.1
                // tolerance this is a stricter due to the

                // Another test case is a CADSurf like NACA,
                //    where it fills the
                // DO WE NEED THIS ?

                m_mesh->m_meshGraph->GetCADAssociation()->Set(edge,
                                                              {EdgeSurf1});

                NekDouble x, y, z;
                edge->GetVertex(0)->GetCoords(x, y, z);

                m_log(VERBOSE) << "edge CASE commonCADCurves.size()==2 for "
                                  "comn.size()  = 2     xyz= "
                               << x << " " << y << " " << z << endl;
            }
        }
        else
        {
            // If more than 2 common CAD Surfaces are present,
            //    we do not
            // associate CADCurve because it is too risky
            // Closest CAD Surf
            m_log(VERBOSE) << "too many common surfaces for Edge association "
                              "(cmn>2) will use the element to associate."
                           << endl;
        }
        // if no CAD Curves are associated, the CAD Surf will be associated
        // through the Face Association.
    }
}

vector<int> ProcessProjectCAD::IntersectCADLinks(
    const vector<SpatialDomains::CADLink> &v1_CADs,
    const vector<SpatialDomains::CADLink> &v2_CADs)
{
    vector<int> vi1, vi2;
    for (auto &link : v1_CADs)
    {
        vi1.push_back(link.Id());
    }
    for (auto &link : v2_CADs)
    {
        vi2.push_back(link.Id());
    }

    sort(vi1.begin(), vi1.end());
    sort(vi2.begin(), vi2.end());

    vector<int> cmn;
    set_intersection(vi1.begin(), vi1.end(), vi2.begin(), vi2.end(),
                     back_inserter(cmn));

    return cmn;
}

void ProcessProjectCAD::ProjectEdges(
    EdgeMap &surfEdges, int order, bgi::rtree<boxI, bgi::quadratic<16>> &rtree)
{
    m_log(VERBOSE) << " Projecting Edges to CAD (CASE3)" << endl;
    // Project the Edges to CAD

    LibUtilities::PointsKey ekey(order + 1,
                                 LibUtilities::eGaussLobattoLegendre);
    Array<OneD, NekDouble> gll;
    LibUtilities::PointsManager()[ekey]->GetPoints(gll);

    // make surface edges high-order
    int cnt  = 0;
    int cnt1 = 0;

    for (auto &[id, edge] : surfEdges)
    {
        // IF the edge is already associated with a CAD surface, HOSurf will do
        // the curving job

        if (m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge) ||
            m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge))
        {
            continue;
        }
        if (lockedNodes.count(edge->GetVertex(0)) ||
            lockedNodes.count(edge->GetVertex(1)))
        {
            continue;
        }
        cnt++;

        vector<SpatialDomains::CADLink> v1CAD =
            m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                edge->GetVertex(0), SpatialDomains::CADType::eSurf);
        vector<SpatialDomains::CADLink> v2CAD =
            m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                edge->GetVertex(1), SpatialDomains::CADType::eSurf);

        vector<int> vi1, vi2, vi1vi2;
        for (size_t j = 0; j < v1CAD.size(); ++j)
        {
            vi1.push_back(v1CAD[j].Id());
        }
        for (size_t j = 0; j < v2CAD.size(); ++j)
        {
            vi2.push_back(v2CAD[j].Id());
        }

        sort(vi1.begin(), vi1.end());
        sort(vi2.begin(), vi2.end());

        vector<int> cmn;
        set_intersection(vi1.begin(), vi1.end(), vi2.begin(), vi2.end(),
                         back_inserter(cmn));
        NekDouble x, y, z, x1, y1, z1;
        edge->GetVertex(0)->GetCoords(x, y, z);
        edge->GetVertex(1)->GetCoords(x1, y1, z1);

        if (cmn.size() == 1 || cmn.size() == 2)
        {
            // THERE IS A BUG HERE THE CIRCLE!
            for (int j = 0; j < cmn.size(); j++)
            {

                SpatialDomains::CADSurfSharedPtr cadSurf =
                    m_mesh->m_meshGraph->GetCAD()->GetSurf(cmn[j]);
                if (cadSurf->IsPlanar())
                {
                    // if its planar dont care
                    continue;
                }

                auto uvb = m_mesh->m_meshGraph->GetCADAssociation()->GetSurfUV(
                    edge->GetVertex(0), cadSurf->GetId());

                auto uve = m_mesh->m_meshGraph->GetCADAssociation()->GetSurfUV(
                    edge->GetVertex(1), cadSurf->GetId());

                // can compare the loction of the projection to the
                // corresponding position of the straight sided edge
                // if the two differ by more than the length of the edge
                // something has gone wrong
                NekDouble len = edge->GetVertex(0)->dist(*edge->GetVertex(1));

                // Create the curve which we will populate
                auto curve =
                    ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                        edge->GetGlobalID(),
                        LibUtilities::PointsManager()[ekey]->GetPointsType());

                // Add vertex1
                curve->m_points.push_back(
                    m_mesh->m_meshGraph->CreateCurveNode(3, 0, x, y, z));

                // Add internal edgenodes
                for (int k = 1; k < order + 1 - 1; k++)
                {
                    std::array<NekDouble, 2> uv = {
                        uvb[0] * (1.0 - gll[k]) / 2.0 +
                            uve[0] * (1.0 + gll[k]) / 2.0,
                        uvb[1] * (1.0 - gll[k]) / 2.0 +
                            uve[1] * (1.0 + gll[k]) / 2.0};
                    std::array<NekDouble, 3> loc =
                        m_mesh->m_meshGraph->GetCAD()->GetSurf(cmn[j])->P(uv);

                    std::array<NekDouble, 3> locT;
                    locT[0] =
                        x * (1.0 - gll[k]) / 2.0 + x1 * (1.0 + gll[k]) / 2.0;
                    locT[1] =
                        y * (1.0 - gll[k]) / 2.0 + y1 * (1.0 + gll[k]) / 2.0;
                    locT[2] =
                        z * (1.0 - gll[k]) / 2.0 + z1 * (1.0 + gll[k]) / 2.0;

                    NekDouble d = sqrt((locT[0] - loc[0]) * (locT[0] - loc[0]) +
                                       (locT[1] - loc[1]) * (locT[1] - loc[1]) +
                                       (locT[2] - loc[2]) * (locT[2] - loc[2]));

                    if (d > len)
                    {
                        curve->m_points.clear();
                        break;
                    }

                    curve->m_points.push_back(
                        m_mesh->m_meshGraph->CreateCurveNode(3, 0, loc[0],
                                                             loc[1], loc[2]));
                }

                // Add v2
                curve->m_points.push_back(
                    m_mesh->m_meshGraph->CreateCurveNode(3, 0, x1, y1, z1));

                // This might introduce a BUG if the curvepoints are cleared
                // initially
                if (curve->m_points.size() < 3)
                {
                    // it suceeded on this surface so skip the other possibility
                    curve->m_points.clear();
                    continue;
                }

                // Assign the curve to the EDGE ?
                edge->SetCurve(curve.get());
                m_mesh->m_meshGraph->GetCurvedEdges()[edge->GetGlobalID()] =
                    std::move(curve);
            }
        }
        else if (cmn.size() == 0)
        {
            // projection, if the projection requires more than two
            // surfaces
            // including the edge nodes, then,  in theory projection
            // shouldnt be used
            vi1vi2.insert(vi1vi2.end(), vi1.begin(), vi1.end());
            vi1vi2.insert(vi1vi2.end(), vi2.begin(), vi2.end());

            // Create the curve which we will populate
            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    edge->GetGlobalID(),
                    LibUtilities::PointsManager()[ekey]->GetPointsType());

            // Add vertex1
            curve->m_points.push_back(
                m_mesh->m_meshGraph->CreateCurveNode(3, 0, x, y, z));

            set<int> sused;
            for (int k = 1; k < order + 1 - 1; k++)
            {
                std::array<NekDouble, 3> locT;
                locT[0] = x * (1.0 - gll[k]) / 2.0 + x1 * (1.0 + gll[k]) / 2.0;
                locT[1] = y * (1.0 - gll[k]) / 2.0 + y1 * (1.0 + gll[k]) / 2.0;
                locT[2] = z * (1.0 - gll[k]) / 2.0 + z1 * (1.0 + gll[k]) / 2.0;

                int s;
                if (!FindAndProject(rtree, locT, s))
                {
                    curve->m_points.clear();
                    m_log(VERBOSE) << "failed to find CAD" << endl;
                    break;
                }

                sused.insert(s);

                if (sused.size() > 2)
                {
                    m_log(WARNING) << "found too many CAD " << endl;
                    curve->m_points.clear();
                    break;
                }

                curve->m_points.push_back(m_mesh->m_meshGraph->CreateCurveNode(
                    3, 0, locT[0], locT[1], locT[2]));
            }

            curve->m_points.push_back(
                m_mesh->m_meshGraph->CreateCurveNode(3, 0, x1, y1, z1));

            // This might introduce a BUG if the curvepoints are cleared
            // initially
            if (curve->m_points.size() < 3)
            {
                // it suceeded on this surface so skip the other possibility
                curve->m_points.clear();
                continue;
            }

            // Assign the curve to the EDGE ?
            edge->SetCurve(curve.get());
            m_mesh->m_meshGraph->GetCurvedEdges()[edge->GetGlobalID()] =
                std::move(curve);

            cnt1++;
        }
        else
        {
            m_log(WARNING)
                << "Too many common cad Surfaces associated to the edge vertex "
                << endl;
        }
    }

    m_log(VERBOSE) << "Edges No CAD N= " << cnt << endl;
    m_log(VERBOSE) << "Edges No CAD Projected N= " << cnt1 << endl;
}

void ProcessProjectCAD::Diagnostics()
{
    m_log(VERBOSE) << endl
                   << " ---------Diagnostics---------     " << endl
                   << endl;

    if (true)
    {
        long int counterElementsNoCAD = 0;
        long int counterEdgesNoCAD    = 0;
        long int counterVerticesNoCAD = 0;

        m_log(VERBOSE) << "         CAD Stats       " << endl;
        m_log(VERBOSE) << "CAD Vertices =           "
                       << m_mesh->m_meshGraph->GetCAD()->GetNumVerts() << endl;
        m_log(VERBOSE) << "CAD Curves =             "
                       << m_mesh->m_meshGraph->GetCAD()->GetNumCurve() << endl;
        m_log(VERBOSE) << "CAD Surfaces =           "
                       << m_mesh->m_meshGraph->GetCAD()->GetNumSurf() << endl;

        m_log(VERBOSE) << "        Mesh Stats       " << endl;
        m_log(VERBOSE) << "Mesh Surface Vertices =  " << surfNodes.size()
                       << endl;
        m_log(VERBOSE) << "Mesh Surface Edges =     " << surfEdges.size()
                       << endl;
        m_log(VERBOSE) << "Mesh Surface Face =      "
                       << m_mesh->m_elementTags[2].size() << endl;

        // Elements without CAD
        for (auto [element, tag] : m_mesh->m_elementTags[2])
        {

            if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(element) ==
                nullptr)
            {
                NekDouble x, y, z;
                counterElementsNoCAD++;
                element->GetVertex(0)->GetCoords(x, y, z);

                m_log(TRACE) << "Element No CAD Vertex1 xyz= " << x << " " << y
                             << " " << z << endl;
            }
        }

        // Edges without CAD
        int cntEdgeCurve = 0, cntEdgeSurf = 0;
        for (auto &[id, edge] : surfEdges)
        {
            if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge) ==
                    nullptr &&
                m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge) ==
                    nullptr)
            {
                counterEdgesNoCAD++;
            }
            else if (m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge) !=
                     nullptr)
            {
                cntEdgeCurve++;
            }
            else
            {
                cntEdgeSurf++;
            }
        }

        // Vertices CAD
        int cntCAD2 = 0, cntCAD1 = 0, cntCAD3orMore = 0;
        int cntCADVertices = 0, cntCADCurves = 0;
        for (auto [id, vertex] : surfNodes)
        {
            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eSurf) == 0 &&
                m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eCurve) == 0)
            {
                counterVerticesNoCAD++;
            }

            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eSurf) == 1)
            {
                cntCAD1++;
            }

            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eSurf) == 2)
            {
                cntCAD2++;
            }

            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eSurf) > 2)
            {
                cntCAD3orMore++;
            }

            if (m_mesh->m_meshGraph->GetCADAssociation()->GetVert(vertex))
            {
                cntCADVertices++;
            }
            if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                    vertex, SpatialDomains::CADType::eCurve) > 0)
            {
                cntCADCurves++;
            }
        }

        // Stats
        m_log(WARNING) << "Vertices No CAD (Includes PlanarSurf) N= "
                       << counterVerticesNoCAD << endl;
        // m_log(WARNING) << "Planar Vertices =                        " <<
        // planarCnt << endl;
        m_log(WARNING) << "Edges without CADObject               N= "
                       << counterEdgesNoCAD << endl;
        m_log(WARNING) << "Faces without CADObject               N= "
                       << counterElementsNoCAD << endl;

        m_log(VERBOSE) << "Vertices 1 CADSurf                    N= " << cntCAD1
                       << endl;
        m_log(VERBOSE) << "Vertices 2 CADSurf                    N= " << cntCAD2
                       << endl;
        m_log(VERBOSE) << "Vertices CAD3 or more Surf            N= "
                       << cntCAD3orMore << endl;
        m_log(VERBOSE) << "Vertices- CADVertex (also have CADSu) N= "
                       << cntCADVertices << endl;
        m_log(VERBOSE) << "Vertices- CADCurve                    N= "
                       << cntCADCurves << endl;
        m_log(VERBOSE) << "Edges CADCurve                        N= "
                       << cntEdgeCurve << endl;
        m_log(VERBOSE) << "Edges CADSurf                         N= "
                       << cntEdgeSurf << endl;

        if (m_config["ho"].beenSet)
        {

            // Surface Edges without CAD
            int cntEdgesWithPoints     = 0;
            int cntHONodesWithCADCurve = 0;
            int cntHONodesWithCADSurf  = 0;
            int cntHONodesNoCAD        = 0;

            for (auto &[id, edge] : surfEdges)
            {
                if (edge->GetCurve() && edge->GetCurve()->m_points.size() > 0)
                {
                    cntEdgesWithPoints++;
                    for (int k = 0; k < 2; k++)
                    {
                        auto *v = edge->GetVertex(k);
                        if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                                v, SpatialDomains::CADType::eCurve) > 0)
                        {
                            cntHONodesWithCADCurve++;
                        }
                        else if (m_mesh->m_meshGraph->GetCADAssociation()
                                     ->Count(
                                         v, SpatialDomains::CADType::eSurf) > 0)
                        {
                            cntHONodesWithCADSurf++;
                        }
                        else
                        {
                            cntHONodesNoCAD++;
                        }
                    }
                }
                else
                {
                }
            }

            // HO faces: curved faces and CAD coverage of their vertices
            int cntFacesWithPoints    = 0;
            int cntHOFaceVertsCADSurf = 0;
            int cntHOFaceVertsNoCAD   = 0;

            // auto &curvedFaces = m_mesh->m_meshGraph->GetCurvedFaces();
            for (auto &[element, tag] : m_mesh->m_elementTags[2])
            {
                // auto it = curvedFaces.find(element->GetGlobalID());

                cntFacesWithPoints++;
                for (int k = 0; k < element->GetCurve()->m_points.size(); k++)
                {
                    auto *v = element->GetCurve()->m_points[k];
                    if (m_mesh->m_meshGraph->GetCADAssociation()->Count(
                            v, SpatialDomains::CADType::eSurf) > 0)
                    {
                        cntHOFaceVertsCADSurf++;
                    }
                    else
                    {
                        cntHOFaceVertsNoCAD++;
                    }
                }
            }

            m_log(VERBOSE) << " HO - information    " << endl;

            m_log(VERBOSE) << " HO Edges               N= "
                           << cntEdgesWithPoints << endl;
            m_log(VERBOSE) << "  HO-nodes on edges with CADCurve N= "
                           << cntHONodesWithCADCurve << endl;
            m_log(VERBOSE) << "  HO-nodes on edges with CADSurf  N= "
                           << cntHONodesWithCADSurf << endl;
            m_log(VERBOSE) << "  HO-nodes on edges - no CAD      N= "
                           << cntHONodesNoCAD << endl;

            m_log(VERBOSE) << " HO Faces               N= "
                           << cntFacesWithPoints << endl;
            m_log(VERBOSE) << "  HO-nodes on faces with CADSurf N= "
                           << cntHOFaceVertsCADSurf << endl;
            m_log(VERBOSE) << "  HO-nodes on faces - no CAD.    N= "
                           << cntHOFaceVertsNoCAD << endl;
        }
    }
}

void ProcessProjectCAD::LinkFaceToCAD(NekDouble tolv1)
{
    for (auto &[element, el_tag] : m_mesh->m_elementTags[2])
    {
        // vector<SpatialDomains::PointGeom> vertices =
        // element->GetVertexList(); vector<SpatialDomains::SegGeom> edges =
        // element->GetEdgeList();
        // CASE1 - all Edges internal to same CADSurf
        bool internal = true;
        for (int i = 1; i < element->GetNumEdges(); i++)
        {

            if ((m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(
                     element->GetEdge(i)) !=
                 m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(
                     element->GetEdge(i - 1))))
            {
                internal = false;
                break;
            }
        }

        if (internal && m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(
                            element->GetEdge(0)))
        {
            SpatialDomains::CADSurfSharedPtr cadSurf =
                m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(
                    element->GetEdge(0));

            m_mesh->m_meshGraph->GetCADAssociation()->Set(element, {cadSurf});
            continue;
        }

        // CASE2 and CASE3 - 2 or more CADSurfs or None (Use vertice CAD)
        vector<vector<int>> cmn;
        for (int i = 1; i < element->GetNumVerts(); i++)
        {

            vector<int> cmn_i = IntersectCADLinks(
                m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                    element->GetVertex(i), SpatialDomains::CADType::eSurf),
                m_mesh->m_meshGraph->GetCADAssociation()->GetLinks(
                    element->GetVertex(i - 1), SpatialDomains::CADType::eSurf));
            if (cmn_i.size() > 0)
            {
                cmn.push_back(cmn_i);
            }
            else
            {
                cmn.clear();
                break;
            }
        }

        if (cmn.size() == 0)
        {
            // CASE 3
            continue;
        }

        std::vector<int> commonCAD = cmn[0];
        for (auto cmn_i : cmn)
        {
            std::sort(cmn_i.begin(), cmn_i.end());
            vector<int> temp;
            std::set_intersection(commonCAD.begin(), commonCAD.end(),
                                  cmn_i.begin(), cmn_i.end(),
                                  std::back_inserter(temp));
            commonCAD = temp;
        }

        // commonCAD CADSurf found in the individual vertices
        if (commonCAD.size() == 1)
        {
            // Internal element based on the
            SpatialDomains::CADSurfSharedPtr cadSurf =
                m_mesh->m_meshGraph->GetCAD()->GetSurf(commonCAD[0]);

            m_mesh->m_meshGraph->GetCADAssociation()->Set(element, {cadSurf});

            for (int j = 0; j < element->GetNumEdges(); j++)
            {
                SpatialDomains::Geometry1D *edge = element->GetEdge(j);
                if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge) ==
                        nullptr &&
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge) ==
                        nullptr)
                {
                    m_mesh->m_meshGraph->GetCADAssociation()->Set(edge,
                                                                  {cadSurf});
                }
            }
        }
        else
        {
            // CASE 2 - 2 or more CADSurfs (max 1-2% of the faces)
            // This could be a trailing edge surface for example
            // Sliver Surface on IFW, etc
            // or very thin surface, where all vertices share >1 CAD Surf
            // Solution :: use edge nodes and face nodes to check which one
            // is closest FaceNode Effect x2, Edge

            // Get center of linear triag/quad
            std::array<NekDouble, 3> center = {0.0, 0.0, 0.0};
            for (int i = 0; i < element->GetNumVerts(); i++)
            {
                NekDouble x, y, z;
                element->GetVertex(i)->GetCoords(x, y, z);
                center[0] += x / element->GetNumVerts();
                center[1] += y / element->GetNumVerts();
                center[2] += z / element->GetNumVerts();
            }

            // Check mid of the distance of the centroids of edges and mid
            // face to every CAD
            NekDouble minDist = tolv1 * 10.0;
            int minID         = -1;
            for (int id : commonCAD)
            {
                std::array<NekDouble, 4> lim;
                SpatialDomains::CADSurfSharedPtr surf =
                    m_mesh->m_meshGraph->GetCAD()->GetSurf(id);
                surf->GetBounds(lim[0], lim[1], lim[2], lim[3]);
                std::array<NekDouble, 3> loc = {0.0, 0.0, 0.0};
                NekDouble distoveral         = 0.0;
                for (int j = 0; j < element->GetNumEdges(); j++)
                {
                    NekDouble x, y, z;
                    NekDouble x1, y1, z1;

                    SpatialDomains::Geometry1D *edge = element->GetEdge(j);
                    edge->GetVertex(0)->GetCoords(x, y, z);
                    edge->GetVertex(1)->GetCoords(x1, y1, z1);

                    loc[0] = (x + x1) * 0.5;
                    loc[1] = (y + y1) * 0.5;
                    loc[2] = (z + z1) * 0.5;

                    NekDouble dist = 1e7;

                    surf->locuv(loc, dist, lim[0], lim[1], lim[2], lim[3]);
                    distoveral += dist;
                }

                NekDouble dist = 1e7;
                surf->locuv(center, dist);
                distoveral += dist;

                if (distoveral < minDist)
                {
                    minID   = id;
                    minDist = distoveral;
                }
            }

            if (minID != -1)
            {
                SpatialDomains::CADSurfSharedPtr surf =
                    m_mesh->m_meshGraph->GetCAD()->GetSurf(minID);

                m_mesh->m_meshGraph->GetCADAssociation()->Set(element, {surf});

                for (int j = 0; j < element->GetNumEdges(); j++)
                {
                    SpatialDomains::Geometry1D *edge = element->GetEdge(j);
                    if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(
                            edge) == nullptr &&
                        m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(
                            edge) == nullptr)
                    {
                        m_mesh->m_meshGraph->GetCADAssociation()->Set(edge,
                                                                      {surf});
                    }
                }
            }

            // Choose the smallest one if within 1/10 min edge
        }
    }
}

void ProcessProjectCAD::LinkHOtoCAD(EdgeMap &surfEdges, NekDouble tolv1)
{
    m_log(VERBOSE) << " Associating HO-nodes to CAD" << endl;
    for (auto &[id, edge] : surfEdges)
    {
        if ((m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge) !=
                 nullptr ||
             m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge) !=
                 nullptr) &&
            edge->GetCurve()->m_points.size() > 0)
        {
            // loop over the edges and assign CADCurve
            if (m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge))
            {
                // CAD Curve
                SpatialDomains::CADCurveSharedPtr curve =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetCurve(edge);

                // loct() takes these as the parameter range to search in,
                // which is what keeps a periodic curve from projecting onto
                // the wrong lap. They were left uninitialised here, unlike
                // the surface cases below.
                std::array<NekDouble, 2> lim;
                curve->GetBounds(lim[0], lim[1]);

                for (auto node : edge->GetCurve()->m_points)
                {
                    NekDouble dist = 1e6;
                    NekDouble x, y, z;
                    node->GetCoords(x, y, z);
                    NekDouble t = curve->loct({x, y, z}, dist, lim[0], lim[1]);
                    if (dist < tolv1)
                    {
                        // Just give the node a parametric location
                        // DO NOT Project the node for the moment !
                        m_mesh->m_meshGraph->GetCADAssociation()->Add(
                            node, {curve, {t, 0.0}});
                    }
                }
            }
            else if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge))
            {
                // CAD Surf
                SpatialDomains::CADSurfSharedPtr surf =
                    m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(edge);

                std::array<NekDouble, 4> lim;
                surf->GetBounds(lim[0], lim[1], lim[2], lim[3]);
                for (auto node : edge->GetCurve()->m_points)
                {
                    NekDouble dist = 1e6;
                    NekDouble x, y, z;
                    node->GetCoords(x, y, z);

                    std::array<NekDouble, 2> uv = surf->locuv(
                        {x, y, z}, dist, lim[0], lim[1], lim[2], lim[3]);
                    if (dist < tolv1)
                    {
                        // Just give the node a parametric location
                        // DO NOT Project the node for the moment !
                        m_mesh->m_meshGraph->GetCADAssociation()->Add(
                            node, {surf, {uv[0], uv[1]}});
                    }
                    else
                    {
                        m_log(VERBOSE)
                            << "HO node distance too large  = " << dist
                            << "   at xyz=" << x << " " << y << " " << z
                            << endl;
                    }
                }
            }
        }
    }

    for (auto &[element, el_tag] : m_mesh->m_elementTags[2])
    {
        if (m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(element))
        {
            SpatialDomains::CADSurfSharedPtr surf =
                m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(element);
            std::array<NekDouble, 4> lim;
            surf->GetBounds(lim[0], lim[1], lim[2], lim[3]);

            // vector<NodeSharedPtr> nodelist;
            // face->GetCurvedNodes(nodelist);
            for (auto node : element->GetCurve()->m_points)
            {
                NekDouble dist = 1e6;
                NekDouble x, y, z;
                node->GetCoords(x, y, z);
                std::array<NekDouble, 2> uv = surf->locuv(
                    {x, y, z}, dist, lim[0], lim[1], lim[2], lim[3]);
                if (dist < tolv1 &&
                    m_mesh->m_meshGraph->GetCADAssociation()->Count(
                        node, SpatialDomains::CADType::eCurve) == 0)
                {
                    // Just give the node a parametric location
                    // DO NOT Project the node for the moment !
                    // Prioritise the CADCurve allocation if present !
                    m_mesh->m_meshGraph->GetCADAssociation()->Add(
                        node, {surf, {uv[0], uv[1]}});
                }
            }
        }
    }
}
} // namespace Nektar::NekMesh
