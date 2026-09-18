////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessBL.cpp
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
//  Description: Refine prismatic or quadrilateral boundary layer elements.
//
////////////////////////////////////////////////////////////////////////////////

#include <string>

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/BLPoints.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <LocalRegions/HexExp.h>
#include <LocalRegions/PrismExp.h>
#include <LocalRegions/QuadExp.h>

#include <NekMesh/Module/MacroElements.h>

#include "ProcessBL.h"
#include <NekMesh/MeshElements/Element.h>

using namespace std;

namespace Nektar::NekMesh
{
ModuleKey ProcessBL::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "bl"), ProcessBL::create,
    "Refines a prismatic boundary layers. Updated version - "
    "FaceNodes + EdgeNodes");

struct SplitMapHelper
{
    int nVerts;
    int dir;
    int oppositeFace;
    int nSides;
    std::array<int, 3> axes;
    std::vector<int> edgesToSplit;
    std::vector<std::array<int, 2>> sideEdgeVerts;
    std::vector<int> bndEdges;
    std::vector<int> sideFaces;
    // Standard-space coordinates of each element vertex.
    std::vector<std::array<int, 3>> stanVerts;
    LibUtilities::PointsKey pkey;
};

ProcessBL::ProcessBL(MeshSharedPtr m) : ProcessModule(m)
{
    // BL mesh configuration.
    m_config["layers"] =
        ConfigOption(false, "2", "Number of layers to refine.");
    m_config["nq"] =
        ConfigOption(false, "5", "Number of points in high order elements.");
    m_config["surf"] =
        ConfigOption(false, "", "Tag identifying surface connected to prism.");
    m_config["r"] =
        ConfigOption(false, "2.0", "Ratio to use in geometry progression.");
    m_config["partitioned"] = ConfigOption(
        true, "0",
        "Deterministically calculate element IDs from macro elements.");
}

ProcessBL::~ProcessBL()
{
}

void ProcessBL::Process()
{
    m_log(VERBOSE) << "Refining boundary layer." << endl;

    if (m_mesh->m_comm && m_mesh->m_comm->GetSize() > 1 &&
        !m_config["partitioned"].as<bool>())
    {
        m_log(WARNING) << "Running the 'bl' module in parallel without the "
                          "'partitioned' option"
                       << endl;
    }

    int dim = m_mesh->m_meshGraph->GetMeshDimension();
    switch (dim)
    {
        case 2:
            BoundaryLayer2D();
            break;

        case 3:
            BoundaryLayer3D();
            break;

        default:
            m_log(FATAL) << "Only 2D and 3D meshes supported." << endl;
            break;
    }

    // Renumber geometry IDs so they are filled from 0. Deterministic
    // (partitioned) ID calculation does not renumber; IDs are computed from the
    // macro elements instead.
    if (!m_config["partitioned"].as<bool>())
    {
        auto &graph = m_mesh->m_meshGraph;

        // Change geom IDs to be filled from 0.
        consolidateIDs<SpatialDomains::PointGeom>(0);

        SpatialDomains::CurveMap edgeCurves;
        consolidateIDs<SpatialDomains::SegGeom>(0, &edgeCurves);
        graph->GetCurvedEdges() = std::move(edgeCurves);

        SpatialDomains::CurveMap faceCurves;
        int newKey = consolidateIDs<SpatialDomains::TriGeom>(0, &faceCurves);
        consolidateIDs<SpatialDomains::QuadGeom>(newKey, &faceCurves);
        graph->GetCurvedFaces() = std::move(faceCurves);

        if (graph->GetMeshDimension() == 3)
        {
            newKey = consolidateIDs<SpatialDomains::HexGeom>(0);
            newKey = consolidateIDs<SpatialDomains::PyrGeom>(newKey);
            newKey = consolidateIDs<SpatialDomains::PrismGeom>(newKey);
            consolidateIDs<SpatialDomains::TetGeom>(newKey);
        }
    }

    ProcessVertices();
    ProcessElements();
    ProcessComposites();
}

void ProcessBL::BoundaryLayer2D()
{
    // This implementation of 2D does not support bidirectional splitting

    // Coarse elements displaced by the refinement below. They are extracted
    // from the graph rather than destroyed, so that a later module can update
    // the mesh dynamically from them; see MacroElements.
    auto &macroElements = m_mesh->GetContext().Get<MacroElements>().entities;

    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    int nodeId                               = graph->GetNvertices();
    int nl                                   = m_config["layers"].as<int>();

    // With a single layer there is nothing to split; ID consolidation is
    // handled in Process().
    if (nl == 1)
    {
        return;
    }

    int nq           = m_config["nq"].as<int>();
    bool partitioned = m_config["partitioned"].as<bool>();
    int maxInt       = std::numeric_limits<int>::max();

    // determine if geometric ratio is string or a constant.
    LibUtilities::Interpreter rEval;
    NekDouble r        = 1;
    int rExprId        = -1;
    bool ratioIsString = false;

    if (m_config["r"].isType<NekDouble>())
    {
        r = m_config["r"].as<NekDouble>();
    }
    else
    {
        std::string rstr = m_config["r"].as<string>();
        rExprId          = rEval.DefineFunction("x y z", rstr);
        ratioIsString    = true;
    }

    // Default PointsType.
    LibUtilities::PointsType pt = LibUtilities::eGaussLobattoLegendre;

    // Map from the global ID of a quad to be split to the local edge ID of the
    // quad which is on the boundary
    map<int, int> splitEls;

    // edgeSplitVertMap associates geometry edge IDs to the (nl+1) vertices
    // which are generated along that edge when a prism is split, and is used to
    // avoid generation of duplicate vertices. It is stored as an unordered map
    // for speed.
    std::unordered_map<int, vector<SpatialDomains::PointGeom *>>
        edgeSplitVertMap;
    // Same as above but for nl edges to avoid duplicates.
    std::unordered_map<int, vector<SpatialDomains::SegGeom *>> edgeSplitEdgeMap;

    string surf = m_config["surf"].as<string>();
    ASSERTL0(surf.size() > 0, "Surface must be specified.");

    vector<unsigned int> surfs;
    ParseUtils::GenerateSeqVector(surf, surfs);

    // Process list of elements to find those that are connected to surf.
    for (auto &pair : m_mesh->m_elementTags[2])
    {
        SpatialDomains::Geometry *el = pair.first;
        int nEdge                    = el->GetNumEdges();

        for (int j = 0; j < nEdge; ++j)
        {
            if (m_mesh->m_elementTags[1].find(el->GetEdge(j)) ==
                m_mesh->m_elementTags[1].end())
            {
                continue;
            }

            int tag = m_mesh->m_elementTags[1][el->GetEdge(j)];
            if (std::find(surfs.begin(), surfs.end(), tag) != surfs.end())
            {
                if (el->GetShapeType() != LibUtilities::eQuadrilateral)
                {
                    m_log(WARNING)
                        << "Found non-quad element to split in "
                        << "surface " << surf << "; ignoring" << endl;
                    continue;
                }

                if (splitEls.count(el->GetGlobalID()) > 0)
                {
                    m_log(WARNING) << "quad already found; ignoring" << endl;
                    continue;
                }

                splitEls[el->GetGlobalID()] = j;
            }
        }
    }

    if (splitEls.size() == 0)
    {
        m_log(WARNING) << "No elements detected to split." << endl;
        return;
    }

    auto elTags = m_mesh->m_elementTags;
    // Iterate over list of elements to be split
    for (auto &[elID, locBndEdgeId] : splitEls)
    {
        auto el = graph->GetQuadGeom(elID);

        // Find local ids of other boundary edges if any
        // We can ignore the edge opposite locBndEdgeId because it's reused
        std::set<int> otherBnds;
        for (int l = 0; l <= 1; ++l)
        {
            int locEdge = (locBndEdgeId + 1 + 2 * l) % 4;
            if (elTags[1].find(el->GetEdge(locEdge)) != elTags[1].end())
            {
                otherBnds.insert(locEdge);
            }
        }

        // Determine whether to use reverse points.
        // (if edges 1 or 2 are on the surface)
        LibUtilities::PointsType t =
            ((locBndEdgeId + 1) % 4) < 2
                ? LibUtilities::eBoundaryLayerPoints
                : LibUtilities::eBoundaryLayerPointsRev;

        if (ratioIsString) // determine value of r base on geom
        {
            NekDouble x, y, z;
            NekDouble x1, y1, z1;
            int nverts = el->GetNumVerts();

            x = y = z = 0.0;

            for (int i = 0; i < nverts; ++i)
            {
                el->GetVertex(i)->GetCoords(x1, y1, z1);
                x += x1;
                y += y1;
                z += z1;
            }
            x /= (NekDouble)nverts;
            y /= (NekDouble)nverts;
            z /= (NekDouble)nverts;
            r = rEval.Evaluate(rExprId, x, y, z, 0.0);
        }

        // Create basis.
        LibUtilities::BasisKey B0(LibUtilities::eModified_A, nq,
                                  LibUtilities::PointsKey(nq, pt));
        LibUtilities::BasisKey B1(LibUtilities::eModified_A, 2,
                                  LibUtilities::PointsKey(nl + 1, t, r));

        // Create local region.
        LocalRegions::QuadExpSharedPtr q;
        if (locBndEdgeId % 2)
        {
            q = MemoryManager<LocalRegions::QuadExp>::AllocateSharedPtr(B1, B0,
                                                                        el);
        }
        else
        {
            q = MemoryManager<LocalRegions::QuadExp>::AllocateSharedPtr(B0, B1,
                                                                        el);
        }

        // Grab co-ordinates.
        Array<OneD, NekDouble> x(nq * (nl + 1), 0.0);
        Array<OneD, NekDouble> y(nq * (nl + 1), 0.0);
        Array<OneD, NekDouble> z(nq * (nl + 1), 0.0);
        q->GetCoords(x, y, z);

        vector<vector<SpatialDomains::PointGeom *>> edgeNodes(2);

        // Loop over edges to be split.
        for (int l = 0; l < 2; ++l)
        {
            int locEdge = (locBndEdgeId + 1 + 2 * l) % 4;
            int edgeId  = el->GetEdge(locEdge)->GetGlobalID();

            // Determine whether we have already generated vertices
            // along this edge.
            auto eIt = edgeSplitVertMap.find(edgeId);

            if (eIt == edgeSplitVertMap.end())
            {
                // If not then resize storage to hold new points.
                edgeNodes[l].resize(nl + 1);

                // Re-use existing vertices at endpoints of edge to
                // avoid duplicating the existing vertices.
                edgeNodes[l][0]  = el->GetVertex(locEdge);
                edgeNodes[l][nl] = el->GetVertex((locEdge + 1) % 4);

                // Determine orientation for partition agreement
                bool edgeFwd = (el->GetEdge(locEdge)->GetVertex(0) ==
                                el->GetVertex(locEdge));

                // Variable geometric ratio
                if (ratioIsString)
                {
                    NekDouble x0, y0;
                    NekDouble x1, y1;
                    NekDouble xm, ym, zm = 0.0;

                    // -> Find edge end and mid points
                    x0 = (*edgeNodes[l][0])[0];
                    y0 = (*edgeNodes[l][0])[1];

                    x1 = (*edgeNodes[l][nl])[0];
                    y1 = (*edgeNodes[l][nl])[1];

                    xm = 0.5 * (x0 + x1);
                    ym = 0.5 * (y0 + y1);

                    // evaluate r factor based on mid point value
                    NekDouble rnew;
                    rnew = rEval.Evaluate(rExprId, xm, ym, zm, 0.0);

                    // Get basis with new r;
                    t = (l == 0) ? LibUtilities::eBoundaryLayerPoints
                                 : LibUtilities::eBoundaryLayerPointsRev;
                    LibUtilities::PointsKey Pkey(nl + 1, t, rnew);
                    LibUtilities::PointsSharedPtr newP =
                        LibUtilities::PointsManager()[Pkey];

                    const Array<OneD, const NekDouble> z = newP->GetZ();

                    // Create new interior nodes based on this new blend
                    for (int k = 1; k < nl; ++k)
                    {
                        xm = 0.5 * (1 + z[k]) * (x1 - x0) + x0;
                        ym = 0.5 * (1 + z[k]) * (y1 - y0) + y0;
                        zm = 0.0;

                        int kc     = edgeFwd ? k : nl - k;
                        int id     = partitioned ? maxInt - (edgeId * nl + kc)
                                                 : nodeId++;
                        auto point = ObjPoolManager<SpatialDomains::PointGeom>::
                            AllocateUniquePtr(graph->GetSpaceDimension(), id,
                                              xm, ym, zm);
                        edgeNodes[l][k] = point.get();
                        graph->AddGeom<SpatialDomains::PointGeom>(
                            edgeNodes[l][k]->GetGlobalID(), std::move(point));
                    }
                }
                else
                {
                    // Create new interior nodes.
                    int pos = 0;
                    for (int k = 1; k < nl; ++k)
                    {
                        switch (locEdge)
                        {
                            case 0:
                                pos = k;
                                break;
                            case 1:
                                pos = nq - 1 + k * nq;
                                break;
                            case 2:
                                pos = nq * (nl + 1) - 1 - k;
                                break;
                            case 3:
                                pos = nq * nl - k * nq;
                                break;
                            default:
                                NEKERROR(ErrorUtil::efatal,
                                         "Quad edge should be < 4.");
                                break;
                        }

                        int kc     = edgeFwd ? k : nl - k;
                        int id     = partitioned ? maxInt - (edgeId * nl + kc)
                                                 : nodeId++;
                        auto point = ObjPoolManager<SpatialDomains::PointGeom>::
                            AllocateUniquePtr(graph->GetSpaceDimension(), id,
                                              x[pos], y[pos], z[pos]);
                        edgeNodes[l][k] = point.get();
                        graph->AddGeom<SpatialDomains::PointGeom>(
                            edgeNodes[l][k]->GetGlobalID(), std::move(point));
                    }
                }

                // Store these edges in edgeSplitVertMap.
                edgeSplitVertMap[edgeId] = edgeNodes[l];
            }
            else
            {
                // Check orientation
                if (eIt->second[0] == el->GetVertex(locEdge))
                {
                    // Same orientation: copy nodes
                    edgeNodes[l] = eIt->second;
                }
                else
                {
                    // Reversed orientation: copy in reversed order
                    edgeNodes[l].resize(nl + 1);
                    for (int k = 0; k < nl + 1; ++k)
                    {
                        edgeNodes[l][k] = eIt->second[nl - k];
                    }
                }
            }
            if (eIt != edgeSplitVertMap.end() ||
                otherBnds.find(locEdge) != otherBnds.end())
            {
                // Parent edge removed from boundary tags and extracted from
                // meshGraph if it's been used for the second time or lies on
                // another boundary
                m_mesh->m_elementTags[1].erase(el->GetEdge(locEdge));
                macroElements.m_segVec.push_back(
                    graph->ExtractGeom<SpatialDomains::SegGeom>(edgeId, true));
                auto it = graph->GetCurvedEdges().find(edgeId);
                if (it != graph->GetCurvedEdges().end())
                {
                    macroElements.m_curveVec.push_back(std::move(it->second));
                    graph->GetCurvedEdges().erase(edgeId);
                }
            }
        }

        // Create element layers, now always from boundary towards interior.
        for (int j = 0; j < nl; ++j)
        {
            // Get corner vertices. "-j" is seen for edgeNodes[1] because
            // working clockwise this was populated from interior towards
            // boundary
            std::vector<SpatialDomains::PointGeom *> nodeList(4);
            switch (locBndEdgeId)
            {
                case 0:
                {
                    nodeList[0] = edgeNodes[1][nl - j];
                    nodeList[1] = edgeNodes[0][j];
                    nodeList[2] = edgeNodes[0][j + 1];
                    nodeList[3] = edgeNodes[1][nl - j - 1];
                    break;
                }
                case 1:
                {
                    nodeList[0] = edgeNodes[1][nl - j - 1];
                    nodeList[1] = edgeNodes[1][nl - j];
                    nodeList[2] = edgeNodes[0][j];
                    nodeList[3] = edgeNodes[0][j + 1];
                    break;
                }
                case 2:
                {
                    nodeList[0] = edgeNodes[0][j + 1];
                    nodeList[1] = edgeNodes[1][nl - j - 1];
                    nodeList[2] = edgeNodes[1][nl - j];
                    nodeList[3] = edgeNodes[0][j];
                    break;
                }
                case 3:
                {
                    nodeList[0] = edgeNodes[0][j];
                    nodeList[1] = edgeNodes[0][j + 1];
                    nodeList[2] = edgeNodes[1][nl - j - 1];
                    nodeList[3] = edgeNodes[1][nl - j];
                    break;
                }
            }

            // Explicit edge (4) and element ids for the new quad, derived
            // deterministically from the macro quad so partitions agree.
            ElmtIds forceIDs;
            if (partitioned)
            {
                forceIDs.edges.resize(4);
                for (int e = 0; e < 4; ++e)
                {
                    forceIDs.edges[e] = maxInt - (el->GetEid(e) * nl + j);
                }
                forceIDs.elmt = maxInt - (el->GetGlobalID() * nl + j);
            }

            // Create element
            ElmtConfig conf(LibUtilities::eQuadrilateral, 1, false, false,
                            false);
            SpatialDomains::Geometry *element =
                GetElementFactory().CreateInstance(
                    LibUtilities::eQuadrilateral, nodeList, m_mesh->m_meshGraph,
                    m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                    nullptr, nullptr, partitioned ? &forceIDs : nullptr);
            auto elmt = static_cast<SpatialDomains::QuadGeom *>(element);

            // Copy over tags for new element and edges on otherBnds
            m_mesh->m_elementTags[2][elmt] = m_mesh->m_elementTags[2][el];
            for (auto &locEdge : otherBnds)
            {
                m_mesh->m_elementTags[1][elmt->GetEdge(locEdge)] =
                    elTags[1][el->GetEdge(locEdge)];
            }

            // Add high order nodes to boundary side edge of child quad.
            if (j > 0)
            {
                auto edge = dynamic_cast<SpatialDomains::SegGeom *>(
                    elmt->GetEdge(locBndEdgeId));
                auto curve =
                    ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                        edge->GetGlobalID(), pt);
                curve->m_points.push_back(edge->GetVertex(0));
                int pos = 0;
                for (int k = 1; k < nq - 1; ++k)
                {
                    switch (locBndEdgeId)
                    {
                        case 0:
                            pos = j * nq + k;
                            break;
                        case 1:
                            pos = nl - j + k * (nl + 1);
                            break;
                        case 2:
                            pos = (nl - j) * nq + (nq - 1) - k;
                            break;
                        case 3:
                            pos = (nl + 1) * (nq - 1) + j - k * (nl + 1);
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "Quad edge should be < 4.");
                            break;
                    }
                    auto point = ObjPoolManager<SpatialDomains::PointGeom>::
                        AllocateUniquePtr(graph->GetSpaceDimension(),
                                          curve->m_curveID, x[pos], y[pos],
                                          0.0);
                    curve->m_points.push_back(point.get());
                    graph->GetAllCurveNodes().push_back(std::move(point));
                }
                curve->m_points.push_back(edge->GetVertex(1));

                StdRegions::Orientation edgeOrient =
                    SpatialDomains::SegGeom::GetEdgeOrientation(
                        *static_cast<SpatialDomains::SegGeom *>(
                            elmt->GetEdge(locBndEdgeId)),
                        *static_cast<SpatialDomains::SegGeom *>(
                            elmt->GetEdge((locBndEdgeId + 1) % 4)));
                if (edgeOrient == StdRegions::eBackwards)
                {
                    std::reverse(curve->m_points.begin() + 1,
                                 curve->m_points.end() - 1);
                }
                edge->SetCurve(curve.get());
                graph->GetCurvedEdges()[edge->GetGlobalID()] = std::move(curve);
            }
        }

        // Macro quad removed from element tags and extracted from meshGraph
        m_mesh->m_elementTags[2].erase(el);
        macroElements.m_quadVec.push_back(
            graph->ExtractGeom<SpatialDomains::QuadGeom>(el->GetGlobalID(),
                                                         true));
        auto it = graph->GetCurvedFaces().find(el->GetGlobalID());
        if (it != graph->GetCurvedFaces().end())
        {
            macroElements.m_curveVec.push_back(std::move(it->second));
            graph->GetCurvedFaces().erase(el->GetGlobalID());
        }
    }
}

void ProcessBL::BoundaryLayer3D()
{
    m_log(VERBOSE) << "Elements before = " << m_mesh->m_elementTags[3].size()
                   << endl;

    // Coarse elements displaced by the refinement below. They are extracted
    // from the graph rather than destroyed, so that a later module can update
    // the mesh dynamically from them; see MacroElements.
    auto &macroElements = m_mesh->GetContext().Get<MacroElements>().entities;

    // A set containing all element types which are valid.
    set<LibUtilities::ShapeType> validElTypes;
    validElTypes.insert(LibUtilities::ePrism);
    validElTypes.insert(LibUtilities::eHexahedron);

    // int nodeId = m_mesh->m_vertexSet.size();
    int nl = m_config["layers"].as<int>();

    // With a single layer there is nothing to split
    if (nl == 1)
    {
        return;
    }

    int nq           = m_config["nq"].as<int>();
    bool partitioned = m_config["partitioned"].as<bool>();
    int maxInt       = std::numeric_limits<int>::max();
    auto &graph      = m_mesh->m_meshGraph;

    if (!m_config["r"].isType<NekDouble>())
    {
        m_log(FATAL) << "R is string only in 2D possible - give Double."
                     << endl;
    }
    NekDouble r = m_config["r"].as<NekDouble>();

    // Default PointsType.
    LibUtilities::PointsKey ekey(
        nq, LibUtilities::eGaussLobattoLegendre); // ePolyEvenlySpaced
                                                  // //eGaussLobattoLegendre
    Array<OneD, NekDouble> gll;
    LibUtilities::PointsManager()[ekey]->GetPoints(gll);

    // Default FaceNode point
    LibUtilities::PointsKey pkeyHex(nq, LibUtilities::eGaussLobattoLegendre);
    LibUtilities::PointsKey pkeyPrism(nq, LibUtilities::eNodalTriElec);
    Array<OneD, NekDouble> u_prismFaceNodes, v_prismFaceNodes;
    LibUtilities::PointsManager()[pkeyPrism]->GetPoints(u_prismFaceNodes,
                                                        v_prismFaceNodes);

    // Map which takes element ID to face on surface. This enables
    // splitting to occur in either y-direction of the prism.
    unordered_map<int, int> splitEls;

    // Set up maps which takes an edge (in nektar++ ordering) and return
    // their offset and stride in the 3d array of collapsed quadrature
    // points. Note that this map includes only the edges that are on
    // the triangular faces as the edges in the normal direction are
    // linear.
    // Each entry is keyed by the local ID of the face being split and encodes,
    // in aggregate-initialiser order:
    //   nVerts, dir, oppositeFace, nSides, axes,
    //   edgesToSplit, sideEdgeVerts, bndEdges, sideFaces, stanVerts, pkey
    map<LibUtilities::ShapeType, map<int, SplitMapHelper>> splitMap;
    auto &hexMap   = splitMap[LibUtilities::eHexahedron];
    auto &prismMap = splitMap[LibUtilities::ePrism];

    // Standard-space (collapsed) coordinates of each element vertex, shared by
    // all splitting directions of a given shape.
    const std::vector<std::array<int, 3>> hexStanVerts = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    const std::vector<std::array<int, 3>> prisStanVerts = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1},
        {-1, 1, -1},  {-1, -1, 1}, {-1, 1, 1}};

    // HEX DIR X
    hexMap[4] = {8,
                 0,
                 2,
                 4,
                 {1, 2, 0},
                 {0, 2, 10, 8},
                 {{0, 1}, {3, 2}, {7, 6}, {4, 5}},
                 {3, 7, 11, 4},
                 {0, 3, 5, 1},
                 hexStanVerts,
                 pkeyHex};
    hexMap[2] = {8,
                 1,
                 4,
                 4,
                 {1, 2, 0},
                 {0, 2, 10, 8},
                 {{1, 0}, {2, 3}, {6, 7}, {5, 4}},
                 {1, 6, 9, 5},
                 {0, 3, 5, 1},
                 hexStanVerts,
                 pkeyHex};

    // HEX DIR Y
    hexMap[1] = {8,
                 0,
                 3,
                 4,
                 {0, 2, 1},
                 {3, 1, 9, 11},
                 {{0, 3}, {1, 2}, {5, 6}, {4, 7}},
                 {0, 5, 8, 4},
                 {0, 2, 5, 4},
                 hexStanVerts,
                 pkeyHex};
    hexMap[3] = {8,
                 1,
                 1,
                 4,
                 {0, 2, 1},
                 {1, 3, 11, 9},
                 {{2, 1}, {3, 0}, {7, 4}, {6, 5}},
                 {2, 7, 10, 6},
                 {0, 4, 5, 2},
                 hexStanVerts,
                 pkeyHex};

    // HEX DIR Z
    hexMap[0] = {8,
                 0,
                 5,
                 4,
                 {0, 1, 2},
                 {4, 5, 6, 7},
                 {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
                 {0, 1, 2, 3},
                 {1, 2, 3, 4},
                 hexStanVerts,
                 pkeyHex};
    hexMap[5] = {8,
                 1,
                 0,
                 4,
                 {0, 1, 2},
                 {4, 5, 6, 7},
                 {{4, 0}, {5, 1}, {6, 2}, {7, 3}},
                 {8, 9, 10, 11},
                 {1, 2, 3, 4},
                 hexStanVerts,
                 pkeyHex};

    // PRISM DIR Y
    prismMap[1] = {6,
                   0,
                   3,
                   3,
                   {0, 2, 1},
                   {3, 1, 8},
                   {{0, 3}, {1, 2}, {4, 5}},
                   {0, 5, 4},
                   {0, 2, 4},
                   prisStanVerts,
                   pkeyPrism};
    prismMap[3] = {6,
                   1,
                   1,
                   3,
                   {0, 2, 1},
                   {1, 3, 8},
                   {{2, 1}, {3, 0}, {5, 4}},
                   {2, 7, 6},
                   {0, 4, 2},
                   prisStanVerts,
                   pkeyPrism};

    string surf = m_config["surf"].as<string>();
    if (surf.size() == 0)
    {
        m_log(WARNING) << "No surfaces detected to split, continuing." << endl;
        return;
    }
    vector<unsigned int> surfs;
    ParseUtils::GenerateSeqVector(surf.c_str(), surfs);

    // If surface is defined, process list of elements to find those
    // that are connected to it.
    for (auto &[el, el_tag] : m_mesh->m_elementTags[3])
    {
        int nSurf = el->GetNumFaces();

        for (int j = 0; j < nSurf; ++j)
        {
            auto eIt = m_mesh->m_elementTags[2].find(el->GetFace(j));
            if (eIt == m_mesh->m_elementTags[2].end())
            {
                continue;
            }

            int face_tag = eIt->second;
            if (std::find(surfs.begin(), surfs.end(), face_tag) != surfs.end())
            {
                if (el->GetShapeType() == LibUtilities::eHexahedron)
                {
                    map<int, SplitMapHelper>::iterator f =
                        splitMap[LibUtilities::eHexahedron].find(j);
                    if (f == splitMap[LibUtilities::eHexahedron].end())
                    {
                        m_log(WARNING) << "Splitting hex on face " << j
                                       << " unsupported" << endl;
                        continue;
                    }

                    if (splitEls.count(el->GetGlobalID()) > 0)
                    {
                        m_log(WARNING) << "Hex already found; "
                                       << "ignoring" << endl;
                    }

                    splitEls[el->GetGlobalID()] = j;
                }
                else if (el->GetShapeType() == LibUtilities::ePrism)
                {
                    map<int, SplitMapHelper>::iterator f =
                        splitMap[LibUtilities::ePrism].find(j);
                    if (f == splitMap[LibUtilities::ePrism].end())
                    {
                        m_log(WARNING) << "Splitting prism on face " << j
                                       << " unsupported" << endl;
                        continue;
                    }

                    if (splitEls.count(el->GetGlobalID()) > 0)
                    {
                        m_log(WARNING) << "Prism already found; "
                                       << "ignoring" << endl;
                    }

                    splitEls[el->GetGlobalID()] = j;
                }
                else if (validElTypes.count(el->GetShapeType()) == 0)
                {
                    m_log(WARNING) << "Unsupported element type "
                                   << "found in surface " << j << "; "
                                   << "ignoring" << endl;
                    continue;
                }
            }
        }
    }

    if (splitEls.size() == 0)
    {
        m_log(WARNING) << "No elements detected to split; continuing." << endl;
        return;
    }

    // 1. Create a copy of the normal edges inside the boundary layer.
    std::unordered_map<int, SpatialDomains::SegGeom *> ElmtEdgesCopy;
    SpatialDomains::EntityHolder holder;

    // locBndFaceID is the local face ID of the face lying on the
    // composite selected to be split
    for (auto &[elID, locBndFaceID] : splitEls)
    {
        auto el            = graph->GetGeometry3D(elID);
        SplitMapHelper &sm = splitMap[el->GetShapeType()][locBndFaceID];

        // 1.2. Insert a copy of the edges (unique)
        for (int j = 0; j < sm.nSides; j++)
        {
            int eID = el->GetEid(sm.edgesToSplit[j]);
            auto it = ElmtEdgesCopy.find(eID);
            if (it == ElmtEdgesCopy.end())
            {
                auto curve =
                    ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                        eID, LibUtilities::eNoPointsType);

                std::array<SpatialDomains::PointGeom *, 2> verts = {
                    el->GetVertex(sm.sideEdgeVerts[j][0]),
                    el->GetVertex(sm.sideEdgeVerts[j][1])};
                auto edgeCopy =
                    ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                        eID, 3, verts, curve.get());

                //  1.2.1 Add the ID and the Edge pointer to the map
                ElmtEdgesCopy[eID] = edgeCopy.get();
                holder.m_curveVec.push_back(std::move(curve));
                holder.m_segVec.push_back(std::move(edgeCopy));
            }
        }
    }

    // 2. Create the edge splitting and fill the ElmtEdgesCopy with BL
    // distributions
    LibUtilities::PointsKey bkey(nl + 1, LibUtilities::eBoundaryLayerPoints, r);
    Array<OneD, NekDouble> BLPoints;
    LibUtilities::PointsManager()[bkey]->GetPoints(BLPoints);

    for (auto &[elID, locBndFaceID] : splitEls)
    {
        auto el            = graph->GetGeometry3D(elID);
        SplitMapHelper &sm = splitMap[el->GetShapeType()][locBndFaceID];

        // 2.4 Loop over the Edges to be split -> need to populate EdgeNodes of
        // the CopyEdges !
        for (int j = 0; j < sm.nSides; j++)
        {
            auto edge = static_cast<SpatialDomains::SegGeom *>(
                el->GetEdge(sm.edgesToSplit[j]));
            SpatialDomains::SegGeom *edgeCopy =
                ElmtEdgesCopy[edge->GetGlobalID()];
            SpatialDomains::Curve *curve = edgeCopy->GetCurve();

            // 2.4.1 Skip all edge copies that already have the BL
            // distribution as EdgeNodes
            if (curve->m_points.size() != 0)
            {
                continue;
            }

            // 2.4.3 Create the EDGE expansion (Important to do it on the
            // original Curved edge to keep the VarOpti curvature in the Edge
            // 1/3/8 of the prism )
            edge->FillGeom();

            StdRegions::StdExpansionSharedPtr xmap = edge->GetXmap();
            Array<OneD, NekDouble> coeffs0         = edge->GetCoeffs(0);
            Array<OneD, NekDouble> coeffs1         = edge->GetCoeffs(1);
            Array<OneD, NekDouble> coeffs2         = edge->GetCoeffs(2);
            Array<OneD, NekDouble> xc(xmap->GetTotPoints());
            Array<OneD, NekDouble> yc(xmap->GetTotPoints());
            Array<OneD, NekDouble> zc(xmap->GetTotPoints());
            xmap->BwdTrans(coeffs0, xc);
            xmap->BwdTrans(coeffs1, yc);
            xmap->BwdTrans(coeffs2, zc);

            // 2.4.4 Generate the new BL vertices, add them to
            // graph PointGeoms, add them as EdgeNodes to the CopyEdge
            curve->m_points.push_back(edgeCopy->GetVertex(0));
            for (int k = 1; k < nl; k++)
            {
                Array<OneD, NekDouble> xp(1);
                xp[0] = edge->GetVertex(0) == edgeCopy->GetVertex(0)
                            ? BLPoints[k]
                            : -1 * BLPoints[k];

                std::array<NekDouble, 3> loc;
                loc[0] = xmap->PhysEvaluate(xp, xc);
                loc[1] = xmap->PhysEvaluate(xp, yc);
                loc[2] = xmap->PhysEvaluate(xp, zc);

                int id;
                if (partitioned)
                {
                    id = maxInt - (el->GetVid(sm.sideEdgeVerts[j][0]) * nl + k);
                }
                else
                {
                    id = NextPointId(graph);
                }

                auto NewBLVertex =
                    graph->CreatePointGeom(3, id, loc[0], loc[1], loc[2]);

                /* TS: m_parentCAD not supported yet for SpatialDomains
                geoms
                // If this edge is attached to some CAD, then perform a
                // reverse projection to determine parametrisation on
                the
                // CAD curve/surface.
                if (edge->m_parentCAD)
                {
                    if (edge->m_parentCAD->GetType() == CADType::eCurve)
                    {
                        CADCurveSharedPtr c =
                            std::dynamic_pointer_cast<CADCurve>(
                                edge->m_parentCAD);
                        NekDouble t;
                        c->loct(loc, t);
                        NewBLVertex->SetCADCurve(c, t);
                    }
                    else if (edge->m_parentCAD->GetType() ==
                CADType::eSurf)
                    {
                        CADSurfSharedPtr s =
                std::dynamic_pointer_cast<CADSurf>( edge->m_parentCAD);
                        auto uv = s->locuv(loc);
                        NewBLVertex->SetCADSurf(s, uv);
                    }
                }
                */

                // Add the Vertex to the edgeCopy edgenodes
                curve->m_points.push_back(NewBLVertex);
            }
            curve->m_points.push_back(edgeCopy->GetVertex(1));
        }
    }

    // 3.Split Elements High-order-> Create edges ; Create New Elements ;
    // Translate the FaceNodes !!!

    std::map<int, std::set<SpatialDomains::Geometry *>> elmtsToRemove;
    for (auto &[elID, locBndFaceID] : splitEls)
    {
        auto el_macro      = graph->GetGeometry3D(elID);
        SplitMapHelper &sm = splitMap[el_macro->GetShapeType()][locBndFaceID];

        // Find local ids of other boundary faces if any
        // Can ignore boundary-opposite face because it's reused from el_macro
        std::set<int> otherBnds;
        for (int f : sm.sideFaces)
        {
            if (m_mesh->m_elementTags[2].find(el_macro->GetFace(f)) !=
                m_mesh->m_elementTags[2].end())
            {
                otherBnds.insert(f);
            }
        }

        // 3.2. Create the expansion of the Macro Prism Element
        el_macro->FillGeom();

        StdRegions::StdExpansionSharedPtr xmap = el_macro->GetXmap();
        Array<OneD, NekDouble> coeffs0         = el_macro->GetCoeffs(0);
        Array<OneD, NekDouble> coeffs1         = el_macro->GetCoeffs(1);
        Array<OneD, NekDouble> coeffs2         = el_macro->GetCoeffs(2);

        Array<OneD, NekDouble> xc(xmap->GetTotPoints());
        Array<OneD, NekDouble> yc(xmap->GetTotPoints());
        Array<OneD, NekDouble> zc(xmap->GetTotPoints());

        // Does Backwards transform from Std to Cartesian based on quadrature
        // nodes
        xmap->BwdTrans(coeffs0, xc); // uses the expansion coeffs in STD to fill
                                     // all Element nodes in Physical spaces
        xmap->BwdTrans(coeffs1, yc);
        xmap->BwdTrans(coeffs2, zc);

        // 3.4 Create Vector of Vertices for the Copy Edges which always start
        // from the Boundary/Curved Face
        vector<vector<SpatialDomains::PointGeom *>> EdgeBL(sm.nSides);
        for (int i = 0; i < sm.nSides; i++)
        {
            EdgeBL[i] = ElmtEdgesCopy[el_macro->GetEid(sm.edgesToSplit[i])]
                            ->GetCurve()
                            ->m_points;
        }

        // 3.5 Create HO Elements
        vector<SpatialDomains::PointGeom *> NodeList(sm.nVerts);
        for (int j = 0; j < nl; j++)
        {
            // 3.5.1 Create the NodeList
            for (int i = 0; i < sm.nSides; i++)
            {
                for (int k : {0, 1})
                {
                    NodeList[sm.sideEdgeVerts[i][k]] = EdgeBL[i][j + k];
                }
            }

            // Explicit edge, face and element ids for the new element.
            ElmtIds forceIDs;
            if (partitioned)
            {
                for (int e = 0; e < el_macro->GetNumEdges(); ++e)
                {
                    forceIDs.edges.push_back(maxInt -
                                             (el_macro->GetEid(e) * nl + j));
                }
                for (int f = 0; f < el_macro->GetNumFaces(); ++f)
                {
                    forceIDs.faces.push_back(maxInt -
                                             (el_macro->GetFid(f) * nl + j));
                }
                forceIDs.elmt = maxInt - (el_macro->GetGlobalID() * nl + j);
            }

            // 3.5.2 Create the Linear Element
            LibUtilities::ShapeType shapeType = el_macro->GetShapeType();
            ElmtConfig conf(shapeType, 1, false, false, false);
            SpatialDomains::Geometry *elmt_new =
                GetElementFactory().CreateInstance(
                    shapeType, NodeList, graph, m_mesh->m_edgeSet,
                    m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr,
                    partitioned ? &forceIDs : nullptr);

            // // 3.5.3 Copy CAD Dependencies
            // elmt_new->m_parentCAD = el_macro->m_parentCAD;

            // Copy over tags for new element and edges on otherBnds
            m_mesh->m_elementTags[3][elmt_new] =
                m_mesh->m_elementTags[3][el_macro];
            for (auto &locFace : otherBnds)
            {
                m_mesh->m_elementTags[2][elmt_new->GetFace(locFace)] =
                    m_mesh->m_elementTags[2][el_macro->GetFace(locFace)];
            }

            // 3.5.4 For the inner layers create HO EdgeNodes
            Array<OneD, NekDouble> xp(3);
            xp[sm.axes[2]] = sm.dir ? -1 * BLPoints[j] : BLPoints[j];

            for (int i = 0; i < sm.nSides; ++i)
            {
                auto edgeNew = static_cast<SpatialDomains::SegGeom *>(
                    elmt_new->GetEdge(sm.bndEdges[i]));
                if (edgeNew->GetCurve() == nullptr && j != 0)
                {
                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(edgeNew->GetGlobalID(),
                                          LibUtilities::PointsManager()[ekey]
                                              ->GetPointsType());

                    curve->m_points.push_back(edgeNew->GetVertex(0));
                    for (int k = 1; k < nq - 1; k++)
                    {
                        auto stanV0 = sm.stanVerts[sm.sideEdgeVerts[i][0]];
                        auto stanV1 =
                            sm.stanVerts[sm.sideEdgeVerts[(i + 1) % sm.nSides]
                                                         [0]];
                        xp[sm.axes[0]] =
                            stanV0[sm.axes[0]] * (1.0 - gll[k]) / 2.0 +
                            stanV1[sm.axes[0]] * (1.0 + gll[k]) / 2.0;
                        xp[sm.axes[1]] =
                            stanV0[sm.axes[1]] * (1.0 - gll[k]) / 2.0 +
                            stanV1[sm.axes[1]] * (1.0 + gll[k]) / 2.0;

                        Array<OneD, NekDouble> loc(3);
                        loc[0] = xmap->PhysEvaluate(xp, xc);
                        loc[1] = xmap->PhysEvaluate(xp, yc);
                        loc[2] = xmap->PhysEvaluate(xp, zc);

                        curve->m_points.push_back(graph->CreateCurveNode(
                            3, 0, loc[0], loc[1], loc[2]));
                    }
                    curve->m_points.push_back(edgeNew->GetVertex(1));

                    if (edgeNew->GetVid(1) ==
                        elmt_new->GetVid(sm.sideEdgeVerts[i][0]))
                    {
                        std::reverse(curve->m_points.begin() + 1,
                                     curve->m_points.end() - 1);
                    }
                    edgeNew->SetCurve(curve.get());
                    graph->GetCurvedEdges()[edgeNew->GetGlobalID()] =
                        std::move(curve);
                }
            }

            // 3.5.6 Add the FaceNodes as a translation of the
            // BL distribution in Xi2 direction
            auto curvedFace = elmt_new->GetFace(locBndFaceID);
            if (curvedFace->GetCurve() == nullptr && j != 0)
            {
                auto curve =
                    ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                        elmt_new->GetFid(locBndFaceID),
                        LibUtilities::PointsManager()[sm.pkey]
                            ->GetPointsType());

                vector<array<int, 3>> stanVs(sm.nSides);
                for (int v = 0; v < curvedFace->GetNumVerts(); v++)
                {
                    int vID = curvedFace->GetVid(v);
                    for (int vp = 0; vp < sm.nVerts; vp++)
                    {
                        if (elmt_new->GetVid(vp) == vID)
                        {
                            stanVs[v] = sm.stanVerts[vp];
                            break;
                        }
                    }
                }

                if (el_macro->GetShapeType() == LibUtilities::ePrism)
                {
                    for (int v = 0; v < curvedFace->GetNumVerts(); v++)
                    {
                        curve->m_points.push_back(curvedFace->GetVertex(v));
                    }
                    for (int k = 3; k < u_prismFaceNodes.size(); k++)
                    {
                        // Barycentric interp over the triangle; N0..N2 are the
                        // weights for the vertices at (xi0,xi1) = (-1,-1),
                        // (1,-1), (-1,1).
                        NekDouble xi0 = u_prismFaceNodes[k];
                        NekDouble xi1 = v_prismFaceNodes[k];
                        NekDouble N0  = -(xi0 + xi1) / 2.0;
                        NekDouble N1  = (xi0 + 1.0) / 2.0;
                        NekDouble N2  = (xi1 + 1.0) / 2.0;

                        xp[sm.axes[0]] = N0 * stanVs[0][sm.axes[0]] +
                                         N1 * stanVs[1][sm.axes[0]] +
                                         N2 * stanVs[2][sm.axes[0]];
                        xp[sm.axes[1]] = N0 * stanVs[0][sm.axes[1]] +
                                         N1 * stanVs[1][sm.axes[1]] +
                                         N2 * stanVs[2][sm.axes[1]];

                        Array<OneD, NekDouble> loc(3);
                        loc[0] = xmap->PhysEvaluate(xp, xc);
                        loc[1] = xmap->PhysEvaluate(xp, yc);
                        loc[2] = xmap->PhysEvaluate(xp, zc);

                        curve->m_points.push_back(graph->CreateCurveNode(
                            3, 0, loc[0], loc[1], loc[2]));
                    }
                }
                else
                {
                    curve->m_points.resize(nq * nq);
                    curve->m_points[0]             = curvedFace->GetVertex(0);
                    curve->m_points[nq - 1]        = curvedFace->GetVertex(1);
                    curve->m_points[(nq * nq) - 1] = curvedFace->GetVertex(2);
                    curve->m_points[nq * (nq - 1)] = curvedFace->GetVertex(3);
                    for (int i = 0; i < nq; i++)
                    {
                        for (int k = 0; k < nq; k++)
                        {
                            if ((i == 0 || i == nq - 1) &&
                                (k == 0 || k == nq - 1))
                            {
                                continue;
                            }
                            else
                            {
                                // Bilinear interp over the quad; N0..N3 are
                                // the weights for the vertices at (xi0,xi1) =
                                // (-1,-1), (1,-1), (1,1), (-1,1). xi0 = gll[k]
                                // runs V0->V1, xi1 = gll[i] runs V0->V3.
                                NekDouble xi0 = gll[k];
                                NekDouble xi1 = gll[i];
                                NekDouble N0  = (1.0 - xi0) * (1.0 - xi1) / 4.0;
                                NekDouble N1  = (1.0 + xi0) * (1.0 - xi1) / 4.0;
                                NekDouble N2  = (1.0 + xi0) * (1.0 + xi1) / 4.0;
                                NekDouble N3  = (1.0 - xi0) * (1.0 + xi1) / 4.0;

                                xp[sm.axes[0]] = N0 * stanVs[0][sm.axes[0]] +
                                                 N1 * stanVs[1][sm.axes[0]] +
                                                 N2 * stanVs[2][sm.axes[0]] +
                                                 N3 * stanVs[3][sm.axes[0]];
                                xp[sm.axes[1]] = N0 * stanVs[0][sm.axes[1]] +
                                                 N1 * stanVs[1][sm.axes[1]] +
                                                 N2 * stanVs[2][sm.axes[1]] +
                                                 N3 * stanVs[3][sm.axes[1]];

                                Array<OneD, NekDouble> loc(3);
                                loc[0] = xmap->PhysEvaluate(xp, xc);
                                loc[1] = xmap->PhysEvaluate(xp, yc);
                                loc[2] = xmap->PhysEvaluate(xp, zc);

                                curve->m_points[(i * nq) + k] =
                                    graph->CreateCurveNode(3, 0, loc[0], loc[1],
                                                           loc[2]);
                            }
                        }
                    }
                }

                curvedFace->SetCurve(curve.get());
                graph->GetCurvedFaces()[elmt_new->GetFid(locBndFaceID)] =
                    std::move(curve);
            }
        }

        // Macro prism removed from element tags and extracted from meshGraph
        elmtsToRemove[3].insert(el_macro);
        for (int e : sm.edgesToSplit)
        {
            elmtsToRemove[1].insert(el_macro->GetEdge(e));
        }
        for (int f : sm.sideFaces)
        {
            elmtsToRemove[2].insert(el_macro->GetFace(f));
        }
    }

    // Move macro elements
    for (auto &el_macro : elmtsToRemove[3])
    {
        m_mesh->m_elementTags[3].erase(el_macro);
        if (el_macro->GetShapeType() == LibUtilities::eHexahedron)
        {
            macroElements.m_hexVec.push_back(
                graph->ExtractGeom<SpatialDomains::HexGeom>(
                    el_macro->GetGlobalID(), true));
        }
        else
        {
            macroElements.m_prismVec.push_back(
                graph->ExtractGeom<SpatialDomains::PrismGeom>(
                    el_macro->GetGlobalID(), true));
        }
    }
    for (auto &quad_macro : elmtsToRemove[2])
    {
        m_mesh->m_elementTags[2].erase(quad_macro);
        macroElements.m_quadVec.push_back(
            graph->ExtractGeom<SpatialDomains::QuadGeom>(
                quad_macro->GetGlobalID(), true));
        auto it = graph->GetCurvedFaces().find(quad_macro->GetGlobalID());
        if (it != graph->GetCurvedFaces().end())
        {
            macroElements.m_curveVec.push_back(std::move(it->second));
            graph->GetCurvedFaces().erase(quad_macro->GetGlobalID());
        }
    }
    for (auto &seg_macro : elmtsToRemove[1])
    {
        macroElements.m_segVec.push_back(
            graph->ExtractGeom<SpatialDomains::SegGeom>(
                seg_macro->GetGlobalID(), true));
        auto it = graph->GetCurvedEdges().find(seg_macro->GetGlobalID());
        if (it != graph->GetCurvedEdges().end())
        {
            macroElements.m_curveVec.push_back(std::move(it->second));
            graph->GetCurvedEdges().erase(seg_macro->GetGlobalID());
        }
    }

    m_log(VERBOSE) << "Elements after  = " << m_mesh->m_elementTags[3].size()
                   << endl;
}
} // namespace Nektar::NekMesh
