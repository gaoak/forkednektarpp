////////////////////////////////////////////////////////////////////////////////
//
//  File: OutputGmsh.cpp
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
//  Description: Gmsh file format output.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/HOAlignment.h>
#include <NekMesh/Module/InputModules/InputGmsh.h>

#include "OutputGmsh.h"

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey OutputGmsh::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "msh"), OutputGmsh::create,
    "Writes Gmsh msh file.");

OutputGmsh::OutputGmsh(MeshSharedPtr m) : OutputModule(m)
{
    // Populate #InputGmsh::elmMap and use this to construct an
    // inverse mapping from %ElmtConfig to Gmsh ID.
    for (auto &it : InputGmsh::GenElmMap())
    {
        m_elmMap[it.second] = it.first;
    }

    m_config["order"] = ConfigOption(false, "-1", "Enforce a polynomial order");
}

OutputGmsh::~OutputGmsh()
{
}

/**
 * @brief Which way round @p edge runs in @p geom's local edge @p edgeId.
 *
 * An edge is shared, and stored in whichever direction the element that first
 * created it saw. eForwards means @p geom's local ordering of that edge agrees
 * with the stored one, eBackwards that its nodes must be reversed before being
 * written in this element's ordering.
 */
StdRegions::Orientation GetEdgeOrient(SpatialDomains::Geometry *geom,
                                      int edgeId,
                                      SpatialDomains::Geometry1D *edge)
{
    int ind0 = -1;
    int ind1 = -1;
    switch (geom->GetShapeType())
    {
        case LibUtilities::eTriangle:
        {
            static const int edgeVerts[3][2] = {{0, 1}, {1, 2}, {2, 0}};
            ind0                             = edgeVerts[edgeId][0];
            ind1                             = edgeVerts[edgeId][1];
            break;
        }
        case LibUtilities::eQuadrilateral:
        {
            static const int edgeVerts[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
            ind0                             = edgeVerts[edgeId][0];
            ind1                             = edgeVerts[edgeId][1];
            break;
        }
        case LibUtilities::eTetrahedron:
        {
            static const int edgeVerts[6][2] = {{0, 1}, {1, 2}, {0, 2},
                                                {0, 3}, {1, 3}, {2, 3}};
            ind0                             = edgeVerts[edgeId][0];
            ind1                             = edgeVerts[edgeId][1];
            break;
        }
        case LibUtilities::ePrism:
        {
            static const int edgeVerts[9][2] = {{0, 1}, {1, 2}, {3, 2},
                                                {0, 3}, {0, 4}, {1, 4},
                                                {2, 5}, {3, 5}, {4, 5}};
            ind0                             = edgeVerts[edgeId][0];
            ind1                             = edgeVerts[edgeId][1];
            break;
        }
        case LibUtilities::ePyramid:
        {
            static const int edgeVerts[8][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3},
                                                {0, 4}, {1, 4}, {2, 4}, {3, 4}};
            ind0                             = edgeVerts[edgeId][0];
            ind1                             = edgeVerts[edgeId][1];
            break;
        }
        case LibUtilities::eHexahedron:
        {
            static const int edgeVerts[12][2] = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5},
                {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}};
            ind0 = edgeVerts[edgeId][0];
            ind1 = edgeVerts[edgeId][1];
            break;
        }
        default:
            ASSERTL1(false, "GetEdgeOrient: unsupported shape type.");
            return StdRegions::eNoOrientation;
    }

    if (edge->GetVertex(0) == geom->GetVertex(ind0))
    {
        return StdRegions::eForwards;
    }
    else if (edge->GetVertex(0) == geom->GetVertex(ind1))
    {
        return StdRegions::eBackwards;
    }
    else
    {
        ASSERTL1(false, "Edge is not connected to this geom.");
    }
    return StdRegions::eNoOrientation;
}

/**
 * @brief The local vertex index of vertex @p j of @p geom's face @p i.
 *
 * The standard element's face-to-vertex tables, which the writer needs to map
 * a face's nodes back onto the element's own vertex numbering.
 */
int GetFaceVertex(SpatialDomains::Geometry *geom, int i, int j)
{
    switch (geom->GetShapeType())
    {
        case LibUtilities::eTetrahedron:
        {
            static const int faceIds[4][3] = {
                {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};
            return faceIds[i][j];
        }
        case LibUtilities::ePyramid:
        {
            static const int faceIds[5][4] = {{0, 1, 2, 3},
                                              {0, 1, 4, -1},
                                              {1, 2, 4, -1},
                                              {3, 2, 4, -1},
                                              {0, 3, 4, -1}};
            return faceIds[i][j];
        }
        case LibUtilities::ePrism:
        {
            static const int faceIds[5][4] = {{0, 1, 2, 3},
                                              {0, 1, 4, -1},
                                              {1, 2, 5, 4},
                                              {3, 2, 5, -1},
                                              {0, 3, 5, 4}};
            return faceIds[i][j];
        }
        case LibUtilities::eHexahedron:
        {
            static const int faceIds[6][4] = {{0, 1, 2, 3}, {0, 1, 5, 4},
                                              {1, 2, 6, 5}, {3, 2, 6, 7},
                                              {0, 3, 7, 4}, {4, 5, 6, 7}};
            return faceIds[i][j];
        }
        default:
            ASSERTL1(false, "GetFaceVertex: unsupported shape type.");
            return -1;
    }
}

/**
 * @brief Process a mesh to output to Gmsh MSH format.
 *
 * Gmsh output is fairly straightforward. The file first contains a
 * list of nodes, followed by a list of elements. Since
 * pointGeoms only contains vertices of the linear elements, we
 * first loop over the elements so that any high-order vertices can be
 * enumerated and then added to the node list. We then print out the
 * list of nodes and finally print the element list.
 */
void OutputGmsh::Process()
{
    std::string filename = m_config["outfile"].as<std::string>();
    m_log(VERBOSE) << "Writing Gmsh file '" << filename << "'." << endl;

    std::unordered_map<int, vector<int>> orderingMap;

    // Open the file stream.
    bool stream_open = OpenStream();
    if (!stream_open)
    {
        return;
    }

    // Write MSH header
    m_mshFile << "$MeshFormat" << endl
              << "2.2 0 8" << endl
              << "$EndMeshFormat" << endl;

    auto graph  = m_mesh->m_meshGraph;
    int meshDim = graph->GetMeshDimension();

    int order = m_config["order"].as<int>();

    if (order != -1)
    {
        m_log(VERBOSE) << "Making mesh of order " << order << endl;
    }
    else
    {
        order = 1;
        // Do first pass over elements of expansion dimension to determine
        // which elements need completion.
        if (meshDim == 1)
        {
            for (auto [id, geom] : graph->GetGeomMap<SpatialDomains::SegGeom>())
            {
                auto curve = geom->GetCurve();
                if (curve != nullptr)
                {
                    order = std::max(
                        order, static_cast<int>(curve->m_points.size()) - 1);
                }
            }
        }
        else
        {
            for (auto &[geom, tag] : m_mesh->m_elementTags[meshDim])
            {
                for (int e = 0; e < geom->GetNumEdges(); e++)
                {
                    auto curve =
                        static_cast<SpatialDomains::SegGeom *>(geom->GetEdge(e))
                            ->GetCurve();
                    if (curve != nullptr)
                    {
                        order = std::max(
                            order,
                            static_cast<int>(curve->m_points.size()) - 1);
                    }
                }
            }
        }
    }

    // Convert this mesh into a high-order mesh of uniform order.
    m_log(VERBOSE) << "Mesh order of " << order << " detected" << endl;

    m_mesh->MakeOrder(order, LibUtilities::ePolyEvenlySpaced, m_log);

    // Write out nodes section.
    int nEdgeIntNodes = graph->GetNumElements(1) * (order - 1);
    int nFaceIntNodes = graph->GetNumGeoms<SpatialDomains::QuadGeom>() *
                            (order - 1) * (order - 1) +
                        graph->GetNumGeoms<SpatialDomains::TriGeom>() *
                            (order - 1) * (order - 2) / 2;

    // Hexahedra whose interior was kept, either read from a file that carried
    // it or built by MakeOrder. Only those with a curve count: the rest have
    // no interior to write.
    int nVolIntNodes = 0;
    for (auto [id, hex] : graph->GetGeomMap<SpatialDomains::HexGeom>())
    {
        if (hex->GetCurve() != nullptr)
        {
            nVolIntNodes += (order - 1) * (order - 1) * (order - 1);
        }
    }
    const int nTetInt = (order - 3) * (order - 2) * (order - 1) / 6;
    for (auto [id, tet] : graph->GetGeomMap<SpatialDomains::TetGeom>())
    {
        if (tet->GetCurve() != nullptr && nTetInt > 0)
        {
            nVolIntNodes += nTetInt;
        }
    }
    const int nPrismInt = (order + 1) * (order + 2) / 2 * (order + 1) - 6 -
                          9 * (order - 1) - 3 * (order - 1) * (order - 1) -
                          2 * (order - 1) * (order - 2) / 2;
    for (auto [id, prism] : graph->GetGeomMap<SpatialDomains::PrismGeom>())
    {
        if (prism->GetCurve() != nullptr && nPrismInt > 0)
        {
            nVolIntNodes += nPrismInt;
        }
    }
    const int nPyrInt = (order - 2) * (order - 1) * (2 * order - 3) / 6;
    for (auto [id, pyr] : graph->GetGeomMap<SpatialDomains::PyrGeom>())
    {
        if (pyr->GetCurve() != nullptr && nPyrInt > 0)
        {
            nVolIntNodes += nPyrInt;
        }
    }

    m_mshFile << "$Nodes" << endl
              << graph->GetNvertices() + nEdgeIntNodes + nFaceIntNodes +
                     nVolIntNodes
              << endl;

    unsigned int nodeId = 1;
    std::unordered_map<SpatialDomains::PointGeom *, unsigned int> newIDs;

    // Gmsh numbers nodes from one and rejects an ID of zero. Every ID written
    // below comes from newIDs, so a node that never reached the writing pass
    // above would be handed gmsh the 0 that operator[] default-constructs.
    // Look them up through this instead, so that is caught rather than
    // written.
    auto nodeID = [&newIDs](SpatialDomains::PointGeom *n) -> unsigned int {
        auto it = newIDs.find(n);
        ASSERTL0(it != newIDs.end(),
                 "Element references a node that was not written to $Nodes.");
        return it->second;
    };
    for (auto [oldID, vert] : graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        m_mshFile << nodeId << " " << scientific << setprecision(10)
                  << (*vert)[0] << " " << (*vert)[1] << " " << (*vert)[2]
                  << endl;
        newIDs[vert] = nodeId;
        nodeId++;
    }
    if (order > 1)
    {
        for (auto [id, seg] : graph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            auto &points = seg->GetCurve()->m_points;
            for (int k = 1; k < points.size() - 1; ++k)
            {
                m_mshFile << nodeId << " " << scientific << setprecision(10)
                          << (*points[k])[0] << " " << (*points[k])[1] << " "
                          << (*points[k])[2] << endl;
                newIDs[points[k]] = nodeId;
                nodeId++;
            }
        }
        for (auto [id, tri] : graph->GetGeomMap<SpatialDomains::TriGeom>())
        {
            auto &points = tri->GetCurve()->m_points;
            for (int k = 3 * order; k < points.size(); ++k)
            {
                m_mshFile << nodeId << " " << scientific << setprecision(10)
                          << (*points[k])[0] << " " << (*points[k])[1] << " "
                          << (*points[k])[2] << endl;
                newIDs[points[k]] = nodeId;
                nodeId++;
            }
        }
        for (auto [id, quad] : graph->GetGeomMap<SpatialDomains::QuadGeom>())
        {
            auto &points = quad->GetCurve()->m_points;
            for (int i = 1; i < order; ++i)
            {
                for (int j = 1; j < order; ++j)
                {
                    auto &point = points[i * (order + 1) + j];
                    m_mshFile << nodeId << " " << scientific << setprecision(10)
                              << (*point)[0] << " " << (*point)[1] << " "
                              << (*point)[2] << endl;
                    newIDs[point] = nodeId;
                    nodeId++;
                }
            }
        }
        // Interior of a tetrahedron, the tail of the nodal ordering its
        // curve is held in.
        if (nTetInt > 0)
        {
            for (auto [id, tet] : graph->GetGeomMap<SpatialDomains::TetGeom>())
            {
                if (tet->GetCurve() == nullptr)
                {
                    continue;
                }

                auto &points = tet->GetCurve()->m_points;
                for (size_t k = points.size() - nTetInt; k < points.size(); ++k)
                {
                    m_mshFile << nodeId << " " << scientific << setprecision(10)
                              << (*points[k])[0] << " " << (*points[k])[1]
                              << " " << (*points[k])[2] << endl;
                    newIDs[points[k]] = nodeId;
                    nodeId++;
                }
            }
        }

        // Interior of a pyramid, the tail of its nodal ordering.
        if (nPyrInt > 0)
        {
            for (auto [id, pyr] : graph->GetGeomMap<SpatialDomains::PyrGeom>())
            {
                if (pyr->GetCurve() == nullptr)
                {
                    continue;
                }

                auto &points = pyr->GetCurve()->m_points;
                for (size_t k = points.size() - nPyrInt; k < points.size(); ++k)
                {
                    m_mshFile << nodeId << " " << scientific << setprecision(10)
                              << (*points[k])[0] << " " << (*points[k])[1]
                              << " " << (*points[k])[2] << endl;
                    newIDs[points[k]] = nodeId;
                    nodeId++;
                }
            }
        }

        // Interior of a prism, the tail of its nodal ordering.
        if (nPrismInt > 0)
        {
            for (auto [id, prism] :
                 graph->GetGeomMap<SpatialDomains::PrismGeom>())
            {
                if (prism->GetCurve() == nullptr)
                {
                    continue;
                }

                auto &points = prism->GetCurve()->m_points;
                for (size_t k = points.size() - nPrismInt; k < points.size();
                     ++k)
                {
                    m_mshFile << nodeId << " " << scientific << setprecision(10)
                              << (*points[k])[0] << " " << (*points[k])[1]
                              << " " << (*points[k])[2] << endl;
                    newIDs[points[k]] = nodeId;
                    nodeId++;
                }
            }
        }

        // Interior of a hexahedron, held as a curve on the element. The
        // points are the whole grid in tensor order, so the interior is what
        // is left once the boundary layer of the grid is stepped over.
        const int n = order + 1;
        for (auto [id, hex] : graph->GetGeomMap<SpatialDomains::HexGeom>())
        {
            if (hex->GetCurve() == nullptr)
            {
                continue;
            }

            auto &points = hex->GetCurve()->m_points;
            for (int k = 1; k < order; ++k)
            {
                for (int j = 1; j < order; ++j)
                {
                    for (int i = 1; i < order; ++i)
                    {
                        auto &point = points[i + n * (j + n * k)];
                        m_mshFile << nodeId << " " << scientific
                                  << setprecision(10) << (*point)[0] << " "
                                  << (*point)[1] << " " << (*point)[2] << endl;
                        newIDs[point] = nodeId;
                        nodeId++;
                    }
                }
            }
        }
    }

    m_mshFile << "$EndNodes" << endl;

    // Write elements section. All other sections are not currently
    // supported (physical names etc).
    m_mshFile << "$Elements" << endl;
    m_mshFile << m_mesh->GetNumTaggedEntities() << endl;

    int elmtId = 1;

    for (int d = 1; d <= 3; ++d)
    {
        for (auto &[geom, tag] : m_mesh->m_elementTags[d])
        {
            bool faceNodes = order > 1;
            auto shapeType = geom->GetShapeType();
            if ((shapeType == LibUtilities::eTriangle ||
                 shapeType == LibUtilities::eTetrahedron) &&
                order == 2)
            {
                faceNodes = false;
            }
            // A three-dimensional element carries its interior as a curve
            // on itself. Where that is present the nodes are written, and the
            // gmsh type has to be the one that expects them: a 27-node
            // hexahedron is a type, a hexahedron missing its centre is not.
            bool volumeNodes =
                d == 3 && order > 1 && geom->GetCurve() != nullptr &&
                (shapeType == LibUtilities::eHexahedron ||
                 (shapeType == LibUtilities::eTetrahedron && order > 3) ||
                 (shapeType == LibUtilities::ePrism && nPrismInt > 0) ||
                 (shapeType == LibUtilities::ePyramid && nPyrInt > 0));
            ElmtConfig conf =
                ElmtConfig(shapeType, order, faceNodes, volumeNodes);
            int elmtType = m_elmMap[conf];
            m_mshFile << elmtId++ << " " << elmtType << " ";

            // Write out the element's tags: the physical entity it belongs
            // to, then the elementary entity, then the number of partitions
            // it is in. Gmsh numbers entities from one and does not accept an
            // elementary entity of zero, so the composite stands in for both,
            // which is what gmsh itself writes for an unpartitioned mesh.
            m_mshFile << "3 " << tag << " " << tag << " 0 ";

            // Finally write out node list. First write vertices, then
            // internal edge nodes, then face nodes.
            vector<int> tags;

            for (int v = 0; v < geom->GetNumVerts(); ++v)
            {
                tags.push_back(nodeID(geom->GetVertex(v)));
            }

            if (order > 1)
            {
                // Process edge-interior points
                for (int e = 0; e < geom->GetNumEdges(); ++e)
                {
                    auto &nodes =
                        static_cast<SpatialDomains::SegGeom *>(geom->GetEdge(e))
                            ->GetCurve()
                            ->m_points;
                    if (GetEdgeOrient(geom, e, geom->GetEdge(e)) ==
                        StdRegions::eForwards)
                    {
                        for (int k = 1; k < nodes.size() - 1; ++k)
                        {
                            tags.push_back(nodeID(nodes[k]));
                        }
                    }
                    else
                    {
                        for (int k = nodes.size() - 2; k >= 1; --k)
                        {
                            tags.push_back(nodeID(nodes[k]));
                        }
                    }
                }

                // Process face-interior points
                for (int f = 0; f < geom->GetNumFaces(); ++f)
                {
                    auto &nodes    = geom->GetFace(f)->GetCurve()->m_points;
                    int nFaceVerts = geom->GetFace(f)->GetNumVerts();
                    vector<int> faceIds(nFaceVerts), volFaceIds(nFaceVerts);

                    for (int k = 0; k < nFaceVerts; ++k)
                    {
                        faceIds[k]    = geom->GetFace(f)->GetVid(k);
                        volFaceIds[k] = geom->GetVid(GetFaceVertex(geom, f, k));
                    }

                    if (nFaceVerts == 3)
                    {
                        auto faceIntNodes =
                            std::vector<SpatialDomains::PointGeom *>(
                                nodes.begin() + 3 * order, nodes.end());

                        HOTriangle<SpatialDomains::PointGeom *> hoTri(
                            faceIds, faceIntNodes);
                        hoTri.Align(volFaceIds);
                        for (int k = 0; k < hoTri.surfVerts.size(); ++k)
                        {
                            tags.push_back(nodeID(hoTri.surfVerts[k]));
                        }
                    }
                    else
                    {
                        std::vector<SpatialDomains::PointGeom *> faceIntNodes(
                            (order - 1) * (order - 1));
                        for (int i = 1; i < order; ++i)
                        {
                            for (int j = 1; j < order; ++j)
                            {
                                faceIntNodes[(i - 1) * (order - 1) + (j - 1)] =
                                    nodes[i * (order + 1) + j];
                            }
                        }

                        HOQuadrilateral<SpatialDomains::PointGeom *> hoQuad(
                            faceIds, faceIntNodes);
                        hoQuad.Align(volFaceIds);
                        for (int k = 0; k < hoQuad.surfVerts.size(); ++k)
                        {
                            tags.push_back(nodeID(hoQuad.surfVerts[k]));
                        }
                    }
                }

                // TODO: implement 3D volume nodes?
                if (d < 3)
                {
                    auto &points = geom->GetCurve()->m_points;
                    switch (geom->GetShapeType())
                    {
                        case LibUtilities::eSegment:
                        {
                            for (int k = 1; k < points.size() - 1; ++k)
                            {
                                tags.push_back(nodeID(points[k]));
                            }
                            break;
                        }
                        case LibUtilities::eTriangle:
                        {
                            for (int k = 3 * order; k < points.size(); ++k)
                            {
                                tags.push_back(nodeID(points[k]));
                            }
                            break;
                        }
                        case LibUtilities::eQuadrilateral:
                        {
                            for (int i = 1; i < order; ++i)
                            {
                                for (int j = 1; j < order; ++j)
                                {
                                    tags.push_back(
                                        nodeID(points[i * (order + 1) + j]));
                                }
                            }
                            break;
                        }
                        default:
                            // Only the shapes above reach here: the enclosing
                            // test excludes 3D elements, whose interior is
                            // written below, and a point carries no curve at
                            // all.
                            break;
                    }
                }
                else if (volumeNodes &&
                         geom->GetShapeType() == LibUtilities::ePyramid)
                {
                    auto &points = geom->GetCurve()->m_points;
                    for (size_t k = points.size() - nPyrInt; k < points.size();
                         ++k)
                    {
                        tags.push_back(nodeID(points[k]));
                    }
                }
                else if (volumeNodes &&
                         geom->GetShapeType() == LibUtilities::ePrism)
                {
                    auto &points = geom->GetCurve()->m_points;
                    for (size_t k = points.size() - nPrismInt;
                         k < points.size(); ++k)
                    {
                        tags.push_back(nodeID(points[k]));
                    }
                }
                else if (volumeNodes &&
                         geom->GetShapeType() == LibUtilities::eTetrahedron)
                {
                    // The interior is the tail of the curve, already in this
                    // element's own frame, so it goes straight on the end of
                    // the list in the order the reader expects to find it.
                    auto &points = geom->GetCurve()->m_points;
                    for (size_t k = points.size() - nTetInt; k < points.size();
                         ++k)
                    {
                        tags.push_back(nodeID(points[k]));
                    }
                }
                else if (volumeNodes)
                {
                    // The curve holds the whole grid in tensor order, so the
                    // interior is what is left once its boundary is stepped
                    // over. The reordering below expects them in the same
                    // order as the grid they came from, slowest index last.
                    auto &points = geom->GetCurve()->m_points;
                    const int n  = order + 1;
                    for (int k = 1; k < order; ++k)
                    {
                        for (int j = 1; j < order; ++j)
                        {
                            for (int i = 1; i < order; ++i)
                            {
                                tags.push_back(
                                    nodeID(points[i + n * (j + n * k)]));
                            }
                        }
                    }
                }
            }

            // Construct inverse of input reordering. First try to find it
            // in our cache.
            auto oIt = orderingMap.find(elmtType);

            // If it's not created, then create it.
            if (oIt == orderingMap.end())
            {
                vector<int> reordering =
                    InputGmsh::CreateReordering(elmtType, m_log);
                vector<int> inv(tags.size());

                ASSERTL1(tags.size() == reordering.size(),
                         "Reordering map size not equal to element tags 1.");

                for (int j = 0; j < tags.size(); ++j)
                {
                    inv[reordering[j]] = j;
                }

                oIt = orderingMap.insert(make_pair(elmtType, inv)).first;
            }

            ASSERTL1(tags.size() == oIt->second.size(),
                     "Reordering map size not equal to element tags 2.");

            // Finally write element nodes.
            for (int j = 0; j < tags.size(); ++j)
            {
                m_mshFile << tags[oIt->second[j]] << " ";
            }

            m_mshFile << endl;
        }
    }
    m_mshFile << "$EndElements" << endl;
}
} // namespace Nektar::NekMesh
