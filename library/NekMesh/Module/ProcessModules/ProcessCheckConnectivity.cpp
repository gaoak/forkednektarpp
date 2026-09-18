////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessCheckConnectivity.cpp
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
//  Description: Checks the connectivity of all elements to
//  ensure no orphaned geometry.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessCheckConnectivity.h"
#include <stack>

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessCheckConnectivity::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "connectivity"),
        ProcessCheckConnectivity::create,
        "Check the connectivity of m_mesh->m_meshGraph to ensure no orphaned "
        "geometry.");

ProcessCheckConnectivity::ProcessCheckConnectivity(MeshSharedPtr m)
    : ProcessModule(m)
{
    m_config["list"] =
        ConfigOption(true, "0", "Print list of orphaned geometries.");
}

ProcessCheckConnectivity::~ProcessCheckConnectivity()
{
}

/**
 * @brief Perform Depth-First Search to check connectivity of elements and
 * warn about any untagged boundary elements.
 */
std::unordered_set<SpatialDomains::Geometry *> connectivityCheck(
    std::unordered_map<SpatialDomains::Geometry *,
                       std::vector<SpatialDomains::Geometry *>>
        sideElementMap,
    SpatialDomains::Geometry *startElmt, GeomTagMap &boundaryTags,
    int &nBndUntagged, bool printList, Logger &log)
{
    std::unordered_map<SpatialDomains::Geometry *,
                       std::unordered_set<SpatialDomains::Geometry *>>
        adjacency;
    for (auto &[part, elmtList] : sideElementMap)
    {
        if (elmtList.size() == 1)
        {
            if (boundaryTags.find(part) == boundaryTags.end())
            {
                if (printList)
                {
                    log << "  - Untagged "
                        << LibUtilities::ShapeTypeMap[part->GetShapeType()]
                        << " " << part->GetGlobalID()
                        << " is only connected to "
                        << LibUtilities::ShapeTypeMap[elmtList[0]
                                                          ->GetShapeType()]
                        << " " << elmtList[0]->GetGlobalID() << "." << endl;
                }
                nBndUntagged++;
            }
            continue;
        }
        for (int i = 0; i < elmtList.size(); i++)
        {
            for (int j = i + 1; j < elmtList.size(); ++j)
            {
                SpatialDomains::Geometry *a = elmtList[i];
                SpatialDomains::Geometry *b = elmtList[j];
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }
    }
    // Depth-first search through adjacency graph
    std::unordered_set<SpatialDomains::Geometry *> visited;
    std::stack<SpatialDomains::Geometry *> stack;
    stack.push(startElmt);
    while (!stack.empty())
    {
        SpatialDomains::Geometry *current = stack.top();
        stack.pop();
        if (visited.count(current))
        {
            continue;
        }
        visited.insert(current);
        for (auto &neighbor : adjacency[current])
        {
            if (!visited.count(neighbor))
            {
                stack.push(neighbor);
            }
        }
    }
    return visited;
}

template <typename T>
void printOrphanedGeoms(
    SpatialDomains::GeomMapView<T> allGeoms,
    std::unordered_map<SpatialDomains::Geometry *,
                       std::vector<SpatialDomains::Geometry *>>
        sideElementMap,
    Logger &log)
{
    for (auto geom : allGeoms)
    {
        if (sideElementMap.find(static_cast<SpatialDomains::Geometry *>(
                geom.second)) == sideElementMap.end())
        {
            log << "  - "
                << LibUtilities::ShapeTypeMap[geom.second->GetShapeType()]
                << " " << geom.first << " is orphaned." << endl;
        }
    }
}
template <typename T>
void printDisconElmts(SpatialDomains::GeomMapView<T> allGeoms,
                      std::unordered_set<SpatialDomains::Geometry *> visited,
                      Logger &log)
{
    for (auto geom : allGeoms)
    {
        if (visited.find(static_cast<SpatialDomains::Geometry *>(
                geom.second)) == visited.end())
        {
            log << "  - "
                << LibUtilities::ShapeTypeMap[geom.second->GetShapeType()]
                << " " << geom.first << " is disconnected." << endl;
        }
    }
}

void ProcessCheckConnectivity::Process()
{
    m_log(VERBOSE) << "Checking connectivity of mesh" << endl;
    bool printList = m_config["list"].beenSet;

    int meshDim = m_mesh->m_meshGraph->GetMeshDimension();
    std::array<int, 4> nOrph;
    int nBndUntagged = 0;
    std::unordered_set<SpatialDomains::Geometry *> visited;

    while (true)
    {
        std::unordered_map<SpatialDomains::Geometry *,
                           std::vector<SpatialDomains::Geometry *>>
            vertElement1dMap;
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            for (int v = 0; v < geom.second->GetNumVerts(); v++)
            {
                vertElement1dMap[geom.second->GetVertex(v)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        nOrph[0] = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>()
                       .size() -
                   vertElement1dMap.size();
        if (printList)
        {
            printOrphanedGeoms<SpatialDomains::PointGeom>(
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>(),
                vertElement1dMap, m_log);
        }
        if (meshDim == 1)
        {
            SpatialDomains::Geometry *startElmt =
                (*(m_mesh->m_elementTags[1].begin())).first;
            visited = connectivityCheck(vertElement1dMap, startElmt,
                                        m_mesh->m_elementTags[0], nBndUntagged,
                                        printList, m_log);
            nOrph[1] =
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>()
                    .size() -
                visited.size();
            break;
        }

        std::unordered_map<SpatialDomains::Geometry *,
                           std::vector<SpatialDomains::Geometry *>>
            edgeElement2dMap;
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
        {
            for (int e = 0; e < geom.second->GetNumEdges(); e++)
            {
                edgeElement2dMap[geom.second->GetEdge(e)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
        {
            for (int e = 0; e < geom.second->GetNumEdges(); e++)
            {
                edgeElement2dMap[geom.second->GetEdge(e)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        nOrph[1] = m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::SegGeom>() -
                   edgeElement2dMap.size();
        if (printList)
        {
            printOrphanedGeoms<SpatialDomains::SegGeom>(
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>(),
                edgeElement2dMap, m_log);
        }
        if (meshDim == 2)
        {
            SpatialDomains::Geometry *startElmt =
                (*(m_mesh->m_elementTags[2].begin())).first;
            visited = connectivityCheck(edgeElement2dMap, startElmt,
                                        m_mesh->m_elementTags[1], nBndUntagged,
                                        printList, m_log);
            nOrph[2] =
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>()
                    .size() +
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>()
                    .size() -
                visited.size();
            break;
        }

        std::unordered_map<SpatialDomains::Geometry *,
                           std::vector<SpatialDomains::Geometry *>>
            faceElement3dMap;
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>())
        {
            for (int f = 0; f < geom.second->GetNumFaces(); f++)
            {
                faceElement3dMap[geom.second->GetFace(f)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>())
        {
            for (int f = 0; f < geom.second->GetNumFaces(); f++)
            {
                faceElement3dMap[geom.second->GetFace(f)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>())
        {
            for (int f = 0; f < geom.second->GetNumFaces(); f++)
            {
                faceElement3dMap[geom.second->GetFace(f)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        for (auto geom :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>())
        {
            for (int f = 0; f < geom.second->GetNumFaces(); f++)
            {
                faceElement3dMap[geom.second->GetFace(f)].push_back(
                    static_cast<SpatialDomains::Geometry *>(geom.second));
            }
        }
        nOrph[2] =
            m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::TriGeom>() +
            m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::QuadGeom>() -
            faceElement3dMap.size();
        if (printList)
        {
            printOrphanedGeoms<SpatialDomains::TriGeom>(
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>(),
                faceElement3dMap, m_log);
            printOrphanedGeoms<SpatialDomains::QuadGeom>(
                m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>(),
                faceElement3dMap, m_log);
        }
        SpatialDomains::Geometry *startElmt =
            (*(m_mesh->m_elementTags[3].begin())).first;
        visited  = connectivityCheck(faceElement3dMap, startElmt,
                                     m_mesh->m_elementTags[2], nBndUntagged,
                                     printList, m_log);
        nOrph[3] = m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::TetGeom>() +
                   m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::PyrGeom>() +
                   m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>()
                       .size() +
                   m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::HexGeom>() -
                   visited.size();
        break;
    }

    if (printList)
    {
        switch (meshDim)
        {
            case 1:
                printDisconElmts<SpatialDomains::SegGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>(),
                    visited, m_log);
                break;
            case 2:
                printDisconElmts<SpatialDomains::TriGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>(),
                    visited, m_log);
                printDisconElmts<SpatialDomains::QuadGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>(),
                    visited, m_log);
                break;
            case 3:
                printDisconElmts<SpatialDomains::TetGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>(),
                    visited, m_log);
                printDisconElmts<SpatialDomains::PyrGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>(),
                    visited, m_log);
                printDisconElmts<SpatialDomains::PrismGeom>(
                    m_mesh->m_meshGraph
                        ->GetGeomMap<SpatialDomains::PrismGeom>(),
                    visited, m_log);
                printDisconElmts<SpatialDomains::HexGeom>(
                    m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>(),
                    visited, m_log);
                break;
            default:
                break;
        }
    }
    for (int dim = 0; dim < meshDim; dim++)
    {
        if (nOrph[dim] != 0)
        {
            m_log(WARNING) << "Detected " << nOrph[dim] << " orphaned "
                           << (nOrph[dim] == 1 ? "geometry" : "geometries")
                           << " of dimension " << dim << "." << endl;
        }
    }
    if (nBndUntagged != 0)
    {
        m_log(WARNING) << "Detected " << nBndUntagged
                       << " untagged boundary element"
                       << (nBndUntagged == 1 ? "" : "s") << "." << endl;
    }
    if (nOrph[meshDim] != 0)
    {
        m_log(WARNING) << "Detected " << nOrph[meshDim]
                       << " disconnected element"
                       << (nOrph[meshDim] == 1 ? "" : "s") << "." << endl;
    }
}
} // namespace Nektar::NekMesh
