////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessCombine.cpp
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
//  Description: Combine two mesh files into one output file.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessCombine.h"
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/HashUtils.hpp>
#include <LibUtilities/BasicUtils/RealComparison.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <NekMesh/MeshElements/Element.h>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace
{
// @TODO: Replicates master's coordinate-based NodeHash with bitwise equality.
// Consider adding a tolerance for non-bitwise-identical coincident nodes.
struct PointHash
{
    std::size_t operator()(Nektar::SpatialDomains::PointGeom *const p) const
    {
        return Nektar::hash_combine((*p)(0), (*p)(1), (*p)(2));
    }
};
struct PointEqual
{
    bool operator()(Nektar::SpatialDomains::PointGeom *const p1,
                    Nektar::SpatialDomains::PointGeom *const p2) const
    {
        return Nektar::LibUtilities::IsRealEqual((*p1)(0), (*p2)(0)) &&
               Nektar::LibUtilities::IsRealEqual((*p1)(1), (*p2)(1)) &&
               Nektar::LibUtilities::IsRealEqual((*p1)(2), (*p2)(2));
    }
};
using PointSet = std::unordered_set<Nektar::SpatialDomains::PointGeom *,
                                    PointHash, PointEqual>;
} // namespace

namespace Nektar::NekMesh
{

ModuleKey ProcessCombine::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "combine"), ProcessCombine::create,
        "Combine two mesh files into one output file.");

ProcessCombine::ProcessCombine(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["file"] = ConfigOption(
        false, "", "Second mesh file to be combined with the input mesh file.");
    m_config["hashpos"] = ConfigOption(
        true, "0",
        "Deduplicate coincident vertices by position. When false (default), "
        "all vertices, edges, faces, and elements are merged without any "
        "deduplication.");
    m_config["delintbnd"] = ConfigOption(
        true, "0",
        "Delete internal boundaries between the two meshes. Requires "
        "hashpos=true, since boundary identification relies on vertex "
        "deduplication to detect shared faces.");
}

void ProcessCombine::Process()
{
    ASSERTL0(m_config["file"].beenSet,
             "The 'file' module parameter must be set.")

    std::string fname = m_config["file"].as<std::string>();

    // Check to see if filename exists.
    if (!fs::exists(fname))
    {
        m_log(FATAL) << "Unable to read file: '" << fname << "'" << std::endl;
    }

    // Process the second mesh input
    ModuleKey module;
    module.first = eInputModule;

    vector<string> tmp1;
    boost::split(tmp1, fname, boost::is_any_of(":"));
    if (tmp1.size() == 1)
    {
        int dot    = tmp1[0].find_last_of('.') + 1;
        string ext = tmp1[0].substr(dot, tmp1[0].length() - dot);

        if (ext == "gz")
        {
            string tmp = tmp1[0].substr(0, tmp1[0].length() - 3);
            dot        = tmp.find_last_of('.') + 1;
            ext        = tmp.substr(dot, tmp.length() - dot);
        }

        module.second = ext;
        tmp1.push_back("infile=" + tmp1[0]);
    }
    else
    {
        module.second = tmp1[1];
        tmp1.push_back("infile=" + tmp1[0]);
    }

    // Copy communicator
    MeshSharedPtr mesh = std::make_shared<Mesh>();
    mesh->m_comm       = m_mesh->m_comm;

    ModuleSharedPtr mod = GetModuleFactory().CreateInstance(module, mesh);
    mod->SetLogger(m_log);
    mod->GetLogger().SetPrefix("ProcessCombine");

    // Set options for this module.
    for (int j = 1; j < tmp1.size(); ++j)
    {
        vector<string> tmp2;
        boost::split(tmp2, tmp1[j], boost::is_any_of("="));

        if (tmp2.size() == 1)
        {
            mod->RegisterConfig(tmp2[0]);
        }
        else if (tmp2.size() == 2)
        {
            mod->RegisterConfig(tmp2[0], tmp2[1]);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal, "ERROR: Invalid module configuration: "
                                        "format is either :arg or :arg=val");
        }
    }

    // Ensure configuration options have been set.
    mod->SetDefaults();

    // Run second input module
    mod->Process();
    MeshSharedPtr mesh2                       = mod->GetMesh();
    SpatialDomains::MeshGraphSharedPtr graph1 = m_mesh->m_meshGraph;
    SpatialDomains::MeshGraphSharedPtr graph2 = mesh2->m_meshGraph;

    // Check dimensions of both meshes are the same
    int meshDim = graph1->GetMeshDimension();
    ASSERTL0(meshDim == graph2->GetMeshDimension(),
             "The expansion dimensions of the meshes being combined must be "
             "the same.")

    // geomRemap maps every graph2 Geometry* to its canonical graph1 equivalent.
    // Shared sub-elements map to the pre-existing graph1 ptr; new ones map to
    // themselves (valid after extraction since unique_ptr move doesn't
    // relocate).
    std::unordered_map<SpatialDomains::Geometry *, SpatialDomains::Geometry *>
        geomRemap;

    bool hashByPos = m_config["hashpos"].as<bool>();

    // Move mesh2 vertices into mesh1, deduplicating by position if
    // hashpos=true.
    int vid = graph1->GetNumElements(0);
    if (hashByPos)
    {
        PointSet pointSet;
        for (auto [id, point] : graph1->GetGeomMap<SpatialDomains::PointGeom>())
        {
            pointSet.insert(point);
        }

        for (auto [oldID, point] :
             graph2->GetGeomMap<SpatialDomains::PointGeom>())
        {
            auto [it, inserted] = pointSet.insert(point);
            if (!inserted)
            {
                geomRemap[point] = *it;
            }
            else
            {
                graph1->AddGeom<SpatialDomains::PointGeom>(
                    vid, graph2->ExtractGeom<SpatialDomains::PointGeom>(oldID));
                point->SetGlobalID(vid++);
                geomRemap[point] = point;
            }
        }
    }
    else
    {
        for (auto [oldID, point] :
             graph2->GetGeomMap<SpatialDomains::PointGeom>())
        {
            graph1->AddGeom<SpatialDomains::PointGeom>(
                vid, graph2->ExtractGeom<SpatialDomains::PointGeom>(oldID));
            point->SetGlobalID(vid++);
            geomRemap[point] = point;
        }
    }

    // Move mesh2 edges into mesh1, deduplicating via m_edgeSet.
    int eid = graph1->GetNumElements(1);
    for (auto [oldID, edge] : graph2->GetGeomMap<SpatialDomains::SegGeom>())
    {
        for (int i = 0; i < 2; i++)
        {
            edge->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                   geomRemap.at(edge->GetVertex(i))));
        }
        std::pair<int, int> edgeKey(edge->GetVertex(0)->GetGlobalID(),
                                    edge->GetVertex(1)->GetGlobalID());
        auto [edgeIt, edgeInserted] = m_mesh->m_edgeSet.emplace(edgeKey, edge);
        if (!edgeInserted)
        {
            geomRemap[edge] = edgeIt->second;
        }
        else
        {
            graph1->AddGeom<SpatialDomains::SegGeom>(
                eid, graph2->ExtractGeom<SpatialDomains::SegGeom>(
                         edge->GetGlobalID()));
            edge->SetGlobalID(eid);
            if (edge->GetCurve() != nullptr)
            {
                edge->GetCurve()->m_curveID = eid;
                graph1->GetCurvedEdges()[eid] =
                    std::move(graph2->GetCurvedEdges()[oldID]);
            }
            geomRemap[edge] = edge;
            eid++;
        }
    }

    // Move mesh2 faces into mesh1, deduplicating via m_faceSet.
    int fid = graph1->GetNumElements(2);
    for (auto [oldID, tri] : graph2->GetGeomMap<SpatialDomains::TriGeom>())
    {
        for (int i = 0; i < 3; i++)
        {
            tri->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                  geomRemap.at(tri->GetVertex(i))));
            tri->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                geomRemap.at(tri->GetEdge(i))));
        }
        std::array<int, 4> faceKey({tri->GetVertex(0)->GetGlobalID(),
                                    tri->GetVertex(1)->GetGlobalID(),
                                    tri->GetVertex(2)->GetGlobalID(), -1});
        auto [faceIt, faceInserted] = m_mesh->m_faceSet.emplace(faceKey, tri);
        if (!faceInserted)
        {
            geomRemap[tri] = faceIt->second;
        }
        else
        {
            graph1->AddGeom<SpatialDomains::TriGeom>(
                fid, graph2->ExtractGeom<SpatialDomains::TriGeom>(
                         tri->GetGlobalID()));
            tri->SetGlobalID(fid);
            if (tri->GetCurve() != nullptr)
            {
                tri->GetCurve()->m_curveID = fid;
                graph1->GetCurvedFaces()[fid] =
                    std::move(graph2->GetCurvedFaces()[oldID]);
            }
            geomRemap[tri] = tri;
            fid++;
        }
    }
    for (auto [oldID, quad] : graph2->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        for (int i = 0; i < 4; i++)
        {
            quad->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                   geomRemap.at(quad->GetVertex(i))));
            quad->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                 geomRemap.at(quad->GetEdge(i))));
        }
        std::array<int, 4> faceKey({quad->GetVertex(0)->GetGlobalID(),
                                    quad->GetVertex(1)->GetGlobalID(),
                                    quad->GetVertex(2)->GetGlobalID(),
                                    quad->GetVertex(3)->GetGlobalID()});
        auto [faceIt, faceInserted] = m_mesh->m_faceSet.emplace(faceKey, quad);
        if (!faceInserted)
        {
            geomRemap[quad] = faceIt->second;
        }
        else
        {
            graph1->AddGeom<SpatialDomains::QuadGeom>(
                fid, graph2->ExtractGeom<SpatialDomains::QuadGeom>(
                         quad->GetGlobalID()));
            quad->SetGlobalID(fid);
            if (quad->GetCurve() != nullptr)
            {
                quad->GetCurve()->m_curveID = fid;
                graph1->GetCurvedFaces()[fid] =
                    std::move(graph2->GetCurvedFaces()[oldID]);
            }
            geomRemap[quad] = quad;
            fid++;
        }
    }

    // Move mesh2 elements into mesh1, redirecting all sub-element pointers.
    int elid = graph1->GetNumElements(3);
    for (auto [oldID, tet] : graph2->GetGeomMap<SpatialDomains::TetGeom>())
    {
        for (int i = 0; i < tet->GetNumVerts(); i++)
        {
            tet->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                  geomRemap.at(tet->GetVertex(i))));
        }
        for (int i = 0; i < tet->GetNumEdges(); i++)
        {
            tet->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                geomRemap.at(tet->GetEdge(i))));
        }
        for (int i = 0; i < tet->GetNumFaces(); i++)
        {
            tet->SetFace(i, static_cast<SpatialDomains::TriGeom *>(
                                geomRemap.at(tet->GetFace(i))));
        }
        graph1->AddGeom<SpatialDomains::TetGeom>(
            elid,
            graph2->ExtractGeom<SpatialDomains::TetGeom>(tet->GetGlobalID()));
        tet->SetGlobalID(elid++);
    }
    for (auto [oldID, pyr] : graph2->GetGeomMap<SpatialDomains::PyrGeom>())
    {
        for (int i = 0; i < pyr->GetNumVerts(); i++)
        {
            pyr->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                  geomRemap.at(pyr->GetVertex(i))));
        }
        for (int i = 0; i < pyr->GetNumEdges(); i++)
        {
            pyr->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                geomRemap.at(pyr->GetEdge(i))));
        }
        for (int i = 0; i < pyr->GetNumFaces(); i++)
        {
            pyr->SetFace(i, static_cast<SpatialDomains::Geometry2D *>(
                                geomRemap.at(pyr->GetFace(i))));
        }
        graph1->AddGeom<SpatialDomains::PyrGeom>(
            elid,
            graph2->ExtractGeom<SpatialDomains::PyrGeom>(pyr->GetGlobalID()));
        pyr->SetGlobalID(elid++);
    }
    for (auto [oldID, prism] : graph2->GetGeomMap<SpatialDomains::PrismGeom>())
    {
        for (int i = 0; i < prism->GetNumVerts(); i++)
        {
            prism->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                    geomRemap.at(prism->GetVertex(i))));
        }
        for (int i = 0; i < prism->GetNumEdges(); i++)
        {
            prism->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                  geomRemap.at(prism->GetEdge(i))));
        }
        for (int i = 0; i < prism->GetNumFaces(); i++)
        {
            prism->SetFace(i, static_cast<SpatialDomains::Geometry2D *>(
                                  geomRemap.at(prism->GetFace(i))));
        }
        graph1->AddGeom<SpatialDomains::PrismGeom>(
            elid, graph2->ExtractGeom<SpatialDomains::PrismGeom>(
                      prism->GetGlobalID()));
        prism->SetGlobalID(elid++);
    }
    for (auto [oldID, hex] : graph2->GetGeomMap<SpatialDomains::HexGeom>())
    {
        for (int i = 0; i < hex->GetNumVerts(); i++)
        {
            hex->SetVertex(i, static_cast<SpatialDomains::PointGeom *>(
                                  geomRemap.at(hex->GetVertex(i))));
        }
        for (int i = 0; i < hex->GetNumEdges(); i++)
        {
            hex->SetEdge(i, static_cast<SpatialDomains::SegGeom *>(
                                geomRemap.at(hex->GetEdge(i))));
        }
        for (int i = 0; i < hex->GetNumFaces(); i++)
        {
            hex->SetFace(i, static_cast<SpatialDomains::QuadGeom *>(
                                geomRemap.at(hex->GetFace(i))));
        }
        graph1->AddGeom<SpatialDomains::HexGeom>(
            elid,
            graph2->ExtractGeom<SpatialDomains::HexGeom>(hex->GetGlobalID()));
        hex->SetGlobalID(elid++);
    }

    // Update composite geom vectors to use graph1 pointers.
    for (auto &[id, comp] : graph2->GetComposites())
    {
        for (auto &g : comp->m_geomVec)
        {
            if (auto it = geomRemap.find(g); it != geomRemap.end())
            {
                g = it->second;
            }
        }
    }

    // Renumber composites and copy in
    std::map<int, int> compRenumber;
    for (auto &[id, comp] : graph2->GetComposites())
    {
        int newId    = id;
        auto findKey = compRenumber.find(id);
        if (findKey != compRenumber.end())
        {
            newId = findKey->second;
        }
        else
        {
            while (graph1->GetComposites().count(newId) ||
                   (newId != id && graph2->GetComposites().count(newId)))
            {
                newId++;
            }
            compRenumber[id] = newId;
        }
        graph1->GetComposites()[newId] = comp;
    }

    bool deleteInteriorBnds = m_config["delintbnd"].as<bool>();
    std::unordered_set<SpatialDomains::Geometry *> removeBnd;
    // Copy over tags, remapping shared-element keys and composite IDs
    for (int d = 0; d <= 3; d++)
    {
        for (auto &[geom, tag] : mesh2->m_elementTags[d])
        {
            auto it     = geomRemap.find(geom);
            auto *canon = (it != geomRemap.end()) ? it->second : geom;

            if (deleteInteriorBnds && d == meshDim - 1 &&
                m_mesh->m_elementTags[d].find(canon) !=
                    m_mesh->m_elementTags[d].end())
            {
                removeBnd.insert(canon);
            }
            else
            {
                m_mesh->m_elementTags[d][canon] = compRenumber.at(tag);
            }
        }
    }

    if (deleteInteriorBnds)
    {
        // Delete removed boundaries from element tags and composite
        for (auto it = graph1->GetComposites().begin();
             it != graph1->GetComposites().end();)
        {
            auto &[id, comp] = *it;
            if (comp->m_geomVec[0]->GetShapeDim() != meshDim - 1)
            {
                ++it;
                continue;
            }

            std::vector<SpatialDomains::Geometry *> newGeomVec;
            for (auto &geom : comp->m_geomVec)
            {
                if (removeBnd.find(geom) == removeBnd.end())
                {
                    newGeomVec.push_back(geom);
                }
                else
                {
                    m_mesh->m_elementTags[meshDim - 1].erase(geom);
                }
            }

            if (newGeomVec.size() == 0)
            {
                it = graph1->GetComposites().erase(it);
            }
            else
            {
                comp->m_geomVec = newGeomVec;
                ++it;
            }
        }
    }

    // Copy over domains, updating map keys
    int newID = 0;
    for (auto &[id, oldDomain] : graph2->GetDomain())
    {
        while (graph1->GetDomain().count(newID))
        {
            newID++;
        }
        SpatialDomains::CompositeMap newDomain;
        for (auto &[oldID, comp] : oldDomain)
        {
            newDomain[compRenumber[oldID]] = comp;
        }
        graph1->GetDomain()[newID] = newDomain;
    }

    // Transfer graph2 curve nodes into graph1 so Curve::m_points raw pointers
    // don't dangle when mesh2 goes out of scope.
    for (auto &node : graph2->GetAllCurveNodes())
    {
        graph1->GetAllCurveNodes().push_back(std::move(node));
    }

    for (auto it = compRenumber.begin(); it != compRenumber.end();)
    {
        bool removed = graph1->GetComposites().find(it->second) ==
                       graph1->GetComposites().end();
        it = (removed || it->first == it->second) ? compRenumber.erase(it)
                                                  : std::next(it);
    }
    if (!compRenumber.empty())
    {
        m_log << "Duplicate composite IDs from mesh 1 detected in mesh 2."
              << endl;
        m_log << "These will be remapped in the output file to:" << endl;

        for (auto &cIt : compRenumber)
        {
            m_log << "- C[" << cIt.first << "] => "
                  << "C[" << cIt.second << "]" << endl;
        }
    }
}
} // namespace Nektar::NekMesh
