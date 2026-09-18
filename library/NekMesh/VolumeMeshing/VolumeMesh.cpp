////////////////////////////////////////////////////////////////////////////////
//
//  File: VolumeMesh.cpp
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
//  Description: Process volume meshing.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ParseUtils.h>

#include "VolumeMesh.h"
#include <NekMesh/SurfaceMeshing/CurveMesh.h>
#include <NekMesh/SurfaceMeshing/FaceMesh.h>
#include <NekMesh/VolumeMeshing/BLMeshing/BLMesh.h>
#include <NekMesh/VolumeMeshing/TetMeshing/TetMesh.h>
#include <SpatialDomains/CADSystem/CADAssociation.h>

using namespace std;
namespace Nektar::NekMesh
{

ModuleKey VolumeMesh::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "volumemesh"), VolumeMesh::create,
    "Generates a volume mesh");

VolumeMesh::VolumeMesh(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["blsurfs"] =
        ConfigOption(false, "0", "Generate prisms on these surfs");
    m_config["blthick"]  = ConfigOption(false, "0", "Prism layer thickness");
    m_config["bllayers"] = ConfigOption(false, "0", "Prism layers");
    m_config["blprog"]   = ConfigOption(false, "0", "Prism progression");
}

VolumeMesh::~VolumeMesh()
{
}

void VolumeMesh::Process()
{
    auto &m_graph = m_mesh->m_meshGraph;

    m_log(VERBOSE) << "Volume meshing:" << endl;

    bool makeBL;
    vector<unsigned int> blSurfs;

    if (m_config["blsurfs"].beenSet)
    {
        makeBL = true;
        ParseUtils::GenerateSeqVector(m_config["blsurfs"].as<string>(),
                                      blSurfs);
    }
    else
    {
        makeBL = false;
    }

    NekDouble prefix = 100;
    if (m_graph->GetCAD()->GetNumSurf() > 100)
    {
        prefix *= 10;
    }

    TetMeshSharedPtr tet;

    if (makeBL)
    {
        m_log(VERBOSE) << "  Performing boundary layer generation." << endl;
        BLMeshSharedPtr blmesh = MemoryManager<BLMesh>::AllocateSharedPtr(
            m_mesh, blSurfs, m_config["blthick"].as<NekDouble>(),
            m_config["bllayers"].as<int>(), m_config["blprog"].as<NekDouble>(),
            prefix + 1, m_log);

        blmesh->Mesh();

        // remesh the correct surfaces
        vector<unsigned int> symsurfs = blmesh->GetSymSurfs();

        vector<SpatialDomains::Geometry *> els;
        for (auto &[el, tag] : m_mesh->m_elementTags[2])
        {
            els.push_back(el);
        }

        for (int i = 0; i < els.size(); i++)
        {
            vector<unsigned int>::iterator f =
                find(symsurfs.begin(), symsurfs.end(),
                     m_graph->GetCADAssociation()->GetSurf(els[i])->GetId());

            if (f == symsurfs.end())
            {
                continue;
            }

            // erase the triag face from the set with key the {vertices -1}
            std::array<int, 4> vids = {els[i]->GetVertex(0)->GetGlobalID(),
                                       els[i]->GetVertex(1)->GetGlobalID(),
                                       els[i]->GetVertex(2)->GetGlobalID(), -1};
            m_mesh->m_faceSet.erase(vids);
            m_mesh->m_elementTags[2].erase(els[i]);
            m_graph->GetCADAssociation()->Remove(els[i]);
            m_graph->ExtractGeom<SpatialDomains::TriGeom>(els[i]->GetGlobalID(),
                                                          true);
        }

        for (int i = 0; i < symsurfs.size(); i++)
        {
            set<int> cIds;
            vector<SpatialDomains::EdgeLoopSharedPtr> e =
                m_graph->GetCAD()->GetSurf(symsurfs[i])->GetEdges();
            for (int k = 0; k < e.size(); k++)
            {
                for (int j = 0; j < e[k]->edges.size(); j++)
                {
                    cIds.insert(e[k]->edges[j]->GetId());
                }
            }

            // find the curve nodes which are on this symsurf
            map<int, vector<SpatialDomains::PointGeom *>> curveNodeMap;
            for (auto [vid, vert] :
                 m_graph->GetGeomMap<SpatialDomains::PointGeom>())
            {
                auto cc = m_graph->GetCADAssociation()->GetLinks(
                    vert, SpatialDomains::CADType::eCurve);
                for (int j = 0; j < cc.size(); j++)
                {
                    set<int>::iterator f = cIds.find(cc[j].Id());
                    if (f != cIds.end())
                    {
                        curveNodeMap[cc[j].Id()].push_back(vert);
                    }
                }
            }

            // need to bubble sort the vectors
            map<int, vector<SpatialDomains::PointGeom *>>::iterator cit;
            for (cit = curveNodeMap.begin(); cit != curveNodeMap.end(); cit++)
            {
                vector<NekDouble> ts;
                for (int i = 0; i < cit->second.size(); i++)
                {
                    ts.push_back(m_graph->GetCADAssociation()->GetCurveT(
                        cit->second[i], cit->first));
                }
                bool repeat = true;
                while (repeat)
                {
                    repeat = false;
                    for (int i = 0; i < ts.size() - 1; i++)
                    {
                        if (ts[i] > ts[i + 1])
                        {
                            swap(ts[i], ts[i + 1]);
                            swap(cit->second[i], cit->second[i + 1]);
                            repeat = true;
                            break;
                        }
                    }
                }
            }

            // create quads
            map<SpatialDomains::PointGeom *, SpatialDomains::PointGeom *> nmap =
                blmesh->GetSymNodes();
            for (cit = curveNodeMap.begin(); cit != curveNodeMap.end(); cit++)
            {
                for (int j = 0; j < cit->second.size() - 1; j++)
                {
                    auto f1 = nmap.find(cit->second[j]);
                    auto f2 = nmap.find(cit->second[j + 1]);

                    if (f1 == nmap.end() || f2 == nmap.end())
                    {
                        continue;
                    }

                    SpatialDomains::PointGeom *n1 = f1->second;
                    SpatialDomains::PointGeom *n2 = f2->second;

                    vector<SpatialDomains::PointGeom *> ns;
                    ns.push_back(cit->second[j]);
                    ns.push_back(n1);
                    ns.push_back(n2);
                    ns.push_back(cit->second[j + 1]);

                    ElmtConfig conf(LibUtilities::eQuadrilateral, 1, false,
                                    false);

                    // quad element will check if quad element exist and reuse
                    SpatialDomains::Geometry *E =
                        GetElementFactory().CreateInstance(
                            LibUtilities::eQuadrilateral, ns, m_graph,
                            m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                            nullptr, nullptr, nullptr);

                    m_graph->GetCADAssociation()->Set(
                        E, {m_graph->GetCAD()->GetSurf(symsurfs[i])});
                    m_mesh->m_elementTags[2][E] = prefix * 2 + symsurfs[i];
                }
            }

            // swap nodes
            for (cit = curveNodeMap.begin(); cit != curveNodeMap.end(); cit++)
            {
                for (int j = 0; j < cit->second.size(); j++)
                {
                    auto f1 = nmap.find(cit->second[j]);
                    if (f1 == nmap.end())
                    {
                        continue;
                    }
                    cit->second[j] = f1->second;
                }
            }
            map<int, CurveMeshSharedPtr> cm;
            for (cit = curveNodeMap.begin(); cit != curveNodeMap.end(); cit++)
            {
                cm[cit->first] = MemoryManager<CurveMesh>::AllocateSharedPtr(
                    cit->first, m_mesh, cit->second, m_log);
            }

            FaceMeshSharedPtr f = MemoryManager<FaceMesh>::AllocateSharedPtr(
                symsurfs[i], m_mesh, cm, symsurfs[i], m_log);
            f->Mesh();
        }

        vector<unsigned int> blsurfs = blmesh->GetBLSurfs();

        // build the surface for tetgen to use.
        vector<SpatialDomains::Geometry *> tetsurface =
            blmesh->GetPseudoSurface();
        for (auto &[el, tag] : m_mesh->m_elementTags[2])
        {
            if (el->GetShapeType() == LibUtilities::eQuadrilateral)
            {
                continue;
            }

            vector<unsigned int>::iterator f =
                find(blsurfs.begin(), blsurfs.end(),
                     m_graph->GetCADAssociation()->GetSurf(el)->GetId());

            if (f == blsurfs.end())
            {
                tetsurface.push_back(el);
            }
        }

        m_log(VERBOSE) << "    - Boundary layer generation complete." << endl;
        m_log(VERBOSE) << "  Tetrahedral mesh generation:" << endl;

        tet = MemoryManager<TetMesh>::AllocateSharedPtr(m_mesh, prefix, m_log,
                                                        tetsurface);
    }
    else
    {
        m_log(VERBOSE) << "  Tetrahedral mesh generation:" << endl;
        tet = MemoryManager<TetMesh>::AllocateSharedPtr(m_mesh, prefix, m_log);
    }

    tet->Mesh();

    ProcessVertices();
    ProcessElements();
    ProcessComposites();
}

} // namespace Nektar::NekMesh
