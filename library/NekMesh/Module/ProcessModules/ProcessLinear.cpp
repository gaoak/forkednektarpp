////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessLinear.cpp
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
//  Description: linearises mesh.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessLinear.h"
#include <NekMesh/MeshElements/Element.h>

using namespace std;

namespace Nektar::NekMesh
{

using namespace Nektar::NekMesh;

namespace
{
/// Gather the union of a set of integer ids across every rank of @p comm
std::set<int> GatherUnion(const LibUtilities::CommSharedPtr &comm,
                          const std::unordered_set<int> &ids)
{
    int nProc = comm->GetSize();
    int rank  = comm->GetRank();

    // Exchange the per-rank counts so every rank can size the receive buffer.
    Array<OneD, int> recvSizes(nProc, 0);
    recvSizes[rank] = ids.size();
    comm->AllReduce(recvSizes, LibUtilities::ReduceSum);

    Array<OneD, int> recvOffsets(nProc, 0);
    int total = 0;
    for (int p = 0; p < nProc; ++p)
    {
        recvOffsets[p] = total;
        total += recvSizes[p];
    }

    Array<OneD, int> sendData(ids.size());
    int i = 0;
    for (int id : ids)
    {
        sendData[i++] = id;
    }

    Array<OneD, int> gathered(total, 0);
    comm->AllGatherv(sendData, gathered, recvSizes, recvOffsets);

    return std::set<int>(gathered.begin(), gathered.end());
}
} // namespace

ModuleKey ProcessLinear::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "linearise"), ProcessLinear::create,
    "Linearises mesh.");

ProcessLinear::ProcessLinear(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["all"] =
        ConfigOption(true, "0", "remove curve nodes for all elements.");
    m_config["invalid"] =
        ConfigOption(false, "0", "remove curve nodes if element is invalid.");
    m_config["prismonly"] = ConfigOption(true, "0", "only acts on prisms");
    m_config["extract"] =
        ConfigOption(false, "", "dump a mesh of the linearised elements");
}

ProcessLinear::~ProcessLinear()
{
}

void ProcessLinear::Process()
{
    m_log(VERBOSE) << "Linearising mesh." << endl;

    bool all      = m_config["all"].as<bool>();
    bool invalid  = m_config["invalid"].beenSet;
    NekDouble thr = m_config["invalid"].as<NekDouble>();
    auto &graph   = m_mesh->m_meshGraph;
    int meshDim   = graph->GetMeshDimension();

    if (!all & !invalid)
    {
        m_log(FATAL) << "Must specify an option: 'all' (to remove all "
                     << "curvature) or invalid (to remove curvature that makes "
                     << "elements invalid)." << endl;
    }

    if (all)
    {
        graph->GetCurvedEdges().clear();
        graph->GetCurvedFaces().clear();
        graph->GetAllCurveNodes().clear();

        // Reset sets m_curve to nullptr if not found
        for (auto &[geom, tag] : m_mesh->m_elementTags[meshDim])
        {
            geom->Reset(m_mesh->m_meshGraph->GetCurvedEdges(),
                        m_mesh->m_meshGraph->GetCurvedFaces());
            geom->FillGeom();
        }

        m_log(VERBOSE) << "Removed all element curvature." << endl;
    }
    else if (invalid)
    {
        map<int, vector<SpatialDomains::Geometry2D *>> eidToFace;
        map<int, vector<SpatialDomains::Geometry *>> eidToElm;
        map<int, vector<SpatialDomains::Geometry *>> fidToElm;

        vector<SpatialDomains::Geometry *> el;
        for (auto &[geom, tag] : m_mesh->m_elementTags[meshDim])
        {
            el.push_back(geom);
        }

        for (auto &geom : el)
        {
            for (int i = 0; i < geom->GetNumEdges(); i++)
            {
                eidToElm[geom->GetEid(i)].push_back(geom);
            }
        }

        if (meshDim > 2)
        {
            for (auto [vids, face] : m_mesh->m_faceSet)
            {
                for (int i = 0; i < face->GetNumEdges(); i++)
                {
                    eidToFace[face->GetEid(i)].push_back(face);
                }
            }

            for (auto &geom : el)
            {
                for (int i = 0; i < geom->GetNumFaces(); i++)
                {
                    fidToElm[geom->GetFid(i)].push_back(geom);
                }
            }
        }

        // Faces are cleared for two distinct reasons that must be kept apart:
        // because the face belongs to an invalid element (clearedFaces), or
        // only because a nearby edge was straightened (clearedEdgeFaces). Only
        // the former mark genuine face-neighbours to re-check; the latter's
        // neighbours are already picked up as edge-neighbours of that edge.
        std::unordered_set<int> clearedEdges, clearedFaces, clearedEdgeFaces;
        std::unordered_set<SpatialDomains::Geometry *> clearedElmts;
        set<SpatialDomains::Geometry *> faceNeigh;
        set<SpatialDomains::Geometry *> edgeNeigh;

        bool parallel  = m_mesh->m_comm && m_mesh->m_comm->GetSize() > 1;
        string message = parallel ? " cleared in initial run on rank " +
                                        to_string(m_mesh->m_comm->GetRank())
                                  : " cleared in initial run.";

        // Ids cleared on the current pass, awaiting the neighbour sweep below.
        std::vector<int> newEdges, newElmtFaces;

        // The first two are recorded for the sweep, but an edge-collateral face
        // has no further neighbours to chase.
        auto clearEdge = [&](int eid) {
            if (clearedEdges.insert(eid).second)
            {
                graph->GetCurvedEdges().erase(eid);
                newEdges.push_back(eid);
            }
        };
        auto clearElmtFace = [&](int fid) {
            if (clearedFaces.insert(fid).second)
            {
                graph->GetCurvedFaces().erase(fid);
                newElmtFaces.push_back(fid);
            }
        };
        auto clearEdgeFace = [&](int fid) {
            if (clearedEdgeFaces.insert(fid).second)
            {
                graph->GetCurvedFaces().erase(fid);
            }
        };

        // Sweep the freshly-cleared ids: straighten the faces around each
        // cleared edge (2D: the neighbouring elements' faces; 3D: the faces
        // sharing the edge) as edge-collateral, and populate edgeNeigh and
        // faceNeigh
        auto propagateCleared = [&]() {
            for (size_t i = 0; i < newEdges.size(); ++i)
            {
                int eid = newEdges[i];
                if (meshDim == 3)
                {
                    auto fit = eidToFace.find(eid);
                    if (fit != eidToFace.end())
                    {
                        for (auto &face : fit->second)
                        {
                            clearEdgeFace(face->GetGlobalID());
                        }
                    }
                }
                auto it = eidToElm.find(eid);
                if (it != eidToElm.end())
                {
                    for (auto &elmt : it->second)
                    {
                        if (meshDim == 2)
                        {
                            clearEdgeFace(elmt->GetGlobalID());
                        }
                        edgeNeigh.insert(elmt);
                    }
                }
            }
            for (size_t i = 0; i < newElmtFaces.size(); ++i)
            {
                auto it = fidToElm.find(newElmtFaces[i]);
                if (it != fidToElm.end())
                {
                    for (auto &elmt : it->second)
                    {
                        faceNeigh.insert(elmt);
                    }
                }
            }
            newEdges.clear();
            newElmtFaces.clear();
        };

        // Iterate over list of elements of expansion dimension. In parallel
        // every rank must take part in the collective communication below on
        // each pass, so the loop is driven by a globally-reduced continuation
        // flag rather than the local queue size (which differs between ranks).
        int keepGoing = 1;
        while (keepGoing)
        {
            vector<SpatialDomains::Geometry *> invalidEls;
            for (auto &geom : el)
            {
                if (m_config["prismonly"].beenSet &&
                    geom->GetShapeType() != LibUtilities::ePrism)
                {
                    continue;
                }

                if (Invalid(geom, thr)) //(!gfac->IsValid())
                {
                    m_log(VERBOSE) << "Element " << geom->GetGlobalID()
                                   << " invalid." << endl;
                    invalidEls.push_back(geom);
                }
            }

            m_log(VERBOSE) << invalidEls.size() << " out of " << el.size()
                           << message << endl;

            // Remove curvature from every invalid element (detected up front so
            // rebuilt xmaps can't affect each other): its faces are cleared as
            // element-faces and its edges as edges. The sweep then straightens
            // the collateral faces around those edges and queues neighbours.
            for (auto &geom : invalidEls)
            {
                for (int i = 0; i < geom->GetNumFaces(); i++)
                {
                    clearElmtFace(geom->GetFid(i));
                }
                for (int i = 0; i < geom->GetNumEdges(); i++)
                {
                    clearEdge(geom->GetEid(i));
                }
                if (meshDim == 1)
                {
                    clearEdge(geom->GetGlobalID());
                }
                else if (meshDim == 2)
                {
                    clearElmtFace(geom->GetGlobalID());
                }
                clearedElmts.insert(geom);
            }
            propagateCleared();

            // Synchronise the cleared entities across all partitions.
            if (parallel)
            {
                for (int eid : GatherUnion(m_mesh->m_comm, clearedEdges))
                {
                    clearEdge(eid);
                }
                for (int fid : GatherUnion(m_mesh->m_comm, clearedFaces))
                {
                    clearElmtFace(fid);
                }
                propagateCleared();
            }

            el.clear();
            for (auto &geom : faceNeigh)
            {
                if (clearedElmts.find(geom) == clearedElmts.end())
                {
                    geom->Reset(graph->GetCurvedEdges(),
                                graph->GetCurvedFaces());
                    el.push_back(geom);
                }
            }
            faceNeigh.clear();
            string msg = " face";

            // Iterate over face neighbours first to avoid linearising
            // temporarily invalid corner touching elements. In parallel this
            // must be a global decision, edgeNeigh is left intact
            // (and keeps accumulating) on the ranks that wait.
            int faceNeighDone = (el.size() == 0) ? 1 : 0;
            if (parallel)
            {
                m_mesh->m_comm->AllReduce(faceNeighDone,
                                          LibUtilities::ReduceMin);
            }
            if (faceNeighDone)
            {
                for (auto &geom : edgeNeigh)
                {
                    if (clearedElmts.find(geom) == clearedElmts.end())
                    {
                        geom->Reset(graph->GetCurvedEdges(),
                                    graph->GetCurvedFaces());
                        el.push_back(geom);
                    }
                }
                edgeNeigh.clear();
                msg = " edge";
            }

            message = parallel ? msg + " neighbours cleared on rank " +
                                     to_string(m_mesh->m_comm->GetRank())
                               : msg + " neighbours cleared.";

            // Continue while any partition still has elements to process, so
            // all ranks enter the collective communication above in lockstep.
            keepGoing = (el.size() > 0) ? 1 : 0;
            if (parallel)
            {
                m_mesh->m_comm->AllReduce(keepGoing, LibUtilities::ReduceMax);
            }
        }

        m_log(VERBOSE) << "Removed curvature from " << clearedElmts.size()
                       << " elements (" << clearedEdges.size() << " edges, "
                       << clearedFaces.size() << " element faces, "
                       << clearedEdgeFaces.size()
                       << " faces around cleared edges)" << endl;

        if (m_config["extract"].beenSet)
        {
            MeshSharedPtr dmp = std::shared_ptr<Mesh>(new Mesh());
            dmp->m_comm       = m_mesh->m_comm;
            dmp->m_meshGraph->SetMeshDimension(meshDim);
            dmp->m_meshGraph->SetSpaceDimension(graph->GetSpaceDimension());

            auto pointGeoms =
                dmp->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>();
            for (auto &geom : clearedElmts)
            {
                std::vector<SpatialDomains::PointGeom *> nodeList;
                for (int i = 0; i < geom->GetNumVerts(); i++)
                {
                    auto vert = geom->GetVertex(i);
                    auto it   = pointGeoms.find(vert->GetGlobalID());
                    if (it == pointGeoms.end())
                    {
                        auto copyVert = dmp->m_meshGraph->CreatePointGeom(
                            graph->GetSpaceDimension(), vert->GetGlobalID(),
                            (*vert)[0], (*vert)[1], (*vert)[2]);
                        nodeList.push_back(copyVert);
                    }
                    else
                    {
                        nodeList.push_back(it->second);
                    }
                }

                // Reuse the source geometry's global IDs in case partitioned
                ElmtIds forceIDs;
                forceIDs.elmt = geom->GetGlobalID();
                forceIDs.edges.resize(geom->GetNumEdges());
                for (int e = 0; e < geom->GetNumEdges(); ++e)
                {
                    forceIDs.edges[e] = geom->GetEid(e);
                }
                if (meshDim == 3)
                {
                    forceIDs.faces.resize(geom->GetNumFaces());
                    for (int f = 0; f < geom->GetNumFaces(); ++f)
                    {
                        forceIDs.faces[f] = geom->GetFid(f);
                    }
                }

                ElmtConfig conf(geom->GetShapeType(), 1, false, false, false);
                auto geomCopy = GetElementFactory().CreateInstance(
                    geom->GetShapeType(), nodeList, dmp->m_meshGraph,
                    dmp->m_edgeSet, dmp->m_faceSet, conf, nullptr, nullptr,
                    nullptr, &forceIDs);

                dmp->m_elementTags[meshDim][geomCopy] =
                    m_mesh->m_elementTags[meshDim][geom];
                for (int i = 0; i < geom->GetNumFacets(); i++)
                {
                    auto it = m_mesh->m_elementTags[meshDim - 1].find(
                        geom->GetFacet(i));
                    if (it != m_mesh->m_elementTags[meshDim - 1].end())
                    {
                        dmp->m_elementTags[meshDim - 1][geomCopy->GetFacet(i)] =
                            it->second;
                    }
                }
            }

            ModuleSharedPtr mod = GetModuleFactory().CreateInstance(
                ModuleKey(eOutputModule, "xml"), dmp);
            mod->SetLogger(m_log);
            mod->RegisterConfig("outfile",
                                m_config["extract"].as<string>().c_str());
            mod->RegisterConfig("uncompress", "1");
            mod->ProcessComposites();
            mod->SetDefaults();
            mod->Process();
        }

        for (auto &geom : clearedElmts)
        {
            geom->Reset(m_mesh->m_meshGraph->GetCurvedEdges(),
                        m_mesh->m_meshGraph->GetCurvedFaces());
            geom->FillGeom();
        }
    }
}

bool ProcessLinear::Invalid(SpatialDomains::Geometry *geom, NekDouble thr)
{
    geom->Setup();
    geom->FillGeom();
    LibUtilities::PointsKeyVector p = geom->GetXmap()->GetPointsKeys();
    // Generate geometric factors.
    SpatialDomains::GeomFactorsUniquePtr gfac = geom->GenGeomFactors(p);

    if (!gfac->IsValid())
    {
        return true;
    }

    auto blankGraph =
        MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    EdgeMap blankEdgeSet;
    FaceMap blankFaceSet;
    std::vector<SpatialDomains::PointGeom *> nodeList;
    for (int i = 0; i < geom->GetNumVerts(); i++)
    {
        nodeList.push_back(geom->GetVertex(i));
    }
    auto geomL = CreateElementLite(geom->GetShapeType(), nodeList, blankGraph,
                                   blankEdgeSet, blankFaceSet);
    // Need to populate edge orientations
    geomL->ResetLite();

    SpatialDomains::DerivStorage deriv         = gfac->GetDeriv(p);
    SpatialDomains::GeomFactorsUniquePtr gfacL = geomL->GenGeomFactors(p);
    SpatialDomains::DerivStorage derivL        = gfacL->GetDeriv(p);

    if (!gfacL->IsValid())
    {
        stringstream err;
        err << "Negative jacobian for linearised while positive for curved "
            << "(element ID = " << geom->GetGlobalID() << ") "
            << "(first vertex ID = " << geom->GetVid(0) << ")";
        NEKERROR(ErrorUtil::ewarning, err.str());
    }

    int meshDim   = m_mesh->m_meshGraph->GetMeshDimension();
    const int pts = deriv[0][0].size();
    Array<OneD, NekDouble> jc(pts);
    Array<OneD, NekDouble> jcL(pts);
    for (int k = 0; k < pts; ++k)
    {
        DNekMat jac(meshDim, meshDim, 0.0, eFULL);
        DNekMat jacL(meshDim, meshDim, 0.0, eFULL);

        for (int l = 0; l < meshDim; ++l)
        {
            for (int j = 0; j < meshDim; ++j)
            {
                jac(j, l)  = deriv[l][j][k];
                jacL(j, l) = derivL[l][j][k];
            }
        }

        if (meshDim == 2)
        {
            jc[k]  = jac(0, 0) * jac(1, 1) - jac(0, 1) * jac(1, 0);
            jcL[k] = jacL(0, 0) * jacL(1, 1) - jacL(0, 1) * jacL(1, 0);
        }
        else if (meshDim == 3)
        {
            jc[k] =
                jac(0, 0) * (jac(1, 1) * jac(2, 2) - jac(2, 1) * jac(1, 2)) -
                jac(0, 1) * (jac(1, 0) * jac(2, 2) - jac(2, 0) * jac(1, 2)) +
                jac(0, 2) * (jac(1, 0) * jac(2, 1) - jac(2, 0) * jac(1, 1));
            jcL[k] = jacL(0, 0) *
                         (jacL(1, 1) * jacL(2, 2) - jacL(2, 1) * jacL(1, 2)) -
                     jacL(0, 1) *
                         (jacL(1, 0) * jacL(2, 2) - jacL(2, 0) * jacL(1, 2)) +
                     jacL(0, 2) *
                         (jacL(1, 0) * jacL(2, 1) - jacL(2, 0) * jacL(1, 1));
        }
    }

    Array<OneD, NekDouble> j(pts);
    Vmath::Vdiv(jc.size(), jc, 1, jcL, 1, j, 1);

    NekDouble scaledJac =
        Vmath::Vmin(j.size(), j, 1) / Vmath::Vmax(j.size(), j, 1);

    if (scaledJac < thr)
    {
        return true;
    }

    return false;
}
} // namespace Nektar::NekMesh
