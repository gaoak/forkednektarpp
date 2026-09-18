///////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessRevolve.cpp
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
//  Description: Revolve a two-dimensional mesh around the y-axis to form a
//  three-dimensional mesh.  Arguments are "angle" and "layers"
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessRevolve.h"
#include <NekMesh/MeshElements/Element.h>

using namespace std;

namespace Nektar::NekMesh
{
ModuleKey ProcessRevolve::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "revolve"), ProcessRevolve::create);

ProcessRevolve::ProcessRevolve(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["layers"] = ConfigOption(false, "24", "Number of layers");
    m_config["angle"] =
        ConfigOption(false, "full", "Angle of revolution (in radians)");
}

ProcessRevolve::~ProcessRevolve()
{
}

void ProcessRevolve::Process()
{
    m_log(VERBOSE) << "Revolving grid." << endl;
    auto m_graph = m_mesh->m_meshGraph;

    if (m_graph->GetSpaceDimension() != 2)
    {
        m_log(FATAL) << "Revolve should only be called for a two dimensional "
                     << "mesh" << endl;
    }

    int nLayers = m_config["layers"].as<int>();
    // Special case needed for full revolutions
    bool full       = m_config["angle"].as<std::string>() == "full";
    NekDouble angle = full ? 2 * M_PI : m_config["angle"].as<NekDouble>();
    if (angle < 0 || angle > 2 * M_PI)
    {
        m_log(FATAL) << "Angle should be between 0 and 2pi" << endl;
    }
    NekDouble dphi = angle / nLayers;
    // Increment space and expansion dimensions.
    int m_dim = 3;
    m_graph->SetSpaceDimension(m_dim);
    m_graph->SetMeshDimension(m_dim);

    m_log(VERBOSE) << "Boundary composites" << endl;
    for (auto &[id, comp] : m_graph->GetComposites())
    {
        if (comp->m_geomVec[0]->GetShapeDim() != 1)
        {
            continue;
        }
        m_log(VERBOSE) << id;
        for (auto &geom : comp->m_geomVec)
        {
            m_log(VERBOSE) << "\t" << geom->GetGlobalID() << " ("
                           << geom->GetVid(0) << ", " << geom->GetVid(1) << ")";
        }
        m_log(VERBOSE) << endl;
    }

    int initNumVerts = m_graph->GetNumGeoms<SpatialDomains::PointGeom>();
    std::vector<std::pair<int, SpatialDomains::PointGeomUniquePtr>> newVerts;

    // Create vertices for subsequent layers.
    for (auto [vertID, vert] : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        vert->SetCoordim(m_dim);
        if ((*vert)[0] <= 0)
        {
            // Disallow x=0 for now
            m_log(FATAL)
                << "All vertices require positive (non-zero)X coordinates."
                << endl;
        }
        NekDouble r0 = sqrt((*vert)[0] * (*vert)[0] + (*vert)[2] * (*vert)[2]);
        for (int i = 1; i < nLayers + 1; i++)
        {
            if (full && i == nLayers)
            {
                // For last layer we will connect up to first layer
                break;
            }
            int newID = i * initNumVerts + vert->GetGlobalID();
            auto newVert =
                ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                    m_dim, newID, r0 * cos(i * dphi), (*vert)[1],
                    r0 * sin(i * dphi));
            newVerts.push_back(
                std::pair<int, SpatialDomains::PointGeomUniquePtr>(
                    {newID, std::move(newVert)}));
        }
    }
    m_graph->BulkAddGeom<SpatialDomains::PointGeom>(newVerts);

    // Copy 2D mesh elements and reset the global mesh[2] for boundary.
    auto initTags = m_mesh->m_elementTags[2];
    m_mesh->m_elementTags[2].clear();

    std::map<int, SpatialDomains::CompositeSharedPtr> faceTagToVolComp;

    // Create the 3D elements
    for (int j = 0; j < nLayers; ++j)
    {
        int next = full ? (j + 1) % nLayers : j + 1;
        for (auto &[geom, tag] : initTags)
        {
            if (faceTagToVolComp.find(tag) == faceTagToVolComp.end())
            {
                faceTagToVolComp[tag] = MemoryManager<
                    SpatialDomains::Composite>::AllocateSharedPtr();
            }

            if (geom->GetShapeType() == LibUtilities::eQuadrilateral)
            {
                // Hex
                vector<SpatialDomains::PointGeom *> nodeList(8);
                nodeList[0] =
                    m_graph->GetPointGeom(geom->GetVid(0) + j * initNumVerts);
                nodeList[1] =
                    m_graph->GetPointGeom(geom->GetVid(1) + j * initNumVerts);
                nodeList[2] =
                    m_graph->GetPointGeom(geom->GetVid(2) + j * initNumVerts);
                nodeList[3] =
                    m_graph->GetPointGeom(geom->GetVid(3) + j * initNumVerts);
                nodeList[4] = m_graph->GetPointGeom(geom->GetVid(0) +
                                                    next * initNumVerts);
                nodeList[5] = m_graph->GetPointGeom(geom->GetVid(1) +
                                                    next * initNumVerts);
                nodeList[6] = m_graph->GetPointGeom(geom->GetVid(2) +
                                                    next * initNumVerts);
                nodeList[7] = m_graph->GetPointGeom(geom->GetVid(3) +
                                                    next * initNumVerts);

                auto hexGeom = CreateElementLite(
                    LibUtilities::eHexahedron, nodeList, m_graph,
                    m_mesh->m_edgeSet, m_mesh->m_faceSet);
                faceTagToVolComp[tag]->m_geomVec.push_back(hexGeom);
                m_mesh->m_elementTags[3][hexGeom] = tag;
            }
            else
            {
                // Prism
                vector<SpatialDomains::PointGeom *> nodeList(6);
                nodeList[0] = m_graph->GetPointGeom(geom->GetVid(0) +
                                                    next * initNumVerts);
                nodeList[1] = m_graph->GetPointGeom(geom->GetVid(1) +
                                                    next * initNumVerts);
                nodeList[2] =
                    m_graph->GetPointGeom(geom->GetVid(1) + j * initNumVerts);
                nodeList[3] =
                    m_graph->GetPointGeom(geom->GetVid(0) + j * initNumVerts);
                nodeList[4] = m_graph->GetPointGeom(geom->GetVid(2) +
                                                    next * initNumVerts);
                nodeList[5] =
                    m_graph->GetPointGeom(geom->GetVid(2) + j * initNumVerts);

                auto prismGeom =
                    CreateElementLite(LibUtilities::ePrism, nodeList, m_graph,
                                      m_mesh->m_edgeSet, m_mesh->m_faceSet);
                faceTagToVolComp[tag]->m_geomVec.push_back(prismGeom);
                m_mesh->m_elementTags[3][prismGeom] = tag;
            }
        }
    }

    ProcessElements();

    // Asign the high-order info (curves)
    std::vector<std::pair<int, SpatialDomains::CurveUniquePtr>> newCurves;
    for (auto &[oldEdgeID, oldCurve] : m_graph->GetCurvedEdges())
    {
        auto oldSeg = m_graph->GetSegGeom(oldEdgeID);
        for (int i = 1; i < nLayers + 1; i++)
        {
            if (full && i == nLayers)
            {
                // For last layer we will connect up to first layer
                break;
            }
            auto idPair = std::make_pair(i * initNumVerts + oldSeg->GetVid(0),
                                         i * initNumVerts + oldSeg->GetVid(1));
            auto it     = m_mesh->m_edgeSet.find(idPair);
            ASSERTL0(it != m_mesh->m_edgeSet.end(), "could not find edge");
            auto newSeg = it->second;

            auto newCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    newSeg->GetGlobalID(), oldCurve->m_ptype);
            for (int p = 0; p < oldCurve->m_points.size(); p++)
            {
                if (oldCurve->m_points[p] == oldSeg->GetVertex(0))
                {
                    newCurve->m_points.push_back(newSeg->GetVertex(0));
                }
                else if (oldCurve->m_points[p] == oldSeg->GetVertex(1))
                {
                    newCurve->m_points.push_back(newSeg->GetVertex(1));
                }
                else
                {
                    NekDouble r0 = sqrt((*oldCurve->m_points[p])[0] *
                                            (*oldCurve->m_points[p])[0] +
                                        (*oldCurve->m_points[p])[2] *
                                            (*oldCurve->m_points[p])[2]);
                    auto point   = ObjPoolManager<SpatialDomains::PointGeom>::
                        AllocateUniquePtr(
                            m_dim, newCurve->m_curveID, r0 * cos(i * dphi),
                            (*oldCurve->m_points[p])[1], r0 * sin(i * dphi));
                    newCurve->m_points.push_back(point.get());
                    m_graph->GetAllCurveNodes().push_back(std::move(point));
                }
            }
            if (i * initNumVerts + oldSeg->GetVid(0) == newSeg->GetVid(1))
            {
                std::reverse(newCurve->m_points.begin(),
                             newCurve->m_points.end());
            }
            it->second->SetCurve(newCurve.get());
            newCurves.push_back(std::pair<int, SpatialDomains::CurveUniquePtr>{
                it->second->GetGlobalID(), std::move(newCurve)});
        }
    }
    m_graph->GetCurvedEdges().insert(std::make_move_iterator(newCurves.begin()),
                                     std::make_move_iterator(newCurves.end()));

    // Create the new boundary composites (2D elements)
    unsigned int maxCompId = 0;
    for (auto &[id, comp] : m_graph->GetComposites())
    {
        maxCompId = std::max<int>(maxCompId, id);
        if (comp->m_geomVec[0]->GetShapeDim() == 1)
        {
            auto newComp =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
            for (auto &geom : comp->m_geomVec)
            {
                for (int j = 0; j < nLayers; ++j)
                {
                    int next = full ? (j + 1) % nLayers : j + 1;
                    std::array<int, 4> vertIDs = {
                        j * initNumVerts + geom->GetVid(0),
                        j * initNumVerts + geom->GetVid(1),
                        next * initNumVerts + geom->GetVid(0),
                        next * initNumVerts + geom->GetVid(1)};
                    auto it = m_mesh->m_faceSet.find(vertIDs);
                    ASSERTL0(it != m_mesh->m_faceSet.end(),
                             "could not find face");
                    newComp->m_geomVec.push_back(it->second);
                    m_mesh->m_elementTags[2][it->second] = id;
                }
            }
            comp = newComp;
        }
        else if (comp->m_geomVec[0]->GetShapeDim() == 2)
        {
            comp = faceTagToVolComp[id];
        }
    }

    // Create composites for the periodic faces (starting-end)
    // no periodic composites if full revolution
    if (!full)
    {
        std::array<SpatialDomains::CompositeSharedPtr, 2> periodicComps = {
            MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr(),
            MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr()};

        for (auto &[geom, tag] : initTags)
        {
            periodicComps[0]->m_geomVec.push_back(geom);
            // Find the periodic face pair with the vertex keys
            std::array<int, 4> vertIDs = {
                nLayers * initNumVerts + geom->GetVid(0),
                nLayers * initNumVerts + geom->GetVid(1),
                nLayers * initNumVerts + geom->GetVid(2), -1};
            if (geom->GetNumVerts() == 4)
            {
                vertIDs[3] = nLayers * initNumVerts + geom->GetVid(3);
            }
            auto it = m_mesh->m_faceSet.find(vertIDs);
            ASSERTL0(it != m_mesh->m_faceSet.end(),
                     "could not find the opposite periodic face");
            periodicComps[1]->m_geomVec.push_back(it->second);
        }

        m_graph->GetComposites()[++maxCompId] = periodicComps[0];
        m_graph->GetComposites()[++maxCompId] = periodicComps[1];
        for (int id = maxCompId - 1; id <= maxCompId; id++)
        {
            for (auto &geom : m_graph->GetComposites()[id]->m_geomVec)
            {
                m_mesh->m_elementTags[2][geom] = id;
            }
        }
    }
}
} // namespace Nektar::NekMesh
