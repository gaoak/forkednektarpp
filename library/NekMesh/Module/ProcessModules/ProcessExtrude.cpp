///////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessExtrude.cpp
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
//  Description: Extrude a two-dimensional mesh to a three-dimensional mesh.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessExtrude.h"

using namespace std;

namespace Nektar::NekMesh
{
ModuleKey ProcessExtrude::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "extrude"), ProcessExtrude::create);

ProcessExtrude::ProcessExtrude(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["layers"] =
        ConfigOption(false, "5", "Number of layers to extrude");
    m_config["length"] = ConfigOption(false, "1.0", "Length of extrusion");
}

ProcessExtrude::~ProcessExtrude()
{
}

void ProcessExtrude::Process()
{
    m_log(VERBOSE) << "Extruding grid." << endl;
    auto graph = m_mesh->m_meshGraph;

    if (graph->GetSpaceDimension() != 2)
    {
        m_log(FATAL) << "Extrude should only be called for a two dimensional "
                     << "mesh" << endl;
    }

    int nLayers      = m_config["layers"].as<int>();
    NekDouble length = m_config["length"].as<NekDouble>();

    NekDouble dz = length / nLayers;

    // Increment space and expansion dimensions.
    int dim = 3;
    graph->SetSpaceDimension(dim);
    graph->SetMeshDimension(dim);

    // Save z plane coordinate
    NekDouble z0 = 0;

    int initNumVerts = graph->GetNumGeoms<SpatialDomains::PointGeom>();
    std::vector<std::pair<int, SpatialDomains::PointGeomUniquePtr>> newVerts;
    // Create vertices for subsequent layers.
    for (auto [vertID, vert] : graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        vert->SetCoordim(dim);
        z0 = (*vert)[2];
        for (int i = 1; i < nLayers + 1; ++i)
        {
            int newID = i * initNumVerts + vert->GetGlobalID();
            auto newVert =
                ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                    dim, newID, (*vert)[0], (*vert)[1], z0 + i * dz);
            newVerts.push_back(
                std::pair<int, SpatialDomains::PointGeomUniquePtr>(
                    {newID, std::move(newVert)}));
        }
    }
    graph->BulkAddGeom<SpatialDomains::PointGeom>(newVerts);

    auto initTags = m_mesh->m_elementTags[2];
    std::map<int, SpatialDomains::CompositeSharedPtr> faceTagToVolComp;
    std::array<SpatialDomains::CompositeSharedPtr, 2> periodicComps = {
        MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr(),
        MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr()};
    // Create 3D elements
    for (auto &[geom, tag] : initTags)
    {
        auto it = faceTagToVolComp.find(tag);
        if (it == faceTagToVolComp.end())
        {
            faceTagToVolComp[tag] =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
        }

        for (int j = 0; j < nLayers; ++j)
        {
            if (geom->GetShapeType() == LibUtilities::eQuadrilateral)
            {
                vector<SpatialDomains::PointGeom *> nodeList(8);
                nodeList[0] =
                    graph->GetPointGeom(geom->GetVid(0) + j * initNumVerts);
                nodeList[1] =
                    graph->GetPointGeom(geom->GetVid(1) + j * initNumVerts);
                nodeList[2] =
                    graph->GetPointGeom(geom->GetVid(2) + j * initNumVerts);
                nodeList[3] =
                    graph->GetPointGeom(geom->GetVid(3) + j * initNumVerts);
                nodeList[4] = graph->GetPointGeom(geom->GetVid(0) +
                                                  (j + 1) * initNumVerts);
                nodeList[5] = graph->GetPointGeom(geom->GetVid(1) +
                                                  (j + 1) * initNumVerts);
                nodeList[6] = graph->GetPointGeom(geom->GetVid(2) +
                                                  (j + 1) * initNumVerts);
                nodeList[7] = graph->GetPointGeom(geom->GetVid(3) +
                                                  (j + 1) * initNumVerts);

                auto hexGeom = CreateElementLite(
                    LibUtilities::eHexahedron, nodeList, graph,
                    m_mesh->m_edgeSet, m_mesh->m_faceSet);
                faceTagToVolComp[tag]->m_geomVec.push_back(hexGeom);
                m_mesh->m_elementTags[3][hexGeom] = tag;
            }
            else
            {
                vector<SpatialDomains::PointGeom *> nodeList(6);
                nodeList[0] = graph->GetPointGeom(geom->GetVid(0) +
                                                  (j + 1) * initNumVerts);
                nodeList[1] = graph->GetPointGeom(geom->GetVid(1) +
                                                  (j + 1) * initNumVerts);
                nodeList[2] =
                    graph->GetPointGeom(geom->GetVid(1) + j * initNumVerts);
                nodeList[3] =
                    graph->GetPointGeom(geom->GetVid(0) + j * initNumVerts);
                nodeList[4] = graph->GetPointGeom(geom->GetVid(2) +
                                                  (j + 1) * initNumVerts);
                nodeList[5] =
                    graph->GetPointGeom(geom->GetVid(2) + j * initNumVerts);

                auto prismGeom =
                    CreateElementLite(LibUtilities::ePrism, nodeList, graph,
                                      m_mesh->m_edgeSet, m_mesh->m_faceSet);
                faceTagToVolComp[tag]->m_geomVec.push_back(prismGeom);
                m_mesh->m_elementTags[3][prismGeom] = tag;
            }
        }

        periodicComps[0]->m_geomVec.push_back(geom);
        // Find opposite periodic face using m_faceSet
        std::array<int, 4> vertIDs =
            (geom->GetNumVerts() == 4)
                ? std::array<int, 4>{nLayers * initNumVerts + geom->GetVid(0),
                                     nLayers * initNumVerts + geom->GetVid(1),
                                     nLayers * initNumVerts + geom->GetVid(2),
                                     nLayers * initNumVerts + geom->GetVid(3)}
                : std::array<int, 4>{nLayers * initNumVerts + geom->GetVid(0),
                                     nLayers * initNumVerts + geom->GetVid(1),
                                     nLayers * initNumVerts + geom->GetVid(2),
                                     -1};
        auto it2 = m_mesh->m_faceSet.find(vertIDs);
        ASSERTL0(it2 != m_mesh->m_faceSet.end(), "could not find face");
        periodicComps[1]->m_geomVec.push_back(it2->second);
    }

    ProcessElements();

    // Copy edge curves
    std::vector<std::pair<int, SpatialDomains::CurveUniquePtr>> newCurves;
    for (auto &[initEdgeID, initCurve] : graph->GetCurvedEdges())
    {
        auto initSeg = graph->GetSegGeom(initEdgeID);
        for (int i = 1; i < nLayers + 1; ++i)
        {
            auto idPair = std::make_pair(i * initNumVerts + initSeg->GetVid(0),
                                         i * initNumVerts + initSeg->GetVid(1));
            auto it     = m_mesh->m_edgeSet.find(idPair);
            ASSERTL0(it != m_mesh->m_edgeSet.end(), "could not find edge");
            auto newSeg = it->second;

            auto newCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    newSeg->GetGlobalID(), initCurve->m_ptype);
            for (int p = 0; p < initCurve->m_points.size(); p++)
            {
                if (initCurve->m_points[p] == initSeg->GetVertex(0))
                {
                    newCurve->m_points.push_back(newSeg->GetVertex(0));
                }
                else if (initCurve->m_points[p] == initSeg->GetVertex(1))
                {
                    newCurve->m_points.push_back(newSeg->GetVertex(1));
                }
                else
                {
                    auto point = ObjPoolManager<SpatialDomains::PointGeom>::
                        AllocateUniquePtr(dim, newCurve->m_curveID,
                                          (*initCurve->m_points[p])[0],
                                          (*initCurve->m_points[p])[1],
                                          (*initCurve->m_points[p])[2] +
                                              i * dz);
                    newCurve->m_points.push_back(point.get());
                    graph->GetAllCurveNodes().push_back(std::move(point));
                }
            }
            if (i * initNumVerts + initSeg->GetVid(0) == newSeg->GetVid(1))
            {
                std::reverse(newCurve->m_points.begin(),
                             newCurve->m_points.end());
            }
            it->second->SetCurve(newCurve.get());
            newCurves.push_back(std::pair<int, SpatialDomains::CurveUniquePtr>{
                it->second->GetGlobalID(), std::move(newCurve)});
        }
    }
    graph->GetCurvedEdges().insert(std::make_move_iterator(newCurves.begin()),
                                   std::make_move_iterator(newCurves.end()));

    // Replace existing comps with higher dimension comps
    unsigned int maxCompId = 0;
    for (auto &[id, comp] : graph->GetComposites())
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
                    std::array<int, 4> vertIDs = {
                        j * initNumVerts + geom->GetVid(0),
                        j * initNumVerts + geom->GetVid(1),
                        (j + 1) * initNumVerts + geom->GetVid(0),
                        (j + 1) * initNumVerts + geom->GetVid(1)};
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

    // Set comps and tags of periodic faces
    graph->GetComposites()[++maxCompId] = periodicComps[0];
    graph->GetComposites()[++maxCompId] = periodicComps[1];
    for (int id = maxCompId - 1; id <= maxCompId; id++)
    {
        for (auto &geom : graph->GetComposites()[id]->m_geomVec)
        {
            m_mesh->m_elementTags[2][geom] = id;
        }
    }
}
} // namespace Nektar::NekMesh
