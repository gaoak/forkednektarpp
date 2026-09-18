////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessExtractSurf.cpp
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
//  Description: Extract one or more surfaces from mesh.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessExtractSurf.h"
#include <LibUtilities/BasicUtils/ParseUtils.h>

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey ProcessExtractSurf::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "extract"), ProcessExtractSurf::create,
        "Process elements to extract a specified surface(s) or composites(s).");

ProcessExtractSurf::ProcessExtractSurf(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["surf"] = ConfigOption(
        false, "NotSet", "Tag identifying surface/composite to process.");
    m_config["detectbnd"] =
        ConfigOption(false, "-1", "Tag to detect boundary composites");
}

ProcessExtractSurf::~ProcessExtractSurf()
{
}

void moveCurve(int &id, SpatialDomains::CurveMap &oldMap,
               SpatialDomains::CurveMap &newMap,
               std::unordered_set<SpatialDomains::PointGeom *> &moveNodes)
{
    auto findIt = oldMap.find(id);
    if (findIt != oldMap.end())
    {
        // add curvature nodes to set (not vertices hence begin()+1 and end()-1)
        for (auto &point : findIt->second->m_points)
        {
            moveNodes.insert(point);
        }
        newMap[id] = std::move(findIt->second);
    }
}

void ProcessExtractSurf::Process()
{
    string surf    = m_config["surf"].as<string>();
    bool detectbnd = m_config["detectbnd"].beenSet;
    auto oldGraph  = m_mesh->m_meshGraph;

    // Obtain vector of surface IDs from string.
    vector<unsigned int> surfs;
    if (!ParseUtils::GenerateSeqVector(surf, surfs))
    {
        m_log(FATAL) << "Failed to interp surf string. Have you specified this "
                     << "string?" << endl;
    }
    sort(surfs.begin(), surfs.end());

    // If we're running in verbose mode print out a list of surfaces.
    m_log(VERBOSE) << "Extracting surface" << (surfs.size() > 1 ? "s" : "")
                   << " " << surf << endl;

    int oldDim = oldGraph->GetMeshDimension();
    int newDim = oldGraph->GetComposite(surfs[0])->m_geomVec[0]->GetShapeDim();
    auto newGraph =
        MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    newGraph->SetMeshDimension(newDim);
    newGraph->SetSpaceDimension(oldGraph->GetSpaceDimension());
    auto &newVerts     = newGraph->GetGeomMap<SpatialDomains::PointGeom>();
    auto &newSegGeoms  = newGraph->GetGeomMap<SpatialDomains::SegGeom>();
    auto &newTriGeoms  = newGraph->GetGeomMap<SpatialDomains::TriGeom>();
    auto &newQuadGeoms = newGraph->GetGeomMap<SpatialDomains::QuadGeom>();
    std::unordered_set<SpatialDomains::PointGeom *> vertSet;
    std::unordered_set<SpatialDomains::PointGeom *> moveCurveNodes;

    // Iterate over list of surface elements.
    for (auto &[elmt, tag] : m_mesh->m_elementTags[newDim])
    {
        // Work out whether this lies on our surface of interest.
        // If it doesn't continue to next element.
        if (std::find(surfs.begin(), surfs.end(), tag) == surfs.end())
        {
            continue;
        }

        int elmtId = elmt->GetGlobalID();
        switch (elmt->GetShapeType())
        {
            case LibUtilities::eSegment:
                newGraph->AddGeom<SpatialDomains::SegGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::SegGeom>(elmtId));
                break;
            case LibUtilities::eTriangle:
                newGraph->AddGeom<SpatialDomains::TriGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::TriGeom>(elmtId));
                break;
            case LibUtilities::eQuadrilateral:
                newGraph->AddGeom<SpatialDomains::QuadGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::QuadGeom>(elmtId));
                break;
            case LibUtilities::eTetrahedron:
                newGraph->AddGeom<SpatialDomains::TetGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::TetGeom>(elmtId));
                break;
            case LibUtilities::ePyramid:
                newGraph->AddGeom<SpatialDomains::PyrGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::PyrGeom>(elmtId));
                break;
            case LibUtilities::ePrism:
                newGraph->AddGeom<SpatialDomains::PrismGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::PrismGeom>(elmtId));
                break;
            case LibUtilities::eHexahedron:
                newGraph->AddGeom<SpatialDomains::HexGeom>(
                    elmtId,
                    oldGraph->ExtractGeom<SpatialDomains::HexGeom>(elmtId));
                break;
            default:
                NEKERROR(ErrorUtil::efatal,
                         "Unexpected shape type to extract.");
                break;
        }

        // Insert surface vertices.
        for (int i = 0; i < elmt->GetNumVerts(); i++)
        {
            if (newVerts.find(elmt->GetVid(i)) == newVerts.end())
            {
                newGraph->AddGeom<SpatialDomains::PointGeom>(
                    elmt->GetVid(i),
                    oldGraph->ExtractGeom<SpatialDomains::PointGeom>(
                        elmt->GetVid(i)));
                vertSet.insert(elmt->GetVertex(i));
            }
        }

        // Identify curvature nodes, move element curves and surface edges in 2D
        if (newDim == 1)
        {
            moveCurve(elmtId, oldGraph->GetCurvedEdges(),
                      newGraph->GetCurvedEdges(), moveCurveNodes);
            continue;
        }
        for (int i = 0; i < elmt->GetNumEdges(); i++)
        {
            int edgeID = elmt->GetEid(i);
            if (newSegGeoms.find(edgeID) == newSegGeoms.end())
            {
                newGraph->AddGeom<SpatialDomains::SegGeom>(
                    edgeID,
                    oldGraph->ExtractGeom<SpatialDomains::SegGeom>(edgeID));
                moveCurve(edgeID, oldGraph->GetCurvedEdges(),
                          newGraph->GetCurvedEdges(), moveCurveNodes);
            }
        }
        if (newDim == 2)
        {
            moveCurve(elmtId, oldGraph->GetCurvedFaces(),
                      newGraph->GetCurvedFaces(), moveCurveNodes);
            continue;
        }
        for (int i = 0; i < elmt->GetNumFaces(); i++)
        {
            int faceID = elmt->GetFid(i);
            if (elmt->GetFace(i)->GetShapeType() == LibUtilities::eTriangle &&
                newTriGeoms.find(faceID) == newTriGeoms.end())
            {
                newGraph->AddGeom<SpatialDomains::TriGeom>(
                    faceID,
                    oldGraph->ExtractGeom<SpatialDomains::TriGeom>(faceID));
                moveCurve(faceID, oldGraph->GetCurvedFaces(),
                          newGraph->GetCurvedFaces(), moveCurveNodes);
            }
            else if (elmt->GetFace(i)->GetShapeType() ==
                         LibUtilities::eQuadrilateral &&
                     newQuadGeoms.find(faceID) == newQuadGeoms.end())
            {
                newGraph->AddGeom<SpatialDomains::QuadGeom>(
                    faceID,
                    oldGraph->ExtractGeom<SpatialDomains::QuadGeom>(faceID));
                moveCurve(faceID, oldGraph->GetCurvedFaces(),
                          newGraph->GetCurvedFaces(), moveCurveNodes);
            }
        }
    }

    // Move curve nodes
    for (auto &node : oldGraph->GetAllCurveNodes())
    {
        if (vertSet.find(node.get()) != vertSet.end())
        {
            continue;
        }
        else if (moveCurveNodes.find(node.get()) != moveCurveNodes.end())
        {
            newGraph->GetAllCurveNodes().push_back(std::move(node));
        }
    }

    int maxId = -1;
    for (auto &[id, oldComp] : oldGraph->GetComposites())
    {
        if (find(surfs.begin(), surfs.end(), id) == surfs.end())
        {
            continue;
        }
        maxId = std::max(maxId, id);
    }
    // Process composites, and repopulate m_elementTags
    auto oldTags = m_mesh->m_elementTags[newDim];
    m_mesh->m_elementTags[oldDim].clear();
    m_mesh->m_elementTags[oldDim - 1].clear();
    for (auto &[id, oldComp] : oldGraph->GetComposites())
    {
        if (find(surfs.begin(), surfs.end(), id) == surfs.end())
        {
            continue;
        }
        // 2D surfaces may contain both quadrilaterals and triangles and so need
        // to be split up.
        if (oldDim == 3 && newDim == 2)
        {
            LibUtilities::ShapeType type1 =
                oldComp->m_geomVec[0]->GetShapeType();
            // composite for first shape type found (tri or quad)
            auto newComp1 =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
            // second composite for if the other shape type is found
            auto newComp2 =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
            for (auto &geom : oldComp->m_geomVec)
            {
                if (geom->GetShapeType() == type1)
                {
                    newComp1->m_geomVec.push_back(geom);
                }
                else
                {
                    newComp2->m_geomVec.push_back(geom);
                }
                m_mesh->m_elementTags[newDim][geom] = oldTags[geom];
            }
            newGraph->GetComposites()[id] = newComp1;
            if (newComp2->m_geomVec.size())
            {
                newGraph->GetComposites()[++maxId] = newComp2;
                for (auto &geom : newComp2->m_geomVec)
                {
                    m_mesh->m_elementTags[newDim][geom] = maxId;
                }
                // Print out mapping information if we split a composite
                m_log(VERBOSE)
                    << "  - Split mixed composite " << id << " into composites "
                    << id << " and " << maxId << "." << endl;
            }
        }
        else
        {
            newGraph->GetComposites()[id] = oldComp;
            for (auto &geom : oldComp->m_geomVec)
            {
                m_mesh->m_elementTags[newDim][geom] = oldTags[geom];
            }
        }
    }

    // Create domains for element composites
    int domID = 0;
    for (auto &[compID, comp] : newGraph->GetComposites())
    {
        newGraph->GetDomain()[domID++] =
            SpatialDomains::CompositeMap({{compID, comp}});
    }

    // Detect composites for boundary edges. This is done by looping over all
    // extracted elements in the new graph and finding boundary sides that only
    // belong to one of these extracted elements.
    if (detectbnd)
    {
        ASSERTL0(
            newDim >= 2,
            "Surface boundary detection only implemented for 2D and 3D meshes");

        std::unordered_set<int> visitedOnce;
        // find edges of face elements that are only used by 1 extracted face
        if (newDim == 2)
        {
            for (auto &[compID, comp] : newGraph->GetComposites())
            {
                for (auto &geom : comp->m_geomVec)
                {
                    for (int e = 0; e < geom->GetNumEdges(); e++)
                    {
                        // if the edge has already been found, erase it
                        if (!visitedOnce.insert(geom->GetEid(e)).second)
                        {
                            visitedOnce.erase(geom->GetEid(e));
                        }
                    }
                }
            }
        }
        else
        {
            for (auto &[compID, comp] : newGraph->GetComposites())
            {
                for (auto &geom : comp->m_geomVec)
                {
                    for (int f = 0; f < geom->GetNumFaces(); f++)
                    {
                        // if the edge has already been found, erase it
                        if (!visitedOnce.insert(geom->GetFid(f)).second)
                        {
                            visitedOnce.erase(geom->GetFid(f));
                        }
                    }
                }
            }
        }

        // Iterate over old newDim composites being dumped to group new
        // boundaries into new boundary composites
        for (auto &[id, oldComp] : oldGraph->GetComposites())
        {
            // continue if extracting this composite or not newDim
            if (find(surfs.begin(), surfs.end(), id) != surfs.end() ||
                oldComp->m_geomVec[0]->GetShapeDim() != newDim)
            {
                continue;
            }
            auto newComp =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
            if (newDim == 2)
            {
                for (auto &geom : oldComp->m_geomVec)
                {
                    for (int e = 0; e < geom->GetNumEdges(); e++)
                    {
                        if (visitedOnce.find(geom->GetEid(e)) !=
                            visitedOnce.end())
                        {
                            newComp->m_geomVec.push_back(
                                newGraph->GetSegGeom(geom->GetEid(e)));
                            visitedOnce.erase(geom->GetEid(e));
                        }
                    }
                }
            }
            else
            {
                for (auto &geom : oldComp->m_geomVec)
                {
                    for (int f = 0; f < geom->GetNumFaces(); f++)
                    {
                        if (visitedOnce.find(geom->GetFid(f)) !=
                            visitedOnce.end())
                        {
                            newComp->m_geomVec.push_back(
                                newGraph->GetGeometry2D(geom->GetFid(f)));
                            visitedOnce.erase(geom->GetFid(f));
                        }
                    }
                }
            }
            if (newComp->m_geomVec.size())
            {
                newGraph->GetComposites()[++maxId] = newComp;
                for (auto &geom : newComp->m_geomVec)
                {
                    m_mesh->m_elementTags[newDim - 1][geom] = maxId;
                }
            }
        }

        // Any remaining elements in visitedOnce should be in old
        // boundary composites, if newDim == oldDim
        for (auto &[id, oldComp] : oldGraph->GetComposites())
        {
            if (oldComp->m_geomVec[0]->GetShapeDim() != newDim - 1)
            {
                continue;
            }

            auto newComp =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
            for (auto &geom : oldComp->m_geomVec)
            {
                if (visitedOnce.find(geom->GetGlobalID()) != visitedOnce.end())
                {
                    newComp->m_geomVec.push_back(geom);
                    visitedOnce.erase(geom->GetGlobalID());
                }
            }
            if (newComp->m_geomVec.size())
            {
                newGraph->GetComposites()[++maxId] = newComp;
                for (auto &geom : newComp->m_geomVec)
                {
                    m_mesh->m_elementTags[newDim - 1][geom] = maxId;
                }
            }
        }

        ASSERTL0(visitedOnce.size() == 0,
                 "Some detected boundary elements not assigned to composites.");
    }

    m_mesh->m_meshGraph = newGraph;
}
} // namespace Nektar::NekMesh
