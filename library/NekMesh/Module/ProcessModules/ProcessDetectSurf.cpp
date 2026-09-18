////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessDetectSurf.cpp
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

#include "ProcessDetectSurf.h"
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <NekMesh/MeshElements/Element.h>

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey ProcessDetectSurf::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "detect"), ProcessDetectSurf::create,
        "Process elements to detect a surface.");

ProcessDetectSurf::ProcessDetectSurf(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["vol"] =
        ConfigOption(false, "-1", "Tag identifying surface to process.");
}

ProcessDetectSurf::~ProcessDetectSurf()
{
}

struct EdgeInfo
{
    int count                     = 0;
    SpatialDomains::SegGeom *edge = nullptr;
    unsigned int group            = 0;
};

void ProcessDetectSurf::Process()
{
    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    if (meshDim > 2)
    {
        m_log(WARNING) << "Surface detection only implemented for 2D meshes;"
                       << "ignoring this module." << endl;
        return;
    }

    string surf = m_config["vol"].as<string>();

    // Obtain vector of surface IDs from string.
    vector<unsigned int> surfs;
    if (surf != "-1")
    {
        ParseUtils::GenerateSeqVector(surf, surfs);
        sort(surfs.begin(), surfs.end());
    }

    // If we're running in verbose mode print out a list of surfaces.
    m_log(VERBOSE) << "ProcessDetectSurf: detecting surfaces";
    if (surfs.size() > 0)
    {
        m_log(VERBOSE) << " for surface" << (surfs.size() == 1 ? "" : "s")
                       << " " << surf << endl;
    }

    map<int, EdgeInfo> edgeCount;
    set<int> doneIds;
    map<int, SpatialDomains::Geometry *> idMap;

    // Which elements share each edge. This used to be read off the edge
    // itself, through Edge::m_elLink, and is cheaper to accumulate here
    // than to maintain globally: the pass below already visits every edge
    // of every element of interest.
    EdgeToElMap edgeToEl;

    // Iterate over list of surface elements.
    for (auto &[elmt, tag] : m_mesh->m_elementTags[meshDim])
    {
        // Work out whether this lies on our surface of interest.
        if (surfs.size() > 0 && !binary_search(surfs.begin(), surfs.end(),
                                               static_cast<unsigned int>(tag)))
        {
            continue;
        }

        // List all edges.
        for (int j = 0; j < elmt->GetNumEdges(); ++j)
        {
            auto *e = static_cast<SpatialDomains::SegGeom *>(elmt->GetEdge(j));
            int eId = e->GetGlobalID();
            edgeCount[eId].count++;
            edgeCount[eId].edge = e;
            edgeToEl[eId].push_back(elmt);
        }

        const int elId = elmt->GetGlobalID();
        doneIds.insert(elId);
        ASSERTL0(idMap.count(elId) == 0, "Shouldn't happen");
        idMap[elId] = elmt;
    }

    unsigned int maxId = 0;

    for (auto &cIt : m_mesh->m_meshGraph->GetComposites())
    {
        maxId = (std::max)(static_cast<unsigned int>(cIt.first), maxId);
    }

    ++maxId;

    while (doneIds.size() > 0)
    {
        SpatialDomains::Geometry *start = idMap[*(doneIds.begin())];

        vector<SpatialDomains::Geometry *> block;
        FindContiguousSurface(start, edgeToEl, doneIds, block);
        ASSERTL0(block.size() > 0, "Contiguous block not found");

        // Loop over all edges in block.
        for (auto *elmt : block)
        {
            for (int j = 0; j < elmt->GetNumEdges(); ++j)
            {
                auto eIt = edgeCount.find(elmt->GetEdge(j)->GetGlobalID());
                ASSERTL0(eIt != edgeCount.end(), "Couldn't find edge");
                eIt->second.group = maxId;
            }
        }

        ++maxId;
    }

    // An edge seen once bounds its block, so tag it with that block's ID.
    // The edges already exist as geometries, so unlike the old code there is
    // no element to create: tagging them and rebuilding the composites is
    // what puts one composite around each detected surface.
    for (auto &eIt : edgeCount)
    {
        if (eIt.second.count > 1)
        {
            continue;
        }

        m_mesh->m_elementTags[meshDim - 1][eIt.second.edge] = eIt.second.group;
    }

    ProcessComposites();
}

void ProcessDetectSurf::FindContiguousSurface(
    SpatialDomains::Geometry *start, const EdgeToElMap &edgeToEl,
    set<int> &doneIds, vector<SpatialDomains::Geometry *> &block)
{
    block.push_back(start);
    doneIds.erase(start->GetGlobalID());

    for (int i = 0; i < start->GetNumEdges(); ++i)
    {
        auto eIt = edgeToEl.find(start->GetEdge(i)->GetGlobalID());
        if (eIt == edgeToEl.end())
        {
            continue;
        }

        for (auto *elmt : eIt->second)
        {
            if (elmt == start || doneIds.count(elmt->GetGlobalID()) == 0)
            {
                continue;
            }

            FindContiguousSurface(elmt, edgeToEl, doneIds, block);
        }
    }
}
} // namespace Nektar::NekMesh
