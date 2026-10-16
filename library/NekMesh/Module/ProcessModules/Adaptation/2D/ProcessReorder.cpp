////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessReorder.cpp
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
//  Description: Split Tets and Triags to ratio 1:2 ; 1:4 ; 1:8
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/Module/ProcessModules/Adaptation/2D/ProcessReorder.h>

#include <LibUtilities/BasicUtils/Progressbar.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h>

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessReorder::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "reorder"), ProcessReorder::create,
        "Performs reordering of the IDs in the XML. ");

ProcessReorder::ProcessReorder(MeshSharedPtr m) : ProcessModule(m)
{
}

ProcessReorder::~ProcessReorder()
{
}

void ProcessReorder::Process()
{
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    const int meshDim                        = graph->GetMeshDimension();

    if (meshDim != 2 && meshDim != 3)
    {
        m_log(FATAL) << "Reordering only supported for 2D and 3D meshes."
                     << endl;
    }

    // Element IDs are also the keys of the graph's geometry maps (and in 2D
    // of the curved faces), so keep old -> new to re-key them afterwards.
    map<int, int> newIds;
    int i = 0;
    for (auto &[element, tag] : m_mesh->m_elementTags[meshDim])
    {
        (void)tag; // unused
        newIds[element->GetGlobalID()] = i;
        element->SetGlobalID(i);
        i++;
    }

    // Rebuild the geometry map of type T with the new ids as keys.
    auto rekey = [&](auto *type) {
        using T = std::remove_pointer_t<decltype(type)>;

        vector<int> oldIds;
        for (auto &geom : graph->GetGeomMap<T>())
        {
            oldIds.push_back(geom.first);
        }

        SpatialDomains::GeomMap<T> newMap;
        for (int id : oldIds)
        {
            auto geom = graph->ExtractGeom<T>(id);
            int newId = geom->GetGlobalID();
            if (newMap.count(newId))
            {
                m_log(FATAL) << "Element id " << newId
                             << " used twice after reordering." << endl;
            }
            newMap[newId] = std::move(geom);
        }
        graph->SetGeomMap<T>(std::move(newMap));
    };

    if (meshDim == 2)
    {
        rekey(static_cast<SpatialDomains::TriGeom *>(nullptr));
        rekey(static_cast<SpatialDomains::QuadGeom *>(nullptr));

        // TriGeom/QuadGeom find their curve by global id, and the writer uses
        // the curve id as FACEID.
        SpatialDomains::CurveMap newCurves;
        for (auto &[id, curve] : graph->GetCurvedFaces())
        {
            auto it   = newIds.find(id);
            int newId = it == newIds.end() ? id : it->second;
            if (newCurves.count(newId))
            {
                m_log(FATAL) << "Curved face id " << newId
                             << " used twice after reordering." << endl;
            }
            curve->m_curveID = newId;
            newCurves[newId] = std::move(curve);
        }
        graph->GetCurvedFaces() = std::move(newCurves);
    }
    else
    {
        rekey(static_cast<SpatialDomains::TetGeom *>(nullptr));
        rekey(static_cast<SpatialDomains::PyrGeom *>(nullptr));
        rekey(static_cast<SpatialDomains::PrismGeom *>(nullptr));
        rekey(static_cast<SpatialDomains::HexGeom *>(nullptr));
    }

    // m_mesh->PrintStats(m_log);
}

} // namespace Nektar::NekMesh
