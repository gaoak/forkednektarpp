////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessBL.h
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
//  Description: Refine boundary layer of elements.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_PROCESSBL
#define UTILITIES_NEKMESH_PROCESSBL

#include <NekMesh/Module/Module.h>

namespace Nektar::NekMesh
{

/**
 * @brief This processing module calculates the Jacobian of elements
 * using %SpatialDomains::GeomFactors and the %Element::GetGeom
 * method. For now it simply prints a list of elements which have
 * negative Jacobian.
 */
class ProcessBL : public NekMesh::ProcessModule
{
public:
    /// Creates an instance of this class
    static std::shared_ptr<Module> create(NekMesh::MeshSharedPtr m)
    {
        return MemoryManager<ProcessBL>::AllocateSharedPtr(m);
    }
    static NekMesh::ModuleKey className;

    ProcessBL(NekMesh::MeshSharedPtr m);
    ~ProcessBL() override;

    std::string GetModuleName() override
    {
        return "ProcessBL";
    }

    // Create the boundary layer in 2D
    void BoundaryLayer2D();
    // Create the boundary layer in 3D
    void BoundaryLayer3D();
    /// Write mesh to output file.
    void Process() override;

private:
    // Consolidate geometry IDs. reorCurves used for 1D & 2D, and so goes
    // unread for the volume instantiations; firstKey used for subsequent shape
    // types in 2D & 3D.
    template <typename T>
    int consolidateIDs(
        int firstKey                                          = 0,
        [[maybe_unused]] SpatialDomains::CurveMap *reorCurves = nullptr)
    {
        auto &graph = m_mesh->m_meshGraph;
        SpatialDomains::GeomMap<T> reorGeoms;
        int newKey = firstKey;
        for (auto [oldKey, rawPtr] : graph->GetGeomMap<T>())
        {
            rawPtr->SetGlobalID(newKey);
            reorGeoms[newKey] = graph->ExtractGeom<T>(oldKey);
            if constexpr (std::is_same_v<T, SpatialDomains::SegGeom>)
            {
                auto it = graph->GetCurvedEdges().find(oldKey);
                if (it != graph->GetCurvedEdges().end())
                {
                    it->second->m_curveID = newKey;
                    (*reorCurves)[newKey] = std::move(it->second);
                }
            }
            else if constexpr (std::is_same_v<T, SpatialDomains::TriGeom> ||
                               std::is_same_v<T, SpatialDomains::QuadGeom>)
            {
                auto it = graph->GetCurvedFaces().find(oldKey);
                if (it != graph->GetCurvedFaces().end())
                {
                    it->second->m_curveID = newKey;
                    (*reorCurves)[newKey] = std::move(it->second);
                }
            }
            newKey++;
        }
        graph->SetGeomMap<T>(std::move(reorGeoms));
        return newKey;
    }
};
} // namespace Nektar::NekMesh

#endif
