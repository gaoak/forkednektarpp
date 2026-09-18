////////////////////////////////////////////////////////////////////////////////
//
//  File: InputGmsh.h
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
//  Description: GMSH converter.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_INPUTGMSH
#define UTILITIES_NEKMESH_INPUTGMSH

#include <NekMesh/Module/Module.h>

namespace Nektar::NekMesh
{

/**
 * Converter for Gmsh files.
 */
class InputGmsh : public NekMesh::InputModule
{
public:
    InputGmsh(NekMesh::MeshSharedPtr m);
    ~InputGmsh() override;
    void Process() override;

    /// Creates an instance of this class
    static NekMesh::ModuleSharedPtr create(NekMesh::MeshSharedPtr m)
    {
        return MemoryManager<InputGmsh>::AllocateSharedPtr(m);
    }
    /// %ModuleKey for class.
    static NekMesh::ModuleKey className;
    /// Exported so that the unit tests can reach it; Windows will not
    /// resolve it from outside libNekMesh otherwise.
    NEKMESH_EXPORT static std::map<unsigned int, NekMesh::ElmtConfig> GenElmMap();

    /**
     * Element map; takes a msh id to an %ElmtConfig object.
     */
    static std::map<unsigned int, NekMesh::ElmtConfig> elmMap;
    /// Exported for the unit tests, as GenElmMap() above.
    NEKMESH_EXPORT static std::vector<int> CreateReordering(
        unsigned int InputGmshEntity, Logger &log);

    std::string GetModuleName() override
    {
        return "InputGmsh";
    }

private:
    int GetNnodes(unsigned int InputGmshEntity);
    static std::vector<int> TriReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> QuadReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> HexReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> PrismReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> PyrReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> TetReordering(NekMesh::ElmtConfig conf);
    static std::vector<int> LineReordering(NekMesh::ElmtConfig conf);

    // Gmsh file version
    NekDouble m_version;
    // Highest tag number
    int m_maxTagId;
    // This map takes each element ID and maps it to a permutation map
    // that is required to take Gmsh element node orderings and map them
    // to Nektar++ orderings.
    std::unordered_map<int, std::vector<int>> m_orderingMap;
    /// Container vertices and curvature nodes are loaded into before sorting
    SpatialDomains::GeomMap<SpatialDomains::PointGeom> m_loadedNodes;
    /// IDs to move from m_loadedNodes into the graph as mesh vertices
    std::set<int> m_vertIDs;
    /// IDs to move from m_loadedNodes into the graph as curvature nodes
    std::unordered_set<int> m_curveNodeIDs;
    /// IDs of triangles created naively, whose orientation a tetrahedron
    /// adopting them should overwrite
    std::unordered_set<int> m_naiveTriIDs;

    void ReadNextNode();
    void CalcNodeDim(int nVertices);
    void ReadNextNodeBlock(int nVertices = 0);
    void SaveNode(int id, NekDouble x = 0, NekDouble y = 0, NekDouble z = 0);
    void ReadNextElement(int tag = 0, int elm_type = 0);
    /// Keep the interior nodes of a high-order element as a curve on it.
    void StoreVolumeNodes(
        SpatialDomains::Geometry *element, const NekMesh::ElmtConfig &conf,
        const std::vector<SpatialDomains::PointGeom *> &gmshPoints);
};
} // namespace Nektar::NekMesh

#endif
