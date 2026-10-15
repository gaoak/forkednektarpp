////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessHAdaptation.h
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

#ifndef UTILITIES_NEKMESH_ProcesHAdaptation
#define UTILITIES_NEKMESH_ProcesHAdaptation

#include <NekMesh/Module/Module.h>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/ElUtil.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <iostream>
#include <map>
#include <vector>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 3, bg::cs::cartesian> point;
typedef bg::model::box<point> box;
typedef std::pair<box, unsigned> boxI;

namespace Nektar::NekMesh
{

class ProcessHAdaptation : public NekMesh::ProcessModule
{
public:
    /// Creates an instance of this class
    static std::shared_ptr<Module> create(NekMesh::MeshSharedPtr m)
    {
        return MemoryManager<ProcessHAdaptation>::AllocateSharedPtr(m);
    }
    static NekMesh::ModuleKey className;

    ProcessHAdaptation(NekMesh::MeshSharedPtr m);
    ~ProcessHAdaptation() override;

    /// Write mesh to output file.
    void Process() override;

    std::string GetModuleName() override
    {
        return "ProcessHAdaptation";
    }
    // Some main variables
    std::vector<int> m_ElementsIDAdapt;
    std::vector<SpatialDomains::SegGeom *> m_EdgesToSplit;
    int order, m_dim, i_max;

    std::unordered_map<SpatialDomains::SegGeom *, SpatialDomains::PointGeom *>
        m_hanginNode;
    std::unordered_map<SpatialDomains::SegGeom *,
                       std::vector<SpatialDomains::SegGeom *>>
        m_childEdges;
    std::unordered_set<SpatialDomains::Geometry *> m_adapted;

    // Mappins
    std::map<std::vector<int>, std::vector<int>> SplitTriangleNodeList;
    std::map<std::vector<int>, std::vector<int>> NewInternalEdgeNodeList;
    std::map<std::vector<int>, std::vector<NekDouble>>
        StdEdgeToStdLocHangingNode;
    std::map<int, std::vector<int>> CaseToLocalEdgeNewElement1;
    std::map<int, std::vector<std::vector<int>>> CaseToLocalEdgeNewElement2;
    std::map<int, std::vector<int>> CaseToLocalEdgeNewElement3;
    std::map<int, std::vector<int>> LegacyLocalEdgeToReuse1;
    std::map<int, std::vector<std::vector<int>>> SplitQuadNodeList;
    std::map<int, std::vector<std::vector<int>>> CaseToQuadEdgesNewElement;
    std::map<int, int> QuadEdgeToOpposite;
    std::unordered_map<int, SpatialDomains::Geometry *> ID_ElementPositionMap;

    // Functins
    typedef Nektar::LibUtilities::PtsFieldSharedPtr PtsFieldSharedPtr;
    PtsFieldSharedPtr m_interpField;
    LibUtilities::Interpolator m_interp;

    Array<OneD, Array<OneD, NekDouble>> inPts;
    LibUtilities::Interpolator GetField(
        Array<OneD, Array<OneD, NekDouble>> inPts);
    virtual void GetDesiredAreaField();
    virtual void InterpolateElementToAdapt();

    // virtual void SplitEdges(MeshSharedPtr &m_mesh, std::vector<EdgeSharedPtr>
    // &m_EdgesToSplit, int order);
    virtual void SplitEdges();
    virtual void SplitEdgesHangingNodes();
    virtual void SplitElementsHangingNodes();
    virtual void IdentifyEdgeToSplit();
    virtual void SwapEdge();
    virtual void CollapseEdge();
    virtual void SplitElements();
    void DefineMaps();

    SpatialDomains::SegGeom *CreateEdge(
        SpatialDomains::PointGeom *n1, SpatialDomains::PointGeom *n2,
        std::vector<SpatialDomains::PointGeom *> &edgeNodes,
        LibUtilities::PointsType curveType);

private:
    void RebuildLinks();
};
} // namespace Nektar::NekMesh

#endif
