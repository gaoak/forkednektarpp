////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessHAdaptationMaps.h
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
#include <NekMesh/Module/Module.h>
#include <NekMesh/Module/ProcessModules/Adaptation/2D/ProcessHAdaptation.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <map>
#include <vector>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 3, bg::cs::cartesian> point;
typedef bg::model::box<point> box;
typedef std::pair<box, unsigned> boxI;

namespace Nektar::NekMesh
{
void ProcessHAdaptation::DefineMaps()
{

    // MAPPINGS - keyed on the local edges that are split (according to
    // standard element). Node IDs index NodeListFull = [vertices 0-2, hanging
    // nodes in local edge order]. StdEdgeToStdLocHangingNode holds for every
    // internal edge {begin xi1, begin xi2, end xi1, end xi2}

    //  HangingNodes N=1
    // TriagEdges100 - Internal Edge always m_element->Edge(3)
    SplitTriangleNodeList[{0}]      = {2, 0, 3, 3, 1, 2};
    NewInternalEdgeNodeList[{0}]    = {2, 3};
    StdEdgeToStdLocHangingNode[{0}] = {-1.0, 1.0, 0.0, -1.0};

    // TriagEdges010 error
    // SplitTriangleNodeList[{1}] = {0,1,3, 0,2,3};
    SplitTriangleNodeList[{1}]      = {0, 1, 3, 3, 2, 0};
    NewInternalEdgeNodeList[{1}]    = {0, 3};
    StdEdgeToStdLocHangingNode[{1}] = {-1.0, -1.0, 0.0, 0.0};

    // TriagEdges001 error,but working ??
    SplitTriangleNodeList[{2}]      = {0, 1, 3, 1, 2, 3};
    NewInternalEdgeNodeList[{2}]    = {1, 3};
    StdEdgeToStdLocHangingNode[{2}] = {1.0, -1.0, -1.0, 0.0};

    //  HangingNodes N=2
    // Internal {edge0->m_n1 , edge0->m_n2 , edge1->m_n1, edge1->m_n2 }
    // TriagEdges110
    // SplitTriangleNodeList[{0, 1}] = {0,2,4,  3,1,4,  0,3,4};
    SplitTriangleNodeList[{0, 1}]      = {4, 2, 0, 3, 1, 4, 0, 3, 4};
    NewInternalEdgeNodeList[{0, 1}]    = {0, 4, 4, 3};
    StdEdgeToStdLocHangingNode[{0, 1}] = {-1.0, -1.0, 0.0, 0.0,
                                          0.0,  0.0,  0.0, -1.0};

    // TriagEdges101
    // SplitTriangleNodeList[{0, 2}] = {4,0,3,  1,2,4,  1,3,4};
    SplitTriangleNodeList[{0, 2}]      = {4, 0, 3, 1, 2, 4, 4, 3, 1};
    NewInternalEdgeNodeList[{0, 2}]    = {3, 4, 4, 1};
    StdEdgeToStdLocHangingNode[{0, 2}] = {0.0,  -1.0, -1.0, 0.0,
                                          -1.0, 0.0,  1.0,  -1.0};

    // TriagEdges011
    SplitTriangleNodeList[{1, 2}]      = {0, 1, 3, 4, 0, 3, 3, 2, 4};
    NewInternalEdgeNodeList[{1, 2}]    = {0, 3, 3, 4};
    StdEdgeToStdLocHangingNode[{1, 2}] = {-1.0, -1.0, 0.0,  0.0,
                                          0.0,  0.0,  -1.0, 0.0};

    //  HangingNodes N=3
    // TriagEdges111
    // SplitTriangleNodeList[{0, 1, 2}] = {5,0,3,  4,1,3,  5,2,4, 5,3,4};
    SplitTriangleNodeList[{0, 1, 2}]   = {5, 0, 3, 3, 1, 4, 4, 2, 5, 5, 3, 4};
    NewInternalEdgeNodeList[{0, 1, 2}] = {3, 5, 3, 4, 4, 5};
    StdEdgeToStdLocHangingNode[{0, 1, 2}] = {
        0.0, -1.0, -1.0, 0.0, // Internal Edge0 3-5
        0.0, -1.0, 0.0,  0.0, // Int Edge1 3-4
        0.0, 0.0,  -1.0, 0.0  // Int Edge2 4-5
    };

    // Mapping of the position of the external Split edge with hanging Node
    // int1 = CASE (which edge is split )
    CaseToLocalEdgeNewElement1[0] = {
        1, 0}; // TriagEdges100 (to the local new element, the split edge will 1
               // for Element 1 and GetEdge(0) for Element 2)
    // CaseToLocalEdgeNewElement1[1] = {1,1} ; //TriagEdges010
    CaseToLocalEdgeNewElement1[1] = {1, 0}; // TriagEdges010
    CaseToLocalEdgeNewElement1[2] = {0, 1}; // TriagEdges001

    // Every internal {Triag 0 Edge 0 , Triag0 Edge1 .. }
    CaseToLocalEdgeNewElement2[0] = {{1, 2, 3}, {0, 1, 4}, {0, 4, 3}}; //
    CaseToLocalEdgeNewElement2[1] = {{2, 0, 3}, {1, 2, 4}, {3, 0, 4}}; //
    CaseToLocalEdgeNewElement2[2] = {{0, 1, 3}, {2, 3, 4}, {1, 2, 4}}; //

    // 4 internal triangles
    CaseToLocalEdgeNewElement3[0] = {
        2, 0, 3}; // that composes [OldEdges, InternalEdges]
    CaseToLocalEdgeNewElement3[1] = {0, 1, 4}; // All the edges are
    CaseToLocalEdgeNewElement3[2] = {1, 2, 5};
    CaseToLocalEdgeNewElement3[3] = {3, 4, 5};

    // Mapping of the position of the Legacy NonSplit edge to the local new
    // element {elmentOld->GetEdge(id1), NewEelemnt0->SetEdge{id1, } ,
    // elmentOld->GetEdge(id2) ,    NewElement1->setedge{id2, }
    LegacyLocalEdgeToReuse1[0] = {2, 0, 1, 1}; // TriagEdges100
    LegacyLocalEdgeToReuse1[1] = {0, 0, 2, 1}; // TriagEdges010
    LegacyLocalEdgeToReuse1[2] = {0, 1, 1, 0}; // TriagEdges001

    // Quadrilateral Maps
    // Map that creates a mapping between the LocalID of the Quad Edge and the
    // ID of the opposite one (TO Propagate Conformity)
    // Map with the ID of the Nodes in NodeList Full [MacroVertices,
    // HangingNode0 , HanhingNode1]
    SplitQuadNodeList[0] = {{5, 3, 0, 4}, {4, 1, 2, 5}};
    SplitQuadNodeList[1] = {{5, 0, 1, 4}, {4, 2, 3, 5}};
    // Map with the Edges to the new elements
    CaseToQuadEdgesNewElement[0] = {{7, 3, 4, 8}, {5, 1, 6, 8}};
    CaseToQuadEdgesNewElement[1] = {{7, 0, 4, 8}, {5, 2, 6, 8}};
}
} // namespace Nektar::NekMesh
