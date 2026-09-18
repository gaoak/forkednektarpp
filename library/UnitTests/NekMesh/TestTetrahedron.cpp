///////////////////////////////////////////////////////////////////////////////
//
//  File: TestTetrahedron.cpp
//
//  For more information, please see: http://www.nektar.info
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
//  Description: Unit tests for the NekMesh tetrahedron element.
//
///////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/Mesh.h>
#include <NekMesh/NekMeshDeclspec.h>

#include <boost/test/unit_test.hpp>

namespace Nektar::NekMeshTetrahedronUnitTest
{
using namespace Nektar::NekMesh;

BOOST_AUTO_TEST_CASE(TestConstruction)
{
    // The element factory builds geometry into a MeshGraph now, deduplicating
    // the faces and edges it creates against the maps it is handed, so a mesh
    // is needed to hold all three.
    MeshSharedPtr mesh = std::make_shared<Mesh>();
    auto &graph        = mesh->m_meshGraph;

    graph->SetMeshDimension(3);
    graph->SetSpaceDimension(3);

    // Setup element node list -- order matters!
    std::vector<SpatialDomains::PointGeom *> nodeList(4);
    nodeList[0] = graph->CreatePointGeom(3, 0, -1., 0., 0.);
    nodeList[1] = graph->CreatePointGeom(3, 1, 1., 0., 0.);
    nodeList[2] = graph->CreatePointGeom(3, 2, 0., 1., 0.);
    nodeList[3] = graph->CreatePointGeom(3, 3, 0., 0.5, 1.);

    // CreateElementLite is a straight-sided element with no reorientation,
    // which is the configuration this test has always used.
    SpatialDomains::Geometry *testTet =
        CreateElementLite(LibUtilities::eTetrahedron, nodeList, graph,
                          mesh->m_edgeSet, mesh->m_faceSet);

    BOOST_REQUIRE(testTet != nullptr);
    BOOST_CHECK_EQUAL(testTet->GetShapeType(), LibUtilities::eTetrahedron);
    BOOST_CHECK_EQUAL(testTet->GetNumVerts(), 4);
    BOOST_CHECK_EQUAL(testTet->GetNumEdges(), 6);
    BOOST_CHECK_EQUAL(testTet->GetNumFaces(), 4);

    NekDouble x, y, z;
    testTet->GetVertex(3)->GetCoords(x, y, z);

    BOOST_CHECK_EQUAL(x, 0.);
    BOOST_CHECK_EQUAL(y, 0.5);
    BOOST_CHECK_EQUAL(z, 1.);

    // The faces and edges the factory generated are owned by the graph, and
    // are shared rather than duplicated: a tetrahedron has four triangular
    // faces meeting along six edges.
    BOOST_CHECK_EQUAL(graph->GetNumGeoms<SpatialDomains::TriGeom>(), 4);
    BOOST_CHECK_EQUAL(graph->GetNumGeoms<SpatialDomains::SegGeom>(), 6);
    BOOST_CHECK_EQUAL(mesh->m_faceSet.size(), 4);
    BOOST_CHECK_EQUAL(mesh->m_edgeSet.size(), 6);
}

} // namespace Nektar::NekMeshTetrahedronUnitTest
