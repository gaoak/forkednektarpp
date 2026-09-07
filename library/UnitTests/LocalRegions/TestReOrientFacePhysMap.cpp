///////////////////////////////////////////////////////////////////////////////
//
// File: TestReOrientFacePhysMap.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/Expansion3D.h>
#include <LocalRegions/HexExp.h>
#include <SpatialDomains/MeshGraph.h>
#include <StdRegions/StdRegions.hpp>
#include <boost/test/unit_test.hpp>

namespace Nektar::Expansion3DTests
{

SpatialDomains::SegGeomUniquePtr CreateSegGeom(unsigned int id,
                                               SpatialDomains::PointGeom *v0,
                                               SpatialDomains::PointGeom *v1)
{
    std::array<SpatialDomains::PointGeom *, 2> vertices = {v0, v1};
    SpatialDomains::SegGeomUniquePtr result(
        new SpatialDomains::SegGeom(id, 3, vertices));
    return result;
}

SpatialDomains::HexGeomUniquePtr CreateHex(
    std::array<SpatialDomains::PointGeom *, 8> v,
    std::array<SpatialDomains::SegGeomUniquePtr, 12> &segVec,
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> &faceVec)
{
    std::array<std::array<int, 2>, 12> edgeVerts = {{{{0, 1}},
                                                     {{1, 2}},
                                                     {{2, 3}},
                                                     {{3, 0}},
                                                     {{0, 4}},
                                                     {{1, 5}},
                                                     {{2, 6}},
                                                     {{3, 7}},
                                                     {{4, 5}},
                                                     {{5, 6}},
                                                     {{6, 7}},
                                                     {{7, 4}}}};
    std::array<std::array<int, 4>, 6> faceEdges  = {{{{0, 1, 2, 3}},
                                                     {{0, 5, 8, 4}},
                                                     {{1, 6, 9, 5}},
                                                     {{2, 6, 10, 7}},
                                                     {{3, 7, 11, 4}},
                                                     {{8, 9, 10, 11}}}};

    // Create segments from vertices
    for (int i = 0; i < 12; ++i)
    {
        segVec[i] = CreateSegGeom(i, v[edgeVerts[i][0]], v[edgeVerts[i][1]]);
    }

    // Create faces from edges
    std::array<SpatialDomains::QuadGeom *, 6> faces;
    for (int i = 0; i < 6; ++i)
    {
        std::array<SpatialDomains::SegGeom *, 4> face;
        for (int j = 0; j < 4; ++j)
        {
            face[j] = segVec[faceEdges[i][j]].get();
        }
        faceVec[i] = SpatialDomains::QuadGeomUniquePtr(
            new SpatialDomains::QuadGeom(i, face));
        faces[i] = faceVec[i].get();
    }

    SpatialDomains::HexGeomUniquePtr hexGeom(
        new SpatialDomains::HexGeom(0, faces));
    return hexGeom;
}

BOOST_AUTO_TEST_CASE(TestReOrientQuadFacePhysMap)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, 1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v6(
        new SpatialDomains::PointGeom(3u, 6u, 1.0, 1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::PointGeom *, 8> v = {
        v0.get(), v1.get(), v2.get(), v3.get(),
        v4.get(), v5.get(), v6.get(), v7.get()};
    std::array<SpatialDomains::SegGeomUniquePtr, 12> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom = CreateHex(v, segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    // Dummy 3D expansion
    Nektar::LocalRegions::HexExpSharedPtr exp3d =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    // Initialiaze 3*3 face id map for quad
    int nq0 = 3;
    int nq1 = 3;
    Array<OneD, int> idmap(nq0 * nq1, -1);
    Array<OneD, int> idmap1(nq0 * nq1, -1);

    Array<OneD, NekDouble> idphys(nq0 * nq1);
    Array<OneD, NekDouble> idphys1(nq0 * nq1);
    for (unsigned i = 0; i < nq0 * nq1; ++i)
    {
        idphys[i] = (NekDouble)i;
    }

    // Test different orientations
    StdRegions::Orientation orient;

    orient = StdRegions::eDir1FwdDir1_Dir2FwdDir2;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 0);
    BOOST_CHECK_EQUAL(idmap[1], 1);
    BOOST_CHECK_EQUAL(idmap[2], 2);
    BOOST_CHECK_EQUAL(idmap[3], 3);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 5);
    BOOST_CHECK_EQUAL(idmap[6], 6);
    BOOST_CHECK_EQUAL(idmap[7], 7);
    BOOST_CHECK_EQUAL(idmap[8], 8);

    // check inverse map
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    //  physval version  - true, false do same
    Vmath::Zero(nq0 * nq1, idphys1, 1);
    exp3d->ReOrientTracePhysVals(orient, idphys, idphys1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idphys1[0], 0);
    BOOST_CHECK_EQUAL(idphys1[1], 1);
    BOOST_CHECK_EQUAL(idphys1[2], 2);
    BOOST_CHECK_EQUAL(idphys1[3], 3);
    BOOST_CHECK_EQUAL(idphys1[4], 4);
    BOOST_CHECK_EQUAL(idphys1[5], 5);
    BOOST_CHECK_EQUAL(idphys1[6], 6);
    BOOST_CHECK_EQUAL(idphys1[7], 7);
    BOOST_CHECK_EQUAL(idphys1[8], 8);

    orient = StdRegions::eDir1FwdDir1_Dir2BwdDir2;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 6);
    BOOST_CHECK_EQUAL(idmap[1], 7);
    BOOST_CHECK_EQUAL(idmap[2], 8);
    BOOST_CHECK_EQUAL(idmap[3], 3);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 5);
    BOOST_CHECK_EQUAL(idmap[6], 0);
    BOOST_CHECK_EQUAL(idmap[7], 1);
    BOOST_CHECK_EQUAL(idmap[8], 2);

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    //  physval version  - true, false do same
    Vmath::Zero(nq0 * nq1, idphys1, 1);
    exp3d->ReOrientTracePhysVals(orient, idphys, idphys1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idphys1[0], 6);
    BOOST_CHECK_EQUAL(idphys1[1], 7);
    BOOST_CHECK_EQUAL(idphys1[2], 8);
    BOOST_CHECK_EQUAL(idphys1[3], 3);
    BOOST_CHECK_EQUAL(idphys1[4], 4);
    BOOST_CHECK_EQUAL(idphys1[5], 5);
    BOOST_CHECK_EQUAL(idphys1[6], 0);
    BOOST_CHECK_EQUAL(idphys1[7], 1);
    BOOST_CHECK_EQUAL(idphys1[8], 2);

    orient = StdRegions::eDir1BwdDir1_Dir2FwdDir2;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 2);
    BOOST_CHECK_EQUAL(idmap[1], 1);
    BOOST_CHECK_EQUAL(idmap[2], 0);
    BOOST_CHECK_EQUAL(idmap[3], 5);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 3);
    BOOST_CHECK_EQUAL(idmap[6], 8);
    BOOST_CHECK_EQUAL(idmap[7], 7);
    BOOST_CHECK_EQUAL(idmap[8], 6);

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    //  physval version  - true, false do same
    Vmath::Zero(nq0 * nq1, idphys1, 1);
    exp3d->ReOrientTracePhysVals(orient, idphys, idphys1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idphys1[0], 2);
    BOOST_CHECK_EQUAL(idphys1[1], 1);
    BOOST_CHECK_EQUAL(idphys1[2], 0);
    BOOST_CHECK_EQUAL(idphys1[3], 5);
    BOOST_CHECK_EQUAL(idphys1[4], 4);
    BOOST_CHECK_EQUAL(idphys1[5], 3);
    BOOST_CHECK_EQUAL(idphys1[6], 8);
    BOOST_CHECK_EQUAL(idphys1[7], 7);
    BOOST_CHECK_EQUAL(idphys1[8], 6);

    orient = StdRegions::eDir1BwdDir1_Dir2BwdDir2;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 8);
    BOOST_CHECK_EQUAL(idmap[1], 7);
    BOOST_CHECK_EQUAL(idmap[2], 6);
    BOOST_CHECK_EQUAL(idmap[3], 5);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 3);
    BOOST_CHECK_EQUAL(idmap[6], 2);
    BOOST_CHECK_EQUAL(idmap[7], 1);
    BOOST_CHECK_EQUAL(idmap[8], 0);

    orient = StdRegions::eDir1BwdDir1_Dir2BwdDir2;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    //  physval version  - true, false do same
    Vmath::Zero(nq0 * nq1, idphys1, 1);
    exp3d->ReOrientTracePhysVals(orient, idphys, idphys1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idphys1[0], 8);
    BOOST_CHECK_EQUAL(idphys1[1], 7);
    BOOST_CHECK_EQUAL(idphys1[2], 6);
    BOOST_CHECK_EQUAL(idphys1[3], 5);
    BOOST_CHECK_EQUAL(idphys1[4], 4);
    BOOST_CHECK_EQUAL(idphys1[5], 3);
    BOOST_CHECK_EQUAL(idphys1[6], 2);
    BOOST_CHECK_EQUAL(idphys1[7], 1);
    BOOST_CHECK_EQUAL(idphys1[8], 0);

    // also use non-uniform nq for traspose cases
    int nq0a = 4;
    int nq1a = 3;
    Array<OneD, int> idmapa(nq0a * nq1a, -1);
    Array<OneD, int> idmap1a(nq0a * nq1a, -1);

    Array<OneD, NekDouble> idphysa(nq0a * nq1a);
    Array<OneD, NekDouble> idphys1a(nq0a * nq1a);
    for (unsigned i = 0; i < nq0a * nq1a; ++i)
    {
        idphysa[i] = (NekDouble)i;
    }

    orient = StdRegions::eDir1FwdDir2_Dir2FwdDir1;

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 0);
    BOOST_CHECK_EQUAL(idmap[1], 3);
    BOOST_CHECK_EQUAL(idmap[2], 6);
    BOOST_CHECK_EQUAL(idmap[3], 1);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 7);
    BOOST_CHECK_EQUAL(idmap[6], 2);
    BOOST_CHECK_EQUAL(idmap[7], 5);
    BOOST_CHECK_EQUAL(idmap[8], 8);

    exp3d->ReOrientTracePhysMap(orient, idmapa, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idmapa[0], 0);
    BOOST_CHECK_EQUAL(idmapa[1], 4);
    BOOST_CHECK_EQUAL(idmapa[2], 8);
    BOOST_CHECK_EQUAL(idmapa[3], 1);
    BOOST_CHECK_EQUAL(idmapa[4], 5);
    BOOST_CHECK_EQUAL(idmapa[5], 9);
    BOOST_CHECK_EQUAL(idmapa[6], 2);
    BOOST_CHECK_EQUAL(idmapa[7], 6);
    BOOST_CHECK_EQUAL(idmapa[8], 10);
    BOOST_CHECK_EQUAL(idmapa[9], 3);
    BOOST_CHECK_EQUAL(idmapa[10], 7);
    BOOST_CHECK_EQUAL(idmapa[11], 11);

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    exp3d->ReOrientTracePhysMap(orient, idmapa, nq0a, nq1a, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1a, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[0]], 0);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[1]], 1);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[2]], 2);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[3]], 3);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[4]], 4);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[5]], 5);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[6]], 6);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[7]], 7);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[8]], 8);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[9]], 9);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[10]], 10);
    BOOST_CHECK_EQUAL(idmapa[idmap1a[11]], 11);

    //  physval versions
    Vmath::Zero(nq0a * nq1a, idphys1a, 1);
    exp3d->ReOrientTracePhysVals(orient, idphysa, idphys1a, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idphys1a[0], 0);
    BOOST_CHECK_EQUAL(idphys1a[1], 4);
    BOOST_CHECK_EQUAL(idphys1a[2], 8);
    BOOST_CHECK_EQUAL(idphys1a[3], 1);
    BOOST_CHECK_EQUAL(idphys1a[4], 5);
    BOOST_CHECK_EQUAL(idphys1a[5], 9);
    BOOST_CHECK_EQUAL(idphys1a[6], 2);
    BOOST_CHECK_EQUAL(idphys1a[7], 6);
    BOOST_CHECK_EQUAL(idphys1a[8], 10);
    BOOST_CHECK_EQUAL(idphys1a[9], 3);
    BOOST_CHECK_EQUAL(idphys1a[10], 7);
    BOOST_CHECK_EQUAL(idphys1a[11], 11);

    Vmath::Zero(nq0 * nq1, idphys1, 1);
    exp3d->ReOrientTracePhysVals(orient, idphys, idphys1, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idphys1[0], 0);
    BOOST_CHECK_EQUAL(idphys1[1], 3);
    BOOST_CHECK_EQUAL(idphys1[2], 6);
    BOOST_CHECK_EQUAL(idphys1[3], 1);
    BOOST_CHECK_EQUAL(idphys1[4], 4);
    BOOST_CHECK_EQUAL(idphys1[5], 7);
    BOOST_CHECK_EQUAL(idphys1[6], 2);
    BOOST_CHECK_EQUAL(idphys1[7], 5);
    BOOST_CHECK_EQUAL(idphys1[8], 8);

    orient = StdRegions::eDir1FwdDir2_Dir2BwdDir1;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 2);
    BOOST_CHECK_EQUAL(idmap[1], 5);
    BOOST_CHECK_EQUAL(idmap[2], 8);
    BOOST_CHECK_EQUAL(idmap[3], 1);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 7);
    BOOST_CHECK_EQUAL(idmap[6], 0);
    BOOST_CHECK_EQUAL(idmap[7], 3);
    BOOST_CHECK_EQUAL(idmap[8], 6);

    exp3d->ReOrientTracePhysMap(orient, idmapa, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idmapa[0], 8);
    BOOST_CHECK_EQUAL(idmapa[1], 4);
    BOOST_CHECK_EQUAL(idmapa[2], 0);
    BOOST_CHECK_EQUAL(idmapa[3], 9);
    BOOST_CHECK_EQUAL(idmapa[4], 5);
    BOOST_CHECK_EQUAL(idmapa[5], 1);
    BOOST_CHECK_EQUAL(idmapa[6], 10);
    BOOST_CHECK_EQUAL(idmapa[7], 6);
    BOOST_CHECK_EQUAL(idmapa[8], 2);
    BOOST_CHECK_EQUAL(idmapa[9], 11);
    BOOST_CHECK_EQUAL(idmapa[10], 7);
    BOOST_CHECK_EQUAL(idmapa[11], 3);

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    Vmath::Zero(nq0a * nq1a, idphys1a, 1);
    exp3d->ReOrientTracePhysVals(orient, idphysa, idphys1a, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idphys1a[0], 8);
    BOOST_CHECK_EQUAL(idphys1a[1], 4);
    BOOST_CHECK_EQUAL(idphys1a[2], 0);
    BOOST_CHECK_EQUAL(idphys1a[3], 9);
    BOOST_CHECK_EQUAL(idphys1a[4], 5);
    BOOST_CHECK_EQUAL(idphys1a[5], 1);
    BOOST_CHECK_EQUAL(idphys1a[6], 10);
    BOOST_CHECK_EQUAL(idphys1a[7], 6);
    BOOST_CHECK_EQUAL(idphys1a[8], 2);
    BOOST_CHECK_EQUAL(idphys1a[9], 11);
    BOOST_CHECK_EQUAL(idphys1a[10], 7);
    BOOST_CHECK_EQUAL(idphys1a[11], 3);

    orient = StdRegions::eDir1BwdDir2_Dir2FwdDir1;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 6);
    BOOST_CHECK_EQUAL(idmap[1], 3);
    BOOST_CHECK_EQUAL(idmap[2], 0);
    BOOST_CHECK_EQUAL(idmap[3], 7);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 1);
    BOOST_CHECK_EQUAL(idmap[6], 8);
    BOOST_CHECK_EQUAL(idmap[7], 5);
    BOOST_CHECK_EQUAL(idmap[8], 2);

    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    Vmath::Zero(nq0 * nq1, idphys1a, 1);
    exp3d->ReOrientTracePhysVals(orient, idphysa, idphys1a, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idphys1a[0], 3);
    BOOST_CHECK_EQUAL(idphys1a[1], 7);
    BOOST_CHECK_EQUAL(idphys1a[2], 11);
    BOOST_CHECK_EQUAL(idphys1a[3], 2);
    BOOST_CHECK_EQUAL(idphys1a[4], 6);
    BOOST_CHECK_EQUAL(idphys1a[5], 10);
    BOOST_CHECK_EQUAL(idphys1a[6], 1);
    BOOST_CHECK_EQUAL(idphys1a[7], 5);
    BOOST_CHECK_EQUAL(idphys1a[8], 9);
    BOOST_CHECK_EQUAL(idphys1a[9], 0);
    BOOST_CHECK_EQUAL(idphys1a[10], 4);
    BOOST_CHECK_EQUAL(idphys1a[11], 8);

    orient = StdRegions::eDir1BwdDir2_Dir2BwdDir1;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    BOOST_CHECK_EQUAL(idmap[0], 8);
    BOOST_CHECK_EQUAL(idmap[1], 5);
    BOOST_CHECK_EQUAL(idmap[2], 2);
    BOOST_CHECK_EQUAL(idmap[3], 7);
    BOOST_CHECK_EQUAL(idmap[4], 4);
    BOOST_CHECK_EQUAL(idmap[5], 1);
    BOOST_CHECK_EQUAL(idmap[6], 6);
    BOOST_CHECK_EQUAL(idmap[7], 3);
    BOOST_CHECK_EQUAL(idmap[8], 0);

    orient = StdRegions::eDir1BwdDir2_Dir2BwdDir1;
    exp3d->ReOrientTracePhysMap(orient, idmap, nq0, nq1, false);
    exp3d->ReOrientTracePhysMap(orient, idmap1, nq0, nq1, true);
    BOOST_CHECK_EQUAL(idmap[idmap1[0]], 0);
    BOOST_CHECK_EQUAL(idmap[idmap1[1]], 1);
    BOOST_CHECK_EQUAL(idmap[idmap1[2]], 2);
    BOOST_CHECK_EQUAL(idmap[idmap1[3]], 3);
    BOOST_CHECK_EQUAL(idmap[idmap1[4]], 4);
    BOOST_CHECK_EQUAL(idmap[idmap1[5]], 5);
    BOOST_CHECK_EQUAL(idmap[idmap1[6]], 6);
    BOOST_CHECK_EQUAL(idmap[idmap1[7]], 7);
    BOOST_CHECK_EQUAL(idmap[idmap1[8]], 8);

    Vmath::Zero(nq0 * nq1, idphys1a, 1);
    exp3d->ReOrientTracePhysVals(orient, idphysa, idphys1a, nq0a, nq1a, true);
    BOOST_CHECK_EQUAL(idphys1a[0], 11);
    BOOST_CHECK_EQUAL(idphys1a[1], 7);
    BOOST_CHECK_EQUAL(idphys1a[2], 3);
    BOOST_CHECK_EQUAL(idphys1a[3], 10);
    BOOST_CHECK_EQUAL(idphys1a[4], 6);
    BOOST_CHECK_EQUAL(idphys1a[5], 2);
    BOOST_CHECK_EQUAL(idphys1a[6], 9);
    BOOST_CHECK_EQUAL(idphys1a[7], 5);
    BOOST_CHECK_EQUAL(idphys1a[8], 1);
    BOOST_CHECK_EQUAL(idphys1a[9], 8);
    BOOST_CHECK_EQUAL(idphys1a[10], 4);
    BOOST_CHECK_EQUAL(idphys1a[11], 0);
}

} // namespace Nektar::Expansion3DTests
