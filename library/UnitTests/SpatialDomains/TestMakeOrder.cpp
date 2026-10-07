///////////////////////////////////////////////////////////////////////////////
//
// File: TestMakeOrder.cpp
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
// Description: Tests that MakeOrder places nodes where the mapping says.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <array>
#include <set>
#include <sstream>
#include <vector>

namespace Nektar::MakeOrderTests
{

/// Storage for the entities a hexahedron is built from, which must outlive it.
struct HexParts
{
    std::array<SpatialDomains::PointGeomUniquePtr, 8> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 12> edges;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faces;
    SpatialDomains::HexGeomUniquePtr hex;
    /// Nodes and curves handed back by MakeOrder, which the graph would own.
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
    std::vector<SpatialDomains::CurveUniquePtr> curves;
};

/// Map a face's local edge slot to the edge it should take, for a face
/// numbered @p rot steps around and optionally traversed the other way. Both
/// keep the edge list a closed loop, so the face is the same face, numbered as
/// a neighbouring element might have numbered it.
static int FaceSlot(int j, int n, int rot, bool reverse)
{
    const int cyc = reverse ? (n - j) % n : j;
    return (cyc + rot) % n;
}

static void BuildHex(HexParts &p,
                     const std::array<std::array<NekDouble, 3>, 8> &x,
                     const std::array<int, 6> &faceRot = {0, 0, 0, 0, 0, 0},
                     bool reverse                      = false)
{
    for (int i = 0; i < 8; ++i)
    {
        p.verts[i] = SpatialDomains::PointGeomUniquePtr(
            new SpatialDomains::PointGeom(3u, i, x[i][0], x[i][1], x[i][2]));
    }

    const std::array<std::array<int, 2>, 12> edgeVerts = {{{{0, 1}},
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
    for (int i = 0; i < 12; ++i)
    {
        std::array<SpatialDomains::PointGeom *, 2> ev = {
            p.verts[edgeVerts[i][0]].get(), p.verts[edgeVerts[i][1]].get()};
        p.edges[i] = SpatialDomains::SegGeomUniquePtr(
            new SpatialDomains::SegGeom(i, 3, ev));
    }

    const std::array<std::array<int, 4>, 6> faceEdges = {{{{0, 1, 2, 3}},
                                                          {{0, 5, 8, 4}},
                                                          {{1, 6, 9, 5}},
                                                          {{2, 6, 10, 7}},
                                                          {{3, 7, 11, 4}},
                                                          {{8, 9, 10, 11}}}};

    std::array<SpatialDomains::QuadGeom *, 6> faces;
    for (int i = 0; i < 6; ++i)
    {
        // Rotating the edge list moves the face's own first vertex, which is
        // how a face built by a neighbouring element appears: the same face,
        // numbered differently, so the hexahedron sees a non-trivial
        // orientation for it.
        std::array<SpatialDomains::SegGeom *, 4> f;
        for (int j = 0; j < 4; ++j)
        {
            f[j] = p.edges[faceEdges[i][FaceSlot(j, 4, faceRot[i], reverse)]]
                       .get();
        }
        p.faces[i] = SpatialDomains::QuadGeomUniquePtr(
            new SpatialDomains::QuadGeom(i, f));
        faces[i] = p.faces[i].get();
    }

    p.hex =
        SpatialDomains::HexGeomUniquePtr(new SpatialDomains::HexGeom(0, faces));
    p.hex->Setup();
}

/// Elevate in the order the contract requires: edges, then faces, then the
/// volume, keeping everything the calls hand back alive.
static void Elevate(HexParts &p, int order, LibUtilities::PointsType pType)
{
    auto keep =
        [&p](std::pair<SpatialDomains::CurveUniquePtr,
                       std::vector<SpatialDomains::PointGeomUniquePtr>> &&cd) {
            p.curves.push_back(std::move(cd.first));
            for (auto &n : cd.second)
            {
                p.nodes.push_back(std::move(n));
            }
        };

    // MakeOrder evaluates the existing mapping, so the geometry has to be
    // filled before it is elevated.
    for (int i = 0; i < 12; ++i)
    {
        p.edges[i]->FillGeom();
        keep(p.edges[i]->MakeOrder(order, pType));
    }
    for (int i = 0; i < 6; ++i)
    {
        p.faces[i]->FillGeom();
        keep(p.faces[i]->MakeOrder(order, pType));
    }
    p.hex->FillGeom();
    keep(p.hex->MakeOrder(order, pType));
}

// Every node in the volume curve must sit exactly where the element's own
// mapping puts that tensor grid point. This is what pins down the grid
// ordering and, for the boundary nodes taken from the face curves, the face
// orientation handling: get either wrong and a node lands at the wrong index,
// which shows up here as a mismatch against GetCoord.
BOOST_AUTO_TEST_CASE(TestHexMakeOrderNodesLieOnMapping)
{
    // Deliberately asymmetric, so that a mis-oriented face cannot coincide
    // with the correct answer.
    HexParts p;
    BuildHex(p, {{{{0.0, 0.0, 0.0}},
                  {{1.0, 0.0, 0.0}},
                  {{1.3, 1.0, 0.0}},
                  {{0.0, 1.1, 0.0}},
                  {{0.1, 0.2, 1.0}},
                  {{1.0, 0.0, 1.2}},
                  {{1.4, 1.2, 1.0}},
                  {{0.0, 1.0, 1.3}}}});

    const int order                   = 4;
    const LibUtilities::PointsType pT = LibUtilities::eGaussLobattoLegendre;
    const int nPoints                 = order + 1;

    Elevate(p, order, pT);

    Array<OneD, NekDouble> px;
    LibUtilities::PointsManager()[LibUtilities::PointsKey(nPoints, pT)]
        ->GetPoints(px);

    SpatialDomains::Curve *c = p.hex->GetCurve();
    BOOST_REQUIRE(c != nullptr);
    BOOST_REQUIRE_EQUAL(c->m_points.size(),
                        static_cast<size_t>(nPoints * nPoints * nPoints));

    NekDouble worst = 0.0;
    for (int d2 = 0; d2 < nPoints; ++d2)
    {
        for (int d1 = 0; d1 < nPoints; ++d1)
        {
            for (int d0 = 0; d0 < nPoints; ++d0)
            {
                Array<OneD, NekDouble> xi(3);
                xi[0] = px[d0];
                xi[1] = px[d1];
                xi[2] = px[d2];

                SpatialDomains::PointGeom *node =
                    c->m_points[d0 + nPoints * (d1 + nPoints * d2)];
                BOOST_REQUIRE(node != nullptr);

                for (int j = 0; j < 3; ++j)
                {
                    NekDouble expected = p.hex->GetCoord(j, xi);
                    worst = std::max(worst, std::abs((*node)(j)-expected));
                }
            }
        }
    }

    BOOST_CHECK_SMALL(worst, 1e-9);
}

// The same check, but with each face numbered as a neighbour would have
// numbered it. This is the case that actually exercises the orientation
// handling: with every face built in canonical order all six orientations come
// out as the identity and the reordering is never used.
BOOST_AUTO_TEST_CASE(TestHexMakeOrderRotatedFaces)
{
    HexParts p;
    BuildHex(p,
             {{{{0.0, 0.0, 0.0}},
               {{1.0, 0.0, 0.0}},
               {{1.3, 1.0, 0.0}},
               {{0.0, 1.1, 0.0}},
               {{0.1, 0.2, 1.0}},
               {{1.0, 0.0, 1.2}},
               {{1.4, 1.2, 1.0}},
               {{0.0, 1.0, 1.3}}}},
             {0, 1, 2, 3, 1, 2});

    const int order                   = 4;
    const LibUtilities::PointsType pT = LibUtilities::eGaussLobattoLegendre;
    const int nPoints                 = order + 1;

    // At least one face must be non-trivially oriented, or this test is the
    // previous one over again.
    int nonIdentity = 0;
    for (int f = 0; f < 6; ++f)
    {
        if (p.hex->GetForient(f) != StdRegions::eDir1FwdDir1_Dir2FwdDir2)
        {
            ++nonIdentity;
        }
    }
    BOOST_REQUIRE(nonIdentity > 0);

    Elevate(p, order, pT);

    Array<OneD, NekDouble> px;
    LibUtilities::PointsManager()[LibUtilities::PointsKey(nPoints, pT)]
        ->GetPoints(px);

    SpatialDomains::Curve *c = p.hex->GetCurve();
    BOOST_REQUIRE(c != nullptr);

    NekDouble worst = 0.0;
    for (int d2 = 0; d2 < nPoints; ++d2)
    {
        for (int d1 = 0; d1 < nPoints; ++d1)
        {
            for (int d0 = 0; d0 < nPoints; ++d0)
            {
                Array<OneD, NekDouble> xi(3);
                xi[0] = px[d0];
                xi[1] = px[d1];
                xi[2] = px[d2];

                SpatialDomains::PointGeom *node =
                    c->m_points[d0 + nPoints * (d1 + nPoints * d2)];
                BOOST_REQUIRE(node != nullptr);

                for (int j = 0; j < 3; ++j)
                {
                    worst = std::max(
                        worst, std::abs((*node)(j)-p.hex->GetCoord(j, xi)));
                }
            }
        }
    }

    BOOST_CHECK_SMALL(worst, 1e-9);
}

// The boundary of the volume curve must be the very same nodes the faces
// hold, not copies of them, or neighbouring elements would drift apart.
BOOST_AUTO_TEST_CASE(TestHexMakeOrderSharesFaceNodes)
{
    HexParts p;
    BuildHex(p, {{{{0.0, 0.0, 0.0}},
                  {{1.0, 0.0, 0.0}},
                  {{1.3, 1.0, 0.0}},
                  {{0.0, 1.1, 0.0}},
                  {{0.1, 0.2, 1.0}},
                  {{1.0, 0.0, 1.2}},
                  {{1.4, 1.2, 1.0}},
                  {{0.0, 1.0, 1.3}}}});

    const int order = 3;
    Elevate(p, order, LibUtilities::eGaussLobattoLegendre);

    SpatialDomains::Curve *c = p.hex->GetCurve();
    for (int f = 0; f < 6; ++f)
    {
        SpatialDomains::Curve *fc = p.faces[f]->GetCurve();
        BOOST_REQUIRE(fc != nullptr);
        for (auto *fn : fc->m_points)
        {
            BOOST_CHECK(std::find(c->m_points.begin(), c->m_points.end(), fn) !=
                        c->m_points.end());
        }
    }
}

/// Storage for the entities a tetrahedron is built from.
struct TetParts
{
    std::array<SpatialDomains::PointGeomUniquePtr, 4> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 6> edges;
    std::array<SpatialDomains::TriGeomUniquePtr, 4> faces;
    SpatialDomains::TetGeomUniquePtr tet;
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
    std::vector<SpatialDomains::CurveUniquePtr> curves;
};

static void BuildTet(TetParts &p,
                     const std::array<std::array<NekDouble, 3>, 4> &x,
                     const std::array<int, 4> &faceRot = {0, 0, 0, 0},
                     bool reverse                      = false)
{
    for (int i = 0; i < 4; ++i)
    {
        p.verts[i] = SpatialDomains::PointGeomUniquePtr(
            new SpatialDomains::PointGeom(3u, i, x[i][0], x[i][1], x[i][2]));
    }

    const std::array<std::array<int, 2>, 6> edgeVerts = {
        {{{0, 1}}, {{1, 2}}, {{0, 2}}, {{0, 3}}, {{1, 3}}, {{2, 3}}}};
    for (int i = 0; i < 6; ++i)
    {
        std::array<SpatialDomains::PointGeom *, 2> ev = {
            p.verts[edgeVerts[i][0]].get(), p.verts[edgeVerts[i][1]].get()};
        p.edges[i] = SpatialDomains::SegGeomUniquePtr(
            new SpatialDomains::SegGeom(i, 3, ev));
    }

    const std::array<std::array<int, 3>, 4> faceEdges = {
        {{{0, 1, 2}}, {{0, 4, 3}}, {{1, 5, 4}}, {{2, 5, 3}}}};

    std::array<SpatialDomains::TriGeom *, 4> faces;
    for (int i = 0; i < 4; ++i)
    {
        // As for the hexahedron, rotating a face's edge list numbers it the
        // way a neighbouring element would have.
        std::array<SpatialDomains::SegGeom *, 3> f;
        for (int j = 0; j < 3; ++j)
        {
            f[j] = p.edges[faceEdges[i][FaceSlot(j, 3, faceRot[i], reverse)]]
                       .get();
        }
        p.faces[i] =
            SpatialDomains::TriGeomUniquePtr(new SpatialDomains::TriGeom(i, f));
        faces[i] = p.faces[i].get();
    }

    p.tet =
        SpatialDomains::TetGeomUniquePtr(new SpatialDomains::TetGeom(0, faces));
    p.tet->Setup();
}

static void ElevateTet(TetParts &p, int order)
{
    auto keep =
        [&p](std::pair<SpatialDomains::CurveUniquePtr,
                       std::vector<SpatialDomains::PointGeomUniquePtr>> &&cd) {
            p.curves.push_back(std::move(cd.first));
            for (auto &n : cd.second)
            {
                p.nodes.push_back(std::move(n));
            }
        };

    // Fill everything first, then elevate everything: MakeOrder reads the
    // existing mapping and deliberately does not fill on its own behalf.
    for (int i = 0; i < 6; ++i)
    {
        p.edges[i]->FillGeom();
    }
    for (int i = 0; i < 4; ++i)
    {
        p.faces[i]->FillGeom();
    }
    p.tet->FillGeom();

    for (int i = 0; i < 6; ++i)
    {
        keep(p.edges[i]->MakeOrder(order, LibUtilities::ePolyEvenlySpaced));
    }
    for (int i = 0; i < 4; ++i)
    {
        keep(p.faces[i]->MakeOrder(order, LibUtilities::eNodalTriEvenlySpaced));
    }
    keep(p.tet->MakeOrder(order, LibUtilities::eNodalTetEvenlySpaced));
}

/// Every node of a nodal tetrahedron must sit where the mapping puts the
/// corresponding point of the distribution. Because the distribution fixes the
/// order of vertices, edge interiors, face interiors and volume interior, this
/// single check pins down the whole layout: the vertex placement, the edge
/// orientation, the face alignment and the interior all at once.
static NekDouble WorstTetNode(TetParts &p, int order)
{
    const int nPoints = order + 1;
    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsManager()
        [LibUtilities::PointsKey(nPoints, LibUtilities::eNodalTetEvenlySpaced)]
            ->GetPoints(px, py, pz);

    SpatialDomains::Curve *c = p.tet->GetCurve();
    BOOST_REQUIRE(c != nullptr);
    const int nTetPts = nPoints * (nPoints + 1) * (nPoints + 2) / 6;
    BOOST_REQUIRE_EQUAL(c->m_points.size(), static_cast<size_t>(nTetPts));

    NekDouble worst = 0.0;
    for (int i = 0; i < nTetPts; ++i)
    {
        Array<OneD, NekDouble> xi(3);
        xi[0] = px[i];
        xi[1] = py[i];
        xi[2] = pz[i];

        BOOST_REQUIRE(c->m_points[i] != nullptr);
        for (int j = 0; j < 3; ++j)
        {
            worst = std::max(
                worst, std::abs((*c->m_points[i])(j)-p.tet->GetCoord(j, xi)));
        }
    }
    return worst;
}

static const std::array<std::array<NekDouble, 3>, 4> skewTet = {
    {{{0.0, 0.0, 0.0}},
     {{1.0, 0.0, 0.0}},
     {{0.2, 1.1, 0.0}},
     {{0.3, 0.2, 1.2}}}};

BOOST_AUTO_TEST_CASE(TestTetMakeOrderNodesLieOnMapping)
{
    TetParts p;
    BuildTet(p, skewTet);
    const int order = 5; // high enough to have interior nodes
    ElevateTet(p, order);
    BOOST_CHECK_SMALL(WorstTetNode(p, order), 1e-9);
}

/// Storage for the entities a prism is built from.
struct PrismParts
{
    std::array<SpatialDomains::PointGeomUniquePtr, 6> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 9> edges;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> tris;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quads;
    SpatialDomains::PrismGeomUniquePtr prism;
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
    std::vector<SpatialDomains::CurveUniquePtr> curves;
};

static void BuildPrism(PrismParts &p,
                       const std::array<std::array<NekDouble, 3>, 6> &x,
                       const std::array<int, 5> &faceRot = {0, 0, 0, 0, 0},
                       bool reverse                      = false)
{
    for (int i = 0; i < 6; ++i)
    {
        p.verts[i] = SpatialDomains::PointGeomUniquePtr(
            new SpatialDomains::PointGeom(3u, i, x[i][0], x[i][1], x[i][2]));
    }

    const std::array<std::array<int, 2>, 9> edgeVerts = {{{{0, 1}},
                                                          {{1, 2}},
                                                          {{3, 2}},
                                                          {{0, 3}},
                                                          {{0, 4}},
                                                          {{1, 4}},
                                                          {{2, 5}},
                                                          {{3, 5}},
                                                          {{4, 5}}}};
    for (int i = 0; i < 9; ++i)
    {
        std::array<SpatialDomains::PointGeom *, 2> ev = {
            p.verts[edgeVerts[i][0]].get(), p.verts[edgeVerts[i][1]].get()};
        p.edges[i] = SpatialDomains::SegGeomUniquePtr(
            new SpatialDomains::SegGeom(i, 3, ev));
    }

    const std::array<std::array<int, 4>, 5> faceEdges = {{{{0, 1, 2, 3}},
                                                          {{0, 5, 4, -1}},
                                                          {{1, 6, 8, 5}},
                                                          {{2, 6, 7, -1}},
                                                          {{3, 7, 8, 4}}}};

    std::array<SpatialDomains::Geometry2D *, 5> faces;
    for (int i = 0; i < 5; ++i)
    {
        if (i % 2 == 0)
        {
            std::array<SpatialDomains::SegGeom *, 4> f;
            for (int j = 0; j < 4; ++j)
            {
                f[j] =
                    p.edges[faceEdges[i][FaceSlot(j, 4, faceRot[i], reverse)]]
                        .get();
            }
            p.quads[i / 2] = SpatialDomains::QuadGeomUniquePtr(
                new SpatialDomains::QuadGeom(i, f));
            faces[i] = p.quads[i / 2].get();
        }
        else
        {
            std::array<SpatialDomains::SegGeom *, 3> f;
            for (int j = 0; j < 3; ++j)
            {
                f[j] =
                    p.edges[faceEdges[i][FaceSlot(j, 3, faceRot[i], reverse)]]
                        .get();
            }
            p.tris[i / 2] = SpatialDomains::TriGeomUniquePtr(
                new SpatialDomains::TriGeom(i, f));
            faces[i] = p.tris[i / 2].get();
        }
    }

    p.prism = SpatialDomains::PrismGeomUniquePtr(
        new SpatialDomains::PrismGeom(0, faces));
    p.prism->Setup();
}

/**
 * @brief The point distributions Mesh::MakeOrder pairs with each other, by
 * the one it uses for the prism itself.
 */
struct PrismPointTypes
{
    LibUtilities::PointsType seg, tri, quad, prism;
};

static const PrismPointTypes prismEvenlySpaced = {
    LibUtilities::ePolyEvenlySpaced, LibUtilities::eNodalTriEvenlySpaced,
    LibUtilities::ePolyEvenlySpaced, LibUtilities::eNodalPrismEvenlySpaced};

static const PrismPointTypes prismElectrostatic = {
    LibUtilities::eGaussLobattoLegendre, LibUtilities::eNodalTriElec,
    LibUtilities::eGaussLobattoLegendre, LibUtilities::eNodalPrismElec};

static NekDouble ElevateAndMeasurePrism(PrismParts &p, int order,
                                        const PrismPointTypes &pt);

BOOST_AUTO_TEST_CASE(TestPrismMakeOrderNodesLieOnMapping)
{
    // Both nodal prism distributions. They lay their nodes out the same way
    // now, but they did not always: eNodalPrismEvenlySpaced transposed
    // vertices 2 and 3 with respect to the standard element and
    // eNodalPrismElec ran two of the edges the other way, so MakeOrder had to
    // follow whichever it was handed. Only the evenly spaced one was covered
    // here, and the electrostatic one, which is what a mesh raised in order
    // through NekMesh actually uses, came out with its quadrilateral faces
    // folded over on themselves.
    for (const PrismPointTypes &pt : {prismEvenlySpaced, prismElectrostatic})
    {
        PrismParts p;
        BuildPrism(p, {{{{0.0, 0.0, 0.0}},
                        {{1.0, 0.0, 0.0}},
                        {{1.2, 1.0, 0.0}},
                        {{0.0, 1.1, 0.0}},
                        {{0.1, 0.2, 1.0}},
                        {{0.0, 1.0, 1.2}}}});

        BOOST_CHECK_SMALL(ElevateAndMeasurePrism(p, 5, pt), 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Pyramid
// ---------------------------------------------------------------------------

/// Storage for the entities a pyramid is built from, which must outlive it.
struct PyrParts
{
    std::array<SpatialDomains::PointGeomUniquePtr, 5> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 8> edges;
    SpatialDomains::QuadGeomUniquePtr base;
    std::array<SpatialDomains::TriGeomUniquePtr, 4> tris;
    SpatialDomains::PyrGeomUniquePtr pyr;
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
    std::vector<SpatialDomains::CurveUniquePtr> curves;
};

static void BuildPyr(PyrParts &p,
                     const std::array<std::array<NekDouble, 3>, 5> &x)
{
    for (int i = 0; i < 5; ++i)
    {
        p.verts[i] = SpatialDomains::PointGeomUniquePtr(
            new SpatialDomains::PointGeom(3, i, x[i][0], x[i][1], x[i][2]));
    }

    // PyrGeom::SetUpEdgeOrientation()'s table.
    const std::array<std::array<int, 2>, 8> edgeVerts = {{{{0, 1}},
                                                          {{1, 2}},
                                                          {{3, 2}},
                                                          {{0, 3}},
                                                          {{0, 4}},
                                                          {{1, 4}},
                                                          {{2, 4}},
                                                          {{3, 4}}}};
    for (int i = 0; i < 8; ++i)
    {
        std::array<SpatialDomains::PointGeom *, 2> ev = {
            p.verts[edgeVerts[i][0]].get(), p.verts[edgeVerts[i][1]].get()};
        p.edges[i] = SpatialDomains::SegGeomUniquePtr(
            new SpatialDomains::SegGeom(i, 3, ev));
    }

    // The edges bounding each face, in the cyclic order the face walks them.
    const std::array<std::array<int, 4>, 5> faceEdges = {{{{0, 1, 2, 3}},
                                                          {{0, 5, 4, -1}},
                                                          {{1, 6, 5, -1}},
                                                          {{2, 6, 7, -1}},
                                                          {{3, 7, 4, -1}}}};

    std::array<SpatialDomains::Geometry2D *, 5> faces;
    {
        std::array<SpatialDomains::SegGeom *, 4> f;
        for (int j = 0; j < 4; ++j)
        {
            f[j] = p.edges[faceEdges[0][j]].get();
        }
        p.base = SpatialDomains::QuadGeomUniquePtr(
            new SpatialDomains::QuadGeom(0, f));
        faces[0] = p.base.get();
    }
    for (int i = 1; i < 5; ++i)
    {
        std::array<SpatialDomains::SegGeom *, 3> f;
        for (int j = 0; j < 3; ++j)
        {
            f[j] = p.edges[faceEdges[i][j]].get();
        }
        p.tris[i - 1] =
            SpatialDomains::TriGeomUniquePtr(new SpatialDomains::TriGeom(i, f));
        faces[i] = p.tris[i - 1].get();
    }

    p.pyr =
        SpatialDomains::PyrGeomUniquePtr(new SpatialDomains::PyrGeom(0, faces));
    p.pyr->Setup();
}

static NekDouble ElevateAndMeasurePyr(PyrParts &p, int order)
{
    const int nPoints = order + 1;
    auto keep =
        [&p](std::pair<SpatialDomains::CurveUniquePtr,
                       std::vector<SpatialDomains::PointGeomUniquePtr>> &&cd) {
            p.curves.push_back(std::move(cd.first));
            for (auto &n : cd.second)
            {
                p.nodes.push_back(std::move(n));
            }
        };

    for (int i = 0; i < 8; ++i)
    {
        p.edges[i]->FillGeom();
    }
    p.base->FillGeom();
    for (int i = 0; i < 4; ++i)
    {
        p.tris[i]->FillGeom();
    }
    p.pyr->FillGeom();

    for (int i = 0; i < 8; ++i)
    {
        keep(p.edges[i]->MakeOrder(order, LibUtilities::ePolyEvenlySpaced));
    }
    keep(p.base->MakeOrder(order, LibUtilities::ePolyEvenlySpaced));
    for (int i = 0; i < 4; ++i)
    {
        keep(p.tris[i]->MakeOrder(order, LibUtilities::eNodalTriEvenlySpaced));
    }
    keep(p.pyr->MakeOrder(order, LibUtilities::eNodalPyrEvenlySpaced));

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsManager()
        [LibUtilities::PointsKey(nPoints, LibUtilities::eNodalPyrEvenlySpaced)]
            ->GetPoints(px, py, pz);

    SpatialDomains::Curve *c = p.pyr->GetCurve();
    const int nPts           = nPoints * (nPoints + 1) * (2 * nPoints + 1) / 6;
    NekDouble worst          = 0.0;
    for (int i = 0; i < nPts; ++i)
    {
        Array<OneD, NekDouble> xi(3);
        xi[0] = px[i];
        xi[1] = py[i];
        xi[2] = pz[i];
        for (int j = 0; j < 3; ++j)
        {
            worst = std::max(
                worst, std::abs((*c->m_points[i])(j)-p.pyr->GetCoord(j, xi)));
        }
    }
    return worst;
}

BOOST_AUTO_TEST_CASE(TestPyrMakeOrderNodesLieOnMapping)
{
    // The pyramid has only the one nodal distribution, and MakeOrder reads
    // the element's vertices, edges and faces straight into its blocks. If
    // the distribution and PyrGeom ever disagree about which corner is vertex
    // 2, or which way an edge runs, the nodes land somewhere the element's
    // own mapping does not put them.
    PyrParts p;
    BuildPyr(p, {{{{0.0, 0.0, 0.0}},
                  {{1.0, 0.0, 0.0}},
                  {{1.2, 1.0, 0.0}},
                  {{0.0, 1.1, 0.0}},
                  {{0.4, 0.5, 1.0}}}});

    BOOST_CHECK_SMALL(ElevateAndMeasurePyr(p, 5), 1e-9);
}

// ---------------------------------------------------------------------------
// Face orientation sweep
//
// A standalone element has canonically numbered faces, so every face
// orientation comes out as the identity and none of the reindexing in
// MakeOrder is exercised -- the earlier tests pass even with it removed.
// Renumbering each face, by walking its edge list from a different start and
// in either direction, is the same face seen as a neighbouring element would
// see it, and makes the element compute real orientations for it.
//
// This sweeps those numberings for all three shapes, checks the round trip
// holds for every arrangement the constructors accept, and reports which
// orientations were actually reached. Arrangements the constructors reject are
// skipped: TetGeom::SetUpLocalEdges, for one, requires each face's local edge
// 0 to be shared with face 0, which pins most of them.
// ---------------------------------------------------------------------------

static NekDouble ElevateAndMeasureHex(HexParts &p, int order)
{
    const LibUtilities::PointsType pT = LibUtilities::eGaussLobattoLegendre;
    const int nPoints                 = order + 1;
    Elevate(p, order, pT);

    Array<OneD, NekDouble> px;
    LibUtilities::PointsManager()[LibUtilities::PointsKey(nPoints, pT)]
        ->GetPoints(px);

    SpatialDomains::Curve *c = p.hex->GetCurve();
    NekDouble worst          = 0.0;
    for (int d2 = 0; d2 < nPoints; ++d2)
    {
        for (int d1 = 0; d1 < nPoints; ++d1)
        {
            for (int d0 = 0; d0 < nPoints; ++d0)
            {
                Array<OneD, NekDouble> xi(3);
                xi[0] = px[d0];
                xi[1] = px[d1];
                xi[2] = px[d2];
                SpatialDomains::PointGeom *node =
                    c->m_points[d0 + nPoints * (d1 + nPoints * d2)];
                for (int j = 0; j < 3; ++j)
                {
                    worst = std::max(
                        worst, std::abs((*node)(j)-p.hex->GetCoord(j, xi)));
                }
            }
        }
    }
    return worst;
}

static NekDouble ElevateAndMeasurePrism(PrismParts &p, int order,
                                        const PrismPointTypes &pt)
{
    const int nPoints = order + 1;
    auto keep =
        [&p](std::pair<SpatialDomains::CurveUniquePtr,
                       std::vector<SpatialDomains::PointGeomUniquePtr>> &&cd) {
            p.curves.push_back(std::move(cd.first));
            for (auto &n : cd.second)
            {
                p.nodes.push_back(std::move(n));
            }
        };

    for (int i = 0; i < 9; ++i)
    {
        p.edges[i]->FillGeom();
    }
    for (int i = 0; i < 2; ++i)
    {
        p.tris[i]->FillGeom();
    }
    for (int i = 0; i < 3; ++i)
    {
        p.quads[i]->FillGeom();
    }
    p.prism->FillGeom();

    for (int i = 0; i < 9; ++i)
    {
        keep(p.edges[i]->MakeOrder(order, pt.seg));
    }
    for (int i = 0; i < 2; ++i)
    {
        keep(p.tris[i]->MakeOrder(order, pt.tri));
    }
    for (int i = 0; i < 3; ++i)
    {
        keep(p.quads[i]->MakeOrder(order, pt.quad));
    }
    keep(p.prism->MakeOrder(order, pt.prism));

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsManager()[LibUtilities::PointsKey(nPoints, pt.prism)]
        ->GetPoints(px, py, pz);

    SpatialDomains::Curve *c = p.prism->GetCurve();
    const int nPts           = nPoints * (nPoints + 1) / 2 * nPoints;
    NekDouble worst          = 0.0;
    for (int i = 0; i < nPts; ++i)
    {
        Array<OneD, NekDouble> xi(3);
        xi[0] = px[i];
        xi[1] = py[i];
        xi[2] = pz[i];
        for (int j = 0; j < 3; ++j)
        {
            worst = std::max(
                worst, std::abs((*c->m_points[i])(j)-p.prism->GetCoord(j, xi)));
        }
    }
    return worst;
}

static const std::array<std::array<NekDouble, 3>, 8> skewHex = {
    {{{0.0, 0.0, 0.0}},
     {{1.0, 0.0, 0.0}},
     {{1.3, 1.0, 0.0}},
     {{0.0, 1.1, 0.0}},
     {{0.1, 0.2, 1.0}},
     {{1.0, 0.0, 1.2}},
     {{1.4, 1.2, 1.0}},
     {{0.0, 1.0, 1.3}}}};

static const std::array<std::array<NekDouble, 3>, 6> skewPrism = {
    {{{0.0, 0.0, 0.0}},
     {{1.0, 0.0, 0.0}},
     {{1.2, 1.0, 0.0}},
     {{0.0, 1.1, 0.0}},
     {{0.1, 0.2, 1.0}},
     {{0.0, 1.0, 1.2}}}};

BOOST_AUTO_TEST_CASE(TestMakeOrderFaceOrientationSweep)
{
    const int order = 5;

    std::set<int> hexSeen, tetSeen, prismSeen;
    int hexBuilt = 0, tetBuilt = 0, prismBuilt = 0;

    for (int reverse = 0; reverse < 2; ++reverse)
    {
        for (int rot = 0; rot < 4; ++rot)
        {
            // Hexahedron: quadrilateral faces, four rotations each way.
            try
            {
                HexParts p;
                std::array<int, 6> r;
                r.fill(rot);
                BuildHex(p, skewHex, r, reverse != 0);
                for (int f = 0; f < 6; ++f)
                {
                    hexSeen.insert(p.hex->GetForient(f));
                }
                BOOST_CHECK_SMALL(ElevateAndMeasureHex(p, order), 1e-9);
                ++hexBuilt;
            }
            catch (const std::exception &)
            {
                // arrangement rejected by the constructor
            }

            // Prism: mixed faces, so rotate quadrilaterals by rot and
            // triangles by rot modulo three.
            try
            {
                PrismParts p;
                std::array<int, 5> r = {rot, rot % 3, rot, rot % 3, rot};
                BuildPrism(p, skewPrism, r, reverse != 0);
                for (int f = 0; f < 5; ++f)
                {
                    prismSeen.insert(p.prism->GetForient(f));
                }
                BOOST_CHECK_SMALL(
                    ElevateAndMeasurePrism(p, order, prismEvenlySpaced), 1e-9);
                ++prismBuilt;
            }
            catch (const std::exception &)
            {
            }

            if (rot < 3)
            {
                // Tetrahedron: triangular faces, three rotations each way.
                try
                {
                    TetParts p;
                    std::array<int, 4> r;
                    r.fill(rot);
                    BuildTet(p, skewTet, r, reverse != 0);
                    for (int f = 0; f < 4; ++f)
                    {
                        tetSeen.insert(p.tet->GetForient(f));
                    }
                    ElevateTet(p, order);
                    BOOST_CHECK_SMALL(WorstTetNode(p, order), 1e-9);
                    ++tetBuilt;
                }
                catch (const std::exception &)
                {
                }
            }
        }
    }

    auto report = [](const char *name, int built, const std::set<int> &seen) {
        std::ostringstream ss;
        ss << "  " << name << ": " << built << " arrangements accepted, "
           << seen.size() << " distinct face orientations reached (";
        for (int o : seen)
        {
            ss << o << " ";
        }
        ss << ")";
        BOOST_TEST_MESSAGE(ss.str());
    };
    report("hexahedron", hexBuilt, hexSeen);
    report("tetrahedron", tetBuilt, tetSeen);
    report("prism", prismBuilt, prismSeen);

    // Each shape must reach at least one non-identity orientation, or the
    // reindexing it performs is still untested.
    for (auto *seen : {&hexSeen, &tetSeen, &prismSeen})
    {
        BOOST_CHECK(seen->size() > 1 ||
                    (seen->size() == 1 &&
                     *seen->begin() != StdRegions::eDir1FwdDir1_Dir2FwdDir2));
    }
}

} // namespace Nektar::MakeOrderTests
