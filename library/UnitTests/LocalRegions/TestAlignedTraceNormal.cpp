///////////////////////////////////////////////////////////////////////////////
//
// File: TestAlignedTraceNormal.cpp
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// Description: Tests for trace normals sampled on boundary expansions.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <LocalRegions/HexExp.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TetExp.h>
#include <LocalRegions/TriExp.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <cmath>
#include <set>
#include <vector>

namespace Nektar::AlignedTraceNormalTests
{

struct CurvedTet
{
    std::array<SpatialDomains::PointGeomUniquePtr, 4> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 6> edges;
    std::array<SpatialDomains::TriGeomUniquePtr, 4> faces;
    SpatialDomains::TetGeomUniquePtr tet;
    SpatialDomains::CurveMap curvedEdges;
    SpatialDomains::CurveMap curvedFaces;
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
};

/// Build a tetrahedron with a curved face 0. Setting bit i of @p flip
/// traverses face i (1-3) the other way round, keeping its first edge on face
/// 0 as TetGeom requires, which changes the orientation the element sees.
static CurvedTet MakeCurvedTet(const int flip = 0)
{
    CurvedTet p;
    const std::array<std::array<NekDouble, 3>, 4> x = {{{{0.0, 0.0, 0.0}},
                                                        {{1.0, 0.0, 0.0}},
                                                        {{0.1, 1.0, 0.0}},
                                                        {{0.2, 0.1, 1.1}}}};

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
    std::array<SpatialDomains::TriGeom *, 4> facePtrs;
    for (int i = 0; i < 4; ++i)
    {
        std::array<SpatialDomains::SegGeom *, 3> fe;
        for (int j = 0; j < 3; ++j)
        {
            const int cyc = (flip >> i) & 1 ? (3 - j) % 3 : j;
            fe[j]         = p.edges[faceEdges[i][cyc]].get();
        }
        p.faces[i] = SpatialDomains::TriGeomUniquePtr(
            new SpatialDomains::TriGeom(i, fe));
        facePtrs[i] = p.faces[i].get();
    }
    p.tet = SpatialDomains::TetGeomUniquePtr(
        new SpatialDomains::TetGeom(0, facePtrs));
    p.tet->Setup();

    for (auto &edge : p.edges)
    {
        edge->FillGeom();
        auto curveAndNodes =
            edge->MakeOrder(3, LibUtilities::ePolyEvenlySpaced);
        for (auto &node : curveAndNodes.second)
        {
            p.nodes.push_back(std::move(node));
        }
        p.curvedEdges[edge->GetGlobalID()] = std::move(curveAndNodes.first);
    }
    for (auto &face : p.faces)
    {
        face->FillGeom();
        auto curveAndNodes =
            face->MakeOrder(3, LibUtilities::eNodalTriEvenlySpaced);
        for (auto &node : curveAndNodes.second)
        {
            p.nodes.push_back(std::move(node));
        }
        p.curvedFaces[face->GetGlobalID()] = std::move(curveAndNodes.first);
    }

    // At order three the final triangular node is the single face interior
    // node. Moving it out of the face plane makes the normal vary over the
    // face; the other faces are bent too, so that their orientation matters.
    const std::array<std::array<NekDouble, 3>, 4> bump = {
        {{{0.0, 0.0, 0.2}},
         {{0.02, -0.1, 0.03}},
         {{0.08, 0.05, 0.04}},
         {{-0.1, 0.02, 0.01}}}};
    for (int i = 0; i < 4; ++i)
    {
        SpatialDomains::PointGeom *interior = p.curvedFaces[i]->m_points.back();
        interior->UpdatePosition((*interior)(0) + bump[i][0],
                                 (*interior)(1) + bump[i][1],
                                 (*interior)(2) + bump[i][2]);
    }
    p.tet->Reset(p.curvedEdges, p.curvedFaces);
    return p;
}

BOOST_AUTO_TEST_CASE(TestDeformedFaceNormalWithDifferentQuadrature)
{
    CurvedTet geom = MakeCurvedTet();

    using namespace LibUtilities;
    const BasisKey high0(eModified_A, 5, PointsKey(7, eGaussLobattoLegendre));
    const BasisKey high1(eModified_B, 5, PointsKey(6, eGaussRadauMAlpha1Beta0));
    const BasisKey high2(eModified_C, 5, PointsKey(6, eGaussRadauMAlpha2Beta0));
    auto volume = MemoryManager<LocalRegions::TetExp>::AllocateSharedPtr(
        high0, high1, high2, geom.tet.get());

    const BasisKey low0(eModified_A, 3, PointsKey(4, eGaussLobattoLegendre));
    const BasisKey low1(eModified_B, 3, PointsKey(3, eGaussRadauMAlpha1Beta0));
    auto boundary = MemoryManager<LocalRegions::TriExp>::AllocateSharedPtr(
        low0, low1, geom.faces[0].get());

    LocalRegions::ExpansionSharedPtr adjacent = volume;
    boundary->SetAdjacentElementExp(0, adjacent);

    BOOST_REQUIRE_EQUAL(boundary->GetGeomFactors()->GetGtype(),
                        SpatialDomains::eDeformed);
    BOOST_REQUIRE(volume->GetTraceBasisKey(0, 0) != low0 ||
                  volume->GetTraceBasisKey(0, 1) != low1);

    const int nq = boundary->GetTotPoints();
    Array<OneD, NekDouble> fx(nq), fy(nq), fz(nq);
    for (int i = 0; i < nq; ++i)
    {
        fx[i] = 0.3 + 0.02 * i;
        fy[i] = -0.4 + 0.01 * i;
        fz[i] = 0.8 - 0.03 * i;
    }

    Array<OneD, NekDouble> actual(boundary->GetNcoeffs());
    boundary->NormVectorIProductWRTBase(fx, fy, fz, actual);

    const auto &normal = volume->GetTraceNormal(0);
    Array<OneD, NekDouble> nx(nq), ny(nq), nz(nq);
    Interp2D(volume->GetTraceBasisKey(0, 0), volume->GetTraceBasisKey(0, 1),
             normal[0], low0, low1, nx);
    Interp2D(volume->GetTraceBasisKey(0, 0), volume->GetTraceBasisKey(0, 1),
             normal[1], low0, low1, ny);
    Interp2D(volume->GetTraceBasisKey(0, 0), volume->GetTraceBasisKey(0, 1),
             normal[2], low0, low1, nz);

    Array<OneD, NekDouble> flux(nq);
    for (int i = 0; i < nq; ++i)
    {
        flux[i] = nx[i] * fx[i] + ny[i] * fy[i] + nz[i] * fz[i];
    }
    Array<OneD, NekDouble> expected(boundary->GetNcoeffs());
    boundary->IProductWRTBase(flux, expected);

    for (int i = 0; i < expected.size(); ++i)
    {
        BOOST_CHECK_SMALL(actual[i] - expected[i], 1.0e-11);
    }
}

/// Check that the normals of @p face, sampled on @p bnd, are unit vectors
/// orthogonal to the boundary expansion's own tangents and point from
/// @p centre towards the face. A normal assigned to the wrong point of a curved
/// face fails the orthogonality check.
static void CheckAlignedNormals(LocalRegions::Expansion &elmt, const int face,
                                LocalRegions::Expansion &bnd,
                                const std::array<NekDouble, 3> &centre)
{
    const int nq                       = bnd.GetTotPoints();
    const LocalRegions::NormalVector n = elmt.GetAlignedTraceNormal(face, bnd);
    BOOST_REQUIRE_EQUAL(n.size(), 3);

    std::array<Array<OneD, NekDouble>, 3> x, t0, t1;
    for (int d = 0; d < 3; ++d)
    {
        BOOST_REQUIRE_EQUAL(n[d].size(), nq);
        x[d]  = Array<OneD, NekDouble>(nq);
        t0[d] = Array<OneD, NekDouble>(nq);
        t1[d] = Array<OneD, NekDouble>(nq);
    }
    bnd.GetCoords(x[0], x[1], x[2]);
    for (int d = 0; d < 3; ++d)
    {
        bnd.StdPhysDeriv(x[d], t0[d], t1[d]);
    }

    std::array<NekDouble, 3> faceCentre;
    for (int d = 0; d < 3; ++d)
    {
        faceCentre[d] = Vmath::Vsum(nq, x[d], 1) / nq;
    }

    for (int i = 0; i < nq; ++i)
    {
        NekDouble nn = 0.0, nt0 = 0.0, nt1 = 0.0, tt0 = 0.0, tt1 = 0.0;
        NekDouble out = 0.0;
        for (int d = 0; d < 3; ++d)
        {
            nn += n[d][i] * n[d][i];
            nt0 += n[d][i] * t0[d][i];
            nt1 += n[d][i] * t1[d][i];
            tt0 += t0[d][i] * t0[d][i];
            tt1 += t1[d][i] * t1[d][i];
            out += n[d][i] * (faceCentre[d] - centre[d]);
        }
        BOOST_CHECK_SMALL(nn - 1.0, 1.0e-12);
        BOOST_CHECK_SMALL(nt0 / std::sqrt(tt0), 1.0e-10);
        BOOST_CHECK_SMALL(nt1 / std::sqrt(tt1), 1.0e-10);
        BOOST_CHECK_GT(out, 0.0);
    }
}

static std::array<NekDouble, 3> Centre(LocalRegions::Expansion &elmt)
{
    const int nq = elmt.GetTotPoints();
    Array<OneD, NekDouble> x(nq), y(nq), z(nq);
    elmt.GetCoords(x, y, z);
    return {Vmath::Vsum(nq, x, 1) / nq, Vmath::Vsum(nq, y, 1) / nq,
            Vmath::Vsum(nq, z, 1) / nq};
}

BOOST_AUTO_TEST_CASE(TestAlignedTriFaceNormals)
{
    using namespace LibUtilities;
    const BasisKey b0(eModified_A, 5, PointsKey(7, eGaussLobattoLegendre));
    const BasisKey b1(eModified_B, 5, PointsKey(6, eGaussRadauMAlpha1Beta0));
    const BasisKey b2(eModified_C, 5, PointsKey(6, eGaussRadauMAlpha2Beta0));

    std::set<StdRegions::Orientation> orients;
    for (int flip = 0; flip < 16; flip += 2)
    {
        CurvedTet geom = MakeCurvedTet(flip);
        auto volume    = MemoryManager<LocalRegions::TetExp>::AllocateSharedPtr(
            b0, b1, b2, geom.tet.get());
        const std::array<NekDouble, 3> centre = Centre(*volume);

        for (int f = 0; f < 4; ++f)
        {
            orients.insert(volume->GetTraceOrient(f));
            LocalRegions::ExpansionSharedPtr bnd =
                volume->GetAlignedTraceExp(f);
            CheckAlignedNormals(*volume, f, *bnd, centre);
        }
    }

    // The faces must not all share the element's frame, or the check above
    // could not detect a missing reorientation.
    BOOST_CHECK_GT(orients.size(), 1);
}

/// Storage for a hexahedron with curved faces; entities must outlive it.
struct CurvedHex
{
    std::array<SpatialDomains::PointGeomUniquePtr, 8> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 12> edges;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faces;
    SpatialDomains::HexGeomUniquePtr hex;
    SpatialDomains::CurveMap curvedEdges;
    SpatialDomains::CurveMap curvedFaces;
    std::vector<SpatialDomains::PointGeomUniquePtr> nodes;
};

/// Build a hexahedron whose faces are numbered @p rot steps around (and
/// optionally reversed), so that the element sees non-trivial and transposed
/// face orientations, then curve every edge and face with an asymmetric
/// perturbation of their high-order nodes.
static void MakeCurvedHex(CurvedHex &p, const int rot, const bool reverse)
{
    const std::array<std::array<NekDouble, 3>, 8> x = {{{{0.0, 0.0, 0.0}},
                                                        {{1.0, 0.0, 0.0}},
                                                        {{1.3, 1.0, 0.0}},
                                                        {{0.0, 1.1, 0.0}},
                                                        {{0.1, 0.2, 1.0}},
                                                        {{1.0, 0.0, 1.2}},
                                                        {{1.4, 1.2, 1.0}},
                                                        {{0.0, 1.0, 1.3}}}};
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
    std::array<SpatialDomains::QuadGeom *, 6> facePtrs;
    for (int i = 0; i < 6; ++i)
    {
        // Rotating or reversing the closed edge loop renumbers the face as a
        // neighbouring element might have, giving it a different orientation.
        std::array<SpatialDomains::SegGeom *, 4> fe;
        for (int j = 0; j < 4; ++j)
        {
            const int cyc = reverse ? (4 - j) % 4 : j;
            fe[j]         = p.edges[faceEdges[i][(cyc + rot + i) % 4]].get();
        }
        p.faces[i] = SpatialDomains::QuadGeomUniquePtr(
            new SpatialDomains::QuadGeom(i, fe));
        facePtrs[i] = p.faces[i].get();
    }
    p.hex = SpatialDomains::HexGeomUniquePtr(
        new SpatialDomains::HexGeom(0, facePtrs));
    p.hex->Setup();

    std::vector<SpatialDomains::Curve *> curves;
    auto keep = [&](SpatialDomains::Geometry *geom,
                    SpatialDomains::CurveMap &curved) {
        geom->FillGeom();
        auto curveAndNodes =
            geom->MakeOrder(3, LibUtilities::ePolyEvenlySpaced);
        for (auto &node : curveAndNodes.second)
        {
            p.nodes.push_back(std::move(node));
        }
        curves.push_back(curveAndNodes.first.get());
        curved[geom->GetGlobalID()] = std::move(curveAndNodes.first);
    };
    for (auto &edge : p.edges)
    {
        keep(edge.get(), p.curvedEdges);
    }
    for (auto &face : p.faces)
    {
        keep(face.get(), p.curvedFaces);
    }

    // Move every node that is not a vertex by a smooth, asymmetric function
    // of its position. Nodes shared by an edge and a face start at the same
    // place and so stay coincident.
    std::set<SpatialDomains::PointGeom *> moved;
    for (auto *curve : curves)
    {
        for (auto *node : curve->m_points)
        {
            bool isVertex = false;
            for (auto &v : p.verts)
            {
                isVertex = isVertex || node->dist(*v) < 1.0e-12;
            }
            if (isVertex || !moved.insert(node).second)
            {
                continue;
            }
            const NekDouble px = (*node)(0), py = (*node)(1), pz = (*node)(2);
            node->UpdatePosition(
                px + 0.06 * std::sin(2.1 * px + 0.3 * py + 1.7 * pz + 0.5),
                py + 0.06 * std::sin(0.4 * px + 2.3 * py + 0.9 * pz + 1.1),
                pz + 0.06 * std::sin(1.5 * px + 0.8 * py + 2.2 * pz + 0.2));
        }
    }
    p.hex->Reset(p.curvedEdges, p.curvedFaces);
}

BOOST_AUTO_TEST_CASE(TestAlignedQuadFaceNormals)
{
    using namespace LibUtilities;
    // Different point counts per direction, so that a transposed face sees
    // its extents exchanged.
    const BasisKey b0(eModified_A, 4, PointsKey(5, eGaussLobattoLegendre));
    const BasisKey b1(eModified_A, 4, PointsKey(6, eGaussLobattoLegendre));
    const BasisKey b2(eModified_A, 4, PointsKey(7, eGaussLobattoLegendre));

    std::set<StdRegions::Orientation> orients;
    for (const bool reverse : {false, true})
    {
        for (int rot = 0; rot < 4; ++rot)
        {
            CurvedHex geom;
            MakeCurvedHex(geom, rot, reverse);
            auto volume =
                MemoryManager<LocalRegions::HexExp>::AllocateSharedPtr(
                    b0, b1, b2, geom.hex.get());
            const std::array<NekDouble, 3> centre = Centre(*volume);

            for (int f = 0; f < 6; ++f)
            {
                orients.insert(volume->GetTraceOrient(f));
                LocalRegions::ExpansionSharedPtr bnd =
                    volume->GetAlignedTraceExp(f);
                BOOST_REQUIRE_EQUAL(bnd->GetGeomFactors()->GetGtype(),
                                    SpatialDomains::eDeformed);
                CheckAlignedNormals(*volume, f, *bnd, centre);

                // NormVectorIProductWRTBase must use the same aligned normals.
                LocalRegions::ExpansionSharedPtr adjacent = volume;
                bnd->SetAdjacentElementExp(f, adjacent);
                const int nq = bnd->GetTotPoints();
                const LocalRegions::NormalVector n =
                    volume->GetAlignedTraceNormal(f, *bnd);
                Array<OneD, NekDouble> fx(nq), fy(nq), fz(nq), flux(nq);
                for (int i = 0; i < nq; ++i)
                {
                    fx[i] = 0.3 + 0.02 * i;
                    fy[i] = -0.4 + 0.01 * i;
                    fz[i] = 0.8 - 0.03 * i;
                    flux[i] =
                        n[0][i] * fx[i] + n[1][i] * fy[i] + n[2][i] * fz[i];
                }
                Array<OneD, NekDouble> actual(bnd->GetNcoeffs());
                Array<OneD, NekDouble> expected(bnd->GetNcoeffs());
                bnd->NormVectorIProductWRTBase(fx, fy, fz, actual);
                bnd->IProductWRTBase(flux, expected);
                for (int i = 0; i < expected.size(); ++i)
                {
                    BOOST_CHECK_SMALL(actual[i] - expected[i], 1.0e-12);
                }
            }
        }
    }

    // Exercise both non-trivial and transposing orientations.
    BOOST_CHECK_GT(orients.size(), 2);
    BOOST_CHECK(orients.lower_bound(StdRegions::eDir1FwdDir2_Dir2FwdDir1) !=
                orients.end());
}

} // namespace Nektar::AlignedTraceNormalTests
