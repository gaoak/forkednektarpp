///////////////////////////////////////////////////////////////////////////////
//
// File: TestPyrInverseMap.cpp
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
// Description: Round-trip tests for the pyramid inverse mapping.
//
///////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/GeometryLocator.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <array>

namespace Nektar::PyrInverseMapTests
{

/// Storage for the entities a pyramid is built from, which must outlive it.
struct PyrParts
{
    std::array<SpatialDomains::PointGeomUniquePtr, 5> verts;
    std::array<SpatialDomains::SegGeomUniquePtr, 8> edges;
    std::array<SpatialDomains::TriGeomUniquePtr, 4> tris;
    SpatialDomains::QuadGeomUniquePtr quad;
    SpatialDomains::PyrGeomUniquePtr pyr;
};

/// Build a straight-sided pyramid from five vertex positions.
static void BuildPyr(PyrParts &p,
                     const std::array<std::array<NekDouble, 3>, 5> &x)
{
    for (int i = 0; i < 5; ++i)
    {
        p.verts[i] = SpatialDomains::PointGeomUniquePtr(
            new SpatialDomains::PointGeom(3u, i, x[i][0], x[i][1], x[i][2]));
    }

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

    const std::array<std::array<int, 4>, 5> faceEdges = {{{{0, 1, 2, 3}},
                                                          {{0, 5, 4, -1}},
                                                          {{1, 6, 5, -1}},
                                                          {{2, 7, 6, -1}},
                                                          {{3, 7, 4, -1}}}};

    std::array<SpatialDomains::Geometry2D *, 5> faces;
    for (int i = 0; i < 5; ++i)
    {
        if (i == 0)
        {
            std::array<SpatialDomains::SegGeom *, 4> f;
            for (int j = 0; j < 4; ++j)
            {
                f[j] = p.edges[faceEdges[i][j]].get();
            }
            p.quad = SpatialDomains::QuadGeomUniquePtr(
                new SpatialDomains::QuadGeom(i, f));
            faces[i] = p.quad.get();
        }
        else
        {
            std::array<SpatialDomains::SegGeom *, 3> f;
            for (int j = 0; j < 3; ++j)
            {
                f[j] = p.edges[faceEdges[i][j]].get();
            }
            p.tris[i - 1] = SpatialDomains::TriGeomUniquePtr(
                new SpatialDomains::TriGeom(i, f));
            faces[i] = p.tris[i - 1].get();
        }
    }

    p.pyr =
        SpatialDomains::PyrGeomUniquePtr(new SpatialDomains::PyrGeom(0, faces));
    p.pyr->Setup();
}

/// Map a local coordinate forward and back, and report the largest component
/// error over a set of interior points.
static NekDouble WorstRoundTrip(SpatialDomains::Geometry *geom)
{
    auto locator = SpatialDomains::GeometryLocator::Create(geom);

    // Points strictly inside the reference pyramid, which is bounded by
    // xi_i >= -1 and xi_0 + xi_2 <= 0, xi_1 + xi_2 <= 0.
    const std::array<std::array<NekDouble, 3>, 6> probes = {
        {{{-0.5, -0.5, 0.0}},
         {{0.0, 0.0, -0.5}},
         {{-0.2, -0.3, 0.3}},
         {{-0.6, 0.2, 0.5}},
         {{-0.25, -0.25, 0.25}},
         {{0.3, -0.4, -0.6}}}};

    NekDouble worst = 0.0;
    for (auto &probe : probes)
    {
        Array<OneD, NekDouble> xi(3), x(3), back(3, 0.0);
        for (int d = 0; d < 3; ++d)
        {
            xi[d] = probe[d];
        }

        // Forward: the chi mapping, as GetCoord evaluates it.
        for (int d = 0; d < 3; ++d)
        {
            x[d] = geom->GetCoord(d, xi);
        }

        // Inverse: whatever route the locator chooses.
        locator->GetLocCoords(x, back);

        for (int d = 0; d < 3; ++d)
        {
            worst = std::max(worst, std::abs(back[d] - xi[d]));
        }
    }
    return worst;
}

// A pyramid whose base is a parallelogram has an affine mapping, which the
// linear inverse recovers exactly.
BOOST_AUTO_TEST_CASE(TestPyrRoundTripParallelogramBase)
{
    PyrParts p;
    BuildPyr(p, {{{{0.0, 0.0, 0.0}},
                  {{1.0, 0.0, 0.0}},
                  {{1.0, 1.0, 0.0}},
                  {{0.0, 1.0, 0.0}},
                  {{0.5, 0.5, 1.0}}}});

    BOOST_CHECK_EQUAL(p.pyr->CalcGeomType(), SpatialDomains::eRegular);
    BOOST_CHECK_SMALL(WorstRoundTrip(p.pyr.get()), 1e-8);
}

// With a skewed base the mapping is no longer a polynomial in the local
// coordinates: both base directions collapse against 1 - xi_2, so the
// bilinear term picks up a factor 1/(1 - xi_2). Inverting the polynomial form
// instead of the mapping itself used to leave errors of order 1e-2 in the
// interior while still being exact at the vertices.
BOOST_AUTO_TEST_CASE(TestPyrRoundTripSkewedBase)
{
    PyrParts p;
    BuildPyr(p, {{{{0.0, 0.0, 0.0}},
                  {{1.0, 0.0, 0.0}},
                  {{1.0, 1.0, 0.0}},
                  {{0.0, 1.3, 0.0}},
                  {{0.5, 0.5, 1.0}}}});

    BOOST_CHECK_EQUAL(p.pyr->CalcGeomType(), SpatialDomains::eDeformed);
    BOOST_CHECK_SMALL(WorstRoundTrip(p.pyr.get()), 1e-8);
}

} // namespace Nektar::PyrInverseMapTests
