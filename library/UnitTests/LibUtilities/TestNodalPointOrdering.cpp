///////////////////////////////////////////////////////////////////////////////
//
// File: TestNodalPointOrdering.cpp
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
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Description: Tests that the three-dimensional nodal point distributions
// agree with the way SpatialDomains numbers the standard element.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/Points.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <vector>

namespace Nektar::NodalPointOrderingTests
{

/**
 * @brief What a three-dimensional nodal distribution is required to look like.
 *
 * The points come in blocks: the vertices, then the interior of each edge in
 * turn, then the interior of each face, then the interior of the element. The
 * vertices are expected in the order the standard element numbers them, and
 * each edge's block is expected to run from the first of that edge's vertices
 * to the second, both taken from the Geometry class of the shape --
 * TetGeom::SetUpEdgeOrientation() and its siblings, whose tables are mirrored
 * here. MakeOrder() builds an element's curve by reading its vertices, edges
 * and faces straight into those blocks, so a distribution that disagrees makes
 * a tangled element rather than a wrong one, and the disagreement is not
 * visible anywhere nearer than the Jacobian.
 */
struct Shape
{
    const char *name;
    LibUtilities::PointsType pType;
    std::vector<std::array<NekDouble, 3>> verts;
    std::vector<std::array<int, 2>> edges;
    /// Three or four vertices per face; a triangle repeats its first.
    std::vector<std::array<int, 4>> faces;
    /// Number of interior points on a face, given the number in one direction.
    std::vector<int> faceIsQuad;
};

static const std::array<NekDouble, 3> tetV[4] = {
    {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
static const std::array<NekDouble, 3> prismV[6] = {{-1, -1, -1}, {1, -1, -1},
                                                   {1, 1, -1},   {-1, 1, -1},
                                                   {-1, -1, 1},  {-1, 1, 1}};
static const std::array<NekDouble, 3> pyrV[5]   = {
    {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}};

static std::vector<Shape> GetShapes()
{
    const std::vector<std::array<int, 2>> tetE = {{0, 1}, {1, 2}, {0, 2},
                                                  {0, 3}, {1, 3}, {2, 3}};
    const std::vector<std::array<int, 4>> tetF = {
        {0, 1, 2, 0}, {0, 1, 3, 0}, {1, 2, 3, 0}, {0, 2, 3, 0}};

    const std::vector<std::array<int, 2>> prismE = {
        {0, 1}, {1, 2}, {3, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 5}, {3, 5}, {4, 5}};
    const std::vector<std::array<int, 4>> prismF = {
        {0, 1, 2, 3}, {0, 1, 4, 0}, {1, 2, 5, 4}, {3, 2, 5, 0}, {0, 3, 5, 4}};

    const std::vector<std::array<int, 2>> pyrE = {
        {0, 1}, {1, 2}, {3, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
    const std::vector<std::array<int, 4>> pyrF = {
        {0, 1, 2, 3}, {0, 1, 4, 0}, {1, 2, 4, 0}, {3, 2, 4, 0}, {0, 3, 4, 0}};

    std::vector<Shape> shapes;
    for (auto pt :
         {LibUtilities::eNodalTetElec, LibUtilities::eNodalTetEvenlySpaced})
    {
        shapes.push_back({pt == LibUtilities::eNodalTetElec
                              ? "eNodalTetElec"
                              : "eNodalTetEvenlySpaced",
                          pt,
                          {tetV, tetV + 4},
                          tetE,
                          tetF,
                          {0, 0, 0, 0}});
    }
    for (auto pt :
         {LibUtilities::eNodalPrismElec, LibUtilities::eNodalPrismEvenlySpaced})
    {
        shapes.push_back({pt == LibUtilities::eNodalPrismElec
                              ? "eNodalPrismElec"
                              : "eNodalPrismEvenlySpaced",
                          pt,
                          {prismV, prismV + 6},
                          prismE,
                          prismF,
                          {1, 0, 1, 0, 1}});
    }
    shapes.push_back({"eNodalPyrEvenlySpaced",
                      LibUtilities::eNodalPyrEvenlySpaced,
                      {pyrV, pyrV + 5},
                      pyrE,
                      pyrF,
                      {1, 0, 0, 0, 0}});
    return shapes;
}

static std::array<NekDouble, 3> Sub(const std::array<NekDouble, 3> &a,
                                    const std::array<NekDouble, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static NekDouble Dot(const std::array<NekDouble, 3> &a,
                     const std::array<NekDouble, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static std::array<NekDouble, 3> Cross(const std::array<NekDouble, 3> &a,
                                      const std::array<NekDouble, 3> &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

BOOST_AUTO_TEST_CASE(TestNodalVerticesComeFirstInStandardOrder)
{
    for (auto &s : GetShapes())
    {
        for (int n = 2; n <= 7; ++n)
        {
            LibUtilities::PointsKey key(n, s.pType);
            Array<OneD, NekDouble> x, y, z;
            LibUtilities::PointsManager()[key]->GetPoints(x, y, z);

            for (size_t v = 0; v < s.verts.size(); ++v)
            {
                const std::array<NekDouble, 3> p = {x[v], y[v], z[v]};
                BOOST_TEST_CONTEXT(s.name << " at n = " << n << ", vertex "
                                          << v)
                {
                    BOOST_CHECK_SMALL(
                        std::sqrt(Dot(Sub(p, s.verts[v]), Sub(p, s.verts[v]))),
                        1e-12);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestNodalEdgesRunAlongTheGeometrysEdges)
{
    for (auto &s : GetShapes())
    {
        for (int n = 3; n <= 7; ++n)
        {
            LibUtilities::PointsKey key(n, s.pType);
            Array<OneD, NekDouble> x, y, z;
            LibUtilities::PointsManager()[key]->GetPoints(x, y, z);

            const int nv = s.verts.size(), ne = n - 2;
            for (size_t e = 0; e < s.edges.size(); ++e)
            {
                const auto &a  = s.verts[s.edges[e][0]];
                const auto &b  = s.verts[s.edges[e][1]];
                const auto d   = Sub(b, a);
                NekDouble last = 0.0;

                for (int j = 0; j < ne; ++j)
                {
                    const int i                      = nv + e * ne + j;
                    const std::array<NekDouble, 3> p = {x[i], y[i], z[i]};
                    const auto w                     = Sub(p, a);
                    const NekDouble t                = Dot(w, d) / Dot(d, d);
                    const auto off = Sub(w, {t * d[0], t * d[1], t * d[2]});

                    BOOST_TEST_CONTEXT(s.name << " at n = " << n << ", edge "
                                              << e << ", point " << j)
                    {
                        // On the edge, between its vertices, and further along
                        // it than the point before.
                        BOOST_CHECK_SMALL(std::sqrt(Dot(off, off)), 1e-12);
                        BOOST_CHECK(t > 0.0 && t < 1.0);
                        BOOST_CHECK(t > last);
                    }
                    last = t;
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestNodalFacesLieOnTheGeometrysFaces)
{
    for (auto &s : GetShapes())
    {
        for (int n = 4; n <= 7; ++n)
        {
            LibUtilities::PointsKey key(n, s.pType);
            Array<OneD, NekDouble> x, y, z;
            LibUtilities::PointsManager()[key]->GetPoints(x, y, z);

            const int nv = s.verts.size(), ne = n - 2;
            const int nTri  = (n - 2) * (n - 3) / 2;
            const int nQuad = (n - 2) * (n - 2);

            int offset = nv + s.edges.size() * ne;
            for (size_t f = 0; f < s.faces.size(); ++f)
            {
                const auto &v0      = s.verts[s.faces[f][0]];
                const auto nrm      = Cross(Sub(s.verts[s.faces[f][1]], v0),
                                            Sub(s.verts[s.faces[f][2]], v0));
                const NekDouble len = std::sqrt(Dot(nrm, nrm));
                const int sz        = s.faceIsQuad[f] ? nQuad : nTri;

                for (int j = 0; j < sz; ++j)
                {
                    const int i                      = offset + j;
                    const std::array<NekDouble, 3> p = {x[i], y[i], z[i]};
                    BOOST_TEST_CONTEXT(s.name << " at n = " << n << ", face "
                                              << f << ", point " << j)
                    {
                        // Looser than the vertex and edge checks: the
                        // electrostatic points are tabulated to about ten
                        // digits, which shows up on a face that is not
                        // aligned with a coordinate plane.
                        BOOST_CHECK_SMALL(std::abs(Dot(Sub(p, v0), nrm)) / len,
                                          1e-9);
                    }
                }
                offset += sz;
            }
        }
    }
}

} // namespace Nektar::NodalPointOrderingTests
