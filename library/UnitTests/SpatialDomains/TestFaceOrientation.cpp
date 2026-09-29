///////////////////////////////////////////////////////////////////////////////
//
// File: TestFaceOrientation.cpp
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
// Description: Face orientations agree with the coordinate-based algorithm.
//
///////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace Nektar::FaceOrientationTests
{

using namespace SpatialDomains;

typedef std::array<NekDouble, 3> Pt;

/**
 * @brief The orientation of a face, recovered from coordinates.
 *
 * This is the algorithm the geometry classes used before orientation was read
 * off the vertex permutation, kept so that the two can be compared. It builds
 * the two local axes of the face as the element sees them and as the face
 * geometry itself sees them, and decides the orientation from the signs of the
 * dot products, transposing when the axes do not line up.
 *
 * @param ev    Positions of the element's vertices for this face, in the order
 *              its face-vertex table gives them.
 * @param fv    Positions of the face geometry's own vertices, in its order.
 * @param base  Index into @p ev of the vertex the face geometry starts at.
 */
static StdRegions::Orientation ReferenceOrient(const std::vector<Pt> &ev,
                                               const std::vector<Pt> &fv,
                                               int base)
{
    const bool tri = ev.size() == 3;

    auto sub = [](const Pt &a, const Pt &b) {
        return Pt{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto dot = [](const Pt &a, const Pt &b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto len = [&](const Pt &a) { return std::sqrt(dot(a, a)); };

    // The element's two axes, anchored on whichever vertex the face starts at.
    Pt eA, eB;
    if (tri)
    {
        switch (base)
        {
            case 0:
                eA = sub(ev[1], ev[0]);
                eB = sub(ev[2], ev[0]);
                break;
            case 1:
                eA = sub(ev[1], ev[0]);
                eB = sub(ev[2], ev[1]);
                break;
            default:
                eA = sub(ev[1], ev[2]);
                eB = sub(ev[2], ev[0]);
                break;
        }
    }
    else
    {
        switch (base)
        {
            case 0:
                eA = sub(ev[1], ev[0]);
                eB = sub(ev[3], ev[0]);
                break;
            case 1:
                eA = sub(ev[1], ev[0]);
                eB = sub(ev[2], ev[1]);
                break;
            case 2:
                eA = sub(ev[2], ev[3]);
                eB = sub(ev[2], ev[1]);
                break;
            default:
                eA = sub(ev[2], ev[3]);
                eB = sub(ev[3], ev[0]);
                break;
        }
    }

    // The face's own axes.
    const Pt fA = sub(fv[1], fv[0]);
    const Pt fB = sub(fv[tri ? 2 : 3], fv[0]);

    unsigned int orientation = 0;
    NekDouble d1             = dot(eA, fA);
    NekDouble norm           = std::fabs(d1) / len(eA) / len(fA);

    if (std::fabs(norm - 1.0) < NekConstants::kNekZeroTol)
    {
        if (d1 < 0.0)
        {
            orientation += 2;
        }
        if (dot(eB, fB) < 0.0)
        {
            orientation++;
        }
    }
    else
    {
        // The axes are swapped: compare each against the other.
        orientation = 4;
        if (dot(eA, fB) < 0.0)
        {
            orientation += 2;
        }
        if (dot(eB, fA) < 0.0)
        {
            orientation++;
        }
    }

    return static_cast<StdRegions::Orientation>(orientation + 5);
}

/// A shape, as the vertices of each of its faces.
struct ShapeDef
{
    const char *name;
    int nVerts;
    std::vector<std::vector<int>> faces;
};

/// The face-vertex tables the geometry classes use, with the padding entry the
/// triangular faces of a prism and a pyramid carry in the source removed.
static const ShapeDef kTet = {
    "tetrahedron", 4, {{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}}};
static const ShapeDef kPyr = {
    "pyramid", 5, {{0, 1, 2, 3}, {0, 1, 4}, {1, 2, 4}, {3, 2, 4}, {0, 3, 4}}};
static const ShapeDef kPrism = {
    "prism",
    6,
    {{0, 1, 2, 3}, {0, 1, 4}, {1, 2, 5, 4}, {3, 2, 5}, {0, 3, 5, 4}}};
static const ShapeDef kHex = {"hexahedron",
                              8,
                              {{0, 1, 2, 3},
                               {0, 1, 5, 4},
                               {1, 2, 6, 5},
                               {3, 2, 6, 7},
                               {0, 3, 7, 4},
                               {4, 5, 6, 7}}};

/// Storage for one built element; the parts must outlive it.
struct Parts
{
    std::vector<PointGeomUniquePtr> verts;
    std::vector<SegGeomUniquePtr> edges;
    std::vector<TriGeomUniquePtr> tris;
    std::vector<QuadGeomUniquePtr> quads;
    TetGeomUniquePtr tet;
    PyrGeomUniquePtr pyr;
    PrismGeomUniquePtr prism;
    HexGeomUniquePtr hex;
    Geometry3D *elmt = nullptr;
};

/**
 * @brief Build @p shape, with the vertices of face @c f started at its
 * @c rot[f]'th vertex.
 *
 * Rotating the edges a face is built from rotates the vertices the face
 * geometry reports, which is what varies the orientation the element then has
 * to recover. The edges themselves are shared between faces, as they are in a
 * real mesh, so the element still sees one edge where two faces meet.
 */
static void Build(Parts &p, const ShapeDef &shape, const std::vector<Pt> &x,
                  const std::vector<int> &rot)
{
    for (int i = 0; i < shape.nVerts; ++i)
    {
        p.verts.push_back(PointGeomUniquePtr(
            new PointGeom(3u, i, x[i][0], x[i][1], x[i][2])));
    }

    // Collect the edges from the faces, so only the face tables are stated.
    std::map<std::pair<int, int>, int> edgeId;
    auto edgeOf = [&](int a, int b) {
        auto key = std::minmax(a, b);
        auto it  = edgeId.find({key.first, key.second});
        if (it != edgeId.end())
        {
            return it->second;
        }
        int id                        = (int)p.edges.size();
        std::array<PointGeom *, 2> ev = {p.verts[key.first].get(),
                                         p.verts[key.second].get()};
        p.edges.push_back(SegGeomUniquePtr(new SegGeom(id, 3, ev)));
        edgeId[{key.first, key.second}] = id;
        return id;
    };

    std::vector<Geometry2D *> faces;
    for (size_t f = 0; f < shape.faces.size(); ++f)
    {
        const std::vector<int> &fv = shape.faces[f];
        const int n                = (int)fv.size();

        // The face's edges, started at vertex rot[f] % n and traversed either
        // way round. Reversing keeps local edge 0 the same while swapping the
        // vertices the face reports, which is how the element on the far side
        // of a shared face sees it.
        const int start = rot[f] % n;
        const bool flip = rot[f] >= n;
        std::vector<SegGeom *> fe(n);
        for (int j = 0; j < n; ++j)
        {
            const int k = flip ? ((start - j) % n + n) % n : (start + j) % n;
            fe[j]       = p.edges[edgeOf(fv[k], fv[(k + 1) % n])].get();
        }

        if (n == 3)
        {
            std::array<SegGeom *, 3> e = {fe[0], fe[1], fe[2]};
            p.tris.push_back(TriGeomUniquePtr(new TriGeom((int)f, e)));
            faces.push_back(p.tris.back().get());
        }
        else
        {
            std::array<SegGeom *, 4> e = {fe[0], fe[1], fe[2], fe[3]};
            p.quads.push_back(QuadGeomUniquePtr(new QuadGeom((int)f, e)));
            faces.push_back(p.quads.back().get());
        }
    }

    if (shape.nVerts == 4)
    {
        std::array<TriGeom *, 4> f;
        for (int i = 0; i < 4; ++i)
        {
            f[i] = static_cast<TriGeom *>(faces[i]);
        }
        p.tet  = TetGeomUniquePtr(new TetGeom(0, f));
        p.elmt = p.tet.get();
    }
    else if (shape.nVerts == 5)
    {
        std::array<Geometry2D *, 5> f;
        std::copy(faces.begin(), faces.end(), f.begin());
        p.pyr  = PyrGeomUniquePtr(new PyrGeom(0, f));
        p.elmt = p.pyr.get();
    }
    else if (shape.nVerts == 6)
    {
        std::array<Geometry2D *, 5> f;
        std::copy(faces.begin(), faces.end(), f.begin());
        p.prism = PrismGeomUniquePtr(new PrismGeom(0, f));
        p.elmt  = p.prism.get();
    }
    else
    {
        std::array<QuadGeom *, 6> f;
        for (int i = 0; i < 6; ++i)
        {
            f[i] = static_cast<QuadGeom *>(faces[i]);
        }
        p.hex  = HexGeomUniquePtr(new HexGeom(0, f));
        p.elmt = p.hex.get();
    }
}

/// Check every face of one built element against the reference algorithm.
static void CheckElement(const ShapeDef &shape, const std::vector<Pt> &x,
                         const std::vector<int> &rot,
                         std::set<StdRegions::Orientation> &seen)
{
    Parts p;
    Build(p, shape, x, rot);

    for (size_t f = 0; f < shape.faces.size(); ++f)
    {
        const std::vector<int> &fvIdx = shape.faces[f];
        Geometry2D *face              = p.elmt->GetFace((int)f);

        std::vector<Pt> ev, fv;
        int base = -1;
        for (size_t j = 0; j < fvIdx.size(); ++j)
        {
            PointGeom *v = p.elmt->GetVertex(fvIdx[j]);
            ev.push_back(Pt{(*v)(0), (*v)(1), (*v)(2)});
            if (v->GetGlobalID() == face->GetVid(0))
            {
                base = (int)j;
            }
        }
        BOOST_REQUIRE_MESSAGE(base >= 0, "face geometry does not start at any "
                                         "vertex the element has for it");

        for (size_t j = 0; j < fvIdx.size(); ++j)
        {
            PointGeom *v = face->GetVertex((int)j);
            fv.push_back(Pt{(*v)(0), (*v)(1), (*v)(2)});
        }

        const StdRegions::Orientation expected = ReferenceOrient(ev, fv, base);
        const StdRegions::Orientation actual   = p.elmt->GetForient((int)f);
        seen.insert(actual);

        std::ostringstream where;
        where << shape.name << " face " << f << ", rotations [";
        for (size_t i = 0; i < rot.size(); ++i)
        {
            where << rot[i] << (i + 1 < rot.size() ? " " : "");
        }
        where << "]: coordinate algorithm gives "
              << StdRegions::OrientationMap[expected] << ", geometry gives "
              << StdRegions::OrientationMap[actual];

        BOOST_CHECK_MESSAGE(expected == actual, where.str());
    }
}

/**
 * @brief Check every way of starting each face at a different one of its
 * vertices.
 *
 * Not every combination describes a consistent element and the constructors
 * reject those, so the sweep reports how many it managed in order that a
 * change which quietly made them all invalid cannot pass as a clean run.
 */
static void SweepRotations(const ShapeDef &shape, const std::vector<Pt> &x)
{
    const size_t nf = shape.faces.size();
    std::vector<int> rot(nf, 0);

    size_t total = 1;
    for (size_t f = 0; f < nf; ++f)
    {
        total *= 2 * shape.faces[f].size();
    }

    // Enough to cover the shapes with few faces exhaustively, and to sample
    // the hexahedron's quarter of a million arrangements densely.
    const size_t kMaxCases = 30000;
    const size_t stride    = total > kMaxCases ? total / kMaxCases : 1;

    // The rejections are expected, so keep them out of the test log.
    std::ostringstream discard;
    ErrorUtil::SetErrorStream(discard);

    size_t built = 0;
    std::set<StdRegions::Orientation> seen;
    size_t tried = 0;
    for (size_t n = 0; n < total; n += stride)
    {
        ++tried;
        size_t r = n;
        for (size_t f = 0; f < nf; ++f)
        {
            const size_t m = 2 * shape.faces[f].size();
            rot[f]         = (int)(r % m);
            r /= m;
        }

        try
        {
            CheckElement(shape, x, rot, seen);
            ++built;
        }
        catch (const ErrorUtil::NekError &)
        {
            discard.str("");
            continue;
        }
    }

    ErrorUtil::SetErrorStream(std::cerr);

    BOOST_TEST_MESSAGE(std::string(shape.name) + ": " + std::to_string(built) +
                       " of " + std::to_string(tried) +
                       " rotations built, covering " +
                       std::to_string(seen.size()) + " distinct orientations");

    // A sweep that built nothing, or that only ever saw one orientation, would
    // agree with anything and is not a test of this at all.
    BOOST_CHECK_MESSAGE(built > 0, std::string(shape.name) +
                                       ": no rotation produced an element");
    BOOST_CHECK_MESSAGE(seen.size() > 1,
                        std::string(shape.name) +
                            ": only one orientation ever occurred, so the "
                            "comparison proves nothing");
}

BOOST_AUTO_TEST_CASE(TestTetFaceOrientation)
{
    SweepRotations(kTet, {Pt{0.0, 0.0, 0.0}, Pt{1.0, 0.0, 0.0},
                          Pt{0.0, 1.0, 0.0}, Pt{0.0, 0.0, 1.0}});
}

BOOST_AUTO_TEST_CASE(TestPyrFaceOrientation)
{
    SweepRotations(kPyr,
                   {Pt{0.0, 0.0, 0.0}, Pt{1.0, 0.0, 0.0}, Pt{1.0, 1.0, 0.0},
                    Pt{0.0, 1.0, 0.0}, Pt{0.5, 0.5, 1.0}});
}

BOOST_AUTO_TEST_CASE(TestPrismFaceOrientation)
{
    SweepRotations(kPrism,
                   {Pt{0.0, 0.0, 0.0}, Pt{1.0, 0.0, 0.0}, Pt{1.0, 1.0, 0.0},
                    Pt{0.0, 1.0, 0.0}, Pt{0.0, 0.0, 1.0}, Pt{1.0, 0.0, 1.0}});
}

BOOST_AUTO_TEST_CASE(TestHexFaceOrientation)
{
    SweepRotations(kHex,
                   {Pt{0.0, 0.0, 0.0}, Pt{1.0, 0.0, 0.0}, Pt{1.0, 1.0, 0.0},
                    Pt{0.0, 1.0, 0.0}, Pt{0.0, 0.0, 1.0}, Pt{1.0, 0.0, 1.0},
                    Pt{1.0, 1.0, 1.0}, Pt{0.0, 1.0, 1.0}});
}

} // namespace Nektar::FaceOrientationTests
