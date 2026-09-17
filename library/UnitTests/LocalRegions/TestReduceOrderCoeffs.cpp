///////////////////////////////////////////////////////////////////////////////
//
// File: TestReduceOrderCoeffs.cpp
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
// Description: Unit tests for HexExp::ReduceOrderCoeffs.
//
///////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/HexExp.h>
#include <SpatialDomains/MeshGraph.h>
#include <StdRegions/StdHexExp.h>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Nektar::LocalRegionsReduceOrderTests
{

SpatialDomains::SegGeomUniquePtr CreateSegGeom(unsigned int id,
                                               SpatialDomains::PointGeom *v0,
                                               SpatialDomains::PointGeom *v1)
{
    std::array<SpatialDomains::PointGeom *, 2> vertices = {v0, v1};
    return SpatialDomains::SegGeomUniquePtr(
        new SpatialDomains::SegGeom(id, 3, vertices));
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

    for (int i = 0; i < 12; ++i)
    {
        segVec[i] = CreateSegGeom(i, v[edgeVerts[i][0]], v[edgeVerts[i][1]]);
    }

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

    return SpatialDomains::HexGeomUniquePtr(
        new SpatialDomains::HexGeom(0, faces));
}

/**
 * @brief Check HexExp::ReduceOrderCoeffs mode by mode on an affine element.
 *
 * Each mode of the orthogonal standard hexahedron of the same order and
 * points is expressed in the element's basis and reduced to @p numMin: a mode
 * below numMin in every direction must come back unchanged, every other must
 * vanish. The element is a translated unit cube, so its transforms are those
 * of the standard element up to a constant Jacobian.
 */
void CheckHex(const int nm0, const int nm1, const int nm2, const int numMin)
{
    using namespace LibUtilities;

    std::array<SpatialDomains::PointGeomUniquePtr, 8> v;
    const double c[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    std::array<SpatialDomains::PointGeom *, 8> vp;
    for (int i = 0; i < 8; ++i)
    {
        v[i] = SpatialDomains::PointGeomUniquePtr(new SpatialDomains::PointGeom(
            3u, i, 0.5 + c[i][0], 0.25 + c[i][1], -1.0 + c[i][2]));
        vp[i] = v[i].get();
    }
    std::array<SpatialDomains::SegGeomUniquePtr, 12> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom = CreateHex(vp, segVec, faceVec);

    const PointsKey pkey0(nm0 + 1, eGaussLobattoLegendre);
    const PointsKey pkey1(nm1 + 1, eGaussLobattoLegendre);
    const PointsKey pkey2(nm2 + 1, eGaussLobattoLegendre);
    const BasisKey b0(eModified_A, nm0, pkey0), b1(eModified_A, nm1, pkey1),
        b2(eModified_A, nm2, pkey2);
    const BasisKey o0(eOrtho_A, nm0, pkey0), o1(eOrtho_A, nm1, pkey1),
        o2(eOrtho_A, nm2, pkey2);

    auto exp = MemoryManager<LocalRegions::HexExp>::AllocateSharedPtr(
        b0, b1, b2, hexGeom.get());
    auto ortho =
        MemoryManager<StdRegions::StdHexExp>::AllocateSharedPtr(o0, o1, o2);

    const int nm = exp->GetNcoeffs();
    const int nq = exp->GetTotPoints();
    BOOST_REQUIRE_EQUAL(ortho->GetNcoeffs(), nm);
    BOOST_REQUIRE_EQUAL(ortho->GetTotPoints(), nq);

    // Mode (p, q, r) is stored with p fastest; its degree is the largest
    // index.
    std::vector<int> degree(nm);
    for (int r = 0; r < nm2; ++r)
    {
        for (int q = 0; q < nm1; ++q)
        {
            for (int p = 0; p < nm0; ++p)
            {
                degree[(r * nm1 + q) * nm0 + p] = std::max({p, q, r});
            }
        }
    }

    const NekDouble tol = 1.0e-8;
    for (int mode = 0; mode < nm; ++mode)
    {
        Array<OneD, NekDouble> orthoCoeffs(nm, 0.0);
        orthoCoeffs[mode] = 1.0;

        Array<OneD, NekDouble> phys(nq), coeffs(nm), reduced(nm), back(nq);
        ortho->BwdTrans(orthoCoeffs, phys);
        exp->FwdTrans(phys, coeffs);
        exp->ReduceOrderCoeffs(numMin, coeffs, reduced);
        exp->BwdTrans(reduced, back);

        const bool kept = degree[mode] < numMin;
        NekDouble diff  = 0.0;
        for (int q = 0; q < nq; ++q)
        {
            diff = std::max(diff, std::abs(back[q] - (kept ? phys[q] : 0.0)));
        }
        BOOST_CHECK_MESSAGE(diff < tol,
                            "mode " << mode << " of degree " << degree[mode]
                                    << (kept ? " changed by " : " left ")
                                    << diff << " with numMin " << numMin);
    }
}

BOOST_AUTO_TEST_CASE(TestHexReduceOrderCoeffs)
{
    CheckHex(4, 4, 4, 3);
    CheckHex(4, 4, 4, 2);
    // Anisotropic: numMin above the smallest mode count, and below it.
    CheckHex(3, 4, 5, 4);
    CheckHex(5, 4, 3, 3);
}

} // namespace Nektar::LocalRegionsReduceOrderTests
