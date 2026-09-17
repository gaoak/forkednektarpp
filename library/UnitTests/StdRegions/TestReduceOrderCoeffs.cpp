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
// Description: Unit tests for StdExpansion::ReduceOrderCoeffs.
//
///////////////////////////////////////////////////////////////////////////////

#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdSegExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace Nektar::StdRegionsUnitTests
{

/**
 * @brief Check ReduceOrderCoeffs mode by mode.
 *
 * Each mode of the orthogonal expansion @p ortho, of the same shape, order
 * and points as @p exp, is expressed in @p exp's basis and reduced to
 * @p numMin. A mode whose degree, in the sense the shape truncates, is below
 * numMin must come back unchanged; every other mode must vanish.
 */
void CheckReduceOrderCoeffs(const StdRegions::StdExpansionSharedPtr &exp,
                            const StdRegions::StdExpansionSharedPtr &ortho,
                            const std::vector<int> &degree, const int numMin)
{
    const int nm = exp->GetNcoeffs();
    const int nq = exp->GetTotPoints();

    BOOST_REQUIRE_EQUAL(ortho->GetNcoeffs(), nm);
    BOOST_REQUIRE_EQUAL(ortho->GetTotPoints(), nq);
    BOOST_REQUIRE_EQUAL(static_cast<int>(degree.size()), nm);

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
            const NekDouble expected = kept ? phys[q] : 0.0;
            diff = std::max(diff, std::abs(back[q] - expected));
        }
        BOOST_CHECK_MESSAGE(diff < tol,
                            "mode " << mode << " of degree " << degree[mode]
                                    << (kept ? " changed by " : " left ")
                                    << diff << " with numMin " << numMin);
    }
}

BOOST_AUTO_TEST_CASE(TestSegReduceOrderCoeffs)
{
    using namespace LibUtilities;
    const int nm = 5;
    const PointsKey pkey(nm + 1, eGaussLobattoLegendre);
    const BasisKey bkey(eModified_A, nm, pkey);
    const BasisKey okey(eOrtho_A, nm, pkey);

    auto exp   = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(bkey);
    auto ortho = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(okey);

    std::vector<int> degree(nm);
    for (int p = 0; p < nm; ++p)
    {
        degree[p] = p;
    }
    CheckReduceOrderCoeffs(exp, ortho, degree, nm - 1);
    CheckReduceOrderCoeffs(exp, ortho, degree, nm - 2);
}

void CheckQuad(const int nm0, const int nm1, const int numMin)
{
    using namespace LibUtilities;
    const PointsKey pkey0(nm0 + 1, eGaussLobattoLegendre);
    const PointsKey pkey1(nm1 + 1, eGaussLobattoLegendre);
    const BasisKey b0(eModified_A, nm0, pkey0), b1(eModified_A, nm1, pkey1);
    const BasisKey o0(eOrtho_A, nm0, pkey0), o1(eOrtho_A, nm1, pkey1);

    auto exp = MemoryManager<StdRegions::StdQuadExp>::AllocateSharedPtr(b0, b1);
    auto ortho =
        MemoryManager<StdRegions::StdQuadExp>::AllocateSharedPtr(o0, o1);

    // Mode (p, q) is stored with p fastest; its degree is the larger index.
    std::vector<int> degree(nm0 * nm1);
    for (int q = 0; q < nm1; ++q)
    {
        for (int p = 0; p < nm0; ++p)
        {
            degree[q * nm0 + p] = std::max(p, q);
        }
    }
    CheckReduceOrderCoeffs(exp, ortho, degree, numMin);
}

BOOST_AUTO_TEST_CASE(TestQuadReduceOrderCoeffs)
{
    CheckQuad(4, 4, 3);
    CheckQuad(4, 4, 2);
    // Anisotropic: numMin above the smaller mode count, and below it.
    CheckQuad(4, 6, 5);
    CheckQuad(6, 4, 3);
}

void CheckTri(const int nm, const int numMin)
{
    using namespace LibUtilities;
    const PointsKey pkey0(nm + 1, eGaussLobattoLegendre);
    const PointsKey pkey1(nm, eGaussRadauMAlpha1Beta0);
    const BasisKey b0(eModified_A, nm, pkey0), b1(eModified_B, nm, pkey1);
    const BasisKey o0(eOrtho_A, nm, pkey0), o1(eOrtho_B, nm, pkey1);

    auto exp = MemoryManager<StdRegions::StdTriExp>::AllocateSharedPtr(b0, b1);
    auto ortho =
        MemoryManager<StdRegions::StdTriExp>::AllocateSharedPtr(o0, o1);

    // Modes are stored with p outermost and q < nm - p; the degree is p + q.
    std::vector<int> degree;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm - p; ++q)
        {
            degree.push_back(p + q);
        }
    }
    CheckReduceOrderCoeffs(exp, ortho, degree, numMin);
}

BOOST_AUTO_TEST_CASE(TestTriReduceOrderCoeffs)
{
    CheckTri(4, 3);
    // More than one order removed.
    CheckTri(5, 3);
}

void CheckTet(const int nm, const int numMin)
{
    using namespace LibUtilities;
    const PointsKey pkey0(nm + 1, eGaussLobattoLegendre);
    const PointsKey pkey1(nm, eGaussRadauMAlpha1Beta0);
    const PointsKey pkey2(nm, eGaussRadauMAlpha2Beta0);
    const BasisKey b0(eModified_A, nm, pkey0), b1(eModified_B, nm, pkey1),
        b2(eModified_C, nm, pkey2);
    const BasisKey o0(eOrtho_A, nm, pkey0), o1(eOrtho_B, nm, pkey1),
        o2(eOrtho_C, nm, pkey2);

    auto exp =
        MemoryManager<StdRegions::StdTetExp>::AllocateSharedPtr(b0, b1, b2);
    auto ortho =
        MemoryManager<StdRegions::StdTetExp>::AllocateSharedPtr(o0, o1, o2);

    // Modes are stored with p outermost, then q < nm - p, then r < nm - p - q;
    // the degree is p + q + r.
    std::vector<int> degree;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm - p; ++q)
        {
            for (int r = 0; r < nm - p - q; ++r)
            {
                degree.push_back(p + q + r);
            }
        }
    }
    CheckReduceOrderCoeffs(exp, ortho, degree, numMin);
}

BOOST_AUTO_TEST_CASE(TestTetReduceOrderCoeffs)
{
    CheckTet(4, 3);
    CheckTet(5, 3);
}

BOOST_AUTO_TEST_CASE(TestPrismReduceOrderCoeffs)
{
    using namespace LibUtilities;
    const int nm = 4;
    const PointsKey pkey0(nm + 1, eGaussLobattoLegendre);
    const PointsKey pkey2(nm, eGaussRadauMAlpha1Beta0);
    const BasisKey b0(eModified_A, nm, pkey0), b2(eModified_B, nm, pkey2);
    const BasisKey o0(eOrtho_A, nm, pkey0), o2(eOrtho_B, nm, pkey2);

    auto exp =
        MemoryManager<StdRegions::StdPrismExp>::AllocateSharedPtr(b0, b0, b2);
    auto ortho =
        MemoryManager<StdRegions::StdPrismExp>::AllocateSharedPtr(o0, o0, o2);

    // Modes are stored with p outermost, then q, then r < nm - p; the degree
    // is the larger of q and the collapsed pair's total p + r.
    std::vector<int> degree;
    for (int p = 0; p < nm; ++p)
    {
        for (int q = 0; q < nm; ++q)
        {
            for (int r = 0; r < nm - p; ++r)
            {
                degree.push_back(std::max(q, p + r));
            }
        }
    }
    CheckReduceOrderCoeffs(exp, ortho, degree, nm - 1);
}

} // namespace Nektar::StdRegionsUnitTests
