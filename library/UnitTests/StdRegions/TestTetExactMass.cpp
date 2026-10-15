///////////////////////////////////////////////////////////////////////////////
//
// File: TestTetExactMass.cpp
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
// Description: The P3 tetrahedron mass matrix on P+1 points per direction
// with the Gauss-Jacobi(1,0) rule (eGaussGaussAlpha1Beta0) in the collapsed
// directions is exact, where Gauss-Legendre in every direction is not.
//
///////////////////////////////////////////////////////////////////////////////

#include <StdRegions/StdTetExp.h>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>

namespace Nektar::StdRegionsUnitTests
{

using namespace LibUtilities;

/// The P3 orthogonal tet on @p np points per direction of the given types.
static StdRegions::StdTetExpSharedPtr MakeTet(const unsigned int np,
                                              const PointsType p0,
                                              const PointsType p1,
                                              const PointsType p2)
{
    const unsigned int nm = 4;
    const BasisKey b0(eOrtho_A, nm, PointsKey(np, p0));
    const BasisKey b1(eOrtho_B, nm, PointsKey(np, p1));
    const BasisKey b2(eOrtho_C, nm, PointsKey(np, p2));
    return MemoryManager<StdRegions::StdTetExp>::AllocateSharedPtr(b0, b1, b2);
}

/// The largest entry of the difference between the mass matrices of @p a
/// and @p b.
static NekDouble MassDifference(const StdRegions::StdTetExpSharedPtr &a,
                                const StdRegions::StdTetExpSharedPtr &b)
{
    const DNekMatSharedPtr ma = a->GetStdMatrix(
        StdRegions::StdMatrixKey(StdRegions::eMass, eTetrahedron, *a));
    const DNekMatSharedPtr mb = b->GetStdMatrix(
        StdRegions::StdMatrixKey(StdRegions::eMass, eTetrahedron, *b));
    BOOST_REQUIRE_EQUAL(ma->GetRows(), mb->GetRows());
    BOOST_REQUIRE_EQUAL(ma->GetColumns(), mb->GetColumns());

    NekDouble diff = 0.0;
    for (unsigned int i = 0; i < ma->GetRows(); ++i)
    {
        for (unsigned int j = 0; j < ma->GetColumns(); ++j)
        {
            diff = std::max(diff,
                            std::abs(ma->GetValue(i, j) - mb->GetValue(i, j)));
        }
    }
    return diff;
}

BOOST_AUTO_TEST_CASE(TestTetExactMassRules)
{
    // Reference: the default collapsed rules on twice the points, exact
    // well beyond the degree 2P = 6 the P3 mass matrix needs.
    const auto ref = MakeTet(8, eGaussLobattoLegendre, eGaussRadauMAlpha1Beta0,
                             eGaussRadauMAlpha2Beta0);

    // Rule D: Gauss-Legendre in the first two directions, Gauss-Jacobi(1,0)
    // in the third.
    const auto ruleD = MakeTet(4, eGaussGaussLegendre, eGaussGaussLegendre,
                               eGaussGaussAlpha1Beta0);
    BOOST_CHECK_SMALL(MassDifference(ruleD, ref), 1.0e-13);

    // Rule C: Gauss-Jacobi(1,0) in the second and third directions.
    const auto ruleC = MakeTet(4, eGaussGaussLegendre, eGaussGaussAlpha1Beta0,
                               eGaussGaussAlpha1Beta0);
    BOOST_CHECK_SMALL(MassDifference(ruleC, ref), 1.0e-13);

    // Gauss-Legendre in every direction is exact to degree 5 only, so the
    // top modes' mass is wrong: the reason for the rules above.
    const auto gl = MakeTet(4, eGaussGaussLegendre, eGaussGaussLegendre,
                            eGaussGaussLegendre);
    BOOST_CHECK(MassDifference(gl, ref) > 1.0e-3);
}

} // namespace Nektar::StdRegionsUnitTests
