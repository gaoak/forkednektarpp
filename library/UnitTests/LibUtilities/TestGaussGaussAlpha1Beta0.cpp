///////////////////////////////////////////////////////////////////////////////
//
// File: TestGaussGaussAlpha1Beta0.cpp
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
// Description: Unit tests for the eGaussGaussAlpha1Beta0 points: the
// Gauss-Jacobi rule with weight (1 - x), its derivative and its
// interpolation matrices.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/Points.h>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>

namespace Nektar::GaussGaussAlpha1Beta0UnitTests
{

using namespace LibUtilities;

/// The exact value of the integral of (1 - x) x^k over [-1, 1].
static NekDouble ExactMoment(const unsigned int k)
{
    const auto mono = [](const unsigned int m) {
        return (m % 2 == 0) ? 2.0 / (m + 1.0) : 0.0;
    };
    return mono(k) - mono(k + 1);
}

/**
 * @brief The n-point rule integrates (1 - x) x^k exactly for k up to 2n - 1,
 * as a Gauss rule for that weight must, and not for k = 2n; its points are
 * interior and ascending.
 */
BOOST_AUTO_TEST_CASE(TestGaussGaussAlpha1Beta0Quadrature)
{
    for (unsigned int n = 2; n <= 8; ++n)
    {
        const PointsKey key(n, eGaussGaussAlpha1Beta0);
        Array<OneD, const NekDouble> z, w;
        PointsManager()[key]->GetZW(z, w);
        BOOST_REQUIRE_EQUAL(z.size(), n);

        for (unsigned int i = 0; i < n; ++i)
        {
            BOOST_CHECK(z[i] > -1.0 && z[i] < 1.0);
            if (i > 0)
            {
                BOOST_CHECK(z[i] > z[i - 1]);
            }
        }

        for (unsigned int k = 0; k <= 2 * n; ++k)
        {
            NekDouble sum = 0.0;
            for (unsigned int i = 0; i < n; ++i)
            {
                sum += w[i] * std::pow(z[i], k);
            }
            const NekDouble err = std::abs(sum - ExactMoment(k));
            if (k <= 2 * n - 1)
            {
                BOOST_CHECK_SMALL(err, 1.0e-14);
            }
            else
            {
                BOOST_CHECK(err > 1.0e-6);
            }
        }
    }
}

/**
 * @brief The derivative matrix differentiates, and the interpolation
 * matrix to Gauss-Lobatto-Legendre points interpolates, every polynomial
 * of degree up to n - 1 exactly.
 */
BOOST_AUTO_TEST_CASE(TestGaussGaussAlpha1Beta0DerivAndInterp)
{
    for (unsigned int n = 2; n <= 8; ++n)
    {
        const PointsKey key(n, eGaussGaussAlpha1Beta0);
        const auto &pts = PointsManager()[key];
        Array<OneD, const NekDouble> z;
        pts->GetPoints(z);

        const PointsKey toKey(10, eGaussLobattoLegendre);
        Array<OneD, const NekDouble> zTo;
        PointsManager()[toKey]->GetPoints(zTo);

        const auto D = pts->GetD();
        const auto I = pts->GetI(toKey);
        BOOST_REQUIRE_EQUAL(I->GetRows(), 10u);
        BOOST_REQUIRE_EQUAL(I->GetColumns(), n);

        for (unsigned int k = 0; k < n; ++k)
        {
            for (unsigned int i = 0; i < n; ++i)
            {
                NekDouble d = 0.0;
                for (unsigned int j = 0; j < n; ++j)
                {
                    d += D->GetValue(i, j) * std::pow(z[j], k);
                }
                const NekDouble exact =
                    k == 0 ? 0.0 : k * std::pow(z[i], k - 1);
                BOOST_CHECK_SMALL(d - exact, 1.0e-11);
            }

            for (unsigned int i = 0; i < 10; ++i)
            {
                NekDouble v = 0.0;
                for (unsigned int j = 0; j < n; ++j)
                {
                    v += I->GetValue(i, j) * std::pow(z[j], k);
                }
                BOOST_CHECK_SMALL(v - std::pow(zTo[i], k), 1.0e-12);
            }
        }
    }
}

} // namespace Nektar::GaussGaussAlpha1Beta0UnitTests
