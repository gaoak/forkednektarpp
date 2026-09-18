///////////////////////////////////////////////////////////////////////////////
//
// File: TestHOAlignment.cpp
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
// Description: Tests of the high-order node alignment helpers.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/HOAlignment.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Nektar::NekMeshHOAlignmentUnitTest
{
using namespace Nektar::SpatialDomains;

/**
 * @brief Barycentric position of every node of a nodal tetrahedron.
 *
 * As whole numbers of 1/(np-1), so that they can be compared exactly. These
 * weights are what says where a node belongs once the vertices are relabelled,
 * so they are also what the test checks the alignment against.
 */
static std::vector<std::array<int, 4>> TetWeights(int np)
{
    LibUtilities::PointsKey pKey(np, LibUtilities::eNodalTetEvenlySpaced);
    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    const int nTot = np * (np + 1) * (np + 2) / 6;
    std::vector<std::array<int, 4>> w(nTot);
    for (int m = 0; m < nTot; ++m)
    {
        w[m][1] = (int)std::lround(0.5 * (1.0 + px[m]) * (np - 1));
        w[m][2] = (int)std::lround(0.5 * (1.0 + py[m]) * (np - 1));
        w[m][3] = (int)std::lround(0.5 * (1.0 + pz[m]) * (np - 1));
        w[m][0] = (np - 1) - w[m][1] - w[m][2] - w[m][3];
    }
    return w;
}

/**
 * @brief Where every node of a nodal prism sits.
 *
 * The first three are its position in the triangular cross-section and the
 * last is how far along the axis it is, again as whole numbers.
 */
static std::vector<std::array<int, 4>> PrismPlaces(int np)
{
    LibUtilities::PointsKey pKey(np, LibUtilities::eNodalPrismEvenlySpaced);
    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    const int nTot = np * (np + 1) / 2 * np;
    std::vector<std::array<int, 4>> w(nTot);
    for (int m = 0; m < nTot; ++m)
    {
        w[m][1] = (int)std::lround(0.5 * (1.0 + px[m]) * (np - 1));
        w[m][2] = (int)std::lround(0.5 * (1.0 + pz[m]) * (np - 1));
        w[m][0] = (np - 1) - w[m][1] - w[m][2];
        w[m][3] = (int)std::lround(0.5 * (1.0 + py[m]) * (np - 1));
    }
    return w;
}

/**
 * @brief Aligning a tetrahedron puts every node where its weights say.
 *
 * Checked for all 24 ways the four vertices can be relabelled, with the nodes
 * carrying their own index so that where each one ended up can be seen.
 */
BOOST_AUTO_TEST_CASE(TestHOTetrahedronAlign)
{
    for (int np = 2; np <= 6; ++np)
    {
        const auto w   = TetWeights(np);
        const int nTot = w.size();

        std::array<int, 4> perm = {0, 1, 2, 3};
        do
        {
            std::vector<int> stored = {10, 11, 12, 13}, want(4), nodes(nTot);
            for (int i = 0; i < 4; ++i)
            {
                want[i] = stored[perm[i]];
            }
            for (int m = 0; m < nTot; ++m)
            {
                nodes[m] = m;
            }

            HOTetrahedron<int> hoTet(stored, nodes, np);
            hoTet.Align(want);

            for (int m = 0; m < nTot; ++m)
            {
                std::array<int, 4> have{};
                for (int i = 0; i < 4; ++i)
                {
                    have[perm[i]] = w[m][i];
                }

                const int expected =
                    std::find(w.begin(), w.end(), have) - w.begin();

                BOOST_CHECK_MESSAGE(hoTet.volVerts[m] == expected,
                                    "Tetrahedron of "
                                        << nTot << " nodes, slot " << m
                                        << ": got " << hoTet.volVerts[m]
                                        << ", expected " << expected);
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
    }
}

/**
 * @brief The interior of a tetrahedron aligns on its own too.
 *
 * Given only the interior the order cannot be worked out from the number of
 * nodes -- the interior of one order has as many as the whole of another -- so
 * this is the case where it is deduced rather than given.
 */
BOOST_AUTO_TEST_CASE(TestHOTetrahedronInteriorAlign)
{
    for (int np = 5; np <= 7; ++np)
    {
        const auto w    = TetWeights(np);
        const int nTot  = w.size();
        const int nInt  = (np - 4) * (np - 3) * (np - 2) / 6;
        const int first = nTot - nInt;

        std::array<int, 4> perm = {0, 1, 2, 3};
        do
        {
            std::vector<int> stored = {10, 11, 12, 13}, want(4), nodes(nInt);
            for (int i = 0; i < 4; ++i)
            {
                want[i] = stored[perm[i]];
            }
            for (int s = 0; s < nInt; ++s)
            {
                nodes[s] = s;
            }

            // No point count given: the order is deduced from the interior.
            HOTetrahedron<int> hoTet(stored, nodes);
            hoTet.Align(want);

            for (int s = 0; s < nInt; ++s)
            {
                std::array<int, 4> have{};
                for (int i = 0; i < 4; ++i)
                {
                    have[perm[i]] = w[first + s][i];
                }

                const int expected =
                    std::find(w.begin() + first, w.end(), have) - w.begin() -
                    first;

                BOOST_CHECK_MESSAGE(hoTet.volVerts[s] == expected,
                                    "Tetrahedron interior of "
                                        << nInt << " nodes, slot " << s
                                        << ": got " << hoTet.volVerts[s]
                                        << ", expected " << expected);
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
    }
}

/**
 * @brief Turning a prism moves every node round the triangle it sits in.
 */
BOOST_AUTO_TEST_CASE(TestHOPrismRotate)
{
    for (int np = 2; np <= 6; ++np)
    {
        const auto w   = PrismPlaces(np);
        const int nTot = w.size();

        for (int nrot = 0; nrot < 3; ++nrot)
        {
            std::vector<int> ids = {10, 11, 12, 13, 14, 15}, nodes(nTot);
            for (int m = 0; m < nTot; ++m)
            {
                nodes[m] = m;
            }

            HOPrism<int> hoPrism(ids, nodes, np);
            hoPrism.Rotate(nrot);

            for (int m = 0; m < nTot; ++m)
            {
                std::array<int, 4> have{};
                for (int c = 0; c < 3; ++c)
                {
                    have[(c + nrot) % 3] = w[m][c];
                }
                have[3] = w[m][3];

                const int expected =
                    std::find(w.begin(), w.end(), have) - w.begin();

                BOOST_CHECK_MESSAGE(hoPrism.volVerts[m] == expected,
                                    "Prism of "
                                        << nTot << " nodes, turned " << nrot
                                        << ", slot " << m << ": got "
                                        << hoPrism.volVerts[m] << ", expected "
                                        << expected);
            }
        }
    }
}

/**
 * @brief Turning a prism three times leaves it as it was.
 */
BOOST_AUTO_TEST_CASE(TestHOPrismThreeTurnsIsIdentity)
{
    for (int np = 2; np <= 6; ++np)
    {
        const int nTot = np * (np + 1) / 2 * np;

        std::vector<int> ids = {10, 11, 12, 13, 14, 15}, nodes(nTot);
        for (int m = 0; m < nTot; ++m)
        {
            nodes[m] = m;
        }

        HOPrism<int> hoPrism(ids, nodes, np);
        hoPrism.Rotate(1);
        hoPrism.Rotate(1);
        hoPrism.Rotate(1);

        for (int m = 0; m < nTot; ++m)
        {
            BOOST_CHECK_EQUAL(hoPrism.volVerts[m], m);
        }
    }
}

} // namespace Nektar::NekMeshHOAlignmentUnitTest
