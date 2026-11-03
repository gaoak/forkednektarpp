///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysLiftProject.cpp
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
// Description: Unit test of the PhysLiftProject operator: equivalence with the
// four-operator chain for every shape, orders 1 to 8, square and non-square
// B, affine and deformed elements, with and without the scale and append.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestPhysLiftProject

#include <UnitTests/MultiRegions/TestPhysLiftProject.hpp>

#include <iostream>

// Orders 1 to 8, each with the expansion type's own points and with one
// point per mode or one more (as many points as modes makes B square on a
// quadrilateral or hexahedron, so the Pointwise form runs). The modified
// basis takes the standard inverse mass densely, the orthogonal one by its
// diagonal where the rule integrates it exactly.
#define TEST_PHYSLIFTPROJECT(test_name, test, tol)                             \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        for (const auto type :                                                 \
             {SpatialDomains::eModified, SpatialDomains::eOrthogonal})         \
        {                                                                      \
            for (int nmodes = 2; nmodes <= 9; ++nmodes)                        \
            {                                                                  \
                for (const int npts : {-1, nmodes, nmodes + 1})                \
                {                                                              \
                    const bool match = RunExpansion(type, nmodes, npts, tol);  \
                    if (!match)                                                \
                    {                                                          \
                        std::cout << "Failed: type " << type << ", modes "     \
                                  << nmodes << ", points " << npts             \
                                  << std::endl;                                \
                    }                                                          \
                    BOOST_TEST(match);                                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        for (const auto &form : m_forms)                                       \
        {                                                                      \
            std::cout << "  reached " << form << std::endl;                    \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuitePhysLiftProject)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_PHYSLIFTPROJECT(physliftproject_seg, Seg, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_seg_curved, SegCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_quad_tri, QuadTri, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_quad_tri_curved, QuadTriCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_hex, Hex, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_hex_curved, HexCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_prism, Prism, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_prism_curved, PrismCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_pyr, Pyr, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_pyr_curved, PyrCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_tet, Tet, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_tet_curved, TetCurved, 1.0E-09)

#endif

BOOST_AUTO_TEST_SUITE_END()
