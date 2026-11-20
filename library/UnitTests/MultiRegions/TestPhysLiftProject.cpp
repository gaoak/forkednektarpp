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

// Hexahedra with as many points as modes, on Gauss-Legendre points (the
// orthogonal basis) and on Gauss-Lobatto-Legendre points (the modified
// basis), affine and deformed: the Device SumFac and SumFacTOP
// implementations lift the trace directly at every point.
BOOST_FIXTURE_TEST_CASE(physliftproject_hex_direct_lift, Hex)
{
    std::cout << "Run: physliftproject_hex_direct_lift" << std::endl;
    for (const auto type :
         {SpatialDomains::eOrthogonal, SpatialDomains::eModified})
    {
        for (int nmodes = 2; nmodes <= 5; ++nmodes)
        {
            BOOST_TEST(RunExpansion(type, nmodes, nmodes, 1.0E-09));
        }
    }
    BOOST_TEST(m_forms.count("Pointwise/affine/direct lift") == 1u);
    BOOST_TEST(m_forms.count("Pointwise/deformed/direct lift") == 1u);
}

// One mode on one point of the orthogonal basis, which makes every shape
// Pointwise: the Device SumFac and SumFacTOP implementations lift the trace
// directly, the collapsed shapes each trace's integral onto the one point.
// The triangles run on a triangle-only mesh: QuadExp::v_ComputeTraceNormal
// reads past an array on a one-point quadrilateral that is not a
// parallelogram, as the quadrilaterals of the mixed meshes are.
#define TEST_PHYSLIFTPROJECT_ONE_POINT(test_name, test, tol, reached)          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        for (const int npts : {-1, 1})                                         \
        {                                                                      \
            BOOST_TEST(                                                        \
                RunExpansion(SpatialDomains::eOrthogonal, 1, npts, tol));      \
        }                                                                      \
        for (const auto &form : m_forms)                                       \
        {                                                                      \
            std::cout << "  reached " << form << std::endl;                    \
        }                                                                      \
        BOOST_TEST(m_forms.count(reached) == 1u);                              \
    }

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_tri, Tri, 1.0E-09,
                               "Pointwise/affine/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_tri_curved, TriCurved,
                               1.0E-09, "Pointwise/deformed/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_prism, Prism, 1.0E-09,
                               "Pointwise/affine/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_prism_curved,
                               PrismCurved, 1.0E-09,
                               "Pointwise/deformed/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_pyr, Pyr, 1.0E-09,
                               "Pointwise/affine/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_pyr_curved, PyrCurved,
                               1.0E-09, "Pointwise/deformed/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_tet, Tet, 1.0E-09,
                               "Pointwise/affine/direct lift")

TEST_PHYSLIFTPROJECT_ONE_POINT(physliftproject_one_point_tet_curved, TetCurved,
                               1.0E-09, "Pointwise/deformed/direct lift")

// Tetrahedra, prisms and pyramids on the default rules of the orthogonal
// basis (on a tetrahedron, Gauss-Jacobi(1,0) in direction 2, so face 0's
// points are not the volume's in its second in-face direction): the Device
// SumFacTOP implementation applies the StdInvMass form through the dense
// projector up to 125 points, and through the fused kernels above.
#define TEST_PHYSLIFTPROJECT_DENSE_TAIL(test_name, test, tol)                  \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        for (int nmodes = 2; nmodes <= 6; ++nmodes)                            \
        {                                                                      \
            BOOST_TEST(                                                        \
                RunExpansion(SpatialDomains::eOrthogonal, nmodes, -1, tol));   \
        }                                                                      \
        BOOST_TEST(m_forms.count("StdInvMass/affine/dense tail") == 1u);       \
        BOOST_TEST(m_forms.count("StdInvMass/affine") == 1u);                  \
    }

TEST_PHYSLIFTPROJECT_DENSE_TAIL(physliftproject_tet_dense_tail, Tet, 1.0E-09)

TEST_PHYSLIFTPROJECT_DENSE_TAIL(physliftproject_prism_dense_tail, Prism,
                                1.0E-09)

TEST_PHYSLIFTPROJECT_DENSE_TAIL(physliftproject_pyr_dense_tail, Pyr, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_prism, Prism, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_prism_curved, PrismCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_pyr, Pyr, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_pyr_curved, PyrCurved, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_tet, Tet, 1.0E-09)

TEST_PHYSLIFTPROJECT(physliftproject_tet_curved, TetCurved, 1.0E-09)

#endif

BOOST_AUTO_TEST_SUITE_END()
