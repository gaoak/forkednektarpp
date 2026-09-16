///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysDeriv.cpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestPhysDeriv

#include "TestPhysDeriv.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_PHYSDERIV(test_name, test, tol)                                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_PHYSDERIV3DH1(test_name, test, tol)                               \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure3DH1(16);                                                     \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuitePhysDeriv)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_PHYSDERIV(physderiv_seg, Seg, 2.0E-12)

TEST_PHYSDERIV(physderiv_seg_sem, SegSEM, 1.0E-12)

TEST_PHYSDERIV(physderiv_seg_3d, Seg3D, 1.0E-12)

TEST_PHYSDERIV(physderiv_quad, Quad, 1.0E-12)

TEST_PHYSDERIV(physderiv_quad_3d, Quad3D, 1.0E-12)

TEST_PHYSDERIV(physderiv_quad_sem, QuadSEM, 1.0E-12)

TEST_PHYSDERIV(physderiv_quad_varp, QuadVarP, 1.0E-12)

TEST_PHYSDERIV(physderiv_tri, Tri, 1.0E-12)

TEST_PHYSDERIV(physderiv_tri_3d, Tri3D, 1.0E-12)

TEST_PHYSDERIV(physderiv_tri_varp, TriVarP, 1.0E-12)

TEST_PHYSDERIV(physderiv_tri_nodal, TriNodal, 1.0E-12)

TEST_PHYSDERIV(physderiv_square_all_elements, SquareAllElements, 1.0E-12)

TEST_PHYSDERIV(physderiv_hex, Hex, 1.0E-12)

TEST_PHYSDERIV(physderiv_hex_sem, HexSEM, 1.0E-12)

TEST_PHYSDERIV(physderiv_hex_varp, HexVarP, 1.0E-12)

TEST_PHYSDERIV(physderiv_prism, Prism, 1.0E-12)

TEST_PHYSDERIV(physderiv_prism_varp, PrismVarP, 2.5E-12)

TEST_PHYSDERIV(physderiv_prism_nodal, PrismNodal, 2.5E-12)

TEST_PHYSDERIV(physderiv_pyr, Pyr, 1.0E-12)

TEST_PHYSDERIV(physderiv_pyr_varp, PyrVarP, 1.0E-12)

TEST_PHYSDERIV(physderiv_tet, Tet, 1.0E-12)

TEST_PHYSDERIV(physderiv_tet_varp, TetVarP, 2.0E-12)

TEST_PHYSDERIV(physderiv_tet_nodal, TetNodal, 2.0E-12)

TEST_PHYSDERIV(physderiv_cube_prism_hex, CubePrismHex, 1.0E-12)

TEST_PHYSDERIV(physderiv_cube_all_elements, CubeAllElements, 1.0E-12)
#if !defined(NEKTAR_ENABLE_DEVICE) || defined(NEKTAR_ENABLE_CUDA)
TEST_PHYSDERIV3DH1(physderiv_quad_3dh1, QuadFFT, 1.0E-10)

TEST_PHYSDERIV3DH1(physderiv_tri_3dh1, TriFFT, 1.0E-10)

TEST_PHYSDERIV3DH1(physderiv_square_all_elements_3dh1, SquareAllElementsFFT,
                   1.0E-10)
#endif
#endif // NEKTAR_ENABLE_DOUBLE_PRECISION

BOOST_AUTO_TEST_SUITE_END()
