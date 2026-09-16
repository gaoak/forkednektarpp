///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvectionDealias.cpp
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

#define BOOST_TEST_MODULE TestAdvectionDealias

#include "TestAdvectionDealias.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>

#define TEST_ADVECTIONDEALIAS(test_name, test, tol)                            \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            RunTestCase(false);                                                \
            BOOST_TEST(Compare(tol));                                          \
            RunTestCase(true);                                                 \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuiteAdvectionDealias)

// Same fixture set as test_advection.cpp's TestVGradU, excluding its disabled
// embedded-coordinate cases. Each fixture tests overwrite and append modes.
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_ADVECTIONDEALIAS(advectiondealias_seg, Seg, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_seg_sem, SegSEM, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_quad, Quad, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_quad_deformed, QuadDeformed, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_quad_sem, QuadSEM, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_quad_varp, QuadVarP, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_tri, Tri, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_tri_varp, TriVarP, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_tri_nodal, TriNodal, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_square_all_elements, SquareAllElements,
                      1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_hex, Hex, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_hex_sem, HexSEM, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_hex_varp, HexVarP, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_prism, Prism, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_prism_varp, PrismVarP, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_prism_nodal, PrismNodal, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_pyr, Pyr, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_pyr_varp, PyrVarP, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_tet, Tet, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_tet_varp, TetVarP, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_tet_nodal, TetNodal, 1.0E-9)

TEST_ADVECTIONDEALIAS(advectiondealias_cube_prism_hex, CubePrismHex, 1.0E-10)

TEST_ADVECTIONDEALIAS(advectiondealias_cube_all_elements, CubeAllElements,
                      1.0E-9)
#endif

BOOST_AUTO_TEST_SUITE_END()
