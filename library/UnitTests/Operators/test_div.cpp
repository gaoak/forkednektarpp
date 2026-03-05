///////////////////////////////////////////////////////////////////////////////
//
// File: test_div.cpp
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

#define BOOST_TEST_MODULE TestDiv

#include "init_divfields.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_DIV(test_name, test, tol)                                         \
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

BOOST_AUTO_TEST_SUITE(TestDiv)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_DIV(div_seg, Seg, 5E-12)

TEST_DIV(div_seg_sem, SegSEM, 1.0E-12)

TEST_DIV(div_quad, Quad, 1.0E-12)

TEST_DIV(div_quad_sem, QuadSEM, 1.0E-12)

TEST_DIV(div_quad_varp, QuadVarP, 1.0E-12)

TEST_DIV(div_tri, Tri, 1.5E-12)

TEST_DIV(div_tri_varp, TriVarP, 4.5E-12)

TEST_DIV(div_tri_nodal, TriNodal, 1.5E-12)

TEST_DIV(div_square_all_elements, SquareAllElements, 2.0E-11)

TEST_DIV(div_hex, Hex, 2.0E-12)

TEST_DIV(div_hex_sem, HexSEM, 1.0E-12)

TEST_DIV(div_hex_varp, HexVarP, 3.0E-12)

TEST_DIV(div_prism, Prism, 2.0E-12)

TEST_DIV(div_prism_varp, PrismVarP, 4.0E-11)

TEST_DIV(div_prism_nodal, PrismNodal, 2.5E-12)

TEST_DIV(div_pyr, Pyr, 9.0E-12)

TEST_DIV(div_pyr_varp, PyrVarP, 2.0E-11)

TEST_DIV(div_tet, Tet, 8.0E-12)

TEST_DIV(div_tet_varp, TetVarP, 8.0E-12)

TEST_DIV(div_tet_nodal, TetNodal, 8.0E-12)

TEST_DIV(div_cube_prism_hex, CubePrismHex, 2.5E-12)

TEST_DIV(div_cube_all_elements, CubeAllElements, 7.0E-12)

#endif

BOOST_AUTO_TEST_SUITE_END()
