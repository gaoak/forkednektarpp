///////////////////////////////////////////////////////////////////////////////
//
// File: TestCurlCurl.cpp
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

#define BOOST_TEST_MODULE TestCurlCurl

#include "TestCurlCurl.hpp"

#include <algorithm>
#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_CURLCURL(test_name, test, tol)                                    \
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

#define TEST_CURLCURL3DH1(test_name, test, tol)                                \
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

BOOST_AUTO_TEST_SUITE(TestSuiteCurlCurl)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

// Curl Curl operator is not defined for 1D
// TEST_CURLCURL(curlcurl_seg, Seg, 5E-12)
//
// TEST_CURLCURL(curlcurl_seg_sem, SegSEM, 1.0E-12)

TEST_CURLCURL(curlcurl_quad, Quad, 3.0E-11)

TEST_CURLCURL(curlcurl_quad_sem, QuadSEM, 6.0E-12)

TEST_CURLCURL(curlcurl_quad_varp, QuadVarP, 1.0E-10)

TEST_CURLCURL(curlcurl_tri, Tri, 2.0E-10)

TEST_CURLCURL(curlcurl_tri_varp, TriVarP, 6.0E-10)

TEST_CURLCURL(curlcurl_tri_nodal, TriNodal, 2.0E-10)

TEST_CURLCURL(curlcurl_square_all_elements, SquareAllElements, 2.0E-10)

TEST_CURLCURL(curlcurl_hex, Hex, 4.0E-11)

TEST_CURLCURL(curlcurl_hex_sem, HexSEM, 4.0E-12)

TEST_CURLCURL(curlcurl_hex_varp, HexVarP, 1.0E-10)

TEST_CURLCURL(curlcurl_prism, Prism, 2.0E-09)

TEST_CURLCURL(curlcurl_prism_varp, PrismVarP, 1.0E-08)

TEST_CURLCURL(curlcurl_prism_nodal, PrismNodal, 1.0E-10)

TEST_CURLCURL(curlcurl_pyr, Pyr, 1.0E-09)

TEST_CURLCURL(curlcurl_pyr_varp, PyrVarP, 5.0E-09)

TEST_CURLCURL(curlcurl_tet, Tet, 3.0E-09)

TEST_CURLCURL(curlcurl_tet_varp, TetVarP, 3.0E-09)

TEST_CURLCURL(curlcurl_tet_nodal, TetNodal, 3.0E-09)

TEST_CURLCURL(curlcurl_cube_prism_hex, CubePrismHex, 4.0E-10)

TEST_CURLCURL(curlcurl_cube_all_elements, CubeAllElements, 2.0E-09)

TEST_CURLCURL3DH1(curlcurl_quad_3dh1, QuadFFT, 1.0E-09)

TEST_CURLCURL3DH1(curlcurl_tri_3dh1, TriFFT, 3.0E-09)

TEST_CURLCURL3DH1(curlcurl_square_all_elements_3dh1, SquareAllElementsFFT,
                  1.0E-08)

#endif

BOOST_AUTO_TEST_SUITE_END()
