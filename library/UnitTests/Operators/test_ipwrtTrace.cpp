///////////////////////////////////////////////////////////////////////////////
//
// File: test_ipwrtTrace.cpp
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

#define BOOST_TEST_MODULE TestIpwrtTrace

#include "init_ipwrtTrace.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>

#define TEST_IPWRTTRACE(test_name, test, tol, nonColl)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase(nonColl);                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
        RunTestCase(nonColl, true);                                            \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_IPWRTTRACE_DIVTEST(test_name, test, tol)                          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        bool divtest = true;                                                   \
        Configure(divtest);                                                    \
        SetTestCaseDivTest();                                                  \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_IPWRTTRACESINGLE(test_name, test, tol, nonColl)                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCaseSingle(nonColl);                                            \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
        RunTestCaseSingle(nonColl, true);                                      \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestIpwrtTrace)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_IPWRTTRACE(ipwrtTrace_seg, Seg, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_seg_gauss, SegGaussPts, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_seg_coll, Seg, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_quad, Quad, 1.0E-12, true)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_quad_divtest, QuadOrtho, 1.0E-12)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_quad_divtest_gauss, QuadOrthoGauss, 1.0E-12)

TEST_IPWRTTRACE(ipwrtTrace_quad_coll, Quad, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_quad_varp, QuadVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_quad_varp_coll, QuadVarP, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_quad_gauss, QuadGaussPts, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_tri, Tri, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_tri_coll, Tri, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_tri_varp, TriVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_tri_varp_coll, TriVarP, 1.0E-12, false)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_tri_divtest, TriOrtho, 1.0E-12)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_tri_divtest_gauss, TriOrthoGauss, 1.0E-12)

TEST_IPWRTTRACE(ipwrtTrace_square_all_elements, SquareAllElements, 1.0E-12,
                true)

TEST_IPWRTTRACE(ipwrtTrace_hex, Hex, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_hex_coll, Hex, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_hex_varp, HexVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_hex_varp_coll, HexVarP, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_prism, Prism, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_prism_coll, Prism, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_prism_varp, PrismVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_prism_varp_coll, PrismVarP, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_pyr, Pyr, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_pyr_colls, Pyr, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_pyr_varp, PyrVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_pyr_varp_colls, PyrVarP, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_tet, Tet, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_tet_coll, Tet, 1.0E-12, false)

TEST_IPWRTTRACE(ipwrtTrace_tet_varp, TetVarP, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTrace_tet_varp_coll, TetVarP, 1.0E-12, false)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_tet_divtest, TetOrtho, 1.0E-12)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_tet_divtest_gauss, TetOrthoGauss, 1.0E-12)

TEST_IPWRTTRACE_DIVTEST(ipwrtTrace_hex_affine_gauss, HexAffineGauss, 1.0E-12)

TEST_IPWRTTRACE(ipwrtTrace_cube_prism_hex, CubePrismHex, 1.0E-12, true)

TEST_IPWRTTRACE(ipwrtTracey_cube_all_elements, CubeAllElements, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_quad_single, Quad, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_quad_single_coll, Quad, 1.0E-12, false)

TEST_IPWRTTRACESINGLE(ipwrtTrace_hex_single, HexFixedP, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_tri_single, Tri, 1.0E-12, false)

TEST_IPWRTTRACESINGLE(ipwrtTrace_tri_single_coll, Tri, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_hex_single_coll, HexFixedP, 1.0E-12, false)

TEST_IPWRTTRACESINGLE(ipwrtTrace_prism_single, PrismFixedP, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_prism_single_coll, PrismFixedP, 1.0E-12, false)

TEST_IPWRTTRACESINGLE(ipwrtTrace_pyr_single, Pyr, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_pyr_single_coll, Pyr, 1.0E-12, false)

TEST_IPWRTTRACESINGLE(ipwrtTrace_tet_single, Tet, 1.0E-12, true)

TEST_IPWRTTRACESINGLE(ipwrtTrace_tet_single_coll, Tet, 1.0E-12, false)

#endif

BOOST_AUTO_TEST_SUITE_END()
