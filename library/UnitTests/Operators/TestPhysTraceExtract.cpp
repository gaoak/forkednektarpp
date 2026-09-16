///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysTraceExtract.cpp
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

#define BOOST_TEST_MODULE TestPhysTraceExtract

#include "TestPhysTraceExtract.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>

#define TEST_PHYSTRACEEXTRACT(test_name, test, tol, nonColl)                   \
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
    }

#define TEST_PHYSTRACEEXTRACTTRACE(test_name, test, tol, nonColl)              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCaseExtractTrace(nonColl);                                      \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuitePhysTraceExtract)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_PHYSTRACEEXTRACT(phystraceext_seg_endpts, Seg, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_seg_endpts_gauss, SegGaussPts, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_endpts, Tri, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_endpts_coll, Tri, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_nodal_endpts, TriNodal, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_nodal_endpts_coll, TriNodal, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tri_nodal_endpts, TriNodal,
                           1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_varp_endpts, TriVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tri_varp_endpts_coll, TriVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tri_endpts, Tri, 1.0E-12, true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tri_endpts_coll, Tri, 1.0E-12,
                           false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_quad_endpts, Quad, 1.0E-12, true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_quad_endpts_coll, Quad, 1.0E-12,
                           false)

TEST_PHYSTRACEEXTRACT(phystraceext_quad_endpts_coll, Quad, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_quad_endpts_gauss, QuadGaussPts, 1.0E-12,
                      true)

TEST_PHYSTRACEEXTRACT(phystraceext_quad_varp_endpts, QuadVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_quad_varp_endpts_coll, QuadVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACT(phystraceext_square_all_elements_endpts,
                      SquareAllElements, 1.0E-12, true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tet_endpts, Tet, 1.0E-12, true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tet_endpts_coll, Tet, 1.0E-12,
                           false)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_endpts, Tet, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_varp_endpts, TetVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_endpts_coll, Tet, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_nodal_endpts, TetNodal, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_nodal_endpts_coll, TetNodal, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_tet_nodal_endpts, TetNodal,
                           1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_tet_varp_endpts_coll, TetVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_pyr_endpts, Pyr, 1.0E-12, true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_pyr_endpts_coll, Pyr, 1.0E-12,
                           false)

TEST_PHYSTRACEEXTRACT(phystraceext_pyr_endpts, Pyr, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_pyr_endpt_colls, Pyr, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_pyr_varp_endpts, PyrVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_pyr_varp_endpt_colls, PyrVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_prism_endpts, PrismFixedP, 1.0E-12,
                           true)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_prism_endpts_coll, PrismFixedP,
                           1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_prism_endpts, Prism, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_prism_endpts_coll, Prism, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_prism_varp_endpts, PrismVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_prism_varp_endpts_coll, PrismVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACTTRACE(phystraceexttrace_hex_endpts, HexFixedP, 1.0E-12,
                           true)

TEST_PHYSTRACEEXTRACT(phystraceext_hex_endpts, Hex, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_hex_endpts_coll, Hex, 1.0E-12, false)

TEST_PHYSTRACEEXTRACT(phystraceext_hex_varp_endpts, HexVarP, 1.0E-12, true)

TEST_PHYSTRACEEXTRACT(phystraceext_hex_varp_endpts_coll, HexVarP, 1.0E-12,
                      false)

TEST_PHYSTRACEEXTRACT(phystraceext_cube_prism_hex_endpts, CubePrismHex, 1.0E-12,
                      true)

TEST_PHYSTRACEEXTRACT(phystraceext_cube_all_elements_endpts, CubeAllElements,
                      1.0E-12, true)
#endif

BOOST_AUTO_TEST_SUITE_END()
