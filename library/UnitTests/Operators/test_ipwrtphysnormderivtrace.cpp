///////////////////////////////////////////////////////////////////////////////
//
// File: test_ipwrtphysnormderivtrace.cpp
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
// Description: Unit tests for the IProductWRTPhysNormalDerivTrace operator.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestIpwrtPhysNormDerivTrace

#include "init_ipwrtphysnormderivtrace.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_IPWRTNORMDERIVTRACE_DIVTEST(test_name, test, tol)                 \
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

// The vector-input lift, sum_k <dphi/dx_k, g_k>, validated against the
// scalar operator through the identity g_k = n_k g. See
// RunVectorIdentityCase().
#define TEST_IPWRTNORMDERIVTRACE_VECTOR(test_name, test, tol)                  \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        bool divtest = true;                                                   \
        Configure(divtest);                                                    \
        SetTestCaseDivTest();                                                  \
        RunVectorIdentityCase(tol);                                            \
    }

BOOST_AUTO_TEST_SUITE(TestIpwrtPhysNormDerivTrace)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_divtest, QuadOrtho,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_divtest_gauss,
                                 QuadOrthoGauss, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tri_divtest, TriOrtho,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_affine, QuadAffine,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_affine_curved,
                                 QuadAffineCurved, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_affine_gauss,
                                 QuadAffineGauss, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_quad_affine_curved_gauss,
                                 QuadAffineCurvedGauss, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_hex_orient_varq,
                                 HexOrientVarQ, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_hex_affine, HexAffine,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_hex_affine_gauss,
                                 HexAffineGauss, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_hex_divtest, Hex, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_prism_divtest, Prism,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_pyr_divtest, Pyr, 1.0E-11)

// Disabled: run/tet_p1.xml declares NUMMODES="1,1,1", and a tet needs more
// than one mode per direction, so GetNcoeffs() trips "Order in 'a'
// direction must be > 1" in a FULLDEBUG build.
// TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tet_p1, TetP1,
//                                  1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_seg_divtest, Seg, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_seg_curved_affine_divtest,
                                 SegCurvedAffine, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_seg_curved_divtest,
                                 SegCurved, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tet_p2, TetP2, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tet_p2_gauss, TetP2Gauss,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tet_divtest, TetOrtho,
                                 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_DIVTEST(ipwrtNormDerivTrace_tet_divtest_gauss,
                                 TetOrthoGauss, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_quad_affine, QuadAffine,
                                1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_quad_affine_curved,
                                QuadAffineCurved, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_tri, TriOrtho, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_hex_orient_varq,
                                HexOrientVarQ, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_prism, Prism, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_pyr, Pyr, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_tet_p2, TetP2, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_seg, Seg, 1.0E-11)

TEST_IPWRTNORMDERIVTRACE_VECTOR(ipwrtVecDerivTrace_seg_curved, SegCurved,
                                1.0E-11)

#endif

BOOST_AUTO_TEST_SUITE_END()
