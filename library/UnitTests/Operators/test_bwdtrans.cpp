///////////////////////////////////////////////////////////////////////////////
//
// File: test_bwdtrans.cpp
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

#define BOOST_TEST_MODULE TestBwdTrans
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorBwdTrans.hpp"
#include "init_bwdtransfields.hpp"

#define TEST_BWDTRANS(test_name, test, tol)                                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());  \
        BwdTrans<>::create(fixt_explist, "StdMat")                             \
            ->apply(*fixt_in, *fixt_out);                                      \
        ExpectedSolution(fixt_expected->GetBlocks(),                           \
                         fixt_expected->GetStorage().GetCPUPtr());             \
        BOOST_TEST(fixt_out->compare(*fixt_expected, tol));                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),               \
                             fixt_expected->GetStorage().GetCPUPtr(), tol);    \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestBwdTrans)

TEST_BWDTRANS(bwdtrans_seg, Seg, 1.0E-12)

TEST_BWDTRANS(bwdtrans_quad, Quad, 1.0E-12)

TEST_BWDTRANS(bwdtrans_tri, Tri, 1.0E-12)

TEST_BWDTRANS(bwdtrans_square_all_elements, SquareAllElements, 1.0E-12)

TEST_BWDTRANS(bwdtrans_hex, Hex, 1.0E-12)

TEST_BWDTRANS(bwdtrans_prism, Prism, 1.0E-12)

TEST_BWDTRANS(bwdtrans_pyr, Pyr, 1.0E-12)

TEST_BWDTRANS(bwdtrans_tet, Tet, 1.0E-12)

TEST_BWDTRANS(bwdtrans_cube_prism_hex, CubePrismHex, 1.0E-12)

TEST_BWDTRANS(bwdtrans_cube_all_elements, CubeAllElements, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
