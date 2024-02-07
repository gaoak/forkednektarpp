///////////////////////////////////////////////////////////////////////////////
//
// File: test_bwdtrans_matfree.cpp
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestBwdTransMatrixFree
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorBwdTrans.hpp"
#include "init_bwdtransfields.hpp"

BOOST_AUTO_TEST_SUITE(TestBwdTransMatFree)

BOOST_FIXTURE_TEST_CASE(bwdtrans_seg, Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_quad, Quad)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_tri, Tri)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_square_all_elements, SquareAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_hex, Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_prism, Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_pyr, Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_tet, Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_cube_prism_hex, CubePrismHex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(bwdtrans_cube_all_elements, CubeAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();
    // compare
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()
