///////////////////////////////////////////////////////////////////////////////
//
// File: test_esdirk.cpp
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

#define BOOST_TEST_MODULE TestESDIRK

#include "init_esdirk.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_ESDIRK(test_name, test, nvar, order)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure(nvar, nvar);                                                 \
        SetTestCase(-10.0);                                                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(CheckOrderOfAccuracy("DIRK_ES", order));                \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestESDIRK)

TEST_ESDIRK(esdirk_order_2, segment_order_2, 2, 2)

TEST_ESDIRK(esdirk_order_3, segment_order_3, 2, 3)

TEST_ESDIRK(esdirk_order_4, segment_order_4, 2, 4)

BOOST_AUTO_TEST_SUITE_END()
