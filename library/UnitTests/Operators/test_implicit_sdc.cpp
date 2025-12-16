///////////////////////////////////////////////////////////////////////////////
//
// File: test_implicit_sdc.cpp
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

#define BOOST_TEST_MODULE TestImplicitSDC

#include "init_timeop.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_SCHEME(test_name, test, scheme, variant, order, freeparam)        \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase(0.0, -10.0);                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(                                                        \
                CheckOrderOfAccuracy(scheme, variant, order, freeparam));      \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestImplicitSDC)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_SCHEME(sdc_order_1, segment, "ImplicitSDC", "Equidistant", 1,
            (std::vector<double>{1.0, 1}))

TEST_SCHEME(sdc_order_2, segment, "ImplicitSDC", "Equidistant", 2,
            (std::vector<double>{1.0, 2}))

TEST_SCHEME(sdc_order_3, segment, "ImplicitSDC", "Equidistant", 3,
            (std::vector<double>{1.0, 3}))

TEST_SCHEME(sdc_order_4, segment, "ImplicitSDC", "Equidistant", 4,
            (std::vector<double>{1.0, 4}))

TEST_SCHEME(sdc_order_5, segment, "ImplicitSDC", "Equidistant", 5,
            (std::vector<double>{1.0, 5}))

TEST_SCHEME(sdc_gll_order_2, segment, "ImplicitSDC", "GaussLobattoLegendre", 2,
            (std::vector<double>{1.0, 2}))

TEST_SCHEME(sdc_gll_order_4, segment, "ImplicitSDC", "GaussLobattoLegendre", 4,
            (std::vector<double>{1.0, 3}))

TEST_SCHEME(sdc_gll_order_6, segment, "ImplicitSDC", "GaussLobattoLegendre", 6,
            (std::vector<double>{1.0, 4}))

TEST_SCHEME(sdc_grl_order_3, segment, "ImplicitSDC", "GaussRadauLegendre", 3,
            (std::vector<double>{1.0, 2}))

TEST_SCHEME(sdc_grl_order_5, segment, "ImplicitSDC", "GaussRadauLegendre", 5,
            (std::vector<double>{1.0, 3}))

TEST_SCHEME(sdc_ggl_order_2, segment, "ImplicitSDC", "GaussGaussLegendre", 2,
            (std::vector<double>{1.0, 1}))

TEST_SCHEME(sdc_ggl_order_4, segment, "ImplicitSDC", "GaussGaussLegendre", 4,
            (std::vector<double>{1.0, 2}))

TEST_SCHEME(sdc_ggl_order_6, segment, "ImplicitSDC", "GaussGaussLegendre", 6,
            (std::vector<double>{1.0, 3}))
#endif

BOOST_AUTO_TEST_SUITE_END()
