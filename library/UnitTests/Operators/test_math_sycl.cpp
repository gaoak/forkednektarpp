///////////////////////////////////////////////////////////////////////////////
//
// File: test_math_sycl.cpp
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

#define BOOST_TEST_MODULE TestMathSYCL

#include "MathKernelsLauncher.hpp"
#include "init_mathkernels.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestMathSYCL)

BOOST_FIXTURE_TEST_CASE(sycl_negkernel, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    neg();

    // SYCL results
    negKernelLauncher(*fixt_in, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_addkernel, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    add();

    // SYCL results
    addKernelLauncher(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_subkernel, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    sub();

    // SYCL results
    subKernelLauncher(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_daxpykernel, MathKernels)
{
    Configure();
    SetTestCase();
    double alpha = 1.5;

    // std results
    daxpy(alpha);

    // SYCL results
    daxpyKernelLauncher(alpha, *fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-14));
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_divkernel, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    div();

    // SYCL results
    divKernelLauncher(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_sum, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = sum();

    // SYCL results
    auto h_out = sumKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << h_out << " Sum = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_max, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = max();

    // SYCL results
    auto h_out = maxKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_min, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = min();

    // SYCL results
    auto h_out = minKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_innerproduct, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = inner_product();

    // SCYL results
    auto h_out = innerproductKernelLauncher(*fixt_in, *fixt_in2);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << std::sqrt(std::abs(h_out))
                  << " ddot = " << std::sqrt(std::abs(out)) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_l1norm, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = l1norm();

    // SYCL results
    auto h_out = l1normKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << h_out << " L1norm = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_l2norm, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = l2norm();

    // SYCL results
    auto h_out = l2normKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_lpnorm, MathKernels)
{
    Configure();
    SetTestCase();

    for (unsigned int p = 1; p < 4; p++)
    {
        // std results
        auto out = lpnorm(p);

        // SYCL results
        auto h_out = lpnormKernelLauncher(p, *fixt_in);

        // Check results
        BOOST_TEST(fabs(h_out - out) < 5.0E-10);
        boost::test_tools::output_test_stream output;
        {
            std::cout << "SYCL = " << std::sqrt(h_out) << " L" << p
                      << "norm = " << std::sqrt(out) << std::endl;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(sycl_linfnorm, MathKernels)
{
    Configure();
    SetTestCase();

    // std results
    auto out = linfnorm();

    // SYCL results
    auto h_out = linfnormKernelLauncher(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "SYCL = " << std::sqrt(h_out)
                  << " Linfnorm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
