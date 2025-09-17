///////////////////////////////////////////////////////////////////////////////
//
// File: test_math.cpp
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

#define BOOST_TEST_MODULE TestMath

#include "init_math.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestMath)

BOOST_FIXTURE_TEST_CASE(abskernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::abs();

    // Backend results
    math.abs(*fixt_in, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(negkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::neg();

    // Backend results
    math.neg(*fixt_in, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sqrtkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::sqrt();

    // Backend results
    math.abs(*fixt_in, *fixt_in2);
    math.sqrt(*fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(addkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::add();

    // Backend results
    math.add(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(subkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::sub();

    // Backend results
    math.sub(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(mulkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::mul();

    // Backend results
    math.mul(*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

/* BOOST_FIXTURE_TEST_CASE(divkernel, MathField)
{
    Configure();
    SetTestCase();

    // std results
    MathField::div();

    // Backend results
    math.div(,*fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}*/

BOOST_FIXTURE_TEST_CASE(daxpykernel, MathField)
{
    Configure();
    SetTestCase();
    double alpha = 1.5;

    // std results
    MathField::daxpy(alpha);

    // Backend results
    math.daxpy(alpha, *fixt_in, *fixt_in2, *fixt_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-14));
    }
}

BOOST_FIXTURE_TEST_CASE(sum, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::sum();

    // Backend results
    double h_out = math.reduceSum(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 1.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Sum = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(max, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::max();

    // Backend results
    double h_out = math.reduceMax(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(min, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::min();

    // Backend results
    double h_out = math.reduceMin(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(innerproduct, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::inner_product();

    // Backend results
    double h_out = math.ddot(*fixt_in, *fixt_in2);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(std::abs(h_out))
                  << " ddot = " << std::sqrt(std::abs(out)) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(l1norm, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::l1norm();

    // Backend results
    double h_out = math.l1norm(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " L1norm = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(l2norm, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::l2norm();

    // Backend results
    double h_out = math.l2norm(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(lpnorm, MathField)
{
    Configure();
    SetTestCase();

    for (unsigned int p = 1; p < 4; p++)
    {
        // std results
        double out = MathField::lpnorm(p);

        // Backend results
        double h_out = math.lpnorm(p, *fixt_in);

        // Check results
        BOOST_TEST(fabs(h_out - out) < 5.0E-09);
        boost::test_tools::output_test_stream output;
        {
            std::cout << "Backend = " << std::sqrt(h_out) << " L" << p
                      << "norm = " << std::sqrt(out) << std::endl;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(linfnorm, MathField)
{
    Configure();
    SetTestCase();

    // std results
    double out = MathField::linfnorm();

    // Backend results
    double h_out = math.linfnorm(*fixt_in);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(h_out)
                  << " Linfnorm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
