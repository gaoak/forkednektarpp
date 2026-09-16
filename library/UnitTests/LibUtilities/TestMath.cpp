///////////////////////////////////////////////////////////////////////////////
//
// File: TestMath.cpp
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

#include "TestMath.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteMath)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_FIXTURE_TEST_CASE(abskernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::abs();

    // Backend results
    m_math.abs(*m_in1, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(negkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::neg();

    // Backend results
    m_math.neg(*m_in1, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(sqrtkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::sqrt();

    // Backend results
    m_math.abs(*m_in1, *m_in2);
    m_math.sqrt(*m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(addkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::add();

    // Backend results
    m_math.add(*m_in1, *m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(subkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::sub();

    // Backend results
    m_math.sub(*m_in1, *m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(mulkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::mul();

    // Backend results
    m_math.mul(*m_in1, *m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}

/* BOOST_FIXTURE_TEST_CASE(divkernel, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    TestMath<double>::div();

    // Backend results
    m_math.div(,*m_in1, *m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-15));
    }
}*/

BOOST_FIXTURE_TEST_CASE(daxpykernel, TestMath<double>)
{
    Configure();
    SetTestCase();
    double alpha = 1.5;

    // std results
    TestMath<double>::daxpy(alpha);

    // Backend results
    m_math.daxpy(alpha, *m_in1, *m_in2, *m_out);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(1.0E-14));
    }
}

BOOST_AUTO_TEST_CASE(ddot_padded_multicomponent)
{
    const std::vector<BlockAttributes<FieldState::Coeff>> blockAttr = {
        {3, 4, 2, 1}};
    Field<double, FieldState::Coeff> x("x", blockAttr, {"u", "v", "w"}, 1);
    Field<double, FieldState::Coeff> y("y", blockAttr, {"u", "v", "w"}, 1);

    x.template Initialize<NektarSpaces::HostSpace>(100.0);
    y.template Initialize<NektarSpaces::HostSpace>(-100.0);

    auto &block   = x.GetBlocks()[0];
    auto realSize = block.GetNumElements() * block.GetNumData();
    auto stride   = block.CompSize();
    auto *xptr    = block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto *yptr =
        y.GetBlocks()[0].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    double expected = 0.0;
    for (unsigned int c = 0; c < x.GetNumComponents(); ++c)
    {
        for (size_t i = 0; i < realSize; ++i)
        {
            xptr[c * stride + i] = 10.0 * c + i + 1.0;
            yptr[c * stride + i] = 1.0 - 0.5 * i + c;
            expected += xptr[c * stride + i] * yptr[c * stride + i];
        }
    }

    Math::MathHelper math;
    BOOST_TEST(std::abs(math.ddot(x, y) - expected) < 1.0E-12);
}

BOOST_AUTO_TEST_CASE(masked_ddot_padded_multicomponent)
{
    const std::vector<BlockAttributes<FieldState::Coeff>> blockAttr = {
        {3, 4, 2, 1}};
    Field<std::uint8_t, FieldState::Coeff> mask("mask", blockAttr,
                                                {"u", "v", "w"}, 1);
    Field<double, FieldState::Coeff> x("x", blockAttr, {"u", "v", "w"}, 1);
    Field<double, FieldState::Coeff> y("y", blockAttr, {"u", "v", "w"}, 1);

    mask.template Initialize<NektarSpaces::HostSpace>(0);
    x.template Initialize<NektarSpaces::HostSpace>(100.0);
    y.template Initialize<NektarSpaces::HostSpace>(-100.0);

    auto &block   = x.GetBlocks()[0];
    auto realSize = block.GetNumElements() * block.GetNumData();
    auto stride   = block.CompSize();
    auto *mptr    = mask.GetBlocks()[0]
                     .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto *xptr = block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto *yptr =
        y.GetBlocks()[0].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    double expected = 0.0;
    for (unsigned int c = 0; c < x.GetNumComponents(); ++c)
    {
        for (size_t i = 0; i < realSize; ++i)
        {
            mptr[c * stride + i] = ((i + c) % 2 == 0) ? 1 : 0;
            xptr[c * stride + i] = 10.0 * c + i + 1.0;
            yptr[c * stride + i] = 1.0 - 0.5 * i + c;
            expected += mptr[c * stride + i] * xptr[c * stride + i] *
                        yptr[c * stride + i];
        }
    }

    Math::MathHelper math;
    BOOST_TEST(std::abs(math.ddot(mask, x, y) - expected) < 1.0E-12);
}

BOOST_FIXTURE_TEST_CASE(sum, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::sum();

    // Backend results
    double h_out = m_math.reduceSum(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 1.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Sum = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(max, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::max();

    // Backend results
    double h_out = m_math.reduceMax(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(min, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::min();

    // Backend results
    double h_out = m_math.reduceMin(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(innerproduct, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::inner_product();

    // Backend results
    double h_out = m_math.ddot(*m_in1, *m_in2);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(std::abs(h_out))
                  << " ddot = " << std::sqrt(std::abs(out)) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(l1norm, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::l1norm();

    // Backend results
    double h_out = m_math.l1norm(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " L1norm = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(l2norm, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::l2norm();

    // Backend results
    double h_out = m_math.l2norm(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(lpnorm, TestMath<double>)
{
    Configure();
    SetTestCase();

    for (unsigned int p = 1; p < 4; p++)
    {
        // std results
        double out = TestMath<double>::lpnorm(p);

        // Backend results
        double h_out = m_math.lpnorm(p, *m_in1);

        // Check results
        BOOST_TEST(fabs(h_out - out) < 5.0E-09);
        boost::test_tools::output_test_stream output;
        {
            std::cout << "Backend = " << std::sqrt(h_out) << " L" << p
                      << "norm = " << std::sqrt(out) << std::endl;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(linfnorm, TestMath<double>)
{
    Configure();
    SetTestCase();

    // std results
    double out = TestMath<double>::linfnorm();

    // Backend results
    double h_out = m_math.linfnorm(*m_in1);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << std::sqrt(h_out)
                  << " Linfnorm = " << std::sqrt(out) << std::endl;
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
