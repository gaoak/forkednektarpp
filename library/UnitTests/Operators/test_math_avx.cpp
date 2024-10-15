///////////////////////////////////////////////////////////////////////////////
//
// File: test_math_avx.cpp
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

#define BOOST_TEST_MODULE TestMathAVX
#include <boost/test/tools/output_test_stream.hpp>

#include "MathKernelsLauncher.hpp"
#include "init_mathkernels.hpp"

#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using MemSpace = NektarSpaces::HostSpace;

BOOST_AUTO_TEST_SUITE(TestMathAVX)

BOOST_FIXTURE_TEST_CASE(avx_negkernel, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    const double *x;
    double *y;

    // std results
    x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    y = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    std::transform(x, x + n, y, [](const double &xi) { return -xi; });

    // Serial results
    y = fixt_out->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    negKernelLauncher(n, x, y);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(*fixt_out, *fixt_expected, 1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(avx_addkernel, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    const double *x, *y;
    double *z;

    // std results
    x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    y = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    std::transform(x, x + n, y, z,
                   [](const double &xi, const double &yi) { return xi + yi; });

    // AVX results
    z = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    addKernelLauncher(n, x, z, z);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(*fixt_out, *fixt_expected, 1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(avx_subkernel, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    const double *x, *y;
    double *z;

    // std results
    x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    y = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    std::transform(x, x + n, y, z,
                   [](const double &xi, const double &yi) { return xi - yi; });

    // AVX results
    z = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    subKernelLauncher(n, x, z, z);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(*fixt_out, *fixt_expected, 1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(avx_daxpykernel, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n     = fixt_in->size();
    double alpha = 1.5;
    const double *x, *y;
    double *z;

    // std results
    x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    y = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    std::transform(x, x + n, y, z, [=](const double &xi, const double &yi) {
        return alpha * xi + yi;
    });

    // AVX results
    z = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    daxpyKernelLauncher(n, alpha, x, z, z);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(*fixt_out, *fixt_expected, 1.0E-14));
    }
}

BOOST_FIXTURE_TEST_CASE(avx_divkernel, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    const double *x, *y;
    double *z;

    // std results
    x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    y = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    std::transform(x, x + n, y, z,
                   [](const double &xi, const double &yi) { return xi / yi; });

    // AVX results
    z = fixt_out->GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    divKernelLauncher(n, x, z, z);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(Compare(*fixt_out, *fixt_expected, 1.0E-15));
    }
}

BOOST_FIXTURE_TEST_CASE(avx_sum, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::accumulate(x, x + n, 0.0);

    // AVX results
    sumKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << h_out << " Sum = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_max, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = *(std::max_element(x, x + n));

    // AVX results
    maxKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_min, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = *(std::min_element(x, x + n));

    // AVX results
    minKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_innerproduct, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::inner_product(x, x + n, x, 0.0);

    // AVX results
    innerproductKernelLauncher(n, x, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << std::sqrt(h_out)
                  << " ddot = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_l1norm, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::accumulate(x, x + n, 0.0,
                          [](const double &acc, const double &val) {
                              return acc + std::abs(val);
                          });

    // AVX results
    l1normKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << h_out << " L1norm = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_l2norm, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::accumulate(
        x, x + n, 0.0,
        [](const double &acc, const double &val) { return acc + val * val; });

    // AVX results
    l2normKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(avx_lpnorm, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    for (int p = 1; p < 4; p++)
    {
        // std results
        x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        out = std::accumulate(x, x + n, 0.0,
                              [&](const double &acc, const double &val) {
                                  return acc + std::pow(std::abs(val), p);
                              });

        // AVX results
        lpnormKernelLauncher(n, p, x, &h_out);

        // Check results
        BOOST_TEST(fabs(h_out - out) < 5.0E-10);
        boost::test_tools::output_test_stream output;
        {
            std::cout << "AVX = " << std::sqrt(h_out) << " L" << p
                      << "norm = " << std::sqrt(out) << std::endl;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(avx_linfnorm, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::accumulate(x, x + n, std::numeric_limits<double>::min(),
                          [](const double &acc, const double &val) {
                              return std::max(std::abs(acc), std::abs(val));
                          });

    // AVX results
    linfnormKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "AVX = " << std::sqrt(h_out)
                  << " Linfnorm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
