///////////////////////////////////////////////////////////////////////////////
//
// File: test_math_cuda.cpp
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

#define BOOST_TEST_MODULE TestMathCUDA
#include <boost/test/tools/output_test_stream.hpp>

#include "MathKernelsLauncher.hpp"
#include "Operators/MemoryRegionDevice.hpp"
#include "init_mathkernels.hpp"

#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using MemSpace = NektarSpaces::DeviceSpace;

BOOST_AUTO_TEST_SUITE(TestMathCUDA)

BOOST_FIXTURE_TEST_CASE(cuda_negkernel, MathKernels)
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
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    y = fixt_cuda_out->template GetPtr<MemSpace, WriteOnly>();
    negKernelLauncher(n, x, y);

    // Check results
    BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_addkernel, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    z = fixt_cuda_out->template GetPtr<MemSpace, ReadWrite>();
    addKernelLauncher(n, x, z, z);

    // Check results
    BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_subkernel, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    z = fixt_cuda_out->template GetPtr<MemSpace, ReadWrite>();
    subKernelLauncher(n, x, z, z);

    // Check results
    BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_daxpykernel, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    z = fixt_cuda_out->template GetPtr<MemSpace, ReadWrite>();
    daxpyKernelLauncher(n, alpha, x, z, z);

    // Check results
    BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, 1.0E-14));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            1.0E-14);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_divkernel, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    z = fixt_cuda_out->template GetPtr<MemSpace, ReadWrite>();
    divKernelLauncher(n, x, z, z);

    // Check results
    BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_sum, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::accumulate(x, x + n, 0.0);

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    sumKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_max, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = *(std::max_element(x, x + n));

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    maxKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_min, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = *(std::min_element(x, x + n));

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    minKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_innerproduct, MathKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixt_in->size();
    double out, h_out;
    const double *x;

    // std results
    x   = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    out = std::inner_product(x, x + n, x, 0.0);

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    innerproductKernelLauncher(n, x, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_l1norm, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    l1normKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << h_out << " L1norm = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_l2norm, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    l2normKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_lpnorm, MathKernels)
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

        // CUDA results
        x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
        lpnormKernelLauncher(n, p, x, &h_out);

        // Check results
        BOOST_TEST(fabs(h_out - out) < 5.0E-10);
        boost::test_tools::output_test_stream output;
        {
            std::cout << "CUDA = " << std::sqrt(h_out) << " L" << p
                      << "norm = " << std::sqrt(out) << std::endl;
        }
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_linfnorm, MathKernels)
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

    // CUDA results
    x = fixt_cuda_in->template GetPtr<MemSpace, ReadOnly>();
    linfnormKernelLauncher(n, x, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-10);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << std::sqrt(h_out)
                  << " L2norm = " << std::sqrt(out) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
