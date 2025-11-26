///////////////////////////////////////////////////////////////////////////////
//
// File: test_parallel_reduce.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University
// (USA), Department of Aeronautics, Imperial College London (UK), and
// Scientific Computing and Imaging Institute, University of Utah (USA).
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

#define BOOST_TEST_MODULE TestReducer

#include "init_parallel_reduce.hpp"

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestReducer)

BOOST_FIXTURE_TEST_CASE(sum, ReducerField)
{
    Configure();
    SetTestCase();

    std::string execName(
        boost::unit_test::framework::master_test_suite().argv[1]);

    // std results
    double out = ReducerField::sum();

    // Backend results
    double h_out = 0.0;
    double tmp   = 0.0;

    for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
    {
        auto x    = (execName == "Device")
                        ? fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::DeviceSpace, ReadOnly>()
                        : fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                    fixt_in->GetBlocks()[blk].GetNumData();
        if (execName == "Serial")
        {
            Nektar::parallel_reduce<NektarSpaces::Serial,
                                    Nektar::ReduceSum<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#if defined(NEKTAR_ENABLE_SIMD)
        else if (execName == "AVX")
        {
            Nektar::parallel_reduce<NektarSpaces::AVX,
                                    Nektar::ReduceSum<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#endif
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
        else if (execName == "Device")
        {
            Nektar::parallel_reduce<NektarSpaces::Device,
                                    Nektar::ReduceSum<double>>(
                0, size, NEKTAR_LAMBDA(size_t i) { return x[i]; }, tmp);
        }
#endif

        h_out += tmp;
    }

    // Check results
    BOOST_TEST(fabs(h_out - out) < 1.0E-11);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Sum = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(max, ReducerField)
{
    Configure();
    SetTestCase();

    std::string execName(
        boost::unit_test::framework::master_test_suite().argv[1]);

    // std results
    double out = ReducerField::max();

    // Backend results
    double h_out = std::numeric_limits<double>::min();
    double tmp   = std::numeric_limits<double>::min();

    for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
    {
        auto x    = (execName == "Device")
                        ? fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::DeviceSpace, ReadOnly>()
                        : fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                    fixt_in->GetBlocks()[blk].GetNumData();
        if (execName == "Serial")
        {
            Nektar::parallel_reduce<NektarSpaces::Serial,
                                    Nektar::ReduceMax<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#if defined(NEKTAR_ENABLE_SIMD)
        else if (execName == "AVX")
        {
            Nektar::parallel_reduce<NektarSpaces::AVX,
                                    Nektar::ReduceMax<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#endif
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
        else if (execName == "Device")
        {
            Nektar::parallel_reduce<NektarSpaces::Device,
                                    Nektar::ReduceMax<double>>(
                0, size, NEKTAR_LAMBDA(size_t i) { return x[i]; }, tmp);
        }
#endif

        h_out = std::max(h_out, tmp);
    }

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Max = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(min, ReducerField)
{
    Configure();
    SetTestCase();

    std::string execName(
        boost::unit_test::framework::master_test_suite().argv[1]);

    // std results
    double out = ReducerField::min();

    // Backend results
    double h_out = std::numeric_limits<double>::max();
    double tmp   = std::numeric_limits<double>::max();

    for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
    {
        auto x    = (execName == "Device")
                        ? fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::DeviceSpace, ReadOnly>()
                        : fixt_in->GetBlocks()[blk]
                           .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                    fixt_in->GetBlocks()[blk].GetNumData();
        if (execName == "Serial")
        {
            Nektar::parallel_reduce<NektarSpaces::Serial,
                                    Nektar::ReduceMin<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#if defined(NEKTAR_ENABLE_SIMD)
        else if (execName == "AVX")
        {
            Nektar::parallel_reduce<NektarSpaces::AVX,
                                    Nektar::ReduceMin<double>>(
                0, size, [=](size_t i) { return x[i]; }, tmp);
        }
#endif
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
        else if (execName == "Device")
        {
            Nektar::parallel_reduce<NektarSpaces::Device,
                                    Nektar::ReduceMin<double>>(
                0, size, NEKTAR_LAMBDA(size_t i) { return x[i]; }, tmp);
        }
#endif

        h_out = std::min(h_out, tmp);
    }

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "Backend = " << h_out << " Min = " << out << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
