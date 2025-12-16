///////////////////////////////////////////////////////////////////////////////
//
// File: test_optimization.cpp
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

#define BOOST_TEST_MODULE TestOptimization

#include "init_optimization.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestOptimization)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_FIXTURE_TEST_CASE(SerialBackend, OptimizationField)
{
    Configure("Serial");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "StdMat");
    }
}

BOOST_FIXTURE_TEST_CASE(AVXBackend, OptimizationField)
{
    Configure("AVX");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}

BOOST_FIXTURE_TEST_CASE(DeviceBackend, OptimizationField)
{
    Configure("Device");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFacTOP");
    }
}

BOOST_FIXTURE_TEST_CASE(SerialBackendOverride, OptimizationField)
{
    Configure("Serial", "SumFac");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}

BOOST_FIXTURE_TEST_CASE(AVXBackendOverride, OptimizationField)
{
    Configure("AVX", "StdMat");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "StdMat");
    }
}

BOOST_FIXTURE_TEST_CASE(DeviceBackendOverride, OptimizationField)
{
    Configure("Device", "SumFac");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
