///////////////////////////////////////////////////////////////////////////////
//
// File: test_cudamathkernels.cpp
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

#define BOOST_TEST_MODULE TestCUDAMathKernels
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include <LibUtilities/BasicUtils/Vmath.hpp>

#include "CUDAMathKernelsLauncher.hpp"
#include "Operators/MemoryRegionCUDA.hpp"
#include "init_cudakernels.hpp"

BOOST_AUTO_TEST_SUITE(TestCUDAKernels)

BOOST_FIXTURE_TEST_CASE(cuda_dotkernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    double out, h_out, *x, *y;

    // Vmath results
    x   = fixt_in->GetStorage().GetCPUPtr();
    y   = fixt_out->GetStorage().GetCPUPtr();
    out = Vmath::Dot(n, x, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    dotKernelLauncher(n, x, y, &h_out);

    // Check results
    BOOST_TEST(fabs(h_out - out) < 5.0E-12);
    boost::test_tools::output_test_stream output;
    {
        std::cout << "CUDA = " << h_out << " Vmath = " << out << std::endl;
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_addkernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vadd(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    addKernelLauncher(n, x, y, y);

    // Check results
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_out->GetStorage().GetCPUPtr(), 1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_subkernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vsub(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    subKernelLauncher(n, x, y, y);

    // Check results
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_out->GetStorage().GetCPUPtr(), 1.0E-15);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_daxpykernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    double *x, *y, alpha = 1.5;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Svtvp(n, alpha, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    daxpyKernelLauncher(n, alpha, x, y, y);

    // Check results
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-14));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_out->GetStorage().GetCPUPtr(), 1.0E-14);
    }
}

BOOST_FIXTURE_TEST_CASE(cuda_vdivkernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vdiv(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    vdivKernelLauncher(n, x, y, y);

    // Check results
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-15));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_out->GetStorage().GetCPUPtr(), 1.0E-15);
    }
}

BOOST_AUTO_TEST_SUITE_END()
