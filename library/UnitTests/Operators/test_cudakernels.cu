#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestCUDAMathKernels
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include <LibUtilities/BasicUtils/Vmath.hpp>

#include "Operators/CUDAMathKernels.cuh"
#include "MemoryRegionCUDA.hpp"
#include "init_cudakernels.hpp"

using namespace Nektar::Operators::detail;

BOOST_AUTO_TEST_SUITE(TestCUDAKernels)

BOOST_FIXTURE_TEST_CASE(cuda_dotkernel, CUDAKernels)
{
    Configure();
    SetTestCase();
    size_t n = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().size();
    size_t gridSize  = 1024;
    size_t blockSize = 32;
    double out, h_out, *d_out, *x, *y;

    // Vmath results
    x   = fixt_in->GetStorage().GetCPUPtr();
    y   = fixt_out->GetStorage().GetCPUPtr();
    out = Vmath::Dot(n, x, 1, y, 1);

    // CUDA results
    cudaMalloc((void **)&d_out, sizeof(double));
    cudaMemset(d_out, 0, sizeof(double));
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    dotKernel<<<gridSize, blockSize, sizeof(double) * blockSize>>>(n, x, y,
                                                                   d_out);
    cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

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
    size_t gridSize  = 1024;
    size_t blockSize = 32;
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vadd(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    addKernel<<<gridSize, blockSize>>>(n, x, y, y);

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
    size_t gridSize  = 1024;
    size_t blockSize = 32;
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vsub(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    subKernel<<<gridSize, blockSize>>>(n, x, y, y);

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
    size_t gridSize  = 1024;
    size_t blockSize = 32;
    double *x, *y, alpha = 1.5;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Svtvp(n, alpha, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    daxpyKernel<<<gridSize, blockSize>>>(n, alpha, x, y, y);

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
    size_t gridSize  = 1024;
    size_t blockSize = 32;
    double *x, *y;

    // Vmath results
    x = fixt_in->GetStorage().GetCPUPtr();
    y = fixt_out->GetStorage().GetCPUPtr();
    Vmath::Vdiv(n, x, 1, y, 1, y, 1);

    // CUDA results
    x = fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    y = fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
    vdivKernel<<<gridSize, blockSize>>>(n, x, y, y);

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
