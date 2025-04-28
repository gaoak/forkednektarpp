///////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerMathKernels.cu
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

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "Operators/MathKernels/MathKernels.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <Operators/Field/Field.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using vec_t = tinysimd::simd<double>;

template <typename TData, bool warmup = false>
void ProfilerReduction(const unsigned int size)
{
    // Initialization.
    Timer timer;
    const unsigned int ntests = 40;
    auto x    = MemoryRegion<TData>::Create("x", size, vec_t::alignment);
    auto y    = MemoryRegion<TData>::Create("y", size, vec_t::alignment);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (unsigned int i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();

    // Serial.
    TData result_serial;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddot<NektarSpaces::Serial>(x, y, &result_serial);
        ASSERTL0((result_serial > 0.0), "Error!");
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // CUDA.
    TData result_cuda;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddot<NektarSpaces::Device>(x, y, &result_cuda);
    }
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
    timer.Stop();
    TData time_cuda = timer.Elapsed().count() / ntests;

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 2 * sizeof(TData) * 1e-9 * size / time_serial
                  << " " << 2 * sizeof(TData) * 1e-9 * size / time_cuda
                  << std::endl;
    }
}

template <typename TData, bool warmup = false>
void ProfilerDaxpy(const unsigned int size)
{
    // Initialization.
    Timer timer;
    const unsigned int ntests = 40;
    auto x    = MemoryRegion<TData>::Create("x", size, vec_t::alignment);
    auto y    = MemoryRegion<TData>::Create("y", size, vec_t::alignment);
    auto z    = MemoryRegion<TData>::Create("z", size, vec_t::alignment);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (unsigned int i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();

    // Serial.
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpy<NektarSpaces::Serial>(3.2, x, y, z);
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // CUDA.
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpy<NektarSpaces::Device>(3.2, x, y, z);
    }
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
    timer.Stop();
    TData time_cuda = timer.Elapsed().count() / ntests;

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 3 * sizeof(TData) * 1e-9 * size / time_serial
                  << " " << 3 * sizeof(TData) * 1e-9 * size / time_cuda
                  << std::endl;
    }
}

int main(void)
{
    // Print GPU properties
    cudaDeviceProp prop;
    CHECK_HIPCUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total Global Memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("  Shared Memory per Block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("  Shared Memory per Multiprocessor (bytes): %ld\n",
           prop.sharedMemPerMultiprocessor);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : Reduction " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial        CUDA" << std::endl;

    // Warm-up
    ProfilerReduction<double, true>(1024 << 18);

    // Benchmark
    for (unsigned int i = 0; i < 18; i++)
    {
        ProfilerReduction<double>(1024 << i);
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : daxpy     " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial        CUDA" << std::endl;

    // Warm-up
    ProfilerDaxpy<double, true>(1024 << 18);

    // Benchmark
    for (unsigned int i = 0; i < 18; i++)
    {
        ProfilerDaxpy<double>(1024 << i);
    }
}
