///////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerMathKernels.cpp
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

#include "Operators/Math/Math.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/Field/Field.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;
using vec_t = tinysimd::simd<double>;

template <typename TData, bool warmup = false>
void ProfilerReduction(const size_t size)
{
    // Initialization.
    Math math;
    Timer timer;
    const unsigned int ntests = 40;
    auto x                    = MemoryRegion<TData>("x", size);
    auto y                    = MemoryRegion<TData>("y", size);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (size_t i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();

    // Serial.
    math = Math("Serial");
    TData result_serial;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_serial = math.ddot(x, y);
        ASSERTL0((result_serial > 0.0), "Error!");
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD)
    math = Math("AVX");
    TData result_avx;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_avx = math.ddot(x, y);
        ASSERTL0((result_avx > 0.0), "Error!");
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
#endif

    // Device.
#if defined(NEKTAR_ENABLE_DEVICE)
    math = Math("Device");
    TData result_device;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_device = math.ddot(x, y);
        ASSERTL0((result_device > 0.0), "Error!");
    }
    timer.Stop();
    TData time_device = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 2 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD)
                  << 2 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
                  << 2 * sizeof(TData) * 1e-9 * size / time_device
#endif
                  << std::endl;
    }
}

template <typename TData, bool warmup = false>
void ProfilerDaxpy(const size_t size)
{
    // Initialization.
    Math math;
    Timer timer;
    const unsigned int ntests = 40;
    auto x                    = MemoryRegion<TData>("x", size);
    auto y                    = MemoryRegion<TData>("y", size);
    auto z                    = MemoryRegion<TData>("z", size);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (size_t i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();

    // Serial.
    math = Math("Serial");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD)
    math = Math("AVX");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
#endif

    // Device.
#if defined(NEKTAR_ENABLE_DEVICE)
    math = Math("Device");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    nekDeviceSynchronize();
    timer.Stop();
    TData time_device = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 3 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD)
                  << 3 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
                  << 3 * sizeof(TData) * 1e-9 * size / time_device
#endif
                  << std::endl;
    }
}

int main(void)
{
    // Print GPU properties
#if defined(NEKTAR_ENABLE_DEVICE)
    PrintDeviceProperties();
#endif

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : Reduction " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD)
    std::cout << "      AVX";
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    std::cout << "      Device";
#endif
    std::cout << std::endl;

    // Warm-up.
    ProfilerReduction<double, true>(2 << 24);

    // Benchmark.
    for (unsigned int i = 0; i < 24; i++)
    {
        ProfilerReduction<double>(4 << i);
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : daxpy     " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD)
    std::cout << "      AVX";
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    std::cout << "      Device";
#endif
    std::cout << std::endl;

    // Warm-up.
    ProfilerDaxpy<double, true>(2 << 24);

    // Benchmark.
    for (unsigned int i = 4; i < 24; i++)
    {
        ProfilerDaxpy<double>(2 << i);
    }
}
