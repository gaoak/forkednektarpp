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

    // Serial.
    TData result_serial;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddotKernel<NektarSpaces::Serial>(size, xptr, yptr, &result_serial);
        ASSERTL0((result_serial > 0.0), "Error!");
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    TData result_avx;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddotKernel<NektarSpaces::AVX>(size, xptr, yptr, &result_avx);
        ASSERTL0((result_avx > 0.0), "Error!");
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
#endif

    // SYCL.
#if defined(NEKTAR_ENABLE_SYCL)
    auto xdptr = x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto ydptr = y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto result_sycl =
        MemoryRegion<TData>::Create("result_sycl", 1, vec_t::alignment);
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddotKernel<NektarSpaces::SYCL>(
            size, xdptr, ydptr,
            result_sycl
                .template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>());
    }
    SYCLQueue::GetInstance().wait();
    timer.Stop();
    TData time_sycl = timer.Elapsed().count() / ntests;
#endif

    // DeviceOnHost.
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
    auto xdptr = x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto ydptr = y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    TData result_deviceonhost;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        ddotKernel<NektarSpaces::DeviceOnHost>(size, xdptr, ydptr,
                                               &result_deviceonhost);
        ASSERTL0((result_deviceonhost > 0.0), "Error!");
    }
    timer.Stop();
    TData time_deviceonhost = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 2 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
                  << 2 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_SYCL)
                  << 2 * sizeof(TData) * 1e-9 * size / time_sycl
#endif
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
                  << 2 * sizeof(TData) * 1e-9 * size / time_deviceonhost
#endif
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
    auto zptr = z.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (unsigned int i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }

    // Serial.
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpyKernel<NektarSpaces::Serial>(size, 3.2, xptr, yptr, zptr);
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpyKernel<NektarSpaces::AVX>(size, 3.2, xptr, yptr, zptr);
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
#endif

    // SYCL.
#if defined(NEKTAR_ENABLE_SYCL)
    auto xdptr = x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto ydptr = y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto zdptr = z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpyKernel<NektarSpaces::SYCL>(size, 3.2, xdptr, ydptr, zdptr);
    }
    SYCLQueue::GetInstance().wait();
    timer.Stop();
    TData time_sycl = timer.Elapsed().count() / ntests;
#endif

    // DeviceOnHost.
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
    auto xdptr = x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto ydptr = y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    auto zdptr = z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        daxpyKernel<NektarSpaces::DeviceOnHost>(size, 3.2, xdptr, ydptr, zdptr);
    }
    timer.Stop();
    TData time_deviceonhost = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 3 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
                  << 3 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_SYCL)
                  << 3 * sizeof(TData) * 1e-9 * size / time_sycl
#endif
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
                  << 3 * sizeof(TData) * 1e-9 * size / time_deviceonhost
#endif
                  << std::endl;
    }
}

int main(void)
{
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : Reduction " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    std::cout << "      AVX";
#endif
#if defined(NEKTAR_ENABLE_SYCL)
    std::cout << "      SYCL";
#endif
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
    std::cout << "      DeviceOnHost";
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
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    std::cout << "      AVX";
#endif
#if defined(NEKTAR_ENABLE_SYCL)
    std::cout << "      SYCL";
#endif
#if defined(NEKTAR_ENABLE_DEVICEONHOST)
    std::cout << "      DeviceOnHost";
#endif
    std::cout << std::endl;

    // Warm-up.
    ProfilerDaxpy<double, true>(2 << 24);

    // Benchmark.
    for (unsigned int i = 0; i < 24; i++)
    {
        ProfilerDaxpy<double>(4 << i);
    }
}
