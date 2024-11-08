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
    Timer timer;
    const unsigned int ntests = 40;
    auto x = MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
        "x", size, vec_t::alignment);
    auto y = MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
        "y", size, vec_t::alignment);

    TData time_serial = 0.0;
    {
        TData result_serial;
        auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::Serial>(
                0, size, [&](unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            ddotKernel<NektarSpaces::Serial>(size, xptr, yptr, &result_serial);
            ASSERTL0((result_serial > 0.0), "Error!");
            timer.Stop();
            time_serial += timer.Elapsed().count();
        }
        time_serial /= ntests;
    }

    TData time_cuda = 0.0;
    {
        auto result_cuda =
            MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
                "result_cuda", 1, vec_t::alignment);
        auto xptr = x.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
        auto yptr = y.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::CUDA>(
                0, size, NEKTAR_LAMBDA(unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            ddotKernel<NektarSpaces::CUDA>(
                size, xptr, yptr,
                result_cuda
                    .template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>());
            cudaDeviceSynchronize();
            ASSERTL0((*result_cuda.template GetPtr<NektarSpaces::HostSpace,
                                                   ReadOnly>() > 0.0),
                     "Error!");
            timer.Stop();
            time_cuda += timer.Elapsed().count();
        }
        time_cuda /= ntests;
    }

    // Display results
    if constexpr (!warmup)
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
    Timer timer;
    const unsigned int ntests = 40;
    auto x = MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
        "x", size, vec_t::alignment);
    auto y = MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
        "y", size, vec_t::alignment);
    auto z = MemoryRegion<TData>::template Create<NektarSpaces::DeviceSpace>(
        "z", size, vec_t::alignment);

    TData time_serial = 0.0;
    {
        auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto zptr = z.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::Serial>(
                0, size, [&](unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            daxpyKernel<NektarSpaces::Serial>(size, 3.2, xptr, yptr, zptr);
            ASSERTL0(zptr[0] > 0.0, "Error!");
            timer.Stop();
            time_serial += timer.Elapsed().count();
        }
        time_serial /= ntests;
    }

    TData time_cuda = 0.0;
    {
        auto xptr = x.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
        auto yptr = y.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
        auto zptr = z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::CUDA>(
                0, size, NEKTAR_LAMBDA(unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            daxpyKernel<NektarSpaces::CUDA>(size, 3.2, xptr, yptr, zptr);
            cudaDeviceSynchronize();
            timer.Stop();
            time_cuda += timer.Elapsed().count();
        }
        time_cuda /= ntests;
    }

    // Display results
    if constexpr (!warmup)
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
