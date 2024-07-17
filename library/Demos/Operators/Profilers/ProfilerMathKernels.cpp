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
    Timer timer;
    const unsigned int ntests = 40;

    TData time_serial = 0.0;
    {
        using MemSpace = NektarSpaces::HostSpace;

        TData result_serial;
        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::Serial>(
                0, size, [&](unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            ddotKernel<NektarSpaces::Serial>(size, xptr, yptr, &result_serial);
            ASSERTL0(result_serial > 0.0, "Error!");
            timer.Stop();
            time_serial += timer.Elapsed().count();
        }
        time_serial /= ntests;
    }

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    TData time_avx = 0.0;
    {
        using MemSpace = NektarSpaces::HostSpace;

        TData result_avx;
        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::AVX>(
                0, size, [&](unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            ddotKernel<NektarSpaces::AVX>(size, xptr, yptr, &result_avx);
            ASSERTL0(result_avx > 0.0, "Error!");
            timer.Stop();
            time_avx += timer.Elapsed().count();
        }
        time_avx /= ntests;
    }
#endif

#if defined(NEKTAR_ENABLE_KOKKOS)
    TData time_kokkos = 0.0;
    {
        using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

        TData result_kokkos;
        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<Kokkos::DefaultExecutionSpace>(
                0, size, KOKKOS_LAMBDA(unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            ddotKernel<Kokkos::DefaultExecutionSpace>(size, xptr, yptr,
                                                      &result_kokkos);
            ASSERTL0(result_kokkos > 0.0, "Error!");
            timer.Stop();
            time_kokkos += timer.Elapsed().count();
        }
        time_kokkos /= ntests;
    }
#endif

    // Display results
    if constexpr (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 2 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
                  << 2 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_KOKKOS)
                  << 2 * sizeof(TData) * 1e-9 * size / time_kokkos
#endif
                  << std::endl;
    }
}

template <typename TData, bool warmup = false>
void ProfilerDaxpy(const unsigned int size)
{
    Timer timer;
    const unsigned int ntests = 40;

    TData time_serial = 0.0;
    {
        using MemSpace = NektarSpaces::HostSpace;

        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto z = MemoryRegion<TData>::template create<MemSpace>(
            "z", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        auto zptr = z.template GetPtr<MemSpace, WriteOnly>();
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

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    TData time_avx = 0.0;
    {
        using MemSpace = NektarSpaces::HostSpace;

        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto z = MemoryRegion<TData>::template create<MemSpace>(
            "z", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        auto zptr = z.template GetPtr<MemSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<NektarSpaces::AVX>(
                0, size, [&](unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            daxpyKernel<NektarSpaces::AVX>(size, 3.2, xptr, yptr, zptr);
            ASSERTL0(zptr[0] > 0.0, "Error!");
            timer.Stop();
            time_avx += timer.Elapsed().count();
        }
        time_avx /= ntests;
    }
#endif

#if defined(NEKTAR_ENABLE_KOKKOS)
    TData time_kokkos = 0.0;
    {
        using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

        auto x = MemoryRegion<TData>::template create<MemSpace>(
            "x", size, vec_t::alignment);
        auto y = MemoryRegion<TData>::template create<MemSpace>(
            "y", size, vec_t::alignment);
        auto z = MemoryRegion<TData>::template create<MemSpace>(
            "z", size, vec_t::alignment);
        auto xptr = x.template GetPtr<MemSpace, WriteOnly>();
        auto yptr = y.template GetPtr<MemSpace, WriteOnly>();
        auto zptr = z.template GetPtr<MemSpace, WriteOnly>();
        for (unsigned int t = 0; t < ntests; ++t)
        {
            Nektar::parallel_for<Kokkos::DefaultExecutionSpace>(
                0, size, KOKKOS_LAMBDA(unsigned int i) {
                    xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191 + t)));
                    yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516 + t)));
                });
            timer.Start();
            daxpyKernel<Kokkos::DefaultExecutionSpace>(size, 3.2, xptr, yptr,
                                                       zptr);
            Kokkos::fence();
            ASSERTL0(
                (z.template GetPtr<NektarSpaces::HostSpace, ReadOnly>()[0] >
                 0.0),
                "Error!");
            timer.Stop();
            time_kokkos += timer.Elapsed().count();
        }
        time_kokkos /= ntests;
    }
#endif

    // Display results
    if constexpr (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 3 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
                  << 3 * sizeof(TData) * 1e-9 * size / time_avx << " "
#endif
#if defined(NEKTAR_ENABLE_KOKKOS)
                  << 3 * sizeof(TData) * 1e-9 * size / time_kokkos
#endif
                  << std::endl;
    }
}

int main(void)
{
#if defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::initialize();
#endif
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : Reduction " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    std::cout << "      AVX";
#endif
#if defined(NEKTAR_ENABLE_KOKKOS)
    std::cout << "      Kokkos";
#endif
    std::cout << std::endl;

    // Warm-up
    ProfilerReduction<double, true>(2 << 24);

    // Benchmark
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
#if defined(NEKTAR_ENABLE_KOKKOS)
    std::cout << "      Kokkos";
#endif
    std::cout << std::endl;

    // Warm-up
    ProfilerDaxpy<double, true>(2 << 24);

    // Benchmark
    for (unsigned int i = 0; i < 24; i++)
    {
        ProfilerDaxpy<double>(4 << i);
    }
#if defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::finalize();
#endif
}
