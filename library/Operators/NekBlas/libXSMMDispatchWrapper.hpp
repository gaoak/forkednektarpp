///////////////////////////////////////////////////////////////////////////////
//
// File: libXSMMDispatchWrapper.hpp
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

#pragma once

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
#include "libxsmm.h"
#else
#define LIBXSMM_PREFETCH_NONE 0
#endif

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"

#if !defined(NEKTAR_ENABLE_SIMD_AVX2) && !defined(NEKTAR_ENABLE_SIMD_AVX512)
template <typename TData> class libxsmm_mmfunction
{
public:
    libxsmm_mmfunction(const int, const int, const int, const int, const TData,
                       const TData, const int);
    void operator()(const TData *a, const TData *b, TData *c);
};
#endif

template <typename TData> struct LibxsmmDispatchWrapper
{
    using KernelFunc = libxsmm_mmfunction<TData>;

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    static inline KernelFunc dispatch(
        const int m, const int n, const int k, const TData alpha,
        const TData beta, const int flags = 0,
        const int prefetch = LIBXSMM_PREFETCH_NONE)
    {
        ASSERTL0(alpha == 1.0, "libxsmm: alpha must be equal to 1.0");
        ASSERTL0(beta == 0.0 || beta == 1.0,
                 "libxsmm: beta must be equal to 0.0 or 1.0");

        return KernelFunc(flags, m, n, k, alpha, beta, prefetch);
    }
#else
    static inline KernelFunc dispatch(
        [[maybe_unused]] const int m, [[maybe_unused]] const int n,
        [[maybe_unused]] const int k, [[maybe_unused]] const double alpha,
        [[maybe_unused]] const double beta,
        [[maybe_unused]] const int flags    = 0,
        [[maybe_unused]] const int prefetch = LIBXSMM_PREFETCH_NONE)
    {
        ASSERTL0(false, "libxsmm requires AVX2/AVX512");

        return KernelFunc(flags, m, n, k, alpha, beta, prefetch);
    }
#endif
};
