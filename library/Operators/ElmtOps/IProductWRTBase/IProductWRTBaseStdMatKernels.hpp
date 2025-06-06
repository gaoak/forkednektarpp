///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseStdMatKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    MultiplyByJacobianKernel(const unsigned int nqTot, const size_t nelmt,
                             const TData *jacptr, const TData *inptr,
                             TData *outptr)
{
    if constexpr (DEFORMED)
    {
        for (size_t i = 0; i < nelmt * nqTot; ++i)
        {
            outptr[i] = jacptr[i] * inptr[i];
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; ++e)
        {
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                outptr[e * nqTot + i] = jacptr[e] * inptr[e * nqTot + i];
            }
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    MultiplyByJacobianKernel(const unsigned int nqTot, const size_t nelmt,
                             const TData *jacptr, const TData *inptr,
                             TData *outptr)
{
    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                outptr[idx] = jacptr[idx] * inptr[idx];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                size_t e    = idx / nqTot;
                outptr[idx] = jacptr[e] * inptr[idx];
            });
    }
}
