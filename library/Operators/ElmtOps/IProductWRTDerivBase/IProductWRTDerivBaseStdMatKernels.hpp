///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseStdMatKernels.hpp
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
    MultiplyByJacobianAndDerivFactorKernel(
        const unsigned int nqTot, const unsigned int ncoord,
        const unsigned int dimension, const size_t nelmt, const TData *jacptr,
        const TData *dfptr, const TData *inptr, TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        for (unsigned int d = 0; d < dimension; d++)
        {
            TData *ptr = outptr + d * nsize;
            for (size_t i = 0; i < nsize; i++)
            {
                ptr[i] = dfptr[ndf * i + d] * inptr[i];
            }
            for (unsigned int k = 1; k < ncoord; ++k)
            {
                for (size_t i = 0; i < nsize; i++)
                {
                    ptr[i] += dfptr[ndf * i + k * dimension + d] *
                              inptr[i + k * nsize];
                }
            }

            for (size_t i = 0; i < nsize; i++)
            {
                ptr[i] *= jacptr[i];
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                TData *ptr = outptr + d * nsize;
                for (unsigned int i = 0; i < nqTot; i++)
                {
                    ptr[nqTot * e + i] =
                        dfptr[ndf * e + d] * inptr[nqTot * e + i];
                }
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    for (unsigned int i = 0; i < nqTot; i++)
                    {
                        ptr[nqTot * e + i] +=
                            dfptr[ndf * e + k * dimension + d] *
                            inptr[nqTot * e + i + k * nsize];
                    }
                }
            }

            for (unsigned int d = 0; d < dimension; d++)
            {
                TData *ptr = outptr + d * nsize;
                for (unsigned int i = 0; i < nqTot; i++)
                {
                    ptr[nqTot * e + i] *= jacptr[e];
                }
            }
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    MultiplyByJacobianAndDerivFactorKernel(
        const unsigned int nqTot, const unsigned int ncoord,
        const unsigned int dimension, const size_t nelmt, const TData *jacptr,
        const TData *dfptr, const TData *inptr, TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nelmt * nqTot;

    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = idx / nqTot;
                for (unsigned int d = 0; d < dimension; d++)
                {
                    TData tmp = dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx] *
                                inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp += dfptr[(ndf - 1) * nqTot * e +
                                     nqTot * (k * dimension + d) + idx] *
                               inptr[idx + k * nsize];
                    }
                    outptr[d * nsize + idx] = tmp * jacptr[idx];
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = idx / nqTot;
                for (unsigned int d = 0; d < dimension; d++)
                {
                    TData tmp = dfptr[(ndf * e + d)] * inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp += dfptr[(ndf * e + k * dimension + d)] *
                               inptr[idx + k * nsize];
                    }
                    outptr[d * nsize + idx] = tmp * jacptr[e];
                }
            });
    }
}
