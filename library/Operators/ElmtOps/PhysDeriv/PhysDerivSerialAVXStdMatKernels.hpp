///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXStdMatKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *dfptr, const TData *inptr,
    TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * inoffset + idx]);
                }
            }
            for (unsigned int k = 0; k < ncoord; k++)
            {
                outptr[k * outoffset + idx] = tmp[k];
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] =
                        dfptr[ndf * e + k * dimension] * inptr[nqTot * e + i];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + d * inoffset]);
                    }
                }
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    outptr[k * outoffset + nqTot * e + i] = tmp[k];
                }
            }
        }
    }
}

template <typename ExecSpace, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivDirFactorKernel(
    const unsigned dir, const unsigned nqTot, const unsigned ncoord,
    const unsigned dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *inptr, TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp;
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            tmp = dfptr[ndf * idx + dir * dimension] * inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp.fma(dfptr[ndf * idx + dir * dimension + d],
                        inptr[d * inoffset + idx]);
            }
            if constexpr (APPEND)
            {
                outptr[idx] += tmp;
            }
            else
            {
                outptr[idx] = tmp;
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                tmp = dfptr[ndf * e + dir * dimension] * inptr[nqTot * e + i];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp.fma(dfptr[ndf * e + dir * dimension + d],
                            inptr[nqTot * e + i + d * inoffset]);
                }
                if constexpr (APPEND)
                {
                    outptr[nqTot * e + i] += tmp;
                }
                else
                {
                    outptr[nqTot * e + i] = tmp;
                }
            }
        }
    }
}

} // namespace Nektar::Operators::detail
