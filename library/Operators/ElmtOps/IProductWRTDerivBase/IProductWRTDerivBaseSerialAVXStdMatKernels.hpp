///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXStdMatKernels.hpp
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

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void JacobianDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outsize, const TData *jacptr, const TData *dfptr,
    const TData *inptr, TData *outptr, const TData scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[ndf * idx + d] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[k * inoffset + idx]);
                }
            }
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outsize + idx] = scale * tmp[d] * jacptr[idx];
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * e + d] * inptr[nqTot * e + i];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + k * inoffset]);
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] =
                        scale * tmp[d] * jacptr[e];
                }
            }
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void JacobianDerivFactorWeightsKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outsize, const TData *jacptr, const TData *dfptr,
    const TData *weights, const TData *inptr, TData *outptr, const TData scale)
{
    const auto ndf = ncoord * dimension;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned i = 0; i < nqTot; i++)
            {
                size_t idx = e * nqTot + i;

                auto wj = jacptr[idx] * weights[i];

                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * idx + d] * inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * idx + k * dimension + d],
                                   inptr[k * inoffset + idx]);
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + idx] = scale * tmp[d] * wj;
                }
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned i = 0; i < nqTot; i++)
            {
                auto wj = jacptr[e] * weights[i];

                for (unsigned d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * e + d] * inptr[nqTot * e + i];
                    for (unsigned k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + k * inoffset]);
                    }
                }
                for (unsigned d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] = scale * tmp[d] * wj;
                }
            }
        }
    }
}
