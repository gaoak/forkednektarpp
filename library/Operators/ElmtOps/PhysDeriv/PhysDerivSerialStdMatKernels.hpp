///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialStdMatKernels.hpp
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

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const unsigned int nelmt, const TData *dfptr,
    const TData *inptr, TData *outptr)
{
    const auto ndf = ncoord * dimension;

    if constexpr (DEFORMED)
    {
        for (unsigned int k = 0; k < ncoord; k++)
        {
            TData *ptr = outptr + k * nqTot * nelmt;
            for (unsigned int i = 0; i < nelmt * nqTot; i++)
            {
                ptr[i] = dfptr[ndf * i + k * dimension] * inptr[i];
            }
            for (unsigned int d = 1; d < dimension; d++)
            {
                for (unsigned int i = 0; i < nelmt * nqTot; i++)
                {
                    ptr[i] += dfptr[ndf * i + k * dimension + d] *
                              inptr[i + d * nqTot * nelmt];
                }
            }
        }
    }
    else
    {
        for (unsigned int e = 0; e < nelmt; e++)
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                TData *ptr = outptr + k * nqTot * nelmt;
                for (unsigned int i = 0; i < nqTot; i++)
                {
                    ptr[nqTot * e + i] =
                        dfptr[ndf * e + k * dimension] * inptr[nqTot * e + i];
                }
                for (unsigned int d = 1; d < dimension; d++)
                {
                    for (unsigned int i = 0; i < nqTot; i++)
                    {
                        ptr[nqTot * e + i] +=
                            dfptr[ndf * e + k * dimension + d] *
                            inptr[nqTot * e + i + d * nqTot * nelmt];
                    }
                }
            }
        }
    }
}
