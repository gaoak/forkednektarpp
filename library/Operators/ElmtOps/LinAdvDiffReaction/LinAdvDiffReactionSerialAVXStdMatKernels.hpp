///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXStdMatKernels.hpp
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

template <typename ExecSpace, typename TData, typename TScalar>
NEK_FORCE_INLINE static void AddAdvectionKernels(
    const size_t nelmt, const unsigned int nqTot, const unsigned int ncoord,
    const size_t advelsize, const size_t derivsize, const TData *advVel,
    const TData *deriv, TData *out, const TScalar scale)
{
    const auto nsize = nelmt * nqTot;
    for (size_t idx = 0; idx < nsize; idx++)
    {
        out[idx] *= scale;
    }
    for (unsigned int d = 0; d < ncoord; d++)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            out[idx].fma(advVel[d * advelsize + idx],
                         deriv[d * derivsize + idx]);
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t insize,
    const size_t outsize, const TScalar *diffCoeff, const TData *jacptr,
    const TData *dfptr, const TData *inptr, TData *outptr, TData *bwdptr,
    const TScalar scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        TData tmp[3], metric[3];
        for (size_t idx = 0; idx < nsize; idx++)
        {
            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] = dfptr[ndf * idx + d] * diffCoeff[k];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k].fma(dfptr[ndf * idx + l * dimension + d],
                                      diffCoeff[l * ncoord + k]);
                    }
                }
                tmp[d] = metric[0] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d].fma(metric[k], inptr[k * insize + idx]);
                }
            }

            // Write.
            auto jac = jacptr[idx];
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outsize + idx] = tmp[d] * jac;
            }
            bwdptr[idx] *= scale * jac;
        }
    }
    else
    {
        TData tmp[3], metric[9];
        for (size_t e = 0; e < nelmt; e++)
        {
            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[d * ncoord + k] = dfptr[ndf * e + d] * diffCoeff[k];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[d * ncoord + k].fma(
                            dfptr[ndf * e + l * dimension + d],
                            diffCoeff[l * ncoord + k]);
                    }
                }
            }

            auto jac = jacptr[e];
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = metric[d * ncoord] * inptr[nqTot * e + i];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(metric[d * ncoord + k],
                                   inptr[nqTot * e + i + k * insize]);
                    }
                }

                // Write.
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] = tmp[d] * jac;
                }
                bwdptr[nqTot * e + i] *= scale * jac;
            }
        }
    }
}
