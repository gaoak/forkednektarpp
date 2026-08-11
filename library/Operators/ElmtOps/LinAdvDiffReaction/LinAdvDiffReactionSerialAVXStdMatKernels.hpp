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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t insize,
    const size_t outsize, const size_t advelsize, const TScalar *diffCoeff,
    const TData *jacptr, const TData *dfptr, const TData *advVel,
    const TData *inptr, TData *outptr, TData *bwdptr, const TScalar scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        TData tmp[3], tmp0, metric[3];
        for (size_t idx = 0; idx < nsize; idx++)
        {
            auto jac = jacptr[idx];

            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * insize + idx]);
                }
                tmp0.fma(advVel[k * advelsize + idx], tmp[k]);
            }

            // Write.
            bwdptr[idx] *= scale * jac;
            bwdptr[idx].fma(tmp0, jac);

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] = dfptr[ndf * idx + d] *
                                diffCoeff[GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k].fma(
                            dfptr[ndf * idx + l * dimension + d],
                            diffCoeff[GetDiffCoeffMap(ncoord, l * ncoord + k)]);
                    }
                }
                tmp0 = metric[0] * tmp[0];
                for (unsigned int k = 1; k < ncoord; k++)
                {
                    tmp0.fma(metric[k], tmp[k]);
                }

                // Write.
                outptr[d * outsize + idx] = tmp0 * jac;
            }
        }
    }
    else
    {
        TData tmp[3], tmp0, metric[9];
        for (size_t e = 0; e < nelmt; e++)
        {
            auto jac = jacptr[e];
            for (unsigned int i = 0; i < nqTot; i++)
            {
                auto idx = nqTot * e + i;
                tmp0     = 0.0;
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    tmp[k] = dfptr[ndf * e + k * dimension] * inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[d * insize + idx]);
                    }
                    tmp0.fma(advVel[k * advelsize + nqTot * e + i], tmp[k]);
                }

                // Write.
                bwdptr[idx] *= scale * jac;
                bwdptr[idx].fma(tmp0, jac);

                // Compute metric.
                for (unsigned int d = 0; d < dimension; d++)
                {
                    if (i == 0)
                    {
                        for (unsigned int k = 0; k < ncoord; ++k)
                        {
                            metric[d * ncoord + k] =
                                dfptr[ndf * e + d] *
                                diffCoeff[GetDiffCoeffMap(ncoord, k)];
                            for (unsigned int l = 1; l < ncoord; ++l)
                            {
                                metric[d * ncoord + k].fma(
                                    dfptr[ndf * e + l * dimension + d],
                                    diffCoeff[GetDiffCoeffMap(ncoord,
                                                              l * ncoord + k)]);
                            }
                        }
                    }
                    tmp0 = metric[d * ncoord] * tmp[0];
                    for (unsigned int k = 1; k < ncoord; k++)
                    {
                        tmp0.fma(metric[d * ncoord + k], tmp[k]);
                    }

                    // Write.
                    outptr[d * outsize + idx] = tmp0 * jac;
                }
            }
        }
    }
}

} // namespace Nektar::Operators::detail
