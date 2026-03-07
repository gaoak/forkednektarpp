///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceStdMatKernels.hpp
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

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

NEK_DEVICE_INLINE static const unsigned int *GetDiffCoeffMapPtr(
    const unsigned int ncoord)
{

    if (ncoord == 1)
    {
        static const std::array<unsigned int, 1> diffCoeff1DMap{0};
        return &diffCoeff1DMap[0];
    }
    else if (ncoord == 2)
    {
        static const std::array<unsigned int, 4> diffCoeff2DMap{0, 1, 1, 2};
        return &diffCoeff2DMap[0];
    }
    else
    {
        static const std::array<unsigned int, 9> diffCoeff3DMap{0, 1, 3, 1, 2,
                                                                4, 3, 4, 5};
        return &diffCoeff3DMap[0];
    }
}

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const size_t adveloffset,
    const TData *diffCoeff, const TData *jacptr, const TData *dfptr,
    const TData *advVel, const TData *inptr, TData *outptr, TData *bwdptr,
    const TData scale, const TthreadBlock &threadBlock)
{
    const auto diffCoeffMapPtr = GetDiffCoeffMapPtr(ncoord);
    const auto ndf             = ncoord * dimension;
    const auto nsize           = nqTot * nelmt * nhomo;

    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        if constexpr (DEFORMED)
        {
            size_t e     = (idx % (nelmt * nqTot)) / nqTot;
            size_t i     = idx % nqTot;
            TData tmp[3] = {0.0}, tmp0 = 0.0, metric[3];

            auto jac = jacptr[nqTot * e + i];

            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * nqTot * e + (k * dimension) * nqTot + i] *
                         inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[ndf * nqTot * e +
                                    (k * dimension + d) * nqTot + i] *
                              inptr[idx + d * inoffset];
                }
                tmp0 += advVel[k * adveloffset + nqTot * e + i] * tmp[k];
            }

            // Write.
            bwdptr[idx] = (scale * bwdptr[idx] + tmp0) * jac;

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] = dfptr[ndf * nqTot * e + d * nqTot + i] *
                                diffCoeff[diffCoeffMapPtr[k]];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[ndf * nqTot * e +
                                           (l * dimension + d) * nqTot + i] *
                                     diffCoeff[diffCoeffMapPtr[l * ncoord + k]];
                    }
                }
                tmp0 = metric[0] * tmp[0];
                for (unsigned int k = 1; k < ncoord; k++)
                {
                    tmp0 += metric[k] * tmp[k];
                }

                // Write.
                outptr[d * outoffset + idx] = tmp0 * jac;
            }
        }
        else
        {
            size_t e     = (idx % (nelmt * nqTot)) / nqTot;
            size_t i     = idx % nqTot;
            TData tmp[3] = {0.0}, tmp0 = 0.0, metric[3];

            auto jac = jacptr[e];

            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * e + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[ndf * e + k * dimension + d] *
                              inptr[idx + d * inoffset];
                }
                tmp0 += advVel[k * adveloffset + nqTot * e + i] * tmp[k];
            }

            // Write.
            bwdptr[idx] = (scale * bwdptr[idx] + tmp0) * jac;

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] =
                        dfptr[(ndf * e + d)] * diffCoeff[diffCoeffMapPtr[k]];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[(ndf * e + l * dimension + d)] *
                                     diffCoeff[diffCoeffMapPtr[l * ncoord + k]];
                    }
                }
                tmp0 = metric[0] * tmp[0];
                for (unsigned int k = 1; k < ncoord; k++)
                {
                    tmp0 += metric[k] * tmp[k];
                }

                // Write.
                outptr[d * outoffset + idx] = tmp0 * jac;
            }
        }
    }
}
#endif

} // namespace Nektar::Operators::detail
