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
#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AddAdvectionKernels(
    const size_t nelmt, const unsigned int nhomo, const unsigned int nqTot,
    const unsigned int ncoord, const size_t adveloffset,
    const size_t derivoffset, const TData *advVel, const TData *deriv,
    TData *out, const TData scale, const TthreadBlock &threadBlock)
{
    const auto nsize = nelmt * nqTot * nhomo;

    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        size_t idx0 = idx % (nelmt * nqTot);
        TData tmp   = 0.0;
        for (unsigned int d = 0; d < ncoord; d++)
        {
            tmp +=
                advVel[d * adveloffset + idx0] * deriv[d * derivoffset + idx];
        }
        out[idx] = scale * out[idx] + tmp;
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const TData *diffCoeff,
    const TData *jacptr, const TData *dfptr, const TData *inptr, TData *outptr,
    TData *bwdptr, const TData scale, const TthreadBlock &threadBlock)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt * nhomo;

    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        if constexpr (DEFORMED)
        {
            size_t e = (idx % (nelmt * nqTot)) / nqTot;
            size_t i = idx % nqTot;
            TData tmp[3], metric[3];

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] =
                        dfptr[ndf * nqTot * e + d * nqTot + i] * diffCoeff[k];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[ndf * nqTot * e +
                                           (l * dimension + d) * nqTot + i] *
                                     diffCoeff[l * ncoord + k];
                    }
                }

                tmp[d] = metric[0] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += metric[k] * inptr[idx + k * inoffset];
                }
            }

            // Write.
            auto jac = jacptr[nqTot * e + i];
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outoffset + idx] = tmp[d] * jac;
            }
            bwdptr[idx] *= scale * jac;
        }
        else
        {
            size_t e = (idx % (nelmt * nqTot)) / nqTot;
            TData tmp[3], metric[3];

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] = dfptr[(ndf * e + d)] * diffCoeff[k];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[(ndf * e + l * dimension + d)] *
                                     diffCoeff[l * ncoord + k];
                    }
                }

                tmp[d] = metric[0] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += metric[k] * inptr[idx + k * inoffset];
                }
            }

            // Write.
            auto jac = jacptr[e];
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outoffset + idx] = tmp[d] * jac;
            }
            bwdptr[idx] *= scale * jac;
        }
    }
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AddAdvectionKernels(
    const size_t nelmt, const unsigned int nhomo, const unsigned int nqTot,
    const unsigned int ncoord, const size_t adveloffset,
    const size_t derivoffset, const TData *advVel, const TData *deriv,
    TData *out, const TData scale)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        (AddAdvectionKernels<>), gridSize, blockSize, 0, nelmt, nhomo, nqTot,
        ncoord, adveloffset, derivoffset, advVel, deriv, out, scale);
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const TData *diffCoeff,
    const TData *jacptr, const TData *dfptr, const TData *inptr, TData *outptr,
    TData *bwdptr, const TData scale)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize =
        (nelmt * nqTot * nhomo + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        (ApplyMetricKernel<DEFORMED>), gridSize, blockSize, 0, nqTot, ncoord,
        dimension, nelmt, nhomo, inoffset, outoffset, diffCoeff, jacptr, dfptr,
        inptr, outptr, bwdptr, scale);
}
#endif
} // namespace Nektar::Operators::detail
