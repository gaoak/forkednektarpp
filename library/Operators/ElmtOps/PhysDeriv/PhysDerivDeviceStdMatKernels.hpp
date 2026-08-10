///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceStdMatKernels.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivFactorKernel(
    const unsigned nqTot, const unsigned ncoord, const unsigned dimension,
    const size_t nelmt, const unsigned nhomo, const size_t inoffset,
    const size_t outoffset, const TData *dfptr, const TData *inptr,
    TData *outptr, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt * nhomo;

    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t idx0 = idx % (nelmt * nqTot);
                size_t e    = idx0 / nqTot;
                TData tmp[3];
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] = dfptr[(ndf - 1) * nqTot * e +
                                   nqTot * (k * dimension) + idx0] *
                             inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k] += dfptr[(ndf - 1) * nqTot * e +
                                        nqTot * (k * dimension + d) + idx0] *
                                  inptr[idx + d * inoffset];
                    }
                }
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    outptr[k * outoffset + idx] = tmp[k];
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = (idx % (nelmt * nqTot)) / nqTot;
                TData tmp[3];
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] = dfptr[(ndf * e + k * dimension)] * inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k] += dfptr[(ndf * e + k * dimension + d)] *
                                  inptr[idx + d * inoffset];
                    }
                }
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    outptr[k * outoffset + idx] = tmp[k];
                }
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivDirFactorKernel(
    const unsigned dir, const unsigned nqTot, const unsigned ncoord,
    const unsigned dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *inptr, TData *outptr,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t idx0 = idx % (nelmt * nqTot);
                size_t e    = idx0 / nqTot;
                TData tmp;
                tmp = dfptr[(ndf - 1) * nqTot * e + nqTot * (dir * dimension) +
                            idx0] *
                      inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp += dfptr[(ndf - 1) * nqTot * e +
                                 nqTot * (dir * dimension + d) + idx0] *
                           inptr[idx + d * inoffset];
                }

                if (APPEND)
                {
                    outptr[idx] += tmp;
                }
                else
                {
                    outptr[idx] = tmp;
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = (idx % (nelmt * nqTot)) / nqTot;
                TData tmp;
                tmp = dfptr[(ndf * e + dir * dimension)] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp += dfptr[(ndf * e + dir * dimension + d)] *
                           inptr[idx + d * inoffset];
                }
                if (APPEND)
                {
                    outptr[idx] += tmp;
                }
                else
                {
                    outptr[idx] = tmp;
                }
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}
