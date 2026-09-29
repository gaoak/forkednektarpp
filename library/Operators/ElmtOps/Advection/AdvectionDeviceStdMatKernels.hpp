///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceStdMatKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void MultiplyByDerivFactorAndAdvecVelKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const TData *dfptr, const TData *advVel,
    const size_t advelsize, const TData *inptr, TData *outptr,
    const TData scale, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt * nhomo;

    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t islot = idx % (nelmt * nqTot);
                size_t e     = islot / nqTot;
                size_t i     = islot - e * nqTot;
                TData tmp[3], tmp0 = 0;
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] =
                        dfptr[ndf * nqTot * e + (k * dimension) * nqTot + i] *
                        inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k] += dfptr[ndf * nqTot * e +
                                        (k * dimension + d) * nqTot + i] *
                                  inptr[idx + d * inoffset];
                    }
                    tmp0 += advVel[k * advelsize + idx] * tmp[k];
                }

                if (APPEND)
                {
                    outptr[idx] += scale * tmp0;
                }
                else
                {
                    outptr[idx] = scale * tmp0;
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = (idx % (nelmt * nqTot)) / nqTot;
                TData tmp[3], tmp0 = 0;
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] = dfptr[(ndf * e + k * dimension)] * inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k] += dfptr[(ndf * e + k * dimension + d)] *
                                  inptr[idx + d * inoffset];
                    }
                    tmp0 += advVel[k * advelsize + idx] * tmp[k];
                }

                if (APPEND)
                {
                    outptr[idx] += scale * tmp0;
                }
                else
                {
                    outptr[idx] = scale * tmp0;
                }
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::Operators::detail
