///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceStdMatKernels.hpp
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
NEK_FORCE_INLINE static void JacobianDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const TData *jacptr,
    const TData *dfptr, const TData *inptr, TData *outptr,
    const unsigned int streamID)
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
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx0] *
                             inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                        nqTot * (k * dimension + d) + idx0] *
                                  inptr[idx + k * inoffset];
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outoffset + idx] = tmp[d] * jacptr[idx0];
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                size_t e = (idx % (nelmt * nqTot)) / nqTot;
                TData tmp[3];
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                                  inptr[idx + k * inoffset];
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outoffset + idx] = tmp[d] * jacptr[e];
                }
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void JacobianDerivFactorWeightsKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const TData *jacptr,
    const TData *dfptr, const TData *weights, const TData *inptr, TData *outptr,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt * nhomo;

    if constexpr (DEFORMED)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                // idx0 is the point's slot on its own plane. Stripping the
                // plane costs the one modulo; the point within the element
                // then comes off idx0 with a multiply and a subtract rather
                // than a second modulo.
                size_t idx0 = idx % (nelmt * nqTot);
                size_t e    = idx0 / nqTot;
                size_t i    = idx0 - e * nqTot;
                TData tmp[3];
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx0] *
                             inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                        nqTot * (k * dimension + d) + idx0] *
                                  inptr[idx + k * inoffset];
                    }
                }
                auto wj = jacptr[idx0] * weights[i];
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outoffset + idx] = tmp[d] * wj;
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nsize, NEKTAR_LAMBDA(const size_t idx) {
                // idx0 is the point's slot on its own plane. Stripping the
                // plane costs the one modulo; the point within the element
                // then comes off idx0 with a multiply and a subtract rather
                // than a second modulo.
                size_t idx0 = idx % (nelmt * nqTot);
                size_t e    = idx0 / nqTot;
                size_t i    = idx0 - e * nqTot;
                TData tmp[3];
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                                  inptr[idx + k * inoffset];
                    }
                }
                auto wj = jacptr[e] * weights[i];
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outoffset + idx] = tmp[d] * wj;
                }
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}
