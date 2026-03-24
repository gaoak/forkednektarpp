///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseCUDASumFacCUBLASHelper.cuh
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{
template <typename ExecSpace, typename TData>
void ApplyDerivWithJac(const unsigned int dim, const unsigned int coordDim,
                       const size_t nelmt, const unsigned int nqTot,
                       const unsigned int nmTot, const size_t blocksize,
                       const unsigned int ndf, const TData *NEK_RESTRICT in,
                       const TData *NEK_RESTRICT deriv,
                       const TData *NEK_RESTRICT jac, TData *out,
                       const bool isDeformed)
{
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                for (unsigned int d = 0; d < dim; d++)
                {
                    // Derivative.
                    auto tmp = deriv[ndf * idx + d] * in[idx];
                    for (unsigned int k = 1; k < coordDim; ++k)
                    {
                        tmp += deriv[ndf * idx + k * dim + d] *
                               in[idx + k * blocksize];
                    }

                    // Jacobian.
                    out[d * nelmt * max(nqTot, nmTot) + idx] = tmp * jac[idx];
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e = idx / nqTot;
                for (unsigned int d = 0; d < dim; d++)
                {
                    // Derivative.
                    auto tmp = deriv[ndf * e + d] * in[idx];
                    for (unsigned int k = 1; k < coordDim; ++k)
                    {
                        tmp += deriv[ndf * e + k * dim + d] *
                               in[idx + k * blocksize];
                    }

                    // Jacobian.
                    out[d * nelmt * max(nqTot, nmTot) + idx] = tmp * jac[e];
                }
            });
    }
}

template <typename ExecSpace, typename TData>
void ApplyDeriv(const unsigned int dim, const unsigned int coordDim,
                const size_t nelmt, const unsigned int nqTot,
                const unsigned int nmTot, const size_t blocksize,
                const unsigned int ndf, const TData *NEK_RESTRICT in,
                const TData *NEK_RESTRICT deriv, TData *out,
                const bool isDeformed)
{
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                for (unsigned int d = 0; d < dim; d++)
                {
                    auto tmp = deriv[ndf * idx + d] * in[idx];
                    for (unsigned int k = 1; k < coordDim; ++k)
                    {
                        tmp += deriv[ndf * idx + k * dim + d] *
                               in[idx + k * blocksize];
                    }
                    out[d * nelmt * max(nqTot, nmTot) + idx] = tmp;
                }
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e = idx / nqTot;
                for (unsigned int d = 0; d < dim; d++)
                {
                    auto tmp = deriv[ndf * e + d] * in[idx];
                    for (unsigned int k = 1; k < coordDim; ++k)
                    {
                        tmp += deriv[ndf * e + k * dim + d] *
                               in[idx + k * blocksize];
                    }
                    out[d * nelmt * max(nqTot, nmTot) + idx] = tmp;
                }
            });
    }
}

template <typename ExecSpace, typename TData>
void ApplyFactorTri(const size_t nelmt, const unsigned int nqTot,
                    const unsigned int nq0, const TData *NEK_RESTRICT f0,
                    const TData *NEK_RESTRICT f1, TData *NEK_RESTRICT in0,
                    const TData *NEK_RESTRICT in1)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq1 = (idx % nqTot) / nq0;
            const unsigned int iq0 = (idx % nqTot) % nq0;

            in0[idx] = in0[idx] * f1[iq1] + in1[idx] * f0[iq0] * f1[iq1];
        });
}

template <typename ExecSpace, typename TData>
void ApplyFactorPrism(const size_t nelmt, const unsigned int nqTot,
                      const unsigned int nq0, const unsigned int nq1,
                      const TData *NEK_RESTRICT f0,
                      const TData *NEK_RESTRICT f3, TData *NEK_RESTRICT in0,
                      const TData *NEK_RESTRICT in1)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            in0[idx] = in0[idx] * f3[iq2] + in1[idx] * f0[iq0] * f3[iq2];
        });
}

template <typename ExecSpace, typename TData>
void ApplyFactorPyr(const size_t nelmt, const unsigned int nqTot,
                    const unsigned int nq0, const unsigned int nq1,
                    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
                    const TData *NEK_RESTRICT f3, TData *NEK_RESTRICT in0,
                    TData *NEK_RESTRICT in1, const TData *NEK_RESTRICT in2)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            in0[idx] = in0[idx] * f3[iq2] + in2[idx] * f0[iq0] * f3[iq2];

            in1[idx] = in1[idx] * f3[iq2] + in2[idx] * f1[iq1] * f3[iq2];
        });
}

template <typename ExecSpace, typename TData>
void ApplyFactorTet(const size_t nelmt, const unsigned int nqTot,
                    const unsigned int nq0, const unsigned int nq1,
                    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
                    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT f3,
                    TData *NEK_RESTRICT in0, TData *NEK_RESTRICT in1,
                    TData *NEK_RESTRICT in2)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            auto tmp = in0[idx];
            tmp += (in1[idx] + in2[idx]) * f0[iq0];
            in0[idx] = tmp * f2[iq1] * f3[iq2];

            tmp = in1[idx];
            tmp += in2[idx] * f1[iq1];
            in1[idx] = tmp * f3[iq2];
        });
}

} // namespace Nektar::Operators::detail
