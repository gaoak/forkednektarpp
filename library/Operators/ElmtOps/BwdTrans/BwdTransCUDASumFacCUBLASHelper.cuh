///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransCUDASumFacCUBLASHelper.cuh
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
void vertexFactorTri(const size_t nelmt, const unsigned int nmTot,
                     const unsigned int nq1, TData *wsp, const TData *inptr,
                     const TData *B1)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nq1, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e       = idx / nq1;
            const unsigned int i = idx % nq1;

            const TData *Bptr = B1 + nq1;
            const TData alpha = inptr[e * nmTot + 1];
            wsp[nq1 * nelmt + e * nq1 + i] += alpha * Bptr[i];
        });
}

template <typename ExecSpace, typename TData>
void vertexFactorPrism(const size_t nelmt, const unsigned int nmTot,
                       const unsigned int nq2, const unsigned int nm0,
                       const unsigned int nm1, const unsigned int nm2,
                       TData *wsp, const TData *inptr, const TData *B2)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nq2, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e       = idx / nq2;
            const unsigned int i = idx % nq2;

            for (unsigned int j = 0; j < nm1; j++)
            {
                const TData alpha = inptr[1 + e * nmTot + j * nm2];
                const TData *Bptr = B2 + nq2;
                wsp[j * nq2 * nelmt * nm0 + nq2 * nelmt + e * nq2 + i] +=
                    alpha * Bptr[i];
            }
        });
}

template <typename ExecSpace, typename TData>
void vertexFactorPyr(const size_t nelmt, const unsigned int nmTot,
                     const unsigned int nq2, const unsigned int nm1, TData *wsp,
                     const TData *inptr, const TData *B2)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nq2, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e       = idx / nq2;
            const unsigned int i = idx % nq2;

            const TData *Bptr = B2 + nq2;
            const TData alpha = inptr[1 + e * nmTot];
            wsp[nq2 * nelmt + e * nq2 + i] += alpha * Bptr[i];
            wsp[nq2 * nelmt * nm1 + e * nq2 + i] += alpha * Bptr[i];
            wsp[nq2 * nelmt * (1 + nm1) + e * nq2 + i] += alpha * Bptr[i];
        });
}

template <typename ExecSpace, typename TData>
void vertexFactorTet1(const size_t nelmt, const unsigned int nmTot,
                      const unsigned int nq2, const unsigned int nm1,
                      TData *wsp, const TData *inptr, const TData *B2)
{
    Nektar::parallel_for<ExecSpace>(
        0, nq2 * nelmt, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e       = idx / nq2;
            const unsigned int i = idx % nq2;

            const TData *Bptr = B2 + nq2;
            const TData alpha = inptr[1 + e * nmTot];
            wsp[nq2 * nelmt + e * nq2 + i] += alpha * Bptr[i];
            wsp[nm1 * nq2 * nelmt + e * nq2 + i] += alpha * Bptr[i];
        });
}

template <typename ExecSpace, typename TData>
void vertexFactorTet2(const size_t nelmt, const unsigned int nq1,
                      const unsigned int nq2, TData *wsp, TData *wsp2,
                      const TData *B1)
{
    Nektar::parallel_for<ExecSpace>(
        0, nq1 * nq2 * nelmt, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e               = idx / (nq2 * nq1);
            const unsigned int remaining = idx % (nq2 * nq1);
            const unsigned int j         = remaining / nq1;
            const unsigned int k         = remaining % nq1;

            const TData *Bptr = B1 + nq1;
            const TData alpha = wsp[nq2 * nelmt + e * nq2 + j];
            wsp2[nq1 * nq2 * nelmt + e * nq1 * nq2 + j * nq1 + k] +=
                alpha * Bptr[k];
        });
}

} // namespace Nektar::Operators::detail
