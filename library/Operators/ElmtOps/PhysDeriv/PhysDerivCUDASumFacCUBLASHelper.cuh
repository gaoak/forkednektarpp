///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivCUDASumFacCUBLASHelper.cuh
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
#include "Operators/NekBlas/NekBlas.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
void ApplyDerivFactor(const unsigned int coordDim, const unsigned int dim,
                      const unsigned int nqTot, const size_t nelmt,
                      const size_t outblocksize, const TData *deriv,
                      const TData *in, TData *out, const bool isDeformed)
{
    const auto ndf = coordDim * dim;

    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot * coordDim, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int k = idx / (nelmt * nqTot);
                const unsigned int i = idx % (nelmt * nqTot);

                auto tmp = deriv[ndf * i + k * dim] * in[i];
                for (unsigned int d = 1; d < dim; d++)
                {
                    tmp += deriv[ndf * i + k * dim + d] *
                           in[i + d * nqTot * nelmt];
                }
                out[k * outblocksize + i] = tmp;
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e = idx / nqTot;

                for (unsigned int k = 0; k < coordDim; k++)
                {
                    auto tmp = deriv[ndf * e + k * dim] * in[idx];
                    for (unsigned int d = 1; d < dim; d++)
                    {
                        tmp += deriv[ndf * e + k * dim + d] *
                               in[idx + d * nqTot * nelmt];
                    }
                    out[k * outblocksize + idx] = tmp;
                }
            });
    }
}

template <typename ExecSpace, typename TData>
void PhysDerivSegKernel(const unsigned int coordDim, const unsigned int dim,
                        const unsigned int nq0, const size_t nelmt,
                        const size_t outblocksize, const TData *__restrict__ d0,
                        const TData *deriv, const TData *in, TData *out,
                        TData *wsp, const bool isDeformed)
{
    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    NekGemm(handle, "N", "N", nq0, nelmt, nq0, 1.0, d0, nq0, in, nq0, 0.0, wsp,
            nq0);

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nq0, nelmt, outblocksize, deriv,
                                wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivQuadKernel(const unsigned int coordDim, const unsigned int dim,
                         const unsigned int nq0, const unsigned int nq1,
                         const unsigned int nqTot, const size_t nelmt,
                         const size_t outblocksize,
                         const TData *__restrict__ d0,
                         const TData *__restrict__ d1,
                         const TData *__restrict__ deriv, const TData *in,
                         TData *out, TData *wsp, const bool isDeformed)
{
    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    NekGemm(handle, "N", "N", nq0, nq1 * nelmt, nq0, 1.0, d0, nq0, in, nq0, 0.0,
            wsp, nq0);

    TData *wsp2 = wsp + nelmt * nqTot;
    for (size_t e = 0, cnt = 0; e < nelmt; e++, cnt += nqTot)
    {
        NekGemm(handle, "N", "T", nq0, nq1, nq1, 1.0, in + cnt, nq0, d1, nq1,
                0.0, wsp2 + cnt, nq0);
    }

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivTriKernel(const unsigned int coordDim, const unsigned int dim,
                        const unsigned int nq0, const unsigned int nq1,
                        const unsigned int nqTot, const size_t nelmt,
                        const size_t outblocksize, const TData *__restrict__ d0,
                        const TData *__restrict__ d1,
                        const TData *__restrict__ f0,
                        const TData *__restrict__ f1,
                        const TData *__restrict__ deriv, const TData *in,
                        TData *out, TData *wsp, const bool isDeformed,
                        std::vector<cudaStream_t> &streams)
{
    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    NekGemm(handle, "N", "N", nq0, nq1 * nelmt, nq0, 1.0, d0, nq0, in, nq0, 0.0,
            wsp, nq0);

    TData *wsp2 = wsp + nelmt * nqTot;
    for (size_t e = 0, cnt = 0; e < nelmt; e++, cnt += nqTot)
    {
        // Set stream.
        cublasSetStream(handle, streams[e % nStreams]);

        NekGemm(handle, "N", "T", nq0, nq1, nq1, 1.0, in + cnt, nq0, d1, nq1,
                0.0, wsp2 + cnt, nq0);
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    // Apply z factor.
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq1 = (idx % nqTot) / nq0;
            const unsigned int iq0 = (idx % nqTot) % nq0;

            wsp[idx] *= f1[iq1];
            wsp2[idx] += wsp[idx] * f0[iq0];
        });

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivHexKernel(const unsigned int coordDim, const unsigned int dim,
                        const unsigned int nq0, const unsigned int nq1,
                        const unsigned int nq2, const unsigned int nqTot,
                        const size_t nelmt, const size_t outblocksize,
                        const TData *__restrict__ d0,
                        const TData *__restrict__ d1,
                        const TData *__restrict__ d2,
                        const TData *__restrict__ deriv, const TData *in,
                        TData *out, TData *wsp, const bool isDeformed)
{
    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    NekGemm(handle, "N", "N", nq0, nq1 * nq2 * nelmt, nq0, 1.0, d0, nq0, in,
            nq0, 0.0, wsp, nq0);

    TData *wsp2 = wsp + nelmt * nqTot;
    TData *wsp3 = wsp2 + nelmt * nqTot;

    NekGemmStridedBatched(handle, "N", "T", nq0, nq1, nq1, 1.0, in, nq0,
                          nq0 * nq1, d1, nq1, 0, 0.0, wsp2, nq0, nq0 * nq1,
                          nelmt * nq2);

    NekGemmStridedBatched(handle, "N", "T", nq0 * nq1, nq2, nq2, 1.0, in,
                          nq0 * nq1, nqTot, d2, nq2, 0, 0.0, wsp3, nq0 * nq1,
                          nqTot, nelmt);

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivPrismKernel(
    const unsigned int coordDim, const unsigned int dim, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nqTot,
    const size_t nelmt, const size_t outblocksize, const TData *__restrict__ d0,
    const TData *__restrict__ d1, const TData *__restrict__ d2,
    const TData *__restrict__ f0, const TData *__restrict__ f3,
    const TData *__restrict__ deriv, const TData *in, TData *out, TData *wsp,
    const bool isDeformed)
{
    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    // Transform in d1.
    NekGemm(handle, "N", "N", nq0, nq1 * nq2 * nelmt, nq0, 1.0, d0, nq0, in,
            nq0, 0.0, wsp, nq0);

    // Set workspaces.
    TData *wsp2 = wsp + nelmt * nqTot;
    TData *wsp3 = wsp2 + nelmt * nqTot;

    // Transform in d2.
    const unsigned int dPts = nq0 * nq1;
    const size_t nBatch     = nq2 * nelmt;

    NekGemmStridedBatched(handle, "N", "T", nq0, nq1, nq1, 1.0, in, nq0, dPts,
                          d1, nq1, 0, 0.0, wsp2, nq0, dPts, nBatch);

    // Transform in d3.
    NekGemmStridedBatched(handle, "N", "T", dPts, nq2, nq2, 1.0, in, dPts,
                          nqTot, d2, nq2, 0, 0.0, wsp3, dPts, nqTot, nelmt);

    // Apply z factor.
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            wsp[idx] *= f3[iq2];
            wsp3[idx] += wsp[idx] * f0[iq0];
        });

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivPyrKernel(
    const unsigned int coordDim, const unsigned int dim, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nqTot,
    const size_t nelmt, const size_t outblocksize, const TData *__restrict__ d0,
    const TData *__restrict__ d1, const TData *__restrict__ d2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f3, const TData *__restrict__ deriv,
    const TData *in, TData *out, TData *wsp, const bool isDeformed,
    std::vector<cudaStream_t> &streams)
{
    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    NekGemm(handle, "N", "N", nq0, nq1 * nq2 * nelmt, nq0, 1.0, d0, nq0, in,
            nq0, 0.0, wsp, nq0);

    TData *wsp2 = wsp + nelmt * nqTot;
    TData *wsp3 = wsp2 + nelmt * nqTot;

    for (size_t e = 0; e < nelmt; ++e)
    {
        for (unsigned int j = 0, cnt = 0; j < nq2; ++j, ++cnt)
        {
            // Set stream.
            cublasSetStream(handle, streams[cnt % nStreams]);

            NekGemm(handle, "N", "T", nq0, nq1, nq1, 1.0,
                    &in[e * nqTot + j * nq0 * nq1], nq0, d1, nq1, 0.0,
                    &wsp2[e * nqTot + j * nq0 * nq1], nq0);
        }

        // Set stream.
        cublasSetStream(handle, streams[e % nStreams]);

        NekGemm(handle, "N", "T", nq0 * nq1, nq2, nq2, 1.0, &in[e * nqTot],
                nq0 * nq1, d2, nq2, 0.0, &wsp3[e * nqTot], nq0 * nq1);
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    // Apply z factor.
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            wsp[idx] *= f3[iq2];
            wsp2[idx] *= f3[iq2];
            wsp3[idx] += wsp[idx] * f0[iq0];
            wsp3[idx] += wsp2[idx] * f1[iq1];
        });

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

template <typename ExecSpace, typename TData>
void PhysDerivTetKernel(
    const unsigned int coordDim, const unsigned int dim, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nqTot,
    const size_t nelmt, const size_t outblocksize, const TData *__restrict__ d0,
    const TData *__restrict__ d1, const TData *__restrict__ d2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f2, const TData *__restrict__ f3,
    const TData *__restrict__ deriv, const TData *in, TData *out, TData *wsp,
    const bool isDeformed)
{
    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    NekGemm(handle, "N", "N", nq0, nq1 * nq2 * nelmt, nq0, 1.0, d0, nq0, in,
            nq0, 0.0, wsp, nq0);

    TData *wsp2 = wsp + nelmt * nqTot;
    TData *wsp3 = wsp2 + nelmt * nqTot;

    // Transform in d2.
    const unsigned int dPts = nq0 * nq1;
    const size_t nBatch     = nq2 * nelmt;

    NekGemmStridedBatched(handle, "N", "T", nq0, nq1, nq1, 1.0, in, nq0, dPts,
                          d1, nq1, 0, 0.0, wsp2, nq0, dPts, nBatch);

    // Transform in d3.
    NekGemmStridedBatched(handle, "N", "T", dPts, nq2, nq2, 1.0, in, dPts,
                          nqTot, d2, nq2, 0, 0.0, wsp3, dPts, nqTot, nelmt);

    // Apply z factor.
    Nektar::parallel_for<ExecSpace>(
        0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
            const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
            const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

            wsp3[idx] += f1[iq1] * f3[iq2] * wsp2[idx];
            wsp2[idx] *= f3[iq2];
            wsp2[idx] += f0[iq0] * f2[iq1] * f3[iq2] * wsp[idx];
            wsp3[idx] += f0[iq0] * f2[iq1] * f3[iq2] * wsp[idx];
            wsp[idx] *= f2[iq1] * f3[iq2];
        });

    ApplyDerivFactor<ExecSpace>(coordDim, dim, nqTot, nelmt, outblocksize,
                                deriv, wsp, out, isDeformed);
}

}; // namespace Nektar::Operators::detail
