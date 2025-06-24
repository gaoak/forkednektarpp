///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseCUDASumFacCUBLASHelper.cuh
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

// #if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
void ApplyJacobian(const size_t nelmt, const unsigned int nqTot,
                   const TData *__restrict__ jac, const TData *__restrict__ in,
                   TData *__restrict__ out, const bool isDeformed)
{
    // Multiply by jacobian.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot,
            NEKTAR_LAMBDA(const size_t idx) { out[idx] = jac[idx] * in[idx]; });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e = idx / nqTot;
                out[idx]       = jac[e] * in[idx];
            });
    }
}

template <typename ExecSpace, typename TData>
void IProductWRTBaseSegSumFacKernel(const unsigned int nq0,
                                    const unsigned int nm0, const size_t nelmt,
                                    const TData *__restrict__ b0,
                                    const TData *__restrict__ w0,
                                    const TData *__restrict__ jac,
                                    const TData *in, TData *out, TData *wsp,
                                    const bool isDeformed)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nq0, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq0 = idx % nq0;
                wsp[idx]               = in[idx] * jac[idx] * w0[iq0];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nq0, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nq0;
                const unsigned int iq0 = idx % nq0;
                wsp[idx]               = in[idx] * jac[e] * w0[iq0];
            });
    }

    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    // Perform matrix-matrix multiply.
    NekGemm(handle, "T", "N", nm0, nelmt, nq0, 1.0, b0, nq0, wsp, nq0, 0.0, out,
            nm0);
}

template <typename ExecSpace, typename TData>
void IProductWRTBaseQuadSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nqTot, const unsigned int nmTot,
    const size_t nelmt, const TData *__restrict__ b0,
    const TData *__restrict__ b1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *in, TData *out, TData *wsp, const bool isDeformed)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq1 = (idx % nqTot) / nq0;
                const unsigned int iq0 = (idx % nqTot) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq1 = (idx % nqTot) / nq0;
                const unsigned int iq0 = (idx % nqTot) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1];
            });
    }

    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    // Set second wsp.
    TData *wsp2 = wsp + max(nmTot, nqTot) * nelmt;

    NekGemm(handle, "T", "N", nq1 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0, nq0,
            0.0, wsp2, nq1 * nelmt);

    if (nelmt > 1)
    {
        NekGemm(handle, "T", "N", nelmt * nm0, nm1, nq1, 1.0, wsp2, nq1, b1,
                nq1, 0.0, wsp, nelmt * nm0);

        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nmTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e       = idx / nmTot;
                const unsigned int i = idx % nmTot;
                out[e * nmTot + i]   = wsp[i * nelmt + e];
            });
    }
    else
    {
        NekGemm(handle, "T", "N", nm0, nm1, nq1, 1.0, wsp2, nq1, b1, nq1, 0.0,
                out, nm0);
    }
}

template <typename ExecSpace, typename TData>
void IProductWRTBaseTriSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nqTot, const unsigned int nmTot,
    const size_t nelmt, const TData *__restrict__ b0,
    const TData *__restrict__ b1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *in, TData *out, TData *wsp, const bool isDeformed,
    const bool isModified, std::vector<cudaStream_t> &streams)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq1 = (idx % nqTot) / nq0;
                const unsigned int iq0 = (idx % nqTot) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq1 = (idx % nqTot) / nq0;
                const unsigned int iq0 = (idx % nqTot) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1];
            });
    }

    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    // Set second wsp.
    TData *wsp2 = wsp + max(nmTot, nqTot) * nelmt;

    NekGemm(handle, "T", "N", nq1 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0, nq0,
            0.0, wsp2, nq1 * nelmt);

    for (unsigned int i = 0, mode = 0; i < nm0; i++)
    {
        // Set stream.
        cublasSetStream(handle, streams[i % nStreams]);

        NekGemm(handle, "T", "N", nm1 - i, nelmt, nq1, 1.0, b1 + mode * nq1,
                nq1, wsp2 + i * nq1 * nelmt, nq1, 0.0, &out[mode], nmTot);
        mode += nm1 - i;
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    if (isModified)
    {
        NekGemv(handle, "T", nq1, nelmt, 1.0, wsp2 + nq1 * nelmt, nq1, b1 + nq1,
                1, 1.0, &out[1], nmTot);
    }
}

template <typename ExecSpace, typename TData>
void IProductWRTBaseHexSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nq2, const unsigned int nm2,
    const unsigned int nqTot, const unsigned int nmTot, const size_t nelmt,
    const TData *__restrict__ b0, const TData *__restrict__ b1,
    const TData *__restrict__ b2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *in, TData *out, TData *wsp,
    const bool isDeformed)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }

    // Fetch handle.
    auto handle = NekHandle<ExecSpace>::GetInstance();

    // Set second and third wsp.
    TData *wsp2 = wsp + nqTot * nelmt;
    TData *wsp3 = wsp2 + nelmt * max(nmTot, nqTot);

    NekGemm(handle, "T", "N", nq1 * nq2 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0,
            nq0, 0.0, wsp2, nq1 * nq2 * nelmt);

    NekGemm(handle, "T", "N", nq2 * nelmt * nm0, nm1, nq1, 1.0, wsp2, nq1, b1,
            nq1, 0.0, wsp3, nq2 * nelmt * nm0);

    NekGemm(handle, "T", "N", nelmt * nm0 * nm1, nm2, nq2, 1.0, wsp3, nq2, b2,
            nq2, 0.0, wsp2, nelmt * nm0 * nm1);

    Nektar::parallel_for<ExecSpace>(
        0, nmTot * nelmt, NEKTAR_LAMBDA(const size_t idx) {
            const unsigned int i = idx / nelmt;
            const unsigned int j = idx % nelmt;

            out[j * nmTot + i] = wsp2[idx];
        });
}

template <typename ExecSpace, typename TData>
void IProductWRTBasePrismSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nq2, const unsigned int nm2,
    const unsigned int nqTot, const unsigned int nmTot, const size_t nelmt,
    const TData *__restrict__ b0, const TData *__restrict__ b1,
    const TData *__restrict__ b2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *in, TData *out, TData *wsp,
    const bool isDeformed, const bool isModified,
    std::vector<cudaStream_t> &streams)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }

    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    // Point to second wsp.
    TData *wsp2 = wsp + nelmt * nq2 * max(nq0 * nq1, nm0 * nm1);

    NekGemm(handle, "T", "N", nq1 * nq2 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0,
            nq0, 0.0, wsp2, nq1 * nq2 * nelmt);

    NekGemm(handle, "T", "N", nq2 * nelmt * nm0, nm1, nq1, 1.0, wsp2, nq1, b1,
            nq1, 0.0, wsp, nq2 * nelmt * nm0);

    unsigned int mode, mode1;
    mode = mode1 = 0;

    for (unsigned int i = 0; i < nm0; i++)
    {
        // Strides.
        const unsigned int m   = nm2 - i;
        const size_t strideWsp = nq2 * nelmt * nm0;

        // Pointers to corresponding block for nm0 iteration.
        const TData *tmpB2  = b2 + mode * nq2;
        const TData *tmpWsp = wsp + i * nq2 * nelmt;
        TData *tmpOut       = out + mode1;

        // Set stream.
        cublasSetStream(handle, streams[i % nStreams]);

        NekGemmStridedBatched(handle, "T", "N", m, nelmt, nq2, 1.0, tmpB2, nq2,
                              0, tmpWsp, nq2, strideWsp, 0.0, tmpOut, nmTot, m,
                              nm1);

        // Advannce mode counters.
        mode1 += m * nm1;
        mode += m;
    }

    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    if (isModified)
    {
        // Starting pointers.
        const TData *tmpWsp = wsp + nq2 * nelmt;
        const TData *tmpB2  = b2 + nq2;
        TData *tmpOut       = out + 1;

        // Strides.
        size_t strideA = nq2 * nelmt * nm0;

        NekGemvStridedBatched(handle, "T", nq2, nelmt, 1.0, tmpWsp, nq2,
                              strideA, tmpB2, 1, 0, 1.0, tmpOut, nmTot, nm2,
                              nm1);
    }
}

template <typename ExecSpace, typename TData>
void IProductWRTBasePyrSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nq2, const unsigned int nm2,
    const unsigned int nqTot, const unsigned int nmTot, const size_t nelmt,
    const TData *__restrict__ b0, const TData *__restrict__ b1,
    const TData *__restrict__ b2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *in, TData *out, TData *wsp,
    const bool isDeformed, const bool isModified,
    const std::vector<cudaStream_t> streams)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }

    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    // Set new wsp.
    TData *wsp2 = wsp + nelmt * nq2 * max(nq0 * nq1, nm0 * nm1);

    // '0' direction.
    NekGemm(handle, "T", "N", nq1 * nq2 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0,
            nq0, 0.0, wsp2, nq1 * nq2 * nelmt);

    // '1' direction.
    unsigned int mode, mode1, cnt;
    mode = 0;

    for (unsigned int i = 0; i < nm0; i++)
    {
        // Set stream.
        cublasSetStream(handle, streams[i % nStreams]);

        NekGemm(handle, "T", "N", nq2 * nelmt, nm1, nq1, 1.0,
                wsp2 + i * nq1 * nq2 * nelmt, nq1, b1, nq1, 0.0,
                wsp + mode * nq2 * nelmt, nq2 * nelmt);

        mode += nm1;
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // '2' direction.
    mode = mode1 = cnt = 0;
    for (unsigned int i = 0; i < nm0; i++)
    {
        for (unsigned int j = 0; j < nm1; j++, cnt++)
        {
            // Set stream.
            cublasSetStream(handle, streams[cnt % nStreams]);

            const unsigned int ijmax = max(i, j);

            NekGemm(handle, "T", "N", nm2 - ijmax, nelmt, nq2, 1.0,
                    b2 + mode * nq2, nq2, wsp + cnt * nq2 * nelmt, nq2, 0.0,
                    out + mode1, nmTot);

            mode += nm2 - ijmax;
            mode1 += nm2 - ijmax;
        }

        for (unsigned int j = nm1; j < nm2; j++)
        {
            const unsigned int ijmax = max(i, j);
            mode += nm2 - ijmax;
        }
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    if (isModified)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(const size_t n) {
                TData sum1 = 0.0;
                TData sum2 = 0.0;
                TData sum3 = 0.0;

                for (unsigned int i = 0; i < nq2; ++i)
                {

                    sum1 += (b2[i + nq2] * wsp[nq2 * nelmt + n * nq2 + i]);
                    sum2 +=
                        (b2[i + nq2] * wsp[nq2 * nm1 * nelmt + n * nq2 + i]);
                    sum3 += (b2[i + nq2] *
                             wsp[nq2 * (nm1 + 1) * nelmt + n * nq2 + i]);
                }

                out[1 + n * nmTot] += sum1;
                out[1 + n * nmTot] += sum2;
                out[1 + n * nmTot] += sum3;
            });
    }
}

template <typename ExecSpace, typename TData>
void IProductWRTBaseTetSumFacKernel(
    const unsigned int nq0, const unsigned int nm0, const unsigned int nq1,
    const unsigned int nm1, const unsigned int nq2, const unsigned int nm2,
    const unsigned int nqTot, const unsigned int nmTot, const size_t nelmt,
    const TData *__restrict__ b0, const TData *__restrict__ b1,
    const TData *__restrict__ b2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *in, TData *out, TData *wsp,
    const bool isDeformed, const bool isModified,
    const std::vector<cudaStream_t> streams)
{
    // Multiply by Jacobian and weights.
    if (isDeformed)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[idx] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt * nqTot, NEKTAR_LAMBDA(const size_t idx) {
                const size_t e         = idx / nqTot;
                const unsigned int iq2 = (idx % nqTot) / (nq0 * nq1);
                const unsigned int iq1 = ((idx % nqTot) % (nq0 * nq1)) / nq0;
                const unsigned int iq0 = ((idx % nqTot) % (nq0 * nq1)) % nq0;

                wsp[idx] = in[idx] * jac[e] * w0[iq0] * w1[iq1] * w2[iq2];
            });
    }

    // Fetch handle.
    auto handle                 = NekHandle<ExecSpace>::GetInstance();
    const unsigned int nStreams = streams.size();

    // Set second wsp.
    TData *wsp2 =
        wsp + nq2 * nelmt * max(nq0 * nq1, nm0 * (2 * nm1 - nm0 + 1) / 2);

    // '0' direction.
    NekGemm(handle, "T", "N", nq1 * nq2 * nelmt, nm0, nq0, 1.0, wsp, nq0, b0,
            nq0, 0.0, wsp2, nq1 * nq2 * nelmt);

    // '1' direction.
    unsigned int mode, mode1, cnt;
    mode = 0;

    for (unsigned int i = 0; i < nm0; i++)
    {
        // Set stream.
        cublasSetStream(handle, streams[i % nStreams]);

        NekGemm(handle, "T", "N", nq2 * nelmt, nm1 - i, nq1, 1.0,
                wsp2 + i * nq1 * nq2 * nelmt, nq1, b1 + mode * nq1, nq1, 0.0,
                wsp + mode * nq2 * nelmt, nq2 * nelmt);

        mode += nm1 - i;
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    if (isModified)
    {
        const TData *tmpWsp2          = wsp2 + nelmt * nq1 * nq2;
        TData *tmpWsp                 = wsp + nq2 * nelmt;
        const unsigned int strideWsp2 = nq1 * nq2;

        NekGemvStridedBatched(handle, "T", nq1, nq2, 1.0, tmpWsp2, nq1,
                              strideWsp2, b1 + nq1, 1, 0, 1.0, tmpWsp, 1, nq2,
                              nelmt);
    }

    // '2' direction.
    mode = mode1 = cnt = 0;

    for (unsigned int i = 0; i < nm0; i++)
    {
        for (unsigned int j = 0; j < nm1 - i; j++, cnt++)
        {
            cublasSetStream(handle, streams[cnt % nStreams]);

            NekGemm(handle, "T", "N", nm2 - i - j, nelmt, nq2, 1.0,
                    b2 + mode * nq2, nq2, wsp + cnt * nq2 * nelmt, nq2, 0.0,
                    out + mode1, nmTot);

            mode += nm2 - i - j;
            mode1 += nm2 - i - j;
        }

        // Increment mode in case order1 != order 2.
        mode += (nm2 - nm1) * (nm2 - nm1 + 1) / 2;
    }

    // Synchronize all streams.
    for (unsigned int s = 0; s < nStreams; ++s)
    {
        cudaStreamSynchronize(streams[s]);
    }

    // Set back to default stream.
    cublasSetStream(handle, 0);

    if (isModified)
    {
        Nektar::parallel_for<ExecSpace>(
            0, nelmt, NEKTAR_LAMBDA(const size_t n) {
                TData sum1 = 0.0;
                TData sum2 = 0.0;

                for (unsigned int i = 0; i < nq2; ++i)
                {
                    sum1 += (b2[i + nq2] * wsp[nq2 * nelmt + n * nq2 + i]);

                    sum2 +=
                        (b2[i + nq2] * wsp[nq2 * nm1 * nelmt + n * nq2 + i]);
                }

                out[1 + n * nmTot] += sum1;
                out[1 + n * nmTot] += sum2;
            });
    }
}

} // namespace Nektar::Operators::detail

// #endif
