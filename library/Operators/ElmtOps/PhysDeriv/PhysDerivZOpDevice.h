///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZOpDevice.h
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
// Description: CUDA backends for the PhysDerivOp z-derivative.
//
// PhysDerivZOpDevice   - cuFFT pipeline (requires NEKTAR_ENABLE_CUDA).
// PhysDerivZOpDeviceDx - fused cuFFTDx kernel (additionally
// NEKTAR_USE_CUFFTDX).
//
// All CUDA-specific headers and types are confined to this file. The guard
// NEKTAR_ENABLE_CUDA ensures these are only compiled under a CUDA toolchain;
// HIP/SYCL device builds see an empty translation unit here.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(NEKTAR_ENABLE_CUDA)

#include "Operators/Common/Spaces.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivZOp.h"

#include <LibUtilities/FFT/PhysDerivZCuFFT.h>

#if defined(NEKTAR_USE_CUFFTDX)
#include <LibUtilities/FFT/NekCuFFTDx.h>
#endif

namespace Nektar::Operators
{

/// \brief cuFFT backend for the homogeneous z-derivative (PhysDerivOp).
///
/// Launch() owns the complete per-call pipeline:
///   1. xy-derivatives via blockOp[blk]->Apply() for all blocks.
///   2. When nhomo > 1: cuFFT D2Z + wavenumber multiply + Z2D via
///      PhysDerivZDirect. On the second and subsequent calls the z-pipeline
///      is captured as a CUDA graph and replayed for lower launch overhead.
///      The xy loop always runs outside graph capture.
///
/// Requires NEKTAR_ENABLE_CUDA. The cuFFTDx variant is
/// PhysDerivZOpDeviceDx (below).
template <typename TData>
class PhysDerivZOpDevice : public PhysDerivZOpBase<TData>
{
public:
    PhysDerivZOpDevice() = default;

    ~PhysDerivZOpDevice() override
    {
        if (m_graphExec)
        {
            cudaGraphExecDestroy(m_graphExec);
        }
        if (m_stream)
        {
            cudaStreamDestroy(m_stream);
        }
    }

protected:
    void v_Init(TData beta) override
    {
        int leastPriority    = 0;
        int greatestPriority = 0;
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        cudaStreamCreateWithPriority(&m_stream, cudaStreamDefault,
                                     greatestPriority);
        m_beta           = beta;
        m_hasGraph       = false;
        m_pendingCapture = false;
        m_graphExec      = nullptr;
    }

    void v_Launch(
        std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> &blockOp,
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out, unsigned int nhomo,
        const std::vector<int> &blockNXY) override
    {
#if !defined(NEKTAR_ENABLE_CUDA)
        ASSERTL0(false, "PhysDerivZOp: execStr \"" + execStr +
                            "\" requires NEKTAR_ENABLE_CUDA.");
#endif

        // xy derivatives (always). When nhomo <= 1 there is no z-FFT.
        if (nhomo <= 1 || m_stream == nullptr)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
            }
            return;
        }

        if (m_hasGraph)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
            }
            WaitForBlockProducers(blockOp.size());
            cudaGraphLaunch(m_graphExec, m_stream);
            cudaStreamSynchronize(m_stream);
            return;
        }

        // xy cuBLAS derivatives run outside capture.
        for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
        {
            blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
        }

        // m_stream is a private stream, not registered with CUDAStream, so
        // it does not otherwise participate in the library's per-block
        // stream discipline (block_idx + 1, see e.g.
        // PhysDerivDeviceSumFac.hpp). Without this, nothing guarantees `in`'s
        // producer writes (issued on that per-block stream) are visible before
        // the z-FFT reads below. Done outside graph capture: replay must
        // re-synchronize against fresh input every call, so this cannot be
        // baked into the graph.
        WaitForBlockProducers(blockOp.size());

        if (m_pendingCapture)
        {
            cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
        }

        for (unsigned int n = 0; n < in.GetNumComponents(); ++n)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                auto &inblock        = in.GetBlocks()[blk];
                auto &outblock       = out.GetBlocks()[blk];
                const int compStride = static_cast<int>(inblock.CompSize());
                const TData *phiPtr =
                    inblock.template GetPtr<NektarSpaces::DeviceSpace,
                                            ReadOnly>() +
                    static_cast<std::ptrdiff_t>(n) * compStride *
                        static_cast<std::ptrdiff_t>(nhomo);
                TData *dzPtr =
                    outblock.template GetPtr<NektarSpaces::DeviceSpace,
                                             WriteOnly>() +
                    static_cast<std::ptrdiff_t>(n * 3 + 2) * compStride *
                        static_cast<std::ptrdiff_t>(nhomo);
                LibUtilities::PhysDerivZDirect(
                    phiPtr, dzPtr, static_cast<int>(nhomo), blockNXY[blk],
                    compStride, m_beta, m_stream);
            }
        }

        if (m_pendingCapture)
        {
            cudaGraph_t graph;
            cudaStreamEndCapture(m_stream, &graph);

            bool updated = false;
            if (m_graphExec)
            {
#if CUDART_VERSION >= 12000
                cudaGraphExecUpdateResultInfo info{};
                const cudaError_t ue =
                    cudaGraphExecUpdate(m_graphExec, graph, &info);
                (void)cudaGetLastError();
                updated = (ue == cudaSuccess &&
                           info.result == cudaGraphExecUpdateSuccess);
#else
                cudaGraphNode_t errNode;
                cudaGraphExecUpdateResult updateResult;
                const cudaError_t ue = cudaGraphExecUpdate(
                    m_graphExec, graph, &errNode, &updateResult);
                updated = (ue == cudaSuccess &&
                           updateResult == cudaGraphExecUpdateSuccess);
#endif
            }
            if (!updated)
            {
                if (m_graphExec)
                {
                    cudaGraphExecDestroy(m_graphExec);
                }
#if CUDART_VERSION >= 12000
                cudaGraphInstantiate(&m_graphExec, graph, 0);
#else
                cudaGraphInstantiate(&m_graphExec, graph, nullptr, nullptr, 0);
#endif
            }
            cudaGraphDestroy(graph);
            m_hasGraph       = true;
            m_pendingCapture = false;
        }

        cudaStreamSynchronize(m_stream);
        m_pendingCapture = !m_hasGraph;
    }

private:
    cudaStream_t m_stream       = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_hasGraph             = false;
    bool m_pendingCapture       = false;
    TData m_beta                = 0.0;

    /// \brief Make m_stream wait on every per-block producer stream
    ///        (block_idx + 1) before it reads their data.
    ///
    /// m_stream never appears as any block's own streamID, so nothing else
    /// in the library orders work against it automatically. Recording a
    /// fresh event on each block's stream here (rather than requiring the
    /// producer to have called CUDAStream::RecordEvent itself) is what
    /// makes this safe regardless of what last wrote to `in`.
    void WaitForBlockProducers(unsigned int numBlocks)
    {
        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            const unsigned int streamID = blk + 1;
            CUDAStream::RecordEvent(streamID);
            cudaStreamWaitEvent(m_stream, CUDAStream::GetEvent(streamID));
        }
    }
};

#if defined(NEKTAR_USE_CUFFTDX)

/// \brief cuFFTDx backend for the homogeneous z-derivative (PhysDerivOp).
///
/// Launch() owns the complete per-call pipeline:
///   1. xy-derivatives via blockOp[blk]->Apply() for all blocks.
///   2. When nhomo > 1: fused cuFFTDx D2Z + wavenumber multiply + Z2D via
///      PhysDerivZDxDirect. On the second and subsequent calls the z-pipeline
///      is captured as a CUDA graph and replayed.
///
/// Requires NEKTAR_ENABLE_CUDA and NEKTAR_USE_CUFFTDX. Reachable only
/// via execStr == "DeviceDx" in PhysDerivZOpBase::Create().
template <typename TData>
class PhysDerivZOpDeviceDx : public PhysDerivZOpBase<TData>
{
public:
    PhysDerivZOpDeviceDx() = default;

    ~PhysDerivZOpDeviceDx() override
    {
        if (m_graphExec)
        {
            cudaGraphExecDestroy(m_graphExec);
        }
        if (m_stream)
        {
            cudaStreamDestroy(m_stream);
        }
    }

protected:
    void v_Init(TData beta) override
    {
        int least    = 0;
        int greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        cudaStreamCreateWithPriority(&m_stream, cudaStreamDefault, greatest);
        m_beta           = beta;
        m_hasGraph       = false;
        m_pendingCapture = false;
        m_graphExec      = nullptr;
    }

    void v_Launch(
        std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> &blockOp,
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out, unsigned int nhomo,
        const std::vector<int> &blockNXY) override
    {
#if !defined(NEKTAR_ENABLE_CUDA)
        ASSERTL0(false, "PhysDerivZOp: execStr \"" + execStr +
                            "\" requires NEKTAR_ENABLE_CUDA.");
#endif

        // xy derivatives (always). When nhomo <= 1 there is no z-FFT.
        if (nhomo <= 1 || m_stream == nullptr)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
            }
            return;
        }

        if (m_hasGraph)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
            }
            WaitForBlockProducers(blockOp.size());
            cudaGraphLaunch(m_graphExec, m_stream);
            cudaStreamSynchronize(m_stream);
            return;
        }

        // xy cuBLAS derivatives run outside capture.
        for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
        {
            blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
        }

        // m_stream is a private stream, not registered with CUDAStream, so
        // it does not otherwise participate in the library's per-block
        // stream discipline (block_idx + 1, see e.g.
        // PhysDerivDeviceSumFac.hpp). Without this, nothing guarantees `in`'s
        // producer writes (issued on that per-block stream) are visible before
        // the z-FFT reads below. Done outside graph capture: replay must
        // re-synchronize against fresh input every call, so this cannot be
        // baked into the graph.
        WaitForBlockProducers(blockOp.size());

        if (m_pendingCapture)
        {
            cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
        }

        for (unsigned int n = 0; n < in.GetNumComponents(); ++n)
        {
            for (unsigned int blk = 0; blk < blockOp.size(); ++blk)
            {
                auto &inblock        = in.GetBlocks()[blk];
                auto &outblock       = out.GetBlocks()[blk];
                const int compStride = static_cast<int>(inblock.CompSize());
                const TData *phiPtr =
                    inblock.template GetPtr<NektarSpaces::DeviceSpace,
                                            ReadOnly>() +
                    static_cast<std::ptrdiff_t>(n) * compStride *
                        static_cast<std::ptrdiff_t>(nhomo);
                TData *dzPtr =
                    outblock.template GetPtr<NektarSpaces::DeviceSpace,
                                             WriteOnly>() +
                    static_cast<std::ptrdiff_t>(n * 3 + 2) * compStride *
                        static_cast<std::ptrdiff_t>(nhomo);
                LibUtilities::PhysDerivZDxDirect(
                    phiPtr, dzPtr, blockNXY[blk], compStride,
                    static_cast<int>(nhomo), m_beta, m_stream);
            }
        }

        if (m_pendingCapture)
        {
            cudaGraph_t graph;
            cudaStreamEndCapture(m_stream, &graph);

            bool updated = false;
            if (m_graphExec)
            {
#if CUDART_VERSION >= 12000
                cudaGraphExecUpdateResultInfo info{};
                const cudaError_t ue =
                    cudaGraphExecUpdate(m_graphExec, graph, &info);
                (void)cudaGetLastError();
                updated = (ue == cudaSuccess &&
                           info.result == cudaGraphExecUpdateSuccess);
#else
                cudaGraphNode_t errNode;
                cudaGraphExecUpdateResult updateResult;
                const cudaError_t ue = cudaGraphExecUpdate(
                    m_graphExec, graph, &errNode, &updateResult);
                updated = (ue == cudaSuccess &&
                           updateResult == cudaGraphExecUpdateSuccess);
#endif
            }
            if (!updated)
            {
                if (m_graphExec)
                {
                    cudaGraphExecDestroy(m_graphExec);
                }
#if CUDART_VERSION >= 12000
                cudaGraphInstantiate(&m_graphExec, graph, 0);
#else
                cudaGraphInstantiate(&m_graphExec, graph, nullptr, nullptr, 0);
#endif
            }
            cudaGraphDestroy(graph);
            m_hasGraph       = true;
            m_pendingCapture = false;
        }

        cudaStreamSynchronize(m_stream);
        m_pendingCapture = !m_hasGraph;
    }

private:
    cudaStream_t m_stream       = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_hasGraph             = false;
    bool m_pendingCapture       = false;
    TData m_beta                = 0.0;

    /// \brief Make m_stream wait on every per-block producer stream
    ///        (block_idx + 1) before it reads their data.
    ///
    /// m_stream never appears as any block's own streamID, so nothing else
    /// in the library orders work against it automatically. Recording a
    /// fresh event on each block's stream here (rather than requiring the
    /// producer to have called CUDAStream::RecordEvent itself) is what
    /// makes this safe regardless of what last wrote to `in`.
    void WaitForBlockProducers(unsigned int numBlocks)
    {
        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            const unsigned int streamID = blk + 1;
            CUDAStream::RecordEvent(streamID);
            cudaStreamWaitEvent(m_stream, CUDAStream::GetEvent(streamID));
        }
    }
};

#endif // NEKTAR_USE_CUFFTDX

} // namespace Nektar::Operators

#endif // NEKTAR_ENABLE_CUDA
