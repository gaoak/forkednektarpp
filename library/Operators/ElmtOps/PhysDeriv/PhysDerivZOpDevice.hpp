///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZOpDevice.hpp
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
// Description: Device (cuFFT/cuFFTDx) backend for the PhysDerivOp
// z-derivative.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#if defined(NEKTAR_ENABLE_CUDA)
#include <LibUtilities/FFT/PhysDerivZCuFFT.h>
#endif

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_CUDA)

/// \brief Device backend for the homogeneous z-derivative.
///
/// Launch() applies a D2Z + wavenumber multiply + Z2D pipeline per block via
/// PhysDerivZDirect. From the second call on the pipeline is captured as a
/// CUDA graph and replayed for lower launch overhead.
template <typename ExecSpace, typename TData>
class PhysDerivZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>>>
{
public:
    /// \param expansionList  Unused; the device pipeline works straight off
    ///                       the block pointers. Present so that both
    ///                       backends are constructed the same way.
    PhysDerivZOpImpl(
        [[maybe_unused]] const MultiRegions::ExpListSharedPtr &expansionList)
    {
    }

    ~PhysDerivZOpImpl()
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

    // Non-copyable and non-movable.
    PhysDerivZOpImpl(const PhysDerivZOpImpl &)            = delete;
    PhysDerivZOpImpl &operator=(const PhysDerivZOpImpl &) = delete;
    PhysDerivZOpImpl(PhysDerivZOpImpl &&)                 = delete;
    PhysDerivZOpImpl &operator=(PhysDerivZOpImpl &&)      = delete;

    void Init(TData beta)
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

    void Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();
        const unsigned int numBlocks =
            static_cast<unsigned int>(in.GetBlocks().size());

        if (m_hasGraph)
        {
            WaitForBlockProducers(numBlocks);
            cudaGraphLaunch(m_graphExec, m_stream);
            cudaStreamSynchronize(m_stream);
            return;
        }

        // m_stream is private and not registered with CUDAStream, so nothing
        // otherwise orders it against the per-block producer streams whose
        // writes the z-FFT reads below. Kept outside graph capture: replay
        // must re-synchronize against fresh input on every call.
        WaitForBlockProducers(numBlocks);

        if (m_pendingCapture)
        {
            cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
        }

        for (unsigned int n = 0; n < in.GetNumComponents(); ++n)
        {
            for (unsigned int blk = 0; blk < numBlocks; ++blk)
            {
                auto &inblock           = in.GetBlocks()[blk];
                auto &outblock          = out.GetBlocks()[blk];
                const size_t compStride = inblock.CompSize();
                const TData *phiPtr =
                    inblock.template GetPtr<NektarSpaces::DeviceSpace,
                                            ReadOnly>() +
                    n * compStride * nhomo;
                TData *dzPtr =
                    outblock.template GetPtr<NektarSpaces::DeviceSpace,
                                             WriteOnly>() +
                    (n * 3 + 2) * compStride * nhomo;
                LibUtilities::PhysDerivZDirect(phiPtr, dzPtr, nhomo, compStride,
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
    /// The event is recorded here rather than left to the producer, so the
    /// ordering holds regardless of what last wrote to `in`.
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

#else

/// \brief Device backend for a build with no device z-FFT.
///
/// Without NEKTAR_ENABLE_CUDA there is no device transform to call, so every
/// entry point reports a fatal error rather than leaving the output field's
/// z-component unwritten. The specialization has to exist either way, because
/// PhysDerivOpImpl is templated on the execution space alone.
template <typename ExecSpace, typename TData>
class PhysDerivZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>>>
{
public:
    PhysDerivZOpImpl(
        [[maybe_unused]] const MultiRegions::ExpListSharedPtr &expansionList)
    {
    }

    // Non-copyable and non-movable.
    PhysDerivZOpImpl(const PhysDerivZOpImpl &)            = delete;
    PhysDerivZOpImpl &operator=(const PhysDerivZOpImpl &) = delete;
    PhysDerivZOpImpl(PhysDerivZOpImpl &&)                 = delete;
    PhysDerivZOpImpl &operator=(PhysDerivZOpImpl &&)      = delete;

    void Init([[maybe_unused]] TData beta)
    {
        Unavailable();
    }

    void Launch(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        Unavailable();
    }

private:
    /// \brief Report that this build cannot take the z-derivative.
    static void Unavailable()
    {
        NEKERROR(ErrorUtil::efatal,
                 "PhysDerivOp: the homogeneous z-derivative on the Device "
                 "execution space needs a CUDA build. Rebuild with "
                 "NEKTAR_ENABLE_CUDA, or run this 3DH1 case on the Serial or "
                 "AVX execution space.");
    }
};

#endif // NEKTAR_ENABLE_CUDA

} // namespace Nektar::Operators::detail
