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

#include <vector>

// Brings in the host API of whichever backend is enabled, and with it
// CUDAStream / HIPStream / SYCLQueue and CHECK_HIPCUDA_ERROR.
#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/FFT/PhysDerivZDeviceFFT.h>

namespace Nektar::Operators::detail
{

/// \brief Device backend for the homogeneous z-derivative.
///
/// Launch() applies a D2Z + wavenumber multiply + Z2D pipeline per block via
/// PhysDerivZDirect, over cuFFT, hipFFT or oneMath. Under CUDA and HIP the
/// pipeline is captured as a device graph from the second call on and
/// replayed for lower launch overhead; SYCL submits it every call, leaving
/// m_graphExec null and m_pendingCapture false, so the capture and replay
/// paths are never entered there.
template <typename ExecSpace, typename TData>
class PhysDerivZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>>>
{
public:
    /// \param expansionList  Unused; the device pipeline works straight off
    ///                       the block pointers. Present so that both
    ///                       backends are constructed the same way.
    PhysDerivZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(homoExpList,
                 "The homogeneous z-derivative needs an ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / homoExpList->GetHomoLen();

#if defined(NEKTAR_ENABLE_CUDA)
        int leastPriority    = 0;
        int greatestPriority = 0;
        CHECK_HIPCUDA_ERROR(cudaDeviceGetStreamPriorityRange(
            &leastPriority, &greatestPriority));
        CHECK_HIPCUDA_ERROR(cudaStreamCreateWithPriority(
            &m_stream, cudaStreamDefault, greatestPriority));
        m_pendingCapture = false;
        m_graphExec      = nullptr;
#elif defined(NEKTAR_ENABLE_HIP)
        int leastPriority    = 0;
        int greatestPriority = 0;
        CHECK_HIPCUDA_ERROR(
            hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CHECK_HIPCUDA_ERROR(hipStreamCreateWithPriority(
            &m_stream, hipStreamDefault, greatestPriority));
        m_pendingCapture = false;
        m_graphExec      = nullptr;
#elif defined(NEKTAR_ENABLE_SYCL)
        m_stream = &SYCLQueue::GetInstance(0);
#endif
    }

    ~PhysDerivZOpImpl()
    {
        if (m_graphExec)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            (void)cudaGraphExecDestroy(m_graphExec);
#elif defined(NEKTAR_ENABLE_HIP)
            (void)hipGraphExecDestroy(m_graphExec);
#endif
        }
        if (m_stream)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            (void)cudaStreamDestroy(m_stream);
#elif defined(NEKTAR_ENABLE_HIP)
            (void)hipStreamDestroy(m_stream);
#endif
        }
    }

    // Non-copyable and non-movable.
    PhysDerivZOpImpl(const PhysDerivZOpImpl &)            = delete;
    PhysDerivZOpImpl &operator=(const PhysDerivZOpImpl &) = delete;
    PhysDerivZOpImpl(PhysDerivZOpImpl &&)                 = delete;
    PhysDerivZOpImpl &operator=(PhysDerivZOpImpl &&)      = delete;

    void Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();
        const unsigned int nComp = in.GetNumComponents();
        const unsigned int numBlocks =
            static_cast<unsigned int>(in.GetBlocks().size());

        // Nothing otherwise orders m_stream against the per-block producer
        // streams whose writes the z-FFT reads below: under CUDA and HIP it
        // is a private stream unknown to CUDAStream, and under SYCL a queue
        // the producers never submit to. Kept outside graph capture: replay
        // must re-synchronize against fresh input on every call.
        WaitForBlockProducers(numBlocks);

        if (m_graphExec)
        {
            LaunchGraph();
            return;
        }

        if (m_pendingCapture)
        {
            BeginCapture();
        }

        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            auto &inblock           = in.GetBlocks()[blk];
            auto &outblock          = out.GetBlocks()[blk];
            const size_t compStride = inblock.CompSize();
            const TData *inPtr =
                inblock.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
            TData *outPtr =
                outblock
                    .template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();

            for (unsigned int n = 0; n < nComp; ++n)
            {
                // Output component for the z-derivative of input n is 3n + 2.
                const TData *phiPtr = inPtr + n * compStride * nhomo;
                TData *dzPtr        = outPtr + (n * 3 + 2) * compStride * nhomo;
                LibUtilities::PhysDerivZDirect(phiPtr, dzPtr, nhomo, compStride,
                                               compStride, m_beta, m_stream);
            }
        }

        if (m_pendingCapture)
        {
            EndCapture();
        }

        Finalise();
    }

private:
#if defined(NEKTAR_ENABLE_CUDA)
    cudaStream_t m_stream       = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_pendingCapture       = false;
#elif defined(NEKTAR_ENABLE_HIP)
    hipStream_t m_stream       = nullptr;
    hipGraphExec_t m_graphExec = nullptr;
    bool m_pendingCapture      = false;
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue *m_stream = nullptr;
    void *m_graphExec     = nullptr;
    bool m_pendingCapture = false;
#else
    void *m_stream        = nullptr;
    void *m_graphExec     = nullptr;
    bool m_pendingCapture = false;
#endif
    TData m_beta = 0.0;

    /// \brief Make m_stream wait on every per-block producer stream
    ///        (block_idx + 1) before it reads their data.
    ///
    /// Under CUDA and HIP the event is recorded here rather than left to the
    /// producer, so the ordering holds regardless of what last wrote to
    /// `in`. Under SYCL the producers publish their last event through
    /// SYCLQueue, and a barrier on m_stream depending on all of them does the
    /// same job; m_stream is in-order, so the pipeline submitted afterwards
    /// inherits the wait.
    void WaitForBlockProducers([[maybe_unused]] unsigned int numBlocks)
    {
#if defined(NEKTAR_ENABLE_SYCL)
        std::vector<unsigned int> eventIDs(numBlocks);
        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            eventIDs[blk] = blk + 1;
        }
        SetStreamDependencies<NektarSpaces::Device>(0, eventIDs);
#else
        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            const unsigned int streamID = blk + 1;
#if defined(NEKTAR_ENABLE_CUDA)
            CUDAStream::RecordEvent(streamID);
            CHECK_HIPCUDA_ERROR(
                cudaStreamWaitEvent(m_stream, CUDAStream::GetEvent(streamID)));
#elif defined(NEKTAR_ENABLE_HIP)
            HIPStream::RecordEvent(streamID);
            CHECK_HIPCUDA_ERROR(
                hipStreamWaitEvent(m_stream, HIPStream::GetEvent(streamID)));
#else
            (void)streamID;
#endif
        }
#endif
    }

    void LaunchGraph(void)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaGraphLaunch(m_graphExec, m_stream));
        CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(m_stream));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipGraphLaunch(m_graphExec, m_stream));
        CHECK_HIPCUDA_ERROR(hipStreamSynchronize(m_stream));
#endif
    }

    void BeginCapture(void)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipStreamBeginCapture(m_stream, hipStreamCaptureModeRelaxed));
#endif
    }

    void EndCapture(void)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaGraph_t graph;
        CHECK_HIPCUDA_ERROR(cudaStreamEndCapture(m_stream, &graph));
#if CUDART_VERSION >= 12000
        CHECK_HIPCUDA_ERROR(cudaGraphInstantiate(&m_graphExec, graph, 0));
#else
        CHECK_HIPCUDA_ERROR(
            cudaGraphInstantiate(&m_graphExec, graph, nullptr, nullptr, 0));
#endif
        CHECK_HIPCUDA_ERROR(cudaGraphDestroy(graph));
#elif defined(NEKTAR_ENABLE_HIP)
        hipGraph_t graph;
        CHECK_HIPCUDA_ERROR(hipStreamEndCapture(m_stream, &graph));
        CHECK_HIPCUDA_ERROR(
            hipGraphInstantiate(&m_graphExec, graph, nullptr, nullptr, 0));
        CHECK_HIPCUDA_ERROR(hipGraphDestroy(graph));
#endif
    }

    void Finalise(void)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(m_stream));
        m_pendingCapture = (m_graphExec == nullptr);
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipStreamSynchronize(m_stream));
        m_pendingCapture = (m_graphExec == nullptr);
#elif defined(NEKTAR_ENABLE_SYCL)
        m_stream->wait();
#endif
    }
};

} // namespace Nektar::Operators::detail
