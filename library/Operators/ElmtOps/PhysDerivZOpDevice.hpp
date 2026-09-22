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
/// PhysDerivZDirect, over cuFFT, hipFFT or oneMath. Each block is submitted
/// to its own per-block stream (block index + 1), the same stream its
/// producers and consumers use, so no cross-stream ordering is needed and the
/// blocks can overlap. Under CUDA and HIP each block's pipeline is captured
/// as its own device graph on the first call -- the graph is launched right
/// after it is instantiated, so that call behaves like any other -- and
/// replayed from the second on for lower launch overhead. Everything the
/// capture must not see (block allocation, host-to-device transfers, plan
/// creation) is resolved before it opens. SYCL has no capture, so there
/// m_graphExec stays empty and every call submits the pipeline directly.
template <typename ExecSpace, typename TData>
class PhysDerivZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>>>
{
public:
    /// \param expansionList  Read once for the homogeneous length; the device
    ///                       pipeline otherwise works straight off the block
    ///                       pointers.
    PhysDerivZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(homoExpList,
                 "The homogeneous z-derivative needs an ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / homoExpList->GetHomoLen();
    }

    ~PhysDerivZOpImpl()
    {
        for ([[maybe_unused]] auto &graph : m_graphExec)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            (void)cudaGraphExecDestroy(graph);
#elif defined(NEKTAR_ENABLE_HIP)
            (void)hipGraphExecDestroy(graph);
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

        // Every block has to be captured before any of them can be replayed,
        // and they are all captured on the first call, so a full m_graphExec
        // is what tells a replay apart from that first pass.
        const bool replay = (m_graphExec.size() == numBlocks);

        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            const unsigned int streamID = blk + 1;

            if (replay)
            {
                LaunchGraph(blk);
                continue;
            }

            auto &inblock           = in.GetBlocks()[blk];
            auto &outblock          = out.GetBlocks()[blk];
            const size_t compStride = inblock.CompSize();

            // Resolved before the capture opens: on a first touch these
            // allocate the block storage and copy it in, and a transfer
            // recorded into the graph would replay stale data for ever.
            const TData *inPtr =
                inblock.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>(
                    streamID);
            TData *outPtr =
                outblock.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>(
                    streamID);

            // Likewise for the transform plans and their scratch: creating
            // them allocates and synchronises the stream, neither of which a
            // capture tolerates, so the graph only ever sees the transforms.
            LibUtilities::PhysDerivZPrepare<TData>(nhomo, compStride,
                                                   compStride, streamID);

            BeginCapture(blk);

            for (unsigned int n = 0; n < nComp; ++n)
            {
                // Output component for the z-derivative of input n is 3n + 2.
                const TData *phiPtr = inPtr + n * compStride * nhomo;
                TData *dzPtr        = outPtr + (n * 3 + 2) * compStride * nhomo;
                LibUtilities::PhysDerivZDirect(phiPtr, dzPtr, nhomo, compStride,
                                               compStride, m_beta, streamID);
            }

            // Capture records the pipeline instead of running it, so the
            // freshly instantiated graph has to be launched here for this
            // call to produce the same output as any other.
            EndCapture(blk);
            LaunchGraph(blk);
        }
    }

private:
#if defined(NEKTAR_ENABLE_CUDA)
    std::vector<cudaGraphExec_t> m_graphExec;
#elif defined(NEKTAR_ENABLE_HIP)
    std::vector<hipGraphExec_t> m_graphExec;
#else
    std::vector<void *> m_graphExec;
#endif
    TData m_beta = 0.0;

    void LaunchGraph(const unsigned int blk)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        const unsigned int streamID = blk + 1;
        CHECK_HIPCUDA_ERROR(cudaGraphLaunch(m_graphExec[blk],
                                            CUDAStream::GetInstance(streamID)));
#elif defined(NEKTAR_ENABLE_HIP)
        const unsigned int streamID = blk + 1;
        CHECK_HIPCUDA_ERROR(
            hipGraphLaunch(m_graphExec[blk], HIPStream::GetInstance(streamID)));
#else
        (void)blk;
#endif
    }

    void BeginCapture(const unsigned int blk)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        const unsigned int streamID = blk + 1;
        CHECK_HIPCUDA_ERROR(cudaStreamBeginCapture(
            CUDAStream::GetInstance(streamID), cudaStreamCaptureModeRelaxed));
#elif defined(NEKTAR_ENABLE_HIP)
        const unsigned int streamID = blk + 1;
        CHECK_HIPCUDA_ERROR(hipStreamBeginCapture(
            HIPStream::GetInstance(streamID), hipStreamCaptureModeRelaxed));
#else
        (void)blk;
#endif
    }

    /// Keep @p graphExec as block @p blk's graph, replacing and destroying
    /// whatever it held before, so that a block can be captured again once an
    /// address the graph was built against has moved.
    template <typename TGraphExec>
    void StoreGraph([[maybe_unused]] const unsigned int blk,
                    [[maybe_unused]] TGraphExec graphExec)
    {
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
        if (blk < m_graphExec.size())
        {
#if defined(NEKTAR_ENABLE_CUDA)
            (void)cudaGraphExecDestroy(m_graphExec[blk]);
#else
            (void)hipGraphExecDestroy(m_graphExec[blk]);
#endif
            m_graphExec[blk] = graphExec;
        }
        else
        {
            m_graphExec.push_back(graphExec);
        }
#endif
    }

    void EndCapture(const unsigned int blk)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        const unsigned int streamID = blk + 1;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        CHECK_HIPCUDA_ERROR(
            cudaStreamEndCapture(CUDAStream::GetInstance(streamID), &graph));
#if CUDART_VERSION >= 12000
        CHECK_HIPCUDA_ERROR(cudaGraphInstantiate(&graphExec, graph, 0));
#else
        CHECK_HIPCUDA_ERROR(
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
#endif
        StoreGraph(blk, graphExec);
        CHECK_HIPCUDA_ERROR(cudaGraphDestroy(graph));
#elif defined(NEKTAR_ENABLE_HIP)
        const unsigned int streamID = blk + 1;
        hipGraph_t graph;
        hipGraphExec_t graphExec;
        CHECK_HIPCUDA_ERROR(
            hipStreamEndCapture(HIPStream::GetInstance(streamID), &graph));
        CHECK_HIPCUDA_ERROR(
            hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        StoreGraph(blk, graphExec);
        CHECK_HIPCUDA_ERROR(hipGraphDestroy(graph));
#else
        (void)blk;
#endif
    }
};

} // namespace Nektar::Operators::detail
