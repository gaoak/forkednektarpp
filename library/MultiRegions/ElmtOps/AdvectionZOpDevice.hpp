///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionZOpDevice.hpp
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
// Description: Device (cuFFT/cuFFTDx) backend for the AdvectionOp z-term.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

// Brings in the host API of whichever backend is enabled, and with it
// CUDAStream / HIPStream / SYCLQueue and CHECK_HIPCUDA_ERROR.
#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Math/MathKernels.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/FFT/DerivZDeviceFFT.h>

#include <MultiRegions/Common/BlockOperator.hpp>

namespace Nektar::MultiRegions::detail
{

/// \brief Device backend for the homogeneous part of the advection term.
///
/// Launch() applies a D2Z + wavenumber multiply + Z2D pipeline per block via
/// DerivZDirect, over cuFFT, hipFFT or oneMath. The transform writes to a
/// scratch buffer rather than to the output, because the xy backend has
/// already left scale * (u dphi/dx + v dphi/dy) there and scale * w dphi/dz
/// has to be added to it; mulKernel scales the derivative by w in place and
/// daxpyKernel adds the result on. The transform's own APPEND mode would not
/// serve here: the derivative has to be weighted by w before it reaches the
/// output. The scratch is the per-stream workspace the block operators share,
/// so this pass allocates nothing of its own.
/// Each block is submitted to its own per-block stream (block index + 1), the
/// same stream its producers and consumers use, so no cross-stream ordering
/// is needed and the blocks can overlap. Under CUDA and HIP each block's
/// pipeline is captured as its own device graph on the first call -- the
/// graph is launched right after it is instantiated, so that call behaves
/// like any other -- and replayed from the second on for lower launch
/// overhead. Everything the capture must not see (block allocation,
/// host-to-device transfers, workspace growth, plan creation) is resolved
/// before it opens, and because the shared workspace moves when another
/// operator outgrows it, a block whose workspace address has changed is
/// captured afresh instead of replayed. SYCL has no capture, so there
/// m_graphExec stays empty and every call submits the pipeline directly.
///
/// The scale and the advection velocity's device pointer are both baked into
/// the capture, so a SetScale() or a SetAdvVel() naming a different Field
/// after the first call is picked up only by rebuilding the operator. Passing
/// the same Field with new contents is fine: only its address is recorded.
template <typename ExecSpace, typename TData>
class AdvectionZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>>>
{
public:
    /// \param expansionList  Read once for the homogeneous length; the device
    ///                       pipeline otherwise works straight off the block
    ///                       pointers.
    AdvectionZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(homoExpList,
                 "The homogeneous advection needs an ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / homoExpList->GetHomoLen();
    }

    ~AdvectionZOpImpl()
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
    AdvectionZOpImpl(const AdvectionZOpImpl &)            = delete;
    AdvectionZOpImpl &operator=(const AdvectionZOpImpl &) = delete;
    AdvectionZOpImpl(AdvectionZOpImpl &&)                 = delete;
    AdvectionZOpImpl &operator=(AdvectionZOpImpl &&)      = delete;

    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        m_advVel = &advVel;
    }

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

        // The advection velocity along the homogeneous direction is the last
        // component; the xy backend has already consumed the others.
        const unsigned int wComp = m_advVel->GetNumComponents() - 1;

        for (unsigned int blk = 0; blk < numBlocks; ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inblock           = in.GetBlocks()[blk];
            auto &outblock          = out.GetBlocks()[blk];
            auto &velblock          = m_advVel->GetBlocks()[blk];
            const size_t compStride = inblock.CompSize();
            const size_t nsize      = compStride * nhomo;

            // The xy backend reshapes the advection velocity into its own
            // interleave format and leaves it there, so it is realigned with
            // the input before being read. This sits ahead of the replay
            // shortcut and outside the capture: the xy backend interleaves it
            // again on every call, so the realignment has to run on every
            // call too, while the graph records only the transforms.
            const auto interleaveWidth = inblock.GetInterleaveWidth();
            if (velblock.GetInterleaveWidth() != interleaveWidth)
            {
                auto velRWPtr =
                    velblock
                        .template GetPtr<NektarSpaces::DeviceSpace, ReadWrite>(
                            streamID);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, velblock.GetInterleaveWidth(),
                    velblock.GetNumElementsWithPadding() *
                        velblock.GetNumComponents() *
                        velblock.GetNumHomoModes(),
                    velblock.GetNumData(), velRWPtr, streamID);
                velblock.template SetInterleaveWidth<TData>(interleaveWidth);
            }

            // Scratch for the transform, taken from the per-stream workspace
            // the block operators share. That workspace is reallocated
            // whenever one of them asks for more than it currently holds, and
            // a captured graph keeps the address it was built against, so a
            // block is replayed only while its workspace has not moved and is
            // captured again when it has.
            TData *dzPtr = BlockOperator<TData>::template GetStaticWorkSpace<
                NektarSpaces::DeviceSpace>(nsize, streamID);

            if (replay && m_wsp[blk] == dzPtr)
            {
                LaunchGraph(blk);
                continue;
            }

            // Resolved before the capture opens: on a first touch these
            // allocate the block storage and copy it in, and a transfer
            // recorded into the graph would replay stale data for ever.
            const TData *inPtr =
                inblock.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>(
                    streamID);
            TData *outPtr =
                outblock.template GetPtr<NektarSpaces::DeviceSpace, ReadWrite>(
                    streamID);
            const TData *velPtr =
                velblock.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>(
                    streamID);

            // Likewise for the transform plans and their own scratch:
            // creating them allocates and synchronises the stream, neither of
            // which a capture tolerates, so the graph only ever sees the
            // transforms.
            LibUtilities::DerivZPrepare<TData, false>(nhomo, compStride,
                                                      compStride, streamID);

            // Record which workspace this capture is built against.
            if (m_wsp.size() > blk)
            {
                m_wsp[blk] = dzPtr;
            }
            else
            {
                m_wsp.push_back(dzPtr);
            }

            BeginCapture(blk);

            const TData *wPtr = velPtr + wComp * nsize;

            for (unsigned int n = 0; n < nComp; ++n)
            {
                const TData *phiPtr = inPtr + n * nsize;
                TData *advPtr       = outPtr + n * nsize;

                LibUtilities::DerivZDirect<
                    TData, LibUtilities::DerivZOrder::First, false>(
                    phiPtr, dzPtr, nhomo, compStride, compStride, m_beta,
                    streamID);

                // The output already holds the xy part of the advection
                // term. Both steps run over the whole component, padding
                // included: a padded xy slot only ever feeds itself.
                Math::mulKernel<ExecSpace>(nsize, wPtr, dzPtr, dzPtr, streamID);
                Math::daxpyKernel<ExecSpace>(nsize, m_scale, dzPtr, advPtr,
                                             advPtr, streamID);
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
    std::vector<TData *> m_wsp;
    LibUtilities::Field<TData, FieldState::Phys> *m_advVel = nullptr;
    TData m_beta                                           = 0.0;
    TData m_scale                                          = 1.0;

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

} // namespace Nektar::MultiRegions::detail
