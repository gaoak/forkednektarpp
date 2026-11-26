///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtBlockOpDeviceSumFacTOP.hpp
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
// Description: Device SumFacTOP implementation of the per-block fused element-
// local advection-diffusion operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtBlockOpDeviceSumFacTOP.hpp
 * @brief Device SumFacTOP implementation of the per-block fused
 * element-local advection-diffusion operator.
 *
 * @details
 * One kernel launch per block and application, one element per thread
 * block and one thread per point; the kernel is AdvDiffFusedElmtKernel() in
 * AdvDiffFusedElmtBlockOpDeviceSumFacTOPKernels.hpp. The Device has this
 * implementation only: CMake registers it under SumFacTOP alone
 * (ADD_POLICY_ELMT_BLOCK_OPERATOR).
 *
 * The implementation builds on what the Device implementations of the fused
 * element-local operators share (detail::AdvectionFusedElmtDeviceBlock,
 * AdvectionFusedElmtBlockOpDeviceSumFacTOP.hpp): the basis data and the
 * advection fields. It adds the trace
 * normals, the normal viscous flux on the traces and the gradient trace of
 * the listed elements.
 *
 * A block is refused, so that the caller takes the separate operators, when
 * its elements have more points than a thread block has threads, or when
 * the kernel's shared memory exceeds what a thread block may have.
 *
 * @note The operator is a class template on the flux policy, so the code
 * supplying the policy generates the translation units that include this
 * header, through ADD_POLICY_ELMT_BLOCK_OPERATOR (cmake/NektarCommon.cmake).
 * Other code goes through AdvDiffFusedElmtOp.hpp and the operator factory.
 */

#pragma once

#include <MultiRegions/DataWarehouse/TraceDataWarehouse.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtBlockOp.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtBlockOpDeviceSumFacTOPKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpDeviceSumFacTOP.hpp>

#include <memory>
#include <type_traits>
#include <vector>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The fused element-local advection-diffusion operator of one block
 * in the Device execution space, one element per thread block.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFacTOP.
 * @tparam Physics         The flux policy (AdvDiffFusedElmtKernels.hpp).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename Physics,
          typename TData>
class AdvDiffFusedElmtBlockOpImpl
    : public AdvectionFusedElmtDeviceBlock<ExecSpace, Physics, TData,
                                           AdvDiffFusedElmtBlockOp>
{
    using DeviceBlock = AdvectionFusedElmtDeviceBlock<ExecSpace, Physics, TData,
                                                      AdvDiffFusedElmtBlockOp>;
    using BlockOpBase = AdvDiffFusedElmtBlockOp<TData>;
    using Block       = typename DeviceBlock::Block;
    using MemSpace    = typename DeviceBlock::MemSpace;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NAUX = Physics::NAUX;
    static_assert(std::is_same_v<Implementation, MultiRegions::SumFacTOP>,
                  "The Device implementation of the fused element-local "
                  "operator is SumFacTOP");

public:
    /**
     * @brief Fetch the block's basis data and trace normals, and bind its
     * device stream.
     *
     * @param   block_idx       Index of the block; the stream used for all
     *                          device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvDiffFusedElmtBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DeviceBlock(block_idx, exp, dataWarehouse)
    {
        m_normalsptr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::LocalTraceNormalKey<TData>(
                block_idx, DeviceBlock::m_implInterleaveWidth));
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory: null for a
    /// block the implementation does not take (TakesBlock()) or whose
    /// kernel needs more shared memory than a thread block has.
    /// AdvDiffFusedElmtOp::Create() then returns no operator at all, and its
    /// caller keeps the separate element operators for the whole field:
    /// there is no other Device implementation of the fused pass to fall
    /// back on.
    static std::unique_ptr<MultiRegions::BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        std::unique_ptr<AdvDiffFusedElmtBlockOpImpl> op;
        if (DeviceBlock::TakesBlock(exp))
        {
            op = std::make_unique<AdvDiffFusedElmtBlockOpImpl>(block_idx, exp,
                                                               dataWarehouse);
            if (!DeviceBlock::FitsSharedMemory(
                    AdvDiffFusedElmtSharedMemorySize<NDIM>(
                        op->template NonTemplatedTraceSize<NDIM>(),
                        op->m_packing.nqTotTrace)))
            {
                op.reset();
            }
        }
        return op;
    }

protected:
    /// Each element's outward normals on its traces, at width one.
    const TData *m_normalsptr = nullptr;
    /// The elements whose gradient trace is written, in list order, and
    /// the block's number of elements with padding
    /// (SetGradientTraceElements()); and each padded element's offset in the
    /// gradient trace block, or AdvDiffFusedElmtDeviceFields::noGradTrace,
    /// built for #m_gradTraceInterleaveWidth, the one thing that depends on
    /// the call.
    std::vector<unsigned int> m_gradTraceElements;
    size_t m_gradTraceNumElmts = 0;
    LibUtilities::MemoryRegion<size_t> m_gradTraceOffset;
    unsigned int m_gradTraceInterleaveWidth = 0;
    /// The viscous outputs ApplyFusedElmt() writes, held for that
    /// application alone and null otherwise.
    Block *m_normVisc  = nullptr;
    Block *m_gradTrace = nullptr;

    void v_SetGradientTraceElements(const std::vector<unsigned int> &elements,
                                    const size_t nelmtPad) override
    {
        m_gradTraceElements        = elements;
        m_gradTraceNumElmts        = nelmtPad;
        m_gradTraceInterleaveWidth = 0;
    }

    /**
     * @brief Each of the block's padded elements' offset in the gradient
     * trace block at @p interleaveWidth, the listed elements in list order,
     * interleaved as that block is, in device memory.
     */
    void BuildGradTraceOffsets(const unsigned int interleaveWidth)
    {
        const size_t nqTotTrace = this->m_packing.nqTotTrace;
        std::vector<size_t> offsets(
            m_gradTraceNumElmts,
            AdvDiffFusedElmtDeviceFields<TData>::noGradTrace);
        for (size_t k = 0; k < m_gradTraceElements.size(); ++k)
        {
            offsets[m_gradTraceElements[k]] =
                (k / interleaveWidth) * nqTotTrace * interleaveWidth +
                k % interleaveWidth;
        }
        m_gradTraceOffset =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                offsets, this->m_streamID);
        m_gradTraceInterleaveWidth = interleaveWidth;
    }

    void v_ApplyAdvDiffFusedElmt(Block &state, Block *aux, Block &trace,
                                 Block &traceNormVisc, Block *gradTrace,
                                 Block &out) override
    {
        ASSERTL1(static_cast<bool>(aux) == (NAUX > 0),
                 "The auxiliary fields are given exactly when the policy has "
                 "them.");

        this->m_aux   = aux;
        this->m_trace = &trace;
        m_normVisc    = &traceNormVisc;
        // The listed elements and what depends on them alone were set with
        // the gradient trace (SetGradientTraceElements()); its width is the
        // caller's, and may change between calls.
        m_gradTrace = m_gradTraceElements.empty() ? nullptr : gradTrace;
        if (m_gradTrace &&
            m_gradTrace->GetInterleaveWidth() != m_gradTraceInterleaveWidth)
        {
            BuildGradTraceOffsets(m_gradTrace->GetInterleaveWidth());
        }

        this->ShapeSwitch(this->m_shapeType, [&](auto shape) {
            if constexpr (BlockOpBase::TakesShape(decltype(shape)::value, NDIM))
            {
                this->template ShapeBlock<decltype(shape)::value>(state, out);
            }
            else
            {
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
            }
        });

        this->m_aux   = nullptr;
        this->m_trace = nullptr;
        m_normVisc    = nullptr;
        m_gradTrace   = nullptr;
    }

    // Shape-specific entry points, specialised for each shape of the
    // policy's dimension in the translation units generated from
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /**
     * @brief The fused pass over the block, with the trace size parameter
     * the generated switch chose: one launch covering every element. The
     * gradient trace is written at its own width.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    void OperatorND(Block &inblock, Block &outblock,
                    const TTraceSizeParameter traceSizeParam)
    {
        const unsigned int nqTotTrace = this->m_packing.nqTotTrace;

        AdvDiffFusedElmtDeviceFields<TData> fields;
        this->FillFields(inblock, outblock, fields);
        // The normal viscous flux and the normals are laid out as the trace,
        // and share its component size.
        ASSERTL1(m_normVisc->CompSize() == fields.traceCompSize,
                 "The normal viscous flux shares the trace's component size.");
        fields.normVisc =
            m_normVisc->template GetPtr<MemSpace, WriteOnly>(this->m_streamID);
        fields.normals = m_normalsptr;
        if (m_gradTrace)
        {
            fields.gradTrace =
                m_gradTrace->template GetPtr<MemSpace, ReadWrite>(
                    this->m_streamID);
            fields.gradTraceCompSize = m_gradTrace->CompSize();
            fields.gradTraceWidth    = m_gradTraceInterleaveWidth;
            fields.gradTraceOffset =
                m_gradTraceOffset.template GetPtr<MemSpace, ReadOnly>(
                    this->m_streamID);
        }

        const unsigned int shmemsize =
            sizeof(TData) *
            AdvDiffFusedElmtSharedMemorySize<NDIM>(traceSizeParam, nqTotTrace);
        // One element per thread block and one thread per point, which
        // carries the point's values from stage to stage and so cannot
        // stride: the block may take up to fusedElmtMaxBlockSize threads.
        const auto kernelParams =
            MultiRegions::GetDeviceKernelParameters<MultiRegions::SumFacTOP>(
                this->m_nqTot, fields.nelmtPad, shmemsize, 1u,
                MultiRegions::defaultPassesPerBlock, fusedElmtMaxBlockSize);
        ASSERTL0(kernelParams.blockSize >= this->m_nqTot,
                 "The fused pass keeps one thread on each point of an element, "
                 "and TakesBlock() refuses an element with more points than a "
                 "thread block has threads.");

        // The kernel reads and writes at width one. The state and the
        // auxiliary fields may each arrive at their own width and go back to
        // it; the outputs leave at the state's. The gradient trace is written
        // at its own width in place.
        constexpr unsigned int implInterleaveWidth =
            DeviceBlock::m_implInterleaveWidth;
        const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
        const unsigned int auxInterleaveWidth =
            this->m_aux ? this->m_aux->GetInterleaveWidth()
                        : implInterleaveWidth;
        inblock.ReshapeStorage(implInterleaveWidth, ExecSpace::name,
                               this->m_streamID);
        if (this->m_aux)
        {
            this->m_aux->ReshapeStorage(implInterleaveWidth, ExecSpace::name,
                                        this->m_streamID);
        }

        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (AdvDiffFusedElmtKernel<SHAPE_TYPE, DEFORMED, Physics>),
            kernelParams.gridSize, kernelParams.blockSize,
            kernelParams.shmemSize, this->m_streamID, traceSizeParam,
            this->m_physics, this->m_basisData, fields);

        inblock.ReshapeStorage(inInterleaveWidth, ExecSpace::name,
                               this->m_streamID);
        if (this->m_aux)
        {
            this->m_aux->ReshapeStorage(auxInterleaveWidth, ExecSpace::name,
                                        this->m_streamID);
        }
        for (Block *written : {&outblock, this->m_trace, m_normVisc})
        {
            written->template SetInterleaveWidth<TData>(implInterleaveWidth);
            written->ReshapeStorage(inInterleaveWidth, ExecSpace::name,
                                    this->m_streamID);
        }
    }
};

} // namespace Nektar::SolverCore::detail
