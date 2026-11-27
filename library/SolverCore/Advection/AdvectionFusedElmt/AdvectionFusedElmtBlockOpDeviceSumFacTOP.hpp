///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtBlockOpDeviceSumFacTOP.hpp
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
// local advection operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtBlockOpDeviceSumFacTOP.hpp
 * @brief Device SumFacTOP implementation of the per-block fused
 * element-local advection operator, and what the Device implementations of
 * the fused element-local operators share.
 *
 * @details
 * One kernel launch per block and application, one element per thread
 * block and one thread per point; the kernel is AdvectionFusedElmtKernel() in
 * AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp. The Device has this
 * implementation only: CMake registers it under SumFacTOP alone
 * (ADD_POLICY_ELMT_BLOCK_OPERATOR).
 *
 * detail::AdvectionFusedElmtDeviceBlock holds what the advection and the
 * advection-diffusion implementations share: the basis data fetched from the
 * data warehouse with the keys the separate operators' SumFacTOP
 * implementations use (the derivative factors transposed and every geometric
 * factor at interleave width one) and the fields of a launch. It is
 * templated on the block-operator base it derives from,
 * AdvectionFusedElmtBlockOp or AdvDiffFusedElmtBlockOp. Each implementation's
 * OperatorND() reshapes the inputs to width one and the outputs back around
 * its launch.
 *
 * ApplyFusedElmt() switches on the block's shape into the generated
 * ShapeBlock(), which calls OperatorND() with the trace size parameter of the
 * trace operators' size switch, compile-time inside the configured range and
 * runtime outside it.
 *
 * A block is refused, so that the caller takes the separate operators, when
 * its elements have more points than a thread block has threads, or when
 * the kernel's shared memory exceeds what a thread block may have.
 *
 * @note The operator is a class template on the flux policy, so the code
 * supplying the policy generates the translation units that include this
 * header, through ADD_POLICY_ELMT_BLOCK_OPERATOR (cmake/NektarCommon.cmake).
 * Other code goes through AdvectionFusedElmtOp.hpp and the operator factory.
 */

#pragma once

#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <MultiRegions/ElmtOps/TraceTables.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp>

#include <memory>
#include <type_traits>
#include <vector>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::SolverCore::detail
{

/**
 * @brief What the Device implementations of the fused element-local
 * operators share, one element per thread block.
 *
 * @tparam ExecSpace     NektarSpaces::Device.
 * @tparam Physics       The flux policy.
 * @tparam TData         Floating-point type of the field data.
 * @tparam TBlockOpBase  The block-operator base template:
 *                       AdvectionFusedElmtBlockOp or one derived from it,
 *                       instantiated on @p TData.
 */
template <typename ExecSpace, typename Physics, typename TData,
          template <typename> typename TBlockOpBase>
class AdvectionFusedElmtDeviceBlock : public TBlockOpBase<TData>
{
protected:
    using MemSpace = typename ExecSpace::memory_space;
    using Block    = LibUtilities::BlockAccessor<TData, FieldState::Phys>;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NVAR = Physics::NVAR;
    static constexpr unsigned int NAUX = Physics::NAUX;
    static_assert(NDIM >= 1 && NDIM <= 3,
                  "The fused element-local operators take one-, two- and "
                  "three-dimensional meshes");

    /// Interleave width the kernels read and write at: an element's values
    /// contiguous.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// The flux policy (InitialisePolicy()).
    Physics m_physics;
    /// The basis data the kernels read (FetchBasisData()).
    AdvectionFusedElmtDeviceBasisData<TData> m_basisData;
    /// Derivative factors (transposed) and Jacobians of the block, at width
    /// one.
    const TData *m_dfptr  = nullptr;
    const TData *m_jacptr = nullptr;
    /// The auxiliary fields and the trace ApplyFusedElmt() reads and writes
    /// besides the state and the derivative inner products, held for that
    /// application alone and null otherwise.
    Block *m_aux   = nullptr;
    Block *m_trace = nullptr;

    /**
     * @brief Fetch the block's basis data and bind its device stream.
     *
     * @param   block_idx       Index of the block; the stream used for all
     *                          device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionFusedElmtDeviceBlock(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : TBlockOpBase<TData>(block_idx, exp, dataWarehouse),
          m_streamID(block_idx + 1)
    {
        FetchBasisData(exp, block_idx);
    }

    ~AdvectionFusedElmtDeviceBlock() override = default;

    /// Whether an implementation takes a block of @p exp: an element of the
    /// policy's dimension in a space of the same dimension, of a shape the
    /// kernels are compiled for (TakesShape()), with no more points than a
    /// thread block has threads.
    static bool TakesBlock(const LocalRegions::ExpansionSharedPtr &exp)
    {
        return exp->GetShapeDimension() == NDIM && exp->GetCoordim() == NDIM &&
               TBlockOpBase<TData>::TakesShape(exp->DetShapeType(), NDIM) &&
               static_cast<unsigned int>(exp->GetTotPoints()) <=
                   fusedElmtMaxBlockSize;
    }

    /// Whether @p size elements of shared memory fit the device's shared
    /// memory per thread block.
    static bool FitsSharedMemory(const size_t size)
    {
        return sizeof(TData) * size <=
               GetDeviceProperties::SharedMemoryPerBlock();
    }

    /// The block's basis data (the differentiation matrices, the quadrature
    /// weights, the collapsed-coordinate factors and the trace interpolation
    /// tables) and its geometric factors, as PhysDeriv,
    /// IProductWRTDerivBase and PhysTraceExtract fetch them for SumFacTOP.
    void FetchBasisData(const LocalRegions::ExpansionSharedPtr &exp,
                        const unsigned int block_idx)
    {
        auto &warehouse = this->m_dataWarehouse;
        auto &basisData = m_basisData;

        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const auto bkey          = exp->GetBasis(d)->GetBasisKey();
            basisData.derivMatrix[d] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey,
                                                  LibUtilities::eDerivative));
            basisData.weights[d] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey,
                                                  LibUtilities::eWeights));
        }
        if constexpr (NDIM == 2)
        {
            basisData.factors[0] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[1] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
        }
        else if constexpr (NDIM == 3)
        {
            basisData.factors[0] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[1] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[2] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
            basisData.factors[3] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
        }

        const auto traceTables =
            MultiRegions::detail::FetchTraceTables<TData, MemSpace>(exp,
                                                                    warehouse);
        for (size_t i = 0; i < traceTables.B.size(); ++i)
        {
            basisData.traceBasis[i] = traceTables.B[i];
        }
        for (size_t i = 0; i < traceTables.endPtsCollocated.size(); ++i)
        {
            basisData.endPts[i] = traceTables.endPtsCollocated[i];
        }
        for (size_t i = 0; i < traceTables.isCollocated.size(); ++i)
        {
            basisData.colloc[i] = traceTables.isCollocated[i];
        }

        m_dfptr = warehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
        m_jacptr = warehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    void v_InitialisePolicy(
        const LibUtilities::SessionReaderSharedPtr &session) override
    {
        m_physics = Physics(session);
    }

    /// The advection fields of a launch over @p inblock into @p outblock and
    /// #m_trace, with #m_aux.
    void FillFields(Block &inblock, Block &outblock,
                    AdvectionFusedElmtDeviceFields<TData> &fields)
    {
        fields.nelmt    = inblock.GetNumElements();
        fields.nelmtPad = inblock.GetNumElementsWithPadding();
        // The fields all come from the block's elements: the volume fields
        // share one component size and the element-local traces another.
        ASSERTL1(outblock.CompSize() == inblock.CompSize() &&
                     (!m_aux || m_aux->CompSize() == inblock.CompSize()),
                 "The state, the auxiliary fields and the derivative inner "
                 "products share the block's component size.");
        fields.volCompSize   = inblock.CompSize();
        fields.traceCompSize = m_trace->CompSize();
        fields.state = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        if (m_aux)
        {
            fields.aux = m_aux->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        }
        fields.out = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        fields.trace =
            m_trace->template GetPtr<MemSpace, WriteOnly>(m_streamID);
        fields.df  = m_dfptr;
        fields.jac = m_jacptr;
    }
};

/**
 * @brief The fused element-local advection operator of one block in the
 * Device execution space, one element per thread block.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFacTOP.
 * @tparam Physics         The flux policy (AdvectionFusedElmtKernels.hpp).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename Physics,
          typename TData>
class AdvectionFusedElmtBlockOpImpl
    : public AdvectionFusedElmtDeviceBlock<ExecSpace, Physics, TData,
                                           AdvectionFusedElmtBlockOp>
{
    using DeviceBlock =
        AdvectionFusedElmtDeviceBlock<ExecSpace, Physics, TData,
                                      AdvectionFusedElmtBlockOp>;
    using BlockOpBase = AdvectionFusedElmtBlockOp<TData>;
    using Block       = typename DeviceBlock::Block;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NAUX = Physics::NAUX;
    static_assert(std::is_same_v<Implementation, MultiRegions::SumFacTOP>,
                  "The Device implementation of the fused element-local "
                  "operator is SumFacTOP");

public:
    /**
     * @brief Fetch the block's basis data and bind its device stream.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionFusedElmtBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DeviceBlock(block_idx, exp, dataWarehouse)
    {
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory: null for a
    /// block the implementation does not take (TakesBlock()) or whose
    /// kernel needs more shared memory than a thread block has.
    /// AdvectionFusedElmtOp::Create() then returns no operator at all, and
    /// its caller keeps the separate element operators for the whole field:
    /// there is no other Device implementation of the fused pass to fall
    /// back on.
    static std::unique_ptr<MultiRegions::BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        std::unique_ptr<AdvectionFusedElmtBlockOpImpl> op;
        if (DeviceBlock::TakesBlock(exp))
        {
            op = std::make_unique<AdvectionFusedElmtBlockOpImpl>(block_idx, exp,
                                                                 dataWarehouse);
            if (!DeviceBlock::FitsSharedMemory(
                    AdvectionFusedElmtSharedMemorySize<NDIM>(
                        op->template NonTemplatedTraceSize<NDIM>())))
            {
                op.reset();
            }
        }
        return op;
    }

protected:
    void v_ApplyFusedElmt(Block &state, Block *aux, Block &trace,
                          Block &out) override
    {
        ASSERTL1(static_cast<bool>(aux) == (NAUX > 0),
                 "The auxiliary fields are given exactly when the policy has "
                 "them.");

        this->m_aux   = aux;
        this->m_trace = &trace;

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
    }

    // Shape-specific entry points, specialised for each shape of the
    // policy's dimension in the translation units generated from
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /**
     * @brief The fused pass over the block, with the trace size parameter
     * the generated switch chose: one launch covering every element.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    void OperatorND(Block &inblock, Block &outblock,
                    const TTraceSizeParameter traceSizeParam)
    {
        AdvectionFusedElmtDeviceFields<TData> fields;
        this->FillFields(inblock, outblock, fields);

        const unsigned int shmemsize =
            sizeof(TData) *
            AdvectionFusedElmtSharedMemorySize<NDIM>(traceSizeParam);
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
        // it; the outputs leave at the state's.
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
            (AdvectionFusedElmtKernel<SHAPE_TYPE, DEFORMED, Physics>),
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
        for (Block *written : {&outblock, this->m_trace})
        {
            written->template SetInterleaveWidth<TData>(implInterleaveWidth);
            written->ReshapeStorage(inInterleaveWidth, ExecSpace::name,
                                    this->m_streamID);
        }
    }
};

} // namespace Nektar::SolverCore::detail
