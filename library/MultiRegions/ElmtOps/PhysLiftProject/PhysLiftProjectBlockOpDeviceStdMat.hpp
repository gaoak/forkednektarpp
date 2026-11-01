///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceStdMat.hpp
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
// Description: Device StdMat implementation of the per-block lift-and-project
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceStdMat.hpp
 * @brief Device StdMat block implementation of the lift-and-project
 * operator: the dense projector on affine blocks.
 *
 * @details
 * Each step is a whole-block launch on the block's stream, at width one:
 * the inner products copied into a scratch block and the trace lifted onto
 * them by the Device IProductWRTPhysTrace block operator (appending), then
 * - Pointwise, where B is square: one launch scaling by `scale / (w J)`,
 *   the kernel the SumFacTOP implementation shares
 *   (PhysLiftProjectBlockOpDeviceKernels.hpp);
 * - StdInvMass: the scratch divided by the Jacobian in place
 *   (DivideByJacobian), then one batched product per component, through
 *   NekBlas, with \f$P = B M_{\mathrm{std}}^{-1} B^{\mathsf T}\f$, nqTot x
 *   nqTot, fetched from the data warehouse once per shape, basis and rule
 *   (StdRegions::ePhysProjectStdMat), the scale its alpha.
 * A block of the ElmtInvMass form takes master's four block operators in
 * sequence (PhysLiftProjectBlockOp::ApplyElmtInvMass()).
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; other code goes through PhysLiftProjectOp.hpp and the
 * operator factory.
 */

#pragma once

#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpDeviceKernels.hpp>

#include <memory>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat block implementation of the lift-and-project
 * operator.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::StdMat; selects the shared Pointwise
 *                         kernel.
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysLiftProjectBlockOpImpl : public PhysLiftProjectBlockOp<TData>
{
    using BlockOpBase = PhysLiftProjectBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the quadrature weights on a block of the Pointwise form,
     * and the projector on a block of the StdInvMass form; create master's
     * four block operators on a block of the ElmtInvMass form.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysLiftProjectBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOpBase(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;
        m_form     = PhysLiftProjectFormOf(exp);
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->CreateChain(ExecSpace::name, SumFac::name);
        }
        else
        {
            m_isDeformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            m_nqTot = exp->GetTotPoints();
            m_lift  = IProductWRTPhysTraceBlockOp<TData>::Create(
                block_idx, exp, dataWarehouse, ExecSpace::name, StdMat::name);
            m_lift->SetAppend(true);
            m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianKey<TData>(block_idx,
                                                 m_implInterleaveWidth));
            if (m_form == PhysLiftProjectForm::Pointwise)
            {
                m_weightsptr =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LocalRegions::WeightsKey<TData>(block_idx,
                                                        m_implInterleaveWidth));
            }
            else
            {
                const unsigned int dim = exp->GetShapeDimension();
                std::vector<LibUtilities::BasisKey> basisKeys(
                    dim, LibUtilities::NullBasisKey);
                for (unsigned int d = 0; d < dim; ++d)
                {
                    basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
                }
                LibUtilities::PointsType nodalType =
                    LibUtilities::eNoPointsType;
                if (exp->IsNodalNonTensorialExp())
                {
                    nodalType = exp->GetNodalPointsKey().GetPointsType();
                }
                m_projptr = this->m_dataWarehouse->template GetData<MemSpace>(
                    StdRegions::StdMatKey<TData>(basisKeys, exp->DetShapeType(),
                                                 StdRegions::ePhysProjectStdMat,
                                                 nodalType));
            }
        }
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysLiftProjectBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the GEMM operates at: the batched product
    /// requires contiguous per-element columns, so storage is
    /// non-interleaved.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
    /// The block's form (PhysLiftProjectFormOf()).
    PhysLiftProjectForm m_form;
    bool m_isDeformed    = false;
    unsigned int m_nqTot = 0;
    /// The Jacobians and, Pointwise only, the quadrature weights at every
    /// point, at #m_implInterleaveWidth; StdInvMass only, the projector
    /// \f$B M_{\mathrm{std}}^{-1} B^{\mathsf T}\f$.
    const TData *m_jacptr     = nullptr;
    const TData *m_weightsptr = nullptr;
    const TData *m_projptr    = nullptr;
    /// The lift onto the scratch copy of the inner products.
    std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>> m_lift;
    /// Scratch: the inner products with the trace lifted onto them, sized
    /// on the first Apply() and again if the component count changes.
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Phys>> m_vol;

    /// The Pointwise form at width one: `out (+)= scale / (w J) * vol`, one
    /// launch.
    void ApplyPointwise(const size_t nelmt, const unsigned int ncomp,
                        const TData *volPtr, TData *outPtr)
    {
        const unsigned int blocksize =
            GetDeviceBlockSize<MultiRegions::SumFacTOP>(m_nqTot);
        const unsigned int gridsize =
            GetDeviceGridSize<MultiRegions::SumFacTOP>(nelmt, blocksize, 0);
        if (this->m_append)
        {
            if (m_isDeformed)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (PhysLiftProjectPointwiseKernel<Implementation, true,
                                                    true>),
                    gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                    this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (PhysLiftProjectPointwiseKernel<Implementation, true,
                                                    false>),
                    gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                    this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (PhysLiftProjectPointwiseKernel<Implementation, false,
                                                    true>),
                    gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                    this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (PhysLiftProjectPointwiseKernel<Implementation, false,
                                                    false>),
                    gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                    this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
            }
        }
    }

    /// The StdInvMass form at width one: the scratch divided by the Jacobian
    /// in place, so that the product, the scale its alpha, can accumulate
    /// onto the output.
    void ApplyProjector(const size_t nelmt, const unsigned int ncomp,
                        const size_t volCompSize, const size_t outCompSize,
                        TData *volPtr, TData *outPtr)
    {
        DivideByJacobian<ExecSpace, false>(nelmt, m_nqTot, ncomp, m_jacptr,
                                           volPtr, volPtr, m_streamID);
        TData beta = 0.0;
        if (this->m_append)
        {
            beta = 1.0;
        }
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);
        NekBlas::GemmStridedBatched(handle, "N", "N", m_nqTot, nelmt, m_nqTot,
                                    this->m_scale, m_projptr, m_nqTot, 0,
                                    volPtr, m_nqTot, volCompSize, beta, outPtr,
                                    m_nqTot, outCompSize, ncomp);
    }

    /**
     * @brief The block's operator with the lift of @p trace: the trace is
     * held for this application alone, then the block is applied as by
     * Apply().
     */
    void v_ApplyLift(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &trace,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        this->m_trace = &trace;
        v_Apply(inblock, outblock);
        this->m_trace = nullptr;
    }

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->template ApplyElmtInvMass<MemSpace>(inblock, outblock,
                                                      m_streamID);
        }
        else
        {
            const unsigned int ncomp =
                inblock.GetNumComponents() * inblock.GetNumHomoModes();
            const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
            if (!m_vol ||
                m_vol->GetNumComponents() != inblock.GetNumComponents() ||
                m_vol->GetNumHomoModes() != inblock.GetNumHomoModes())
            {
                m_vol = std::make_unique<
                    LibUtilities::BlockAccessor<TData, FieldState::Phys>>(
                    inblock.GetNumElements(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    1,
                    LibUtilities::MemoryRegion<TData>(inblock.CompSize() *
                                                      ncomp),
                    nullptr, inblock.GetNumComponents(),
                    inblock.GetNumHomoModes());
            }
            m_vol->template Copy<MemSpace>(inblock, m_streamID);
            if (this->m_trace)
            {
                m_lift->Apply(*this->m_trace, *m_vol);
            }

            // The scratch at the implementation's width, and an output
            // appended to.
            const size_t nelmt = inblock.GetNumElementsWithPadding();
            TData *volPtr =
                m_vol->template GetPtr<MemSpace, ReadWrite>(m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, m_vol->GetInterleaveWidth(),
                nelmt * ncomp, m_nqTot, volPtr, m_streamID);
            m_vol->template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            TData *outPtr = nullptr;
            if (this->m_append)
            {
                outPtr =
                    outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, outblock.GetInterleaveWidth(),
                    nelmt * ncomp, m_nqTot, outPtr, m_streamID);
            }
            else
            {
                outPtr =
                    outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
            }

            if (m_form == PhysLiftProjectForm::Pointwise)
            {
                ApplyPointwise(nelmt, ncomp, volPtr, outPtr);
            }
            else
            {
                ApplyProjector(nelmt, ncomp, m_vol->CompSize(),
                               outblock.CompSize(), volPtr, outPtr);
            }

            // The output, written at the implementation's width, back at
            // the input's width.
            LibUtilities::ReshapeStorage<ExecSpace>(
                inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
                m_nqTot, outPtr, m_streamID);
            outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
