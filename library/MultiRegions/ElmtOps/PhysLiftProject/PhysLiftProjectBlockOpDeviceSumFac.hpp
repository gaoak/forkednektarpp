///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceSumFac.hpp
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
// Description: Device implementation of the per-block lift-and-project
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceSumFac.hpp
 * @brief Device block implementation of the lift-and-project operator,
 * registered for SumFac and SumFacTOP.
 *
 * @details
 * A block of the Pointwise or the StdInvMass form is applied by fused
 * launches built from the per-element functions of the IProductWRTPhysTrace,
 * IProductWRTBase and BwdTrans Device kernels, through the standard
 * ShapeBlock and the trace switch (NEKTAR_BLOCKOP_SWITCH_PHYSTRACE):
 * - Pointwise, and StdInvMass with a diagonal standard inverse mass
 *   (StdRegions::StdExpansion::InvMassIsDiagonal()): one launch
 *   (PhysLiftProjectKernelLauncher), the lift, the basis transpose, the
 *   inverse mass with `scale / J` and the backward transform per element;
 * - StdInvMass with a dense standard inverse mass: the lift and the basis
 *   transpose in one launch, the NekBlas batched product with the standard
 *   inverse mass, then `scale / J` and the backward transform in one launch.
 * The kernels read and write the fields at #m_implInterleaveWidth (warpSize
 * for SumFac, one for SumFacTOP); an input, trace or output at the other
 * width is transposed into the workspace (Nektar::transposeBatched) and the
 * output back, so that the operator never writes to its inputs.
 *
 * The nodal shapes, and a SumFacTOP block whose fused launch would not fit
 * in shared memory, take the Device block operators of IProductWRTPhysTrace,
 * IProductWRTBase and BwdTrans in turn instead (ApplySeparate()). A block of
 * the ElmtInvMass form takes master's four block operators in sequence
 * (PhysLiftProjectBlockOp::ApplyElmtInvMass()).
 *
 * The kernels are those of PhysLiftProjectBlockOpDeviceSumFacKernels.hpp
 * (SumFac) and PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp with
 * PhysLiftProjectBlockOpDeviceKernels.hpp (SumFacTOP, the latter shared with
 * StdMat). The kernels of the two implementations have the same names and
 * arguments, each restricted to its Implementation, so that every launch is
 * `Kernel<Implementation, ...>`, as in BwdTransBlockOpDeviceSumFac.hpp.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation, and into the per-shape ShapeBlock units generated from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in; other code goes
 * through PhysLiftProjectOp.hpp and the operator factory.
 */

#pragma once

#include <LibUtilities/BasicUtils/Math/MathKernels.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpDeviceKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/TraceTables.hpp>

#include <algorithm>
#include <memory>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device block implementation of the lift-and-project operator.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP:
 *                         the implementation of the block operators it
 *                         applies, and of its own kernels.
 * @tparam TData           Floating-point type of the field data.
 *
 * Naming, as in the trace operators the switch is shared with: #m_nm holds
 * the volume quadrature counts per direction, #m_nq the trace quadrature
 * counts per face pair, and #m_modes the mode counts.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysLiftProjectBlockOpImpl : public PhysLiftProjectBlockOp<TData>
{
    using BlockOpBase = PhysLiftProjectBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Create the block operators of the block's form and fetch its
     * warehouse data at #m_implInterleaveWidth.
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
        m_streamID  = block_idx + 1;
        m_shapeType = exp->DetShapeType();
        m_dimension = exp->GetShapeDimension();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_form  = PhysLiftProjectFormOf(exp);
        m_nmTot = exp->GetNcoeffs();
        m_nqTot = exp->GetTotPoints();
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->CreateChain(ExecSpace::name, Implementation::name);
        }
        else
        {
            m_lift = IProductWRTPhysTraceBlockOp<TData>::Create(
                block_idx, exp, dataWarehouse, ExecSpace::name,
                Implementation::name);
            m_lift->SetAppend(true);
            m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianKey<TData>(block_idx,
                                                 m_implInterleaveWidth));
            FetchFusedTables(exp, block_idx);
            if (m_form == PhysLiftProjectForm::Pointwise)
            {
                m_weightsptr =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LocalRegions::WeightsKey<TData>(block_idx,
                                                        m_implInterleaveWidth));
            }
            else
            {
                m_basisTranspose = IProductWRTBaseBlockOp<TData>::Create(
                    block_idx, exp, dataWarehouse, ExecSpace::name,
                    Implementation::name);
                m_basisTranspose->SetIntegration(false);
                m_bwdTrans = BwdTransBlockOp<TData>::Create(
                    block_idx, exp, dataWarehouse, ExecSpace::name,
                    Implementation::name);

                // The diagonal where the standard inverse mass is diagonal;
                // otherwise the whole matrix, in the layout the batched product
                // takes at #m_implInterleaveWidth (see ApplyInvMass()).
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
                m_invMassIsDiagonal                = exp->InvMassIsDiagonal();
                StdRegions::StdMatType invMassType = StdRegions::eInvMassStdMat;
                if (m_invMassIsDiagonal)
                {
                    invMassType = StdRegions::eInvMassStdMatDiagonal;
                }
                else if (m_implInterleaveWidth != 1u)
                {
                    invMassType = StdRegions::eInvMassStdMatTranspose;
                }
                m_invMassptr =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        StdRegions::StdMatKey<TData>(basisKeys,
                                                     exp->DetShapeType(),
                                                     invMassType, nodalType));
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
    /// The width the Device block operators of this implementation work at,
    /// and the width the block is carried at from the first step to the
    /// last.
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, MultiRegions::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    unsigned int m_dimension;
    bool m_isDeformed;
    /// The block's form (PhysLiftProjectFormOf()).
    PhysLiftProjectForm m_form;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    /// The Jacobians and, Pointwise only, the quadrature weights at every
    /// point, at #m_implInterleaveWidth.
    const TData *m_jacptr     = nullptr;
    const TData *m_weightsptr = nullptr;
    /// StdInvMass: the standard inverse mass, its diagonal where it is
    /// diagonal (#m_invMassIsDiagonal), otherwise the whole matrix.
    const TData *m_invMassptr = nullptr;
    bool m_invMassIsDiagonal  = false;
    /// The fused launches' tables. #m_nm and #m_nq, the volume and the trace
    /// quadrature counts the generated switch reads, are padded to three
    /// and six entries; #m_liftB, #m_liftW, #m_isCollocated and
    /// #m_endPtsCollocated are the lift's tables (FetchTraceTables()),
    /// #m_basis the basis tables of the basis transpose and the backward
    /// transform, and #m_index the mode index tables the SumFacTOP kernels
    /// of the collapsed shapes read (null otherwise), as Mass fetches them.
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_modes;
    bool m_uniformModes = false;
    bool m_isModified   = false;
    std::vector<const TData *> m_liftB;
    std::vector<const TData *> m_liftW;
    std::vector<bool> m_isCollocated;
    std::vector<bool> m_endPtsCollocated;
    std::vector<const TData *> m_basis;
    std::vector<const unsigned int *> m_index;
    /// The trace Jacobians at #m_implInterleaveWidth; null in one dimension.
    const TData *m_traceJacptr = nullptr;
    /// The block operators of the separate-operator route (ApplySeparate()):
    /// the lift, then the basis transpose and the backward transform.
    std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>> m_lift;
    std::shared_ptr<IProductWRTBaseBlockOp<TData>> m_basisTranspose;
    std::shared_ptr<BwdTransBlockOp<TData>> m_bwdTrans;
    /// Scratch: the inner products with the trace lifted onto them, and the
    /// coefficients before and after the inverse mass, at
    /// #m_implInterleaveWidth; sized on the first Apply() and again if the
    /// component count changes.
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Phys>> m_vol;
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Coeff>>
        m_coef, m_coef2;

    /// Size the scratch blocks after @p inblock: the inner products only
    /// on the separate-operator route (@p vol), the coefficients wherever
    /// the StdInvMass form takes a batched product.
    void SizeScratch(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        const bool vol)
    {
        const unsigned int ncomp = inblock.GetNumComponents();
        const unsigned int nhomo = inblock.GetNumHomoModes();
        const size_t nelmtPad    = inblock.GetNumElementsWithPadding();
        if (vol && !(m_vol && m_vol->GetNumComponents() == ncomp &&
                     m_vol->GetNumHomoModes() == nhomo))
        {
            m_vol = std::make_unique<
                LibUtilities::BlockAccessor<TData, FieldState::Phys>>(
                inblock.GetNumElements(), nelmtPad, inblock.GetNumData(),
                m_implInterleaveWidth,
                LibUtilities::MemoryRegion<TData>(inblock.CompSize() * ncomp *
                                                  nhomo),
                nullptr, ncomp, nhomo);
        }
        if (m_form == PhysLiftProjectForm::StdInvMass &&
            !(m_coef && m_coef->GetNumComponents() == ncomp &&
              m_coef->GetNumHomoModes() == nhomo))
        {
            const size_t coefSize = nelmtPad * m_nmTot * ncomp * nhomo;
            m_coef                = std::make_unique<
                               LibUtilities::BlockAccessor<TData, FieldState::Coeff>>(
                inblock.GetNumElements(), nelmtPad, m_nmTot,
                m_implInterleaveWidth,
                LibUtilities::MemoryRegion<TData>(coefSize), nullptr, ncomp,
                nhomo);
            m_coef2 = std::make_unique<
                LibUtilities::BlockAccessor<TData, FieldState::Coeff>>(
                inblock.GetNumElements(), nelmtPad, m_nmTot,
                m_implInterleaveWidth,
                LibUtilities::MemoryRegion<TData>(coefSize), nullptr, ncomp,
                nhomo);
        }
    }

    /// The fused launches' tables: the lift's as IProductWRTPhysTrace
    /// fetches them, the bases and the mode index tables as Mass does.
    void FetchFusedTables(const LocalRegions::ExpansionSharedPtr &exp,
                          const unsigned int block_idx)
    {
        m_isModified = exp->GetBasisType(0) == LibUtilities::eModified_A;
        for (unsigned int d = 0; d < m_dimension; ++d)
        {
            m_modes.push_back(exp->GetBasisNumModes(d));
            m_basis.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eBasis)));
        }
        m_uniformModes =
            std::all_of(m_modes.begin(), m_modes.end(),
                        [&](const unsigned int m) { return m == m_modes[0]; });

        auto tables =
            FetchTraceTables<TData, MemSpace>(exp, this->m_dataWarehouse);
        m_nm               = std::move(tables.nm);
        m_liftB            = std::move(tables.B);
        m_liftW            = std::move(tables.W);
        m_nq               = std::move(tables.nq);
        m_endPtsCollocated = std::move(tables.endPtsCollocated);
        m_isCollocated     = std::move(tables.isCollocated);
        if (m_dimension > 1)
        {
            m_traceJacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianLocTraceKey<TData>(
                    block_idx, m_implInterleaveWidth));
        }
        else
        {
            m_nq.push_back(m_nm[0]);
            m_isCollocated.push_back(false);
        }
        while (m_nm.size() < 3)
        {
            m_nm.push_back(0);
        }
        while (m_nq.size() < 6)
        {
            m_nq.push_back(0);
        }

        constexpr bool top =
            std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
        const bool tri   = m_shapeType == LibUtilities::Tri;
        const bool tet   = m_shapeType == LibUtilities::Tet;
        const bool prism = m_shapeType == LibUtilities::Prism;
        const bool pyr   = m_shapeType == LibUtilities::Pyr;
        if (m_dimension == 2)
        {
            const unsigned int *index0 = nullptr;
            if (top && tri)
            {
                index0 = this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::ModeIndexKey(m_shapeType, m_modes[0],
                                               m_modes[1], 0));
            }
            m_index.push_back(index0);
        }
        else if (m_dimension == 3)
        {
            for (unsigned int i = 0; i < 4; ++i)
            {
                bool indexing = top && (tet || prism || pyr);
                if (i == 2)
                {
                    indexing = top && (tet || prism);
                }
                else if (i == 3)
                {
                    indexing = top && tet;
                }
                const unsigned int *index = nullptr;
                if (indexing)
                {
                    index = this->m_dataWarehouse->template GetData<MemSpace>(
                        LibUtilities::ModeIndexKey(m_shapeType, m_modes[0],
                                                   m_modes[1], m_modes[2], i));
                }
                m_index.push_back(index);
            }
        }
    }

    /**
     * @brief The product with the dense standard inverse mass of #m_coef
     * into #m_coef2, at #m_implInterleaveWidth: at width one one per
     * component over the elements (out = M^{-1} in, with eInvMassStdMat); at
     * warpSize one per element group, each group's coefficients a
     * warpSize x nmTot matrix (out = in M^{-T}, with
     * eInvMassStdMatTranspose).
     */
    void ApplyStdInvMassProduct(const unsigned int ncomp)
    {
        const size_t nelmt    = m_coef->GetNumElementsWithPadding();
        const size_t compSize = m_coef->CompSize();
        const TData *coefPtr =
            m_coef->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        TData *coef2Ptr =
            m_coef2->template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);
        if constexpr (m_implInterleaveWidth == 1u)
        {
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmt, m_nmTot, TData(1.0),
                m_invMassptr, m_nmTot, 0, coefPtr, m_nmTot, compSize,
                TData(0.0), coef2Ptr, m_nmTot, compSize, ncomp);
        }
        else
        {
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_implInterleaveWidth, m_nmTot, m_nmTot,
                TData(1.0), coefPtr, m_implInterleaveWidth,
                m_implInterleaveWidth * m_nmTot, m_invMassptr, m_nmTot, 0,
                TData(0.0), coef2Ptr, m_implInterleaveWidth,
                m_implInterleaveWidth * m_nmTot,
                ncomp * (nelmt / m_implInterleaveWidth));
        }
        m_coef2->template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    /**
     * @brief The standard inverse mass with `scale / J` of #m_coef into
     * #m_coef2, at #m_implInterleaveWidth: one launch scaling by the
     * diagonal where it is diagonal; otherwise the batched product
     * (ApplyStdInvMassProduct()), then the scale in place. The
     * separate-operator route.
     */
    void ApplyInvMass(const unsigned int ncomp)
    {
        const size_t nelmt = m_coef->GetNumElementsWithPadding();
        const TData *coefPtr =
            m_coef->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        TData *coef2Ptr =
            m_coef2->template GetPtr<MemSpace, WriteOnly>(m_streamID);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(m_nmTot);
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, 0);

        if (m_invMassIsDiagonal)
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysLiftProjectInvMassScaleKernel<Implementation, true>),
                gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nmTot,
                this->m_scale, m_invMassptr, m_jacptr, coefPtr, coef2Ptr);
        }
        else
        {
            ApplyStdInvMassProduct(ncomp);
            const TData *noDiag = nullptr;
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysLiftProjectInvMassScaleKernel<Implementation, false>),
                gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nmTot,
                this->m_scale, noDiag, m_jacptr, coef2Ptr, coef2Ptr);
        }
        m_coef2->template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    /// The Pointwise form from #m_vol into @p outblock, at
    /// #m_implInterleaveWidth.
    template <bool APPEND>
    void ApplyPointwise(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int ncomp)
    {
        const size_t nelmt = outblock.GetNumElementsWithPadding();
        const TData *volPtr =
            m_vol->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        TData *outPtr = nullptr;
        if constexpr (APPEND)
        {
            outPtr = outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID);
        }
        else
        {
            outPtr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        }
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(m_nqTot);
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, 0);
        if (m_isDeformed)
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysLiftProjectPointwiseKernel<Implementation, APPEND, true>),
                gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysLiftProjectPointwiseKernel<Implementation, APPEND, false>),
                gridsize, ncomp, blocksize, 1, m_streamID, nelmt, m_nqTot,
                this->m_scale, m_weightsptr, m_jacptr, volPtr, outPtr);
        }
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
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
            switch (m_shapeType)
            {
                case LibUtilities::Seg:
                    ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                    break;
                case LibUtilities::Quad:
                    ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                    break;
                case LibUtilities::Tri:
                    ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                    break;
                case LibUtilities::NodalTri:
                    ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                    break;
                case LibUtilities::Hex:
                    ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                    break;
                case LibUtilities::Tet:
                    ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                    break;
                case LibUtilities::NodalTet:
                    ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                    break;
                case LibUtilities::Pyr:
                    ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                    break;
                case LibUtilities::Prism:
                    ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                    break;
                case LibUtilities::NodalPrism:
                    ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                    break;
                default:
                    NEKERROR(ErrorUtil::efatal, "shape type not implemented");
            }
        }
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

    // Shape-specific entry points, specialised for each shape in the
    // translation units generated from
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /// Whether the trace size parameter @p TTraceSizeParameter is
    /// compile-time with every volume and trace count the same.
    template <typename TTraceSizeParameter>
    static constexpr bool UniformCounts()
    {
        using T = TTraceSizeParameter;
        if constexpr (IsTemplatedTraceSizeParameter3D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nm2() == n && T::nq00() == n &&
                   T::nq01() == n && T::nq10() == n && T::nq11() == n &&
                   T::nq20() == n && T::nq21() == n;
        }
        else if constexpr (IsTemplatedTraceSizeParameter2D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nq00() == n && T::nq10() == n;
        }
        else
        {
            return IsTemplatedTraceSizeParameter1D<T>::value;
        }
    }

    /// The compile-time size parameter of @p NUMMODES modes and
    /// @p NUMPOINTS points in every direction.
    template <LibUtilities::ShapeType SHAPE_TYPE, unsigned int NUMMODES,
              unsigned int NUMPOINTS>
    static auto TemplatedSizeParameter()
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        if constexpr (DIM == 1)
        {
            return TemplatedSizeParameter1D<NUMMODES, NUMPOINTS>();
        }
        else if constexpr (DIM == 2)
        {
            return TemplatedSizeParameter2D<
                NUMMODES, NUMMODES,
                LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, NUMMODES,
                                                      NUMMODES),
                NUMPOINTS, NUMPOINTS>();
        }
        else
        {
            return TemplatedSizeParameter3D<
                NUMMODES, NUMMODES, NUMMODES,
                LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, NUMMODES,
                                                      NUMMODES, NUMMODES),
                NUMPOINTS, NUMPOINTS, NUMPOINTS>();
        }
    }

    /// The runtime size parameter of the block.
    template <unsigned int DIM> auto NonTemplatedSizeParameter() const
    {
        if constexpr (DIM == 1)
        {
            return NonTemplatedSizeParameter1D(m_modes[0], m_nm[0]);
        }
        else if constexpr (DIM == 2)
        {
            return NonTemplatedSizeParameter2D(m_modes[0], m_modes[1], m_nmTot,
                                               m_nm[0], m_nm[1]);
        }
        else
        {
            return NonTemplatedSizeParameter3D(m_modes[0], m_modes[1],
                                               m_modes[2], m_nmTot, m_nm[0],
                                               m_nm[1], m_nm[2]);
        }
    }

    /// Mode index tables, stage scratches and lift scratches the kernels
    /// take in @p dim dimensions.
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 4;
    }
    static constexpr unsigned int NumStage(const unsigned int dim)
    {
        return dim - 1;
    }

    /**
     * @brief The block's form with the trace size parameter the generated
     * switch chose: compile-time mode counts for the StdInvMass form where
     * every volume and trace count is the same and the modes are as many as
     * the points or one fewer, the runtime ones otherwise. A deformed block
     * reaching here is Pointwise. The nodal shapes take the
     * separate-operator route.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const TTraceSizeParameter traceSizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        static_assert(
            (DIM == 1 && IsTraceSizeParameter1D_v<TTraceSizeParameter>) ||
                (DIM == 2 && IsTraceSizeParameter2D_v<TTraceSizeParameter>) ||
                (DIM == 3 && IsTraceSizeParameter3D_v<TTraceSizeParameter>),
            "OperatorND expects a trace size parameter matching the "
            "dimension of the shape.");

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri ||
                      SHAPE_TYPE == LibUtilities::NodalTet ||
                      SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            ApplySeparate(inblock, outblock);
        }
        else if constexpr (DEFORMED)
        {
            ApplyFused<SHAPE_TYPE, true, PhysLiftProjectForm::Pointwise>(
                inblock, outblock, NonTemplatedSizeParameter<DIM>(),
                traceSizeParam);
        }
        else if (m_form == PhysLiftProjectForm::Pointwise)
        {
            ApplyFused<SHAPE_TYPE, false, PhysLiftProjectForm::Pointwise>(
                inblock, outblock, NonTemplatedSizeParameter<DIM>(),
                traceSizeParam);
        }
        else
        {
            WithStdInvMassSizeParameter<SHAPE_TYPE>(
                traceSizeParam, [&](const auto sizeParam) {
                    this->template ApplyFused<SHAPE_TYPE, false,
                                              PhysLiftProjectForm::StdInvMass>(
                        inblock, outblock, sizeParam, traceSizeParam);
                });
        }
    }

    /**
     * @brief Calls @p func once with the volume size parameter of the
     * StdInvMass form: compile-time counts of N or N - 1 modes and N points
     * in every direction where the trace size parameter is compile-time with
     * every count N and the block has that many modes in every direction,
     * the runtime counts otherwise. N - 1 modes only where that is a valid
     * mode count: one at least, and two for the nodal shapes, whose
     * coefficient counts assert it even in constant evaluation.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter,
              typename TFunc>
    NEK_FORCE_INLINE void WithStdInvMassSizeParameter(const TTraceSizeParameter,
                                                      TFunc &&func) const
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        if constexpr (UniformCounts<TTraceSizeParameter>())
        {
            constexpr unsigned int N = TTraceSizeParameter::nm0();
            constexpr bool nodal     = SHAPE_TYPE == LibUtilities::NodalTri ||
                                   SHAPE_TYPE == LibUtilities::NodalTet ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism;
            constexpr unsigned int minModes = nodal ? 2u : 1u;
            const unsigned int modes        = m_uniformModes ? m_modes[0] : 0u;
            if constexpr (N >= minModes + 1)
            {
                if (modes == N)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N, N>());
                }
                else if (modes == N - 1)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N - 1, N>());
                }
                else
                {
                    func(NonTemplatedSizeParameter<DIM>());
                }
            }
            else if constexpr (N >= minModes)
            {
                if (modes == N)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N, N>());
                }
                else
                {
                    func(NonTemplatedSizeParameter<DIM>());
                }
            }
            else
            {
                func(NonTemplatedSizeParameter<DIM>());
            }
        }
        else
        {
            func(NonTemplatedSizeParameter<DIM>());
        }
    }

    /// Dynamic shared memory of fused kernel @p KERNEL for a block of form
    /// @p FORM, in bytes; zero for SumFac, whose intermediates live in the
    /// global workspace.
    template <LibUtilities::ShapeType SHAPE_TYPE, PhysLiftProjectKernel KERNEL,
              PhysLiftProjectForm FORM, typename TSizeParameter,
              typename TTraceSizeParameter>
    static unsigned int SharedMemoryBytes(
        const TSizeParameter sizeParam,
        const TTraceSizeParameter traceSizeParam)
    {
        return sizeof(TData) *
               PhysLiftProjectSharedMemorySize<SHAPE_TYPE, Implementation,
                                               KERNEL, FORM, TData>(
                   sizeParam, traceSizeParam);
    }

    /// ApplyFusedImpl() with the index sequences of the shape's dimension;
    /// a SumFacTOP block whose largest launch, the fused one or one of the
    /// two either side of a dense standard inverse mass, would exceed the
    /// shared-memory limit takes the separate-operator route instead.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              PhysLiftProjectForm FORM, typename TSizeParameter,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void ApplyFused(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const TSizeParameter sizeParam,
        const TTraceSizeParameter traceSizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        unsigned int shmemsize =
            SharedMemoryBytes<SHAPE_TYPE, PhysLiftProjectKernel::Fused, FORM>(
                sizeParam, traceSizeParam);
        if constexpr (FORM == PhysLiftProjectForm::StdInvMass)
        {
            if (!m_invMassIsDiagonal)
            {
                shmemsize = std::max(
                    SharedMemoryBytes<SHAPE_TYPE,
                                      PhysLiftProjectKernel::LiftBasisTranspose,
                                      FORM>(sizeParam, traceSizeParam),
                    SharedMemoryBytes<
                        SHAPE_TYPE, PhysLiftProjectKernel::ScaleBwdTrans, FORM>(
                        sizeParam, traceSizeParam));
            }
        }
        if (shmemsize > GetDeviceProperties::SharedMemoryPerBlock())
        {
            ApplySeparate(inblock, outblock);
        }
        else
        {
            ApplyFusedImpl<SHAPE_TYPE, DEFORMED, FORM>(
                inblock, outblock, sizeParam, traceSizeParam,
                std::make_integer_sequence<unsigned int, DIM>(),
                std::make_integer_sequence<unsigned int, DIM *(DIM - 1)>(),
                std::make_integer_sequence<unsigned int, NumStage(DIM)>(),
                std::make_integer_sequence<unsigned int, NumIndex(DIM)>());
        }
    }

    /**
     * @brief The fused launches of the block's form.
     *
     * The workspace (BlockOperator's static buffer of the block's stream)
     * holds one region per intermediate, each laid out as a block at
     * #m_implInterleaveWidth: the input, the trace and the output where
     * they arrive at another width, and, for SumFac, the coefficients and
     * the stage and lift scratches the kernels keep there. An input or a
     * trace at another width is transposed into its region and the
     * kernels read it there; an output at another width, or to be left at
     * the input's other width, is written to its region and transposed
     * back, appended to after it is transposed in. The output leaves at
     * the input's width.
     *
     * @tparam FORM  Pointwise or StdInvMass.
     * @tparam ind0  One index per direction.
     * @tparam ind1  One per face pair.
     * @tparam ind2  One per stage and lift scratch.
     * @tparam ind3  One per mode index table.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              PhysLiftProjectForm FORM, typename TSizeParameter,
              typename TTraceSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2, unsigned int... ind3>
    void ApplyFusedImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const TSizeParameter sizeParam,
        const TTraceSizeParameter traceSizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>,
        std::integer_sequence<unsigned int, ind3...>)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        const size_t nelmt = inblock.GetNumElementsWithPadding();
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int nqTotTrace =
            traceSizeParam.template nqTotTrace<SHAPE_TYPE>();
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());

        // The workspace sizes per element: the input, the trace and the
        // output, then SumFac's coefficients, two stage and two lift
        // scratches.
        size_t coefSize = 0, stage0Size = 0, stage1Size = 0, lift0Size = 0,
               lift1Size = 0;
        if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
        {
            coefSize   = m_nmTot;
            stage0Size = PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(
                sizeParam);
            stage1Size = PhysLiftProjectStage1Size<SHAPE_TYPE, Implementation>(
                sizeParam);
            if constexpr (DIM == 2)
            {
                lift0Size =
                    2u * IProductWRTPhysTraceEdgeModeBlockSize(traceSizeParam);
            }
            else if constexpr (DIM == 3)
            {
                lift0Size =
                    IProductWRTPhysTraceFaceModeBlockSize(traceSizeParam);
                lift1Size = IProductWRTPhysTraceFaceScratchSize(traceSizeParam);
            }
        }

        // The workspace pointers, in that order.
        const size_t nslot = nelmt * ncomp;
        TData *inWsp =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                nslot * (2 * m_nqTot + nqTotTrace + coefSize + stage0Size +
                         stage1Size + lift0Size + lift1Size),
                m_streamID);
        TData *traceWsp                    = inWsp + nslot * m_nqTot;
        TData *outWsp                      = traceWsp + nslot * nqTotTrace;
        TData *coefWsp                     = outWsp + nslot * m_nqTot;
        TData *stage0                      = coefWsp + nslot * coefSize;
        TData *stage1                      = stage0 + nslot * stage0Size;
        TData *lift0                       = stage1 + nslot * stage1Size;
        TData *lift1                       = lift0 + nslot * lift0Size;
        [[maybe_unused]] TData *stage[2]   = {stage0, stage1};
        [[maybe_unused]] TData *liftWsp[2] = {lift0, lift1};

        // A block's storage from one interleave width to another, out of
        // place: a copy at the same width, otherwise one batched transpose
        // of each element group, one of the two widths being one.
        auto relayout =
            [&](const unsigned int fromWidth, const unsigned int toWidth,
                const unsigned int numData, const TData *src, TData *dst) {
                if (fromWidth == toWidth)
                {
                    Math::copyKernel<ExecSpace>(nelmt * ncomp * numData, src,
                                                dst, m_streamID);
                }
                else
                {
                    ASSERTL0(fromWidth == 1u || toWidth == 1u,
                             "PhysLiftProject takes blocks at width one or at "
                             "the implementation's width.");
                    unsigned int rows = fromWidth;
                    unsigned int cols = numData;
                    if (fromWidth == 1u)
                    {
                        rows = numData;
                        cols = toWidth;
                    }
                    Nektar::transposeBatched(nelmt * ncomp /
                                                 std::max(fromWidth, toWidth),
                                             rows, cols, src, dst, m_streamID);
                }
            };

        const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
        const TData *inptr =
            inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        if (inInterleaveWidth != m_implInterleaveWidth)
        {
            relayout(inInterleaveWidth, m_implInterleaveWidth, m_nqTot, inptr,
                     inWsp);
            inptr = inWsp;
        }

        const bool lift       = this->m_trace != nullptr;
        const TData *traceptr = nullptr;
        if (lift)
        {
            ASSERTL1(this->m_trace->GetNumElementsWithPadding() == nelmt,
                     "Trace and volume blocks differ in their padding.");
            const unsigned int traceInterleaveWidth =
                this->m_trace->GetInterleaveWidth();
            traceptr =
                this->m_trace->template GetPtr<MemSpace, ReadOnly>(m_streamID);
            if (traceInterleaveWidth != m_implInterleaveWidth)
            {
                relayout(traceInterleaveWidth, m_implInterleaveWidth,
                         nqTotTrace, traceptr, traceWsp);
                traceptr = traceWsp;
            }
        }

        const unsigned int outInterleaveWidth = outblock.GetInterleaveWidth();
        TData *outptr                         = nullptr;
        if (this->m_append)
        {
            outptr = outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID);
        }
        else
        {
            outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        }
        bool outDirect = inInterleaveWidth == m_implInterleaveWidth;
        if (this->m_append)
        {
            outDirect =
                outDirect && outInterleaveWidth == m_implInterleaveWidth;
        }
        TData *kernelOut = outptr;
        if (!outDirect)
        {
            kernelOut = outWsp;
            if (this->m_append)
            {
                relayout(outInterleaveWidth, m_implInterleaveWidth, m_nqTot,
                         outptr, kernelOut);
            }
        }

        bool fused = true;
        if constexpr (FORM == PhysLiftProjectForm::StdInvMass)
        {
            fused = m_invMassIsDiagonal;
        }
        if (fused)
        {
            const unsigned int shmemsize =
                SharedMemoryBytes<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                  FORM>(sizeParam, traceSizeParam);
            const unsigned int gridsize =
                GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);
            if (this->m_append)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (PhysLiftProjectKernelLauncher<SHAPE_TYPE, Implementation,
                                                   FORM, true, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, traceSizeParam, nelmt, this->m_scale, lift,
                    m_isModified, m_index[ind3]..., m_basis[ind0]...,
                    m_liftB[ind0]..., m_liftB[DIM + ind1]..., m_liftW[ind1]...,
                    m_traceJacptr, (bool)m_isCollocated[ind1]...,
                    (bool)m_endPtsCollocated[ind0]..., m_weightsptr, m_jacptr,
                    m_invMassptr, inptr, traceptr, kernelOut, inWsp, coefWsp,
                    stage[ind2]..., liftWsp[ind2]...);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (PhysLiftProjectKernelLauncher<SHAPE_TYPE, Implementation,
                                                   FORM, false, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, traceSizeParam, nelmt, this->m_scale, lift,
                    m_isModified, m_index[ind3]..., m_basis[ind0]...,
                    m_liftB[ind0]..., m_liftB[DIM + ind1]..., m_liftW[ind1]...,
                    m_traceJacptr, (bool)m_isCollocated[ind1]...,
                    (bool)m_endPtsCollocated[ind0]..., m_weightsptr, m_jacptr,
                    m_invMassptr, inptr, traceptr, kernelOut, inWsp, coefWsp,
                    stage[ind2]..., liftWsp[ind2]...);
            }
        }
        else if constexpr (FORM == PhysLiftProjectForm::StdInvMass)
        {
            // The dense standard inverse mass: the lift and the basis
            // transpose, the batched product, then the scale and the
            // backward transform.
            SizeScratch(inblock, false);
            const unsigned int shmemsizeA = SharedMemoryBytes<
                SHAPE_TYPE, PhysLiftProjectKernel::LiftBasisTranspose, FORM>(
                sizeParam, traceSizeParam);
            const unsigned int gridsizeA =
                GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsizeA);
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (PhysLiftProjectLiftBasisTransposeKernelLauncher<
                    SHAPE_TYPE, Implementation>),
                gridsizeA, ncomp, blocksize, 1, shmemsizeA, m_streamID,
                sizeParam, traceSizeParam, nelmt, lift, m_isModified,
                m_index[ind3]..., m_basis[ind0]..., m_liftB[ind0]...,
                m_liftB[DIM + ind1]..., m_liftW[ind1]..., m_traceJacptr,
                (bool)m_isCollocated[ind1]...,
                (bool)m_endPtsCollocated[ind0]..., inptr, traceptr,
                m_coef->template GetPtr<MemSpace, WriteOnly>(m_streamID), inWsp,
                stage[ind2]..., liftWsp[ind2]...);
            m_coef->template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            ApplyStdInvMassProduct(ncomp);
            TData *coef2Ptr =
                m_coef2->template GetPtr<MemSpace, ReadWrite>(m_streamID);
            const unsigned int shmemsizeB =
                SharedMemoryBytes<SHAPE_TYPE,
                                  PhysLiftProjectKernel::ScaleBwdTrans, FORM>(
                    sizeParam, traceSizeParam);
            const unsigned int gridsizeB =
                GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsizeB);
            if (this->m_append)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (PhysLiftProjectScaleBwdTransKernelLauncher<
                        SHAPE_TYPE, Implementation, true>),
                    gridsizeB, ncomp, blocksize, 1, shmemsizeB, m_streamID,
                    sizeParam, traceSizeParam, nelmt, this->m_scale,
                    m_isModified, m_index[ind3]..., m_basis[ind0]..., m_jacptr,
                    coef2Ptr, kernelOut, stage[ind2]...);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (PhysLiftProjectScaleBwdTransKernelLauncher<
                        SHAPE_TYPE, Implementation, false>),
                    gridsizeB, ncomp, blocksize, 1, shmemsizeB, m_streamID,
                    sizeParam, traceSizeParam, nelmt, this->m_scale,
                    m_isModified, m_index[ind3]..., m_basis[ind0]..., m_jacptr,
                    coef2Ptr, kernelOut, stage[ind2]...);
            }
        }

        if (!outDirect)
        {
            relayout(m_implInterleaveWidth, inInterleaveWidth, m_nqTot,
                     kernelOut, outptr);
        }
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    /**
     * @brief The separate-operator route, for the nodal shapes and for a
     * SumFacTOP block whose fused launch would not fit in shared memory:
     * the Device block operators of IProductWRTPhysTrace, IProductWRTBase
     * and BwdTrans in turn, the block carried at #m_implInterleaveWidth
     * from the first step to the last.
     */
    void ApplySeparate(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
        const size_t inSize                  = inblock.CompSize() * ncomp;
        const TData *inPtr =
            inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // The inner products into the scratch at the operators' width: from
        // width one the copy and the interleave are one launch.
        SizeScratch(inblock, true);
        TData *volPtr = m_vol->template GetPtr<MemSpace, WriteOnly>(m_streamID);
        if (inInterleaveWidth == 1u && m_implInterleaveWidth != 1u)
        {
            Nektar::transposeBatched(
                inSize / (inblock.GetNumData() * m_implInterleaveWidth),
                inblock.GetNumData(), m_implInterleaveWidth, inPtr, volPtr,
                m_streamID);
        }
        else
        {
            Math::copyKernel<ExecSpace>(inSize, inPtr, volPtr, m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, inInterleaveWidth,
                inblock.GetNumElementsWithPadding() * ncomp,
                inblock.GetNumData(), volPtr, m_streamID);
        }
        m_vol->template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // The trace at the lift's width, so that it reshapes neither the
        // trace nor the scratch, and the scratch stays at it.
        if (this->m_trace)
        {
            const unsigned int traceInterleaveWidth =
                this->m_trace->GetInterleaveWidth();
            const size_t traceNelmt =
                this->m_trace->GetNumElementsWithPadding() * ncomp;
            TData *tracePtr =
                this->m_trace->template GetPtr<MemSpace, ReadWrite>(m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, traceInterleaveWidth, traceNelmt,
                this->m_trace->GetNumData(), tracePtr, m_streamID);
            this->m_trace->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
            m_lift->Apply(*this->m_trace, *m_vol);
            LibUtilities::ReshapeStorage<ExecSpace>(
                traceInterleaveWidth, m_implInterleaveWidth, traceNelmt,
                this->m_trace->GetNumData(), tracePtr, m_streamID);
            this->m_trace->template SetInterleaveWidth<TData>(
                traceInterleaveWidth);
        }

        // An output appended to at the operators' width as well.
        const size_t outNelmt = outblock.GetNumElementsWithPadding() * ncomp;
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outblock.GetInterleaveWidth(), outNelmt,
                outblock.GetNumData(),
                outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID),
                m_streamID);
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        }

        if (m_form == PhysLiftProjectForm::Pointwise)
        {
            if (this->m_append)
            {
                ApplyPointwise<true>(outblock, ncomp);
            }
            else
            {
                ApplyPointwise<false>(outblock, ncomp);
            }
        }
        else
        {
            // The scale rides on the inverse mass, with 1/J.
            m_basisTranspose->SetScale(TData(1.0));
            m_basisTranspose->Apply(*m_vol, *m_coef);
            ApplyInvMass(ncomp);
            m_bwdTrans->SetAppend(this->m_append);
            m_bwdTrans->Apply(*m_coef2, outblock);
        }

        // The output back at the input's width.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, outNelmt,
            outblock.GetNumData(),
            outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID),
            m_streamID);
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
