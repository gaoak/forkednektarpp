///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasDeviceSumFac.hpp
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
// Description: Fused 3/2-rule dealiased advection, Device SumFac
// implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utility>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasBlockOp.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasDeviceSumFacTOPKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionDealiasBlockOpImpl : public AdvectionDealiasBlockOp<TData>
{
    using BlockOpBase = AdvectionDealiasBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    AdvectionDealiasBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionDealiasBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Native element size.
            m_nq.push_back(exp->GetNumPoints(d));

            // Basis derivative matrix.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        // Fetch geometric factors.
        if (m_dimension == 2)
        {
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacTOP>;
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(
                block_idx, m_implInterleaveWidth, transpose));

        // Fine (over-integrated) point counts.
        m_nqFine = this->GetScaledNumPoints(m_nq, m_dealiasScale);

        // Fetch interpolation basis data.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_Binterp.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(d)->GetBasisKey(), LibUtilities::eInterp,
                        m_nqFine[d])));
            m_Bproject.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(d)->GetBasisKey(),
                        LibUtilities::eGalerkinProject, m_nqFine[d])));
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            AdvectionDealiasBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;
    static constexpr TData m_dealiasScale = TData(1.5);

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_nqFine;
    std::vector<const TData *> m_D;
    std::vector<const TData *> m_f;
    std::vector<const TData *> m_Binterp;
    std::vector<const TData *> m_Bproject;
    const TData *m_dfptr;

    static unsigned int NqTot(const std::vector<unsigned int> &nq)
    {
        unsigned int total = 1;
        for (auto n : nq)
        {
            total *= n;
        }
        return total;
    }

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                break;
            }
            // Quadrilateral
            case LibUtilities::Quad:
            {
                ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                break;
            }
            // Triangle
            case LibUtilities::Tri:
            {
                ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                break;
            }
            // Nodal triangle
            case LibUtilities::NodalTri:
            {
                ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                break;
            }
            // Hexahedron
            case LibUtilities::Hex:
            {
                ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                break;
            }
            // Tetrahedron
            case LibUtilities::Tet:
            {
                ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                break;
            }
            // Nodal tetrahedron
            case LibUtilities::NodalTet:
            {
                ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                break;
            }
            // Pyramid
            case LibUtilities::Pyr:
            {
                ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                break;
            }
            // Nodal prism
            case LibUtilities::NodalPrism:
            {
                ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of collapsed coordinate factors used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl. sizeParam describes the native
    // (pre-interpolation) grid only; the over-integrated "fine" grid is a
    // fixed 3/2 scaling of it and is always looked up from m_nqFine, whether
    // or not sizeParam is templated.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitch1DCoordsCode.h.in.
        static_assert(
            (DIM == 1 && IsPhysSizeParameter1D_v<TPhysSizeParameter>) ||
                (DIM == 2 && IsPhysSizeParameter2D_v<TPhysSizeParameter>) ||
                (DIM == 3 && IsPhysSizeParameter3D_v<TPhysSizeParameter>),
            "OperatorND expects a size parameter matching the dimension of the "
            "shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    // Generic operator implementation. ind0 indexes each direction, ind1 the
    // collapsed coordinate factors used by the kernels.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>(
                    m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr, m_streamID);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Shape size.
        const auto nqTotFine = NqTot(m_nqFine);

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        const auto inCompStride  = inblock.CompSize();
        const auto outCompStride = outblock.CompSize();

        // Initialize advVel pointers.
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        const auto advVelPlaneStride = this->m_advVel->CompSize();

        // Choose the shared-memory tier. SumFacTOP moves what fits of the
        // per-element workspace into block-local memory: the fused pipeline,
        // else the tensor-contraction scratch alone. The fused gate is half
        // the per-block capacity because GetDeviceGridSize divides the
        // per-multiprocessor capacity by the request.
        bool localPipeline = false;
        bool sharedScratch = false;
        if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
        {
            const unsigned int localWspSize =
                AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
                    sizeParam, true, true);
            localPipeline = sizeof(TData) * localWspSize <=
                            GetDeviceProperties::SharedMemoryPerBlock() / 2u;

            if constexpr (!IsPhysSizeParameter1D_v<TPhysSizeParameter>)
            {
                const unsigned int scratchWspSize =
                    AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE,
                                                        Implementation>(
                        sizeParam, false, true);
                sharedScratch = !localPipeline &&
                                sizeof(TData) * scratchWspSize <=
                                    GetDeviceProperties::SharedMemoryPerBlock();
            }
        }

        // Get static workspace pointer. Each plane has its own slab, less
        // the part block-local memory holds.
        const unsigned int nvar  = inblock.GetNumComponents();
        const unsigned int nhomo = inblock.GetNumHomoModes();
        const unsigned int ncomp = nvar * nhomo;
        const size_t wspSize =
            AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
                nelmt, m_coordDim, nvar, sizeParam, localPipeline) -
            nelmt *
                AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
                    sizeParam, localPipeline, sharedScratch);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * nhomo, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            AdvectionDealiasSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam, localPipeline, sharedScratch);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nqTotFine);
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // AdvectionDealias kernel.
        if (this->m_append)
        {
            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);

            if (localPipeline)
            {
                if constexpr (std::is_same_v<Implementation,
                                             Operators::SumFacTOP>)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER(
                        (AdvectionDealiasKernelLauncher<SHAPE_TYPE,
                                                        Implementation, true,
                                                        DEFORMED, true, true>),
                        gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                        sizeParam, nelmt, nvar, nhomo, m_D[ind0]...,
                        m_f[ind1]..., m_dfptr, m_Binterp[ind0]...,
                        m_Bproject[ind0]..., this->m_scale, inptr, inCompStride,
                        advVelPtr, advVelPlaneStride, outptr, outCompStride,
                        wspptr);
                }
            }
            else if (sharedScratch)
            {
                if constexpr (std::is_same_v<Implementation,
                                             Operators::SumFacTOP>)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER(
                        (AdvectionDealiasKernelLauncher<SHAPE_TYPE,
                                                        Implementation, true,
                                                        DEFORMED, false, true>),
                        gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                        sizeParam, nelmt, nvar, nhomo, m_D[ind0]...,
                        m_f[ind1]..., m_dfptr, m_Binterp[ind0]...,
                        m_Bproject[ind0]..., this->m_scale, inptr, inCompStride,
                        advVelPtr, advVelPlaneStride, outptr, outCompStride,
                        wspptr);
                }
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (AdvectionDealiasKernelLauncher<SHAPE_TYPE, Implementation,
                                                    true, DEFORMED, false,
                                                    false>),
                    gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, nvar, nhomo, m_D[ind0]..., m_f[ind1]...,
                    m_dfptr, m_Binterp[ind0]..., m_Bproject[ind0]...,
                    this->m_scale, inptr, inCompStride, advVelPtr,
                    advVelPlaneStride, outptr, outCompStride, wspptr);
            }
        }
        else
        {
            if (localPipeline)
            {
                if constexpr (std::is_same_v<Implementation,
                                             Operators::SumFacTOP>)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER(
                        (AdvectionDealiasKernelLauncher<SHAPE_TYPE,
                                                        Implementation, false,
                                                        DEFORMED, true, true>),
                        gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                        sizeParam, nelmt, nvar, nhomo, m_D[ind0]...,
                        m_f[ind1]..., m_dfptr, m_Binterp[ind0]...,
                        m_Bproject[ind0]..., this->m_scale, inptr, inCompStride,
                        advVelPtr, advVelPlaneStride, outptr, outCompStride,
                        wspptr);
                }
            }
            else if (sharedScratch)
            {
                if constexpr (std::is_same_v<Implementation,
                                             Operators::SumFacTOP>)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER(
                        (AdvectionDealiasKernelLauncher<SHAPE_TYPE,
                                                        Implementation, false,
                                                        DEFORMED, false, true>),
                        gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                        sizeParam, nelmt, nvar, nhomo, m_D[ind0]...,
                        m_f[ind1]..., m_dfptr, m_Binterp[ind0]...,
                        m_Bproject[ind0]..., this->m_scale, inptr, inCompStride,
                        advVelPtr, advVelPlaneStride, outptr, outCompStride,
                        wspptr);
                }
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (AdvectionDealiasKernelLauncher<SHAPE_TYPE, Implementation,
                                                    false, DEFORMED, false,
                                                    false>),
                    gridsize, nhomo, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, nvar, nhomo, m_D[ind0]..., m_f[ind1]...,
                    m_dfptr, m_Binterp[ind0]..., m_Bproject[ind0]...,
                    this->m_scale, inptr, inCompStride, advVelPtr,
                    advVelPlaneStride, outptr, outCompStride, wspptr);
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
