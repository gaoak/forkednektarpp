///////////////////////////////////////////////////////////////////////////////
//
// File: PhysGalerkinProject1DScaledDeviceSumFac.hpp
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
// Description: Device (SumFac/SumFacTOP) Galerkin projection from a scaled
// (finer) quadrature grid back down to the native quadrature grid. Reuses
// the generic BwdTransXxxKernelLauncher machinery (Operators/ElmtOps/BwdTrans)
// - the same tensor-contraction kernels PhysInterp1DScaledOp's device
// backend uses for the forward interpolation direction - since those
// kernels are agnostic to which of nm/nq is larger.
//
// Scope matches the Serial implementation (see
// PhysGalerkinProject1DScaledSerialAVXSumFac.hpp), including its reuse of
// PhysInterp1DScaled's optimisation switch: the switch sweeps the native
// counts in m_nm and the scaled counts m_scale * m_nm derived from them,
// which is this operator's pair of grids too, and OperatorND exchanges the
// mode and quadrature roles with TransposeSizeParameter because the
// kernels here run the reverse (scaled -> native) direction. The switch
// dispatches every 2D shape as a Quad and every 3D shape as a Hex, which
// is what this operator wants: both grids are physical-space
// tensor-product grids on every shape, via the Duffy transform on the
// collapsed-coordinate ones, and neither side is modal, so the
// collapsed-coordinate correction other BwdTrans-derived kernels apply is
// never needed here.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/PhysGalerkinProject1DScaled/PhysGalerkinProject1DScaledBlockOp.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSINTERP1D

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysGalerkinProject1DScaledBlockOpImpl
    : public PhysGalerkinProject1DScaledBlockOp<TData>
{
    using BlockOpBase = PhysGalerkinProject1DScaledBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    PhysGalerkinProject1DScaledBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysGalerkinProject1DScaledBlockOp<TData>(block_idx, exp,
                                                    dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Native (target) quadrature point counts - fixed, independent of
        // the over-integration scale factor.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        m_index = {nullptr, nullptr};
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
        return std::make_unique<PhysGalerkinProject1DScaledBlockOpImpl<
            ExecSpace, Implementation, TData>>(block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    // Named for the roles the switch gives them, not for the direction the
    // kernels run in: m_nm is the native (here target/output) grid the
    // switch sweeps, m_nq the scaled (here source/input) grid it derives as
    // m_scale * m_nm. v_SetScaleFactor fills m_nq with the identical
    // formula PhysInterp1DScaledOp uses for its own m_nq, so the two
    // operators always agree on the size of the grid passed between them.
    // OperatorND exchanges the two roles for the kernels.
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    // The precomputed collapsed coordinate mode index arrays the 3D kernel
    // takes. Only the Seg/Quad/Hex kernels are ever launched here and none
    // of them applies that correction, so both are always null; they exist
    // to keep the launcher call the same shape in every dimension.
    std::vector<const unsigned int *> m_index;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");

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

    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        // The scaled (source) point counts, shared with
        // PhysInterp1DScaledOp so both arrive at the same grid.
        m_nq = this->GetScaledNumPoints(m_nm, this->m_scale);
        m_B.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch the Galerkin projection matrix: from the scaled grid
            // (m_nq[d] points) down to this basis' native quadrature
            // (m_nm[d] points).
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eGalerkinProject, m_nq[d])));
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of precomputed index arrays used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 0 : 2;
    }

    // Generic operator. Exchanges the size parameter's mode and quadrature
    // roles, builds the index sequences from the shape dimension and
    // forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysInterp1D.h.in.
        static_assert((DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                          (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                          (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                      "OperatorND expects a size parameter matching the "
                      "dimension of the shape.");

        // The switch hands over the grids the way PhysInterp1DScaled reads
        // them - native as the mode side, scaled as the quadrature side.
        // This operator runs the other way, and the kernels always take the
        // mode side as their source, so the two roles are exchanged here.
        OperatorNDImpl<SHAPE_TYPE>(
            inblock, outblock, TransposeSizeParameter(sizeParam),
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumIndex(DIM)>());
    }

    // Generic operator implementation. ind0 indexes each direction,
    // ind1 the precomputed index arrays used by the kernels.
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter,
              unsigned int... ind0, unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers. Appending accumulates into the pre-existing
        // output values, so they are read as well as written.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const size_t wspSize =
            BwdTransWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Neither side of this operator is modal, so there is no nodal to
        // modal transform to apply and no collapsed coordinate correction
        // to make.
        const TData *nodToMod = nullptr;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);

            // PhysGalerkinProject1DScaled kernel.
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, true>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, false, m_index[ind1]..., m_B[ind0]..., nodToMod, inptr,
                outptr, wspptr);
        }
        else
        {
            // PhysGalerkinProject1DScaled kernel.
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, false>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, false, m_index[ind1]..., m_B[ind0]..., nodToMod, inptr,
                outptr, wspptr);
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
