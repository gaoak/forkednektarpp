///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFac.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTBaseBlockOpImpl(const unsigned int block_idx,
                               const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eWeights)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalPrism) ||
            (m_shapeType == LibUtilities::eNodalTet))
        {
            std::vector<LibUtilities::BasisKey> basisKeys(
                m_dimension, LibUtilities::NullBasisKey);
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
            }

            LibUtilities::PointsType nodalType =
                (exp->IsNodalNonTensorialExp())
                    ? exp->GetNodalPointsKey().GetPointsType()
                    : LibUtilities::eNoPointsType;

            // Fetch NodalToModal Matrix if required.
            m_nodToMod =
                dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, eNodalToModal, nodalType));
        }
        else
        {
            m_nodToMod = (const TData *)nullptr;
        }

        if (m_dimension == 2)
        {
            // Precompute index, if necessary.
            const bool indexing =
                (m_shapeType == LibUtilities::Tri ||
                 m_shapeType == LibUtilities::NodalTri) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;

            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], 0))
                         : nullptr);
        }
        else if (m_dimension == 3)
        {
            // Precompute index, if necessary.
            const bool indexingTet =
                (m_shapeType == LibUtilities::Tet ||
                 m_shapeType == LibUtilities::NodalTet) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            const bool indexingPrism =
                (m_shapeType == LibUtilities::Prism ||
                 m_shapeType == LibUtilities::NodalPrism) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            const bool indexingPyr =
                m_shapeType == LibUtilities::Pyr &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;

            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       0))
                    : nullptr);

            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       1))
                    : nullptr);

            m_index.push_back(
                (indexingTet || indexingPrism)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       2))
                    : nullptr);
        }

        // Fetch Jacobian data.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
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
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_W;
    std::vector<const unsigned int *> m_index;
    const TData *m_nodToMod;
    const TData *m_jacptr;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Coeff> &outblock) override
    {
        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
            // Nodal Triangles
            case LibUtilities::NodalTri:
            {
                NodalTriBlock(inblock, outblock);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
            // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
            // Nodal Tet
            case LibUtilities::NodalTet:
            {
                NodalTetBlock(inblock, outblock);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            // Nodal Prism
            case LibUtilities::NodalPrism:
            {
                NodalPrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void SegBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Coeff> &outblock);

    void TriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalTriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Coeff> &outblock);

    void QuadBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                   BlockAccessor<TData, FieldState::Coeff> &outblock);

    void HexBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Coeff> &outblock);

    void PrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalPrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                         BlockAccessor<TData, FieldState::Coeff> &outblock);

    void PyrBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Coeff> &outblock);

    void TetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalTetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Coeff> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, NonTemplatedSizeParameter1D(m_nm[0], m_nq[0]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock,
                                         TemplatedSizeParameter1D<nm0, nq0>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter1D>
    NEK_FORCE_INLINE void Operator1D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter1D sizeParam1D)
    {
        constexpr bool Append = false;

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTBaseSharedMemorySize<Implementation>(sizeParam1D);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam1D.nq0());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);

        if (this->m_integration)
        {
            // IProduct kernel.
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase1DKernelLauncher<Implementation, Scale,
                                                     Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam1D, nelmt, m_B[0], m_W[0], m_jacptr, inptr, outptr,
                    1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase1DKernelLauncher<Implementation, Scale,
                                                     Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam1D, nelmt, m_B[0], m_W[0], m_jacptr, inptr, outptr,
                    this->m_scale);
            }
        }
        else
        {
            // IProduct kernel no quadrature
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase1DKernelLauncher<Implementation, Scale,
                                                     Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam1D, nelmt, m_B[0], inptr, outptr, 1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase1DKernelLauncher<Implementation, Scale,
                                                     Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam1D, nelmt, m_B[0], inptr, outptr, this->m_scale);
            }
        }

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, outblock.GetNumData(), outptr,
                                  m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, m_nm[0], m_nm[1]);

        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedSizeParameter2D(m_nm[0], m_nm[1], nmTot, m_nq[0],
                                        m_nq[1]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedSizeParameter2D<nm0, nm1, nmTot, nq0, nq1>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter2D>
    NEK_FORCE_INLINE void Operator2D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter2D sizeParam2D)
    {
        constexpr bool Append = false;

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        auto wspSize = IProductWRTBaseWorkSpaceSize<SHAPE_TYPE, Implementation>(
            nelmt, sizeParam2D);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam2D);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam2D.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);

        if (this->m_integration)
        {
            // IProduct kernel.
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam2D, nelmt, m_isModified, m_index[0], m_B[0],
                    m_B[1], m_W[0], m_W[1], m_nodToMod, m_jacptr, inptr, outptr,
                    wspptr, 1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam2D, nelmt, m_isModified, m_index[0], m_B[0],
                    m_B[1], m_W[0], m_W[1], m_nodToMod, m_jacptr, inptr, outptr,
                    wspptr, this->m_scale);
            }
        }
        else
        {
            // IProduct Kernel with no quadrature
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam2D, nelmt, m_isModified, m_index[0], m_B[0],
                    m_B[1], m_nodToMod, inptr, outptr, wspptr, 1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam2D, nelmt, m_isModified, m_index[0], m_B[0],
                    m_B[1], m_nodToMod, inptr, outptr, wspptr, this->m_scale);
            }
        }

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, outblock.GetNumData(), outptr,
                                  m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
            SHAPE_TYPE, m_nm[0], m_nm[1], m_nm[2]);

        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedSizeParameter3D(m_nm[0], m_nm[1], m_nm[2], nmTot,
                                        m_nq[0], m_nq[1], m_nq[2]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedSizeParameter3D<nm0, nm1, nm2, nmTot, nq0, nq1, nq2>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter3D>
    NEK_FORCE_INLINE void Operator3D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter3D sizeParam3D)
    {
        constexpr bool Append = false;

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        auto wspSize = IProductWRTBaseWorkSpaceSize<SHAPE_TYPE, Implementation>(
            nelmt, sizeParam3D);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam3D);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam3D.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);

        if (this->m_integration)
        {
            // IProduct kernel.
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam3D, nelmt, m_isModified, m_index[0], m_index[1],
                    m_index[2], m_B[0], m_B[1], m_B[2], m_W[0], m_W[1], m_W[2],
                    m_nodToMod, m_jacptr, inptr, outptr, wspptr, 1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam3D, nelmt, m_isModified, m_index[0], m_index[1],
                    m_index[2], m_B[0], m_B[1], m_B[2], m_W[0], m_W[1], m_W[2],
                    m_nodToMod, m_jacptr, inptr, outptr, wspptr, this->m_scale);
            }
        }
        else
        {
            // IProduct kernel - no quadrature
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam3D, nelmt, m_isModified, m_index[0], m_index[1],
                    m_index[2], m_B[0], m_B[1], m_B[2], m_nodToMod, inptr,
                    outptr, wspptr, 1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation,
                                                     Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam3D, nelmt, m_isModified, m_index[0], m_index[1],
                    m_index[2], m_B[0], m_B[1], m_B[2], m_nodToMod, inptr,
                    outptr, wspptr, this->m_scale);
            }
        }

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, inblock.GetNumData(),
                                  (TData *)inptr, m_streamID);
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                  nelmt * ncomp, outblock.GetNumData(), outptr,
                                  m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
