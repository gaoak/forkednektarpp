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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
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
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eWeights)));
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
            m_nodToMod = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eNodalToModal,
                                             nodalType));
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
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], 0))
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
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 0))
                    : nullptr);

            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 1))
                    : nullptr);

            m_index.push_back(
                (indexingTet || indexingPrism)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 2))
                    : nullptr);
        }

        // Fetch Jacobian data.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
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

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
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
    // IProductWRTBaseSumFacBlockOp.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock);

    // Number of precomputed index arrays used by the kernels in dim dimensions.
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 3;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchCode.h.in.
        static_assert((DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                          (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                          (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                      "OperatorND expects a size parameter matching the "
                      "dimension of the shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumIndex(DIM)>());
    }

    // Generic operator implementation. ind0 indexes each direction,
    // ind1 the precomputed index arrays used by the kernels.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
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
            nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        if (this->m_integration)
        {
            // IProduct kernel.
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBaseKernelLauncher<SHAPE_TYPE, Implementation,
                                                   Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, m_isModified, m_index[ind1]...,
                    m_B[ind0]..., m_W[ind0]..., m_nodToMod, m_jacptr, inptr,
                    outptr, wspptr, (TData)1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBaseKernelLauncher<SHAPE_TYPE, Implementation,
                                                   Scale, Append, DEFORMED>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, m_isModified, m_index[ind1]...,
                    m_B[ind0]..., m_W[ind0]..., m_nodToMod, m_jacptr, inptr,
                    outptr, wspptr, this->m_scale);
            }
        }
        else
        {
            // IProduct kernel - no quadrature
            if (this->m_scale == 1.0)
            {
                constexpr bool Scale = false;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBaseKernelLauncher<SHAPE_TYPE, Implementation,
                                                   Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, m_isModified, m_index[ind1]...,
                    m_B[ind0]..., m_nodToMod, inptr, outptr, wspptr,
                    (TData)1.0);
            }
            else
            {
                constexpr bool Scale = true;

                DEVICE_2DGRID_KERNEL_LAUNCHER(
                    (IProductWRTBaseKernelLauncher<SHAPE_TYPE, Implementation,
                                                   Scale, Append>),
                    gridsize, ncomp, blocksize, 1, shmemsize, m_streamID,
                    sizeParam, nelmt, m_isModified, m_index[ind1]...,
                    m_B[ind0]..., m_nodToMod, inptr, outptr, wspptr,
                    this->m_scale);
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
