///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceSumFac.hpp
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

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const unsigned int block_idx,
                                    const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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
            m_DB.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eBasisDerivative)));
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eDerivative)));
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
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));

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
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));

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

        // Fetch Jacobian and deriv factors.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacTOP>;
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, transpose));
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
            IProductWRTDerivBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_DB;
    std::vector<const TData *> m_D;
    std::vector<const TData *> m_W;
    std::vector<const TData *> m_f;
    std::vector<const unsigned int *> m_index;
    MemoryRegion<TData> m_wsp;
    const TData *m_nodToMod;
    const TData *m_jacptr;
    const TData *m_dfptr;

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

    size_t GetWorkspaceSize(const LibUtilities::ShapeType shapeType,
                            const size_t nelmt,
                            [[maybe_unused]] const unsigned int nq0,
                            const unsigned int nq1, const unsigned int nq2,
                            [[maybe_unused]] const unsigned int nm0,
                            [[maybe_unused]] const unsigned int nm1,
                            [[maybe_unused]] const unsigned int nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Seg)
        {
            wspsize = nq0 * nelmt;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if (shapeType == LibUtilities::NodalTri)
        {
            wspsize = (3 * nq0 * nq1 + nq1 + nm0 * (nm0 + 1) / 2) * nelmt;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if (shapeType == LibUtilities::NodalTet)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 +
                       nm0 * (nm0 + 1) * (nm0 + 2) / 6) *
                      nelmt;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if (shapeType == LibUtilities::NodalPrism)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 +
                       nm0 * (nm0 + 1) * nm0 / 2) *
                      nelmt;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(
        const LibUtilities::ShapeType shapeType, const size_t nelmt,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
    {
        auto wspsize =
            GetWorkspaceSize(shapeType, nelmt, nq0, nq1, nq2, nm0, nm1, nm2);

        return MemoryRegion<TData>(wspsize);
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
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nelmt, nq0, 0, 0, nm0, 0, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
        const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<Implementation>(nq0, nm0);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase1DKernelLauncher<Implementation,
                                                      DEFORMED>),
                gridsize, blocksize, shmemsize, 0, m_coordDim, nm0, nq0, nelmt,
                inoffset, m_DB[0], m_W[0], m_dfptr, m_jacptr, inptr, outptr,
                wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nelmt, nq0, 0, 0, nm0, 0, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
        const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<Implementation>(nq0, nm0);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase1DKernelLauncher<Implementation, DEFORMED,
                                                      nm0, nq0>),
                gridsize, blocksize, shmemsize, 0, m_coordDim, nelmt, inoffset,
                m_DB[0], m_W[0], m_dfptr, m_jacptr, inptr, outptr, wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp =
                    SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, 0, nm0, nm1, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nmTot);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                nq0, nq1, nm0, nm1);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase2DKernelLauncher<
                    SHAPE_TYPE, Implementation, DEFORMED>),
                gridsize, blocksize, shmemsize, 0, m_coordDim, nm0, nm1, nmTot,
                nq0, nq1, nelmt, inoffset, m_isModified, m_index[0], m_B[0],
                m_B[1], m_D[0], m_D[1], m_W[0], m_W[1], m_f[0], m_f[1],
                m_nodToMod, m_dfptr, m_jacptr, inptr, outptr, wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp =
                    SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, 0, nm0, nm1, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nmTot);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                nq0, nq1, nm0, nm1);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase2DKernelLauncher<
                    SHAPE_TYPE, Implementation, DEFORMED, nm0, nm1, nmTot, nq0,
                    nq1>),
                gridsize, blocksize, shmemsize, 0, m_coordDim, nelmt, inoffset,
                m_isModified, m_index[0], m_B[0], m_B[1], m_D[0], m_D[1],
                m_W[0], m_W[1], m_f[0], m_f[1], m_nodToMod, m_dfptr, m_jacptr,
                inptr, outptr, wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(), (TData *)inptr + d * inoffset);
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];
        const auto nm2 = m_nm[2];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, nq2, nm0, nm1,
                                     nm2);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nmTot);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                nq0, nq1, nq2, nm0, nm1, nm2);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto inoffset = inblock.CompSize();
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(),
                    (TData *)inptr + d * inblock.CompSize());
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase3DKernelLauncher<
                    SHAPE_TYPE, Implementation, DEFORMED>),
                gridsize, blocksize, shmemsize, 0, nm0, nm1, nm2, nmTot, nq0,
                nq1, nq2, nelmt, inoffset, m_isModified, m_index[0], m_index[1],
                m_index[2], m_B[0], m_B[1], m_B[2], m_D[0], m_D[1], m_D[2],
                m_W[0], m_W[1], m_W[2], m_f[0], m_f[1], m_f[2], m_f[3],
                m_nodToMod, m_dfptr, m_jacptr, inptr, outptr, wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(),
                    (TData *)inptr + d * inblock.CompSize());
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.CompSize();
            outptr += outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, nq2, nm0, nm1,
                                     nm2);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nmTot);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
                nq0, nq1, nq2, nm0, nm1, nm2);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto inoffset = inblock.CompSize();
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, nelmt,
                    inblock.GetNumData(),
                    (TData *)inptr + d * inblock.CompSize());
            }

            // IProduct kernel.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (IProductWRTDerivBase3DKernelLauncher<
                    SHAPE_TYPE, Implementation, DEFORMED, nm0, nm1, nm2, nmTot,
                    nq0, nq1, nq2>),
                gridsize, blocksize, shmemsize, 0, nelmt, inoffset,
                m_isModified, m_index[0], m_index[1], m_index[2], m_B[0],
                m_B[1], m_B[2], m_D[0], m_D[1], m_D[2], m_W[0], m_W[1], m_W[2],
                m_f[0], m_f[1], m_f[2], m_f[3], m_nodToMod, m_dfptr, m_jacptr,
                inptr, outptr, wspptr);

            // Reshape back, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    inblock.GetNumData(),
                    (TData *)inptr + d * inblock.CompSize());
            }
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.CompSize();
            outptr += outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
