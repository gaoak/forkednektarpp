///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFac.hpp
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PhysDerivBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : PhysDerivBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eDerivative)));
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
        }

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacTOP>;
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, transpose));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<const TData *> m_D;
    std::vector<const TData *> m_f;
    const TData *m_dfptr;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
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
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                   BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                         BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        // Shape size.
        const auto nq0 = m_nq[0];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
        const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

        // Loop over components.
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysDeriv1DKernelLauncher<Implementation, DEFORMED>), gridsize,
                blocksize, 0, m_coordDim, nq0, nelmt, outoffset, m_D[0],
                m_dfptr, inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(), (TData *)outptr + d * outoffset);
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                outptr += (m_coordDim - 1) * outoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
        const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

        // Loop over components.
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (PhysDeriv1DKernelLauncher<Implementation, DEFORMED, coordDim,
                                           nq0>),
                gridsize, blocksize, 0, nelmt, outoffset, m_D[0], m_dfptr,
                inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(), (TData *)outptr + d * outoffset);
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                outptr += (m_coordDim - 1) * outoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        // Shape size.
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nq0 * nq1);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (PhysDeriv2DKernelLauncher<SHAPE_TYPE, Implementation,
                                           DEFORMED>),
                gridsize, blocksize, shmemsize, 0, m_coordDim, nq0, nq1, nelmt,
                outoffset, m_D[0], m_D[1], m_f[0], m_f[1], m_dfptr, inptr,
                outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(), (TData *)outptr + d * outoffset);
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                outptr += (m_coordDim - 1) * outoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nq0 * nq1);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (PhysDeriv2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                           coordDim, nq0, nq1>),
                gridsize, blocksize, shmemsize, 0, nelmt, outoffset, m_D[0],
                m_D[1], m_f[0], m_f[1], m_dfptr, inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(), (TData *)outptr + d * outoffset);
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                outptr += (m_coordDim - 1) * outoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        // Shape size.
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nq0 * nq1 * nq2);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                  nq2);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto outoffset = outblock.CompSize();
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation,
                                           DEFORMED>),
                gridsize, blocksize, shmemsize, 0, nq0, nq1, nq2, nelmt,
                outoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2],
                m_f[3], m_dfptr, inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(),
                    (TData *)outptr + d * outblock.CompSize());
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += 3 * outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(nq0 * nq1 * nq2);
        const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                  nq2);
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        // Loop over components.
        const auto outoffset = outblock.CompSize();
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER(
                (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                           nq0, nq1, nq2>),
                gridsize, blocksize, shmemsize, 0, nelmt, outoffset, m_D[0],
                m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3], m_dfptr, inptr,
                outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            for (unsigned int d = 0; d < m_coordDim; d++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmt,
                    outblock.GetNumData(),
                    (TData *)outptr + d * outblock.CompSize());
            }

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += 3 * outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
