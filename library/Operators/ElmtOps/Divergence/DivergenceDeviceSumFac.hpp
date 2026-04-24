///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceDeviceSumFac.hpp
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

#include "Operators/ElmtOps/Divergence/DivergenceBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Divergence/DivergenceDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/Divergence/DivergenceDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        ASSERTL1(m_dimension == m_coordDim,
                 "Setup assuming coordinate dimention is the same as shape "
                 "dimension");

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
            DivergenceBlockOpImpl<ExecSpace, Implementation, TData>>(
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
        Operator1D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplated1DPhysSizeParameters(m_coordDim, m_nq[0]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, Templated1DPhysSizeParameters<coordDim, nq0>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename SizeParameter1D>
    NEK_FORCE_INLINE void Operator1D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        SizeParameter1D sizeParam1D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam1D.nq0());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, 0);

        // Loop over components.
        const auto inoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (Divergence1DKernelLauncher<Implementation, DEFORMED>),
                gridsize, blocksize, 0, sizeParam1D, nelmt, inoffset, m_D[0],
                m_dfptr, inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(),
                                      (TData *)outptr);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplated2DPhysSizeParameters(m_coordDim, m_nq[0], m_nq[1]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            Templated2DPhysSizeParameters<coordDim, nq0, nq1>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename SizeParameter2D>
    NEK_FORCE_INLINE void Operator2D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        SizeParameter2D sizeParam2D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            DivergenceSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam2D.nq0(), sizeParam2D.nq1());
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(
            sizeParam2D.nq0() * sizeParam2D.nq1());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Loop over components.
        const auto inoffset = outblock.CompSize();

        // Reshape u-component, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth, nelmt,
                                  inblock.GetNumData(), (TData *)inptr);

        // Reshape v-component, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + inoffset);

        // Calculate derivative du/dx
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (Divergence2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, 0, sizeParam2D, nelmt, inoffset,
            m_D[0], m_D[1], m_f[0], m_f[1], m_dfptr, inptr, outptr);

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  inblock.GetNumData(), (TData *)inptr);

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + inoffset);

        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  outblock.GetNumData(), (TData *)outptr);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplated3DPhysSizeParameters(m_nq[0], m_nq[1], m_nq[2]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, Templated3DPhysSizeParameters<nq0, nq1, nq2>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename SizeParameter3D>
    NEK_FORCE_INLINE void Operator3D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        SizeParameter3D sizeParam3D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            DivergenceSharedMemorySize<SHAPE_TYPE, Implementation>(
                sizeParam3D.nq0(), sizeParam3D.nq1(), sizeParam3D.nq2());
        const unsigned int blocksize = GetDeviceBlockSize<Implementation>(
            sizeParam3D.nq0() * sizeParam3D.nq1() * sizeParam3D.nq2());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Loop over components.
        const auto inoffset = outblock.CompSize();

        // Reshape u-component, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth, nelmt,
                                  inblock.GetNumData(), (TData *)inptr);

        // Reshape v-component, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + inoffset);

        // Reshape w-component, if necessary.
        ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + 2u * inoffset);

        // Calculate derivative du/dx
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (Divergence3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, 0, sizeParam3D, nelmt, inoffset,
            m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3], m_dfptr,
            inptr, outptr);

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  inblock.GetNumData(), (TData *)inptr);

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + inoffset);

        // Reshape back, if necessary.
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  inblock.GetNumData(),
                                  (TData *)inptr + 2u * inoffset);

        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  outblock.GetNumData(), (TData *)outptr);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
