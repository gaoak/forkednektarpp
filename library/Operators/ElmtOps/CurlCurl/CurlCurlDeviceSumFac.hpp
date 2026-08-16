///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceSumFac.hpp
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
#include "Operators/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp"

#include "Operators/ElmtOps/CurlCurl/CurlCurlDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/CurlCurl/CurlCurlDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void CombineOmega2DKernel(const size_t npts,
                                            const size_t outoffset,
                                            const TData *NEK_RESTRICT grad0,
                                            const TData *NEK_RESTRICT grad1,
                                            TData *NEK_RESTRICT omega,
                                            const TthreadBlock &threadBlock)
{
    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < npts)
    {
        omega[idx] = grad1[idx] - grad0[outoffset + idx];
        idx += getGlobalRange<0>(threadBlock);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void AssembleCurlCurl2DKernel(
    const size_t npts, const size_t outoffset,
    const TData *NEK_RESTRICT gradOmega, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < npts)
    {
        out[idx]             = gradOmega[outoffset + idx];
        out[outoffset + idx] = -gradOmega[idx];
        idx += getGlobalRange<0>(threadBlock);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void CombineOmega3DKernel(
    const size_t npts, const size_t outoffset, const TData *NEK_RESTRICT grad0,
    const TData *NEK_RESTRICT grad1, const TData *NEK_RESTRICT grad2,
    TData *NEK_RESTRICT omega, const TthreadBlock &threadBlock)
{
    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < npts)
    {
        omega[idx] = grad2[outoffset + idx] - grad1[2u * outoffset + idx];
        omega[outoffset + idx]      = grad0[2u * outoffset + idx] - grad2[idx];
        omega[2u * outoffset + idx] = grad1[idx] - grad0[outoffset + idx];
        idx += getGlobalRange<0>(threadBlock);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void AssembleCurlCurl3DKernel(
    const size_t npts, const size_t outoffset,
    const TData *NEK_RESTRICT gradOmega0, const TData *NEK_RESTRICT gradOmega1,
    const TData *NEK_RESTRICT gradOmega2, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < npts)
    {
        out[idx] =
            gradOmega2[outoffset + idx] - gradOmega1[2u * outoffset + idx];
        out[outoffset + idx] =
            gradOmega0[2u * outoffset + idx] - gradOmega2[idx];
        out[2u * outoffset + idx] =
            gradOmega1[idx] - gradOmega0[outoffset + idx];
        idx += getGlobalRange<0>(threadBlock);
    }
}

template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacTOP>;
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(
                block_idx, m_implInterleaveWidth, transpose));
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
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<const TData *> m_D;
    std::vector<const TData *> m_f;
    const TData *m_dfptr;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
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

    void SegBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter1D(m_coordDim, m_nq[0]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, TemplatedPhysSizeParameter1D<coordDim, nq0>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter1D>
    NEK_FORCE_INLINE void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter1D sizeParam1D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

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
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, interleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);

            // Calculate derivative.
            DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (CurlCurl1DKernelLauncher<Implementation, DEFORMED>), gridsize,
                blocksize, m_streamID, sizeParam1D, nelmt, inoffset, m_D[0],
                m_dfptr, inptr, outptr);

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                outblock.GetNumData(), outptr, m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize();
            outptr += outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter2D(m_coordDim, m_nq[0], m_nq[1]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedPhysSizeParameter2D<coordDim, nq0, nq1>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter2D>
    NEK_FORCE_INLINE void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter2D sizeParam2D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            CurlCurlSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam2D);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam2D.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        const auto inoffset = outblock.CompSize();
        auto wsp = BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
            7u * inoffset, m_streamID);
        auto grad0     = wsp;
        auto grad1     = grad0 + 2u * inoffset;
        auto omega     = grad1 + 2u * inoffset;
        auto gradOmega = omega + inoffset;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, 2 * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam2D, nelmt,
            inoffset, m_D[0], m_D[1], m_f[0], m_f[1], m_dfptr, inptr, grad0);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam2D, nelmt,
            inoffset, m_D[0], m_D[1], m_f[0], m_f[1], m_dfptr, inptr + inoffset,
            grad1);
        DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(CombineOmega2DKernel, gridsize,
                                              blocksize, m_streamID, inoffset,
                                              inoffset, grad0, grad1, omega);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam2D, nelmt,
            inoffset, m_D[0], m_D[1], m_f[0], m_f[1], m_dfptr, omega,
            gradOmega);
        DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
            AssembleCurlCurl2DKernel, gridsize, blocksize, m_streamID, inoffset,
            inoffset, gradOmega, outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, 2 * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, 2 * nelmt,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter3D(m_nq[0], m_nq[1], m_nq[2]));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, TemplatedPhysSizeParameter3D<nq0, nq1, nq2>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter3D>
    NEK_FORCE_INLINE void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter3D sizeParam3D)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            CurlCurlSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam3D);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam3D.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        const auto inoffset = outblock.CompSize();
        auto wsp = BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
            21u * inoffset, m_streamID);
        auto grad0      = wsp;
        auto grad1      = grad0 + 3u * inoffset;
        auto grad2      = grad1 + 3u * inoffset;
        auto omega      = grad2 + 3u * inoffset;
        auto gradOmega0 = omega + 3u * inoffset;
        auto gradOmega1 = gradOmega0 + 3u * inoffset;
        auto gradOmega2 = gradOmega1 + 3u * inoffset;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, 3 * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, inptr, grad0);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, inptr + inoffset, grad1);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, inptr + 2u * inoffset, grad2);
        DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
            CombineOmega3DKernel, gridsize, blocksize, m_streamID, inoffset,
            inoffset, grad0, grad1, grad2, omega);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, omega, gradOmega0);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, omega + inoffset, gradOmega1);
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (PhysDeriv3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam3D, nelmt,
            inoffset, m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2], m_f[3],
            m_dfptr, omega + 2u * inoffset, gradOmega2);
        DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
            AssembleCurlCurl3DKernel, gridsize, blocksize, m_streamID, inoffset,
            inoffset, gradOmega0, gradOmega1, gradOmega2, outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, 3 * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, 3 * nelmt,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
