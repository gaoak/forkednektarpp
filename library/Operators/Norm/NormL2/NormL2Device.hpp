///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2Device.hpp
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

#include <Operators/ElmtOps/ElmtBlockOp.hpp>
#include <Operators/Norm/NormL2/NormL2BlockOp.hpp>

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NormL2BlockOpImpl : public NormL2BlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormL2BlockOpImpl(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : NormL2BlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_nqTot     = exp->GetTotPoints();
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Get basis keys for fetching matrices.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();

            // Fetch element size.
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(basisKeys[d],
                                                  LibUtilities::eWeights)));
        }

        // Fetch Jacobian.
        m_jacptr1 = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, 1));
        m_jacptr2 = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx,
                                             NektarSpaces::Device::warpSize));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<NormL2BlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<NormL2BlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed        = false;
    unsigned int m_nqTot     = 0;
    unsigned int m_dimension = 0;
    unsigned int m_coordDim  = 0;
    const TData *m_jacptr1   = nullptr;
    const TData *m_jacptr2   = nullptr;
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_nm;
    std::vector<const TData *> m_W;

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        switch (m_dimension)
        {
            case 1:
            {
                OperatorND<1>(inblock, data);
                break;
            }
            case 2:
            {
                OperatorND<2>(inblock, data);
                break;
            }
            case 3:
            {
                OperatorND<3>(inblock, data);
                break;
            }
            default:
                ASSERTL0(
                    false,
                    "NormL2 Device only implemented for dimension 1, 2 or 3.");
        }
    }

    template <unsigned int NDIM>
    void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data)
    {
        OperatorNDImpl(inblock, data,
                       std::make_integer_sequence<unsigned int, NDIM>());
    }

    template <unsigned int... ind>
    void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data,
        std::integer_sequence<unsigned int, ind...>)
    {
        const auto nelmt    = inblock.GetNumElements();
        const auto nelmtPad = inblock.GetNumElementsWithPadding();
        const auto nComp    = inblock.GetNumComponents();
        const auto nHomo    = inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        // Get interleave parameter and the stride between components.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto compSize        = inblock.CompSize();

        // Set Kernel parameters.
        const unsigned int shmemsize = 0;
        const unsigned int blocksize =
            (interleaveWidth == 1) ? GetDeviceBlockSize<SumFacTOP>(m_nqTot)
                                   : GetDeviceBlockSize<SumFac>(m_nqTot);
        const unsigned int gridsize =
            (interleaveWidth == 1)
                ? GetDeviceGridSize<SumFacTOP>(nelmtPad, blocksize, shmemsize)
                : GetDeviceGridSize<SumFac>(nelmtPad, blocksize, shmemsize);

        // Compute volume.
        if (this->m_normalised)
        {
            if (m_isDeformed)
            {
                if (interleaveWidth == 1)
                {
                    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (VolumeKernelLauncher<1u, true>), gridsize, blocksize,
                        m_streamID, m_nq[ind]..., nelmt, m_W[ind]..., m_jacptr1,
                        dataptr + nComp);
                }
                else
                {
                    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (VolumeKernelLauncher<NektarSpaces::Device::warpSize,
                                              true>),
                        gridsize, blocksize, m_streamID, m_nq[ind]..., nelmt,
                        m_W[ind]..., m_jacptr2, dataptr + nComp);
                }
            }
            else
            {
                if (interleaveWidth == 1)
                {
                    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (VolumeKernelLauncher<1u, false>), gridsize, blocksize,
                        m_streamID, m_nq[ind]..., nelmt, m_W[ind]..., m_jacptr1,
                        dataptr + nComp);
                }
                else
                {
                    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (VolumeKernelLauncher<NektarSpaces::Device::warpSize,
                                              false>),
                        gridsize, blocksize, m_streamID, m_nq[ind]..., nelmt,
                        m_W[ind]..., m_jacptr2, dataptr + nComp);
                }
            }
        }

        // Compute norm.
        if (m_isDeformed)
        {
            if (interleaveWidth == 1)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (IntegralKernelLauncher<IntegralOp::Square, 1u, true>),
                    gridsize, nComp * nHomo, blocksize, 1, m_streamID,
                    m_nq[ind]..., nelmt, compSize, m_W[ind]..., m_jacptr1,
                    inptr, dataptr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (IntegralKernelLauncher<IntegralOp::Square,
                                            NektarSpaces::Device::warpSize,
                                            true>),
                    gridsize, nComp * nHomo, blocksize, 1, m_streamID,
                    m_nq[ind]..., nelmt, compSize, m_W[ind]..., m_jacptr2,
                    inptr, dataptr);
            }
        }
        else
        {
            if (interleaveWidth == 1)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (IntegralKernelLauncher<IntegralOp::Square, 1u, false>),
                    gridsize, nComp * nHomo, blocksize, 1, m_streamID,
                    m_nq[ind]..., nelmt, compSize, m_W[ind]..., m_jacptr1,
                    inptr, dataptr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (IntegralKernelLauncher<IntegralOp::Square,
                                            NektarSpaces::Device::warpSize,
                                            false>),
                    gridsize, nComp * nHomo, blocksize, 1, m_streamID,
                    m_nq[ind]..., nelmt, compSize, m_W[ind]..., m_jacptr2,
                    inptr, dataptr);
            }
        }
    }
};

} // namespace Nektar::Operators::detail
