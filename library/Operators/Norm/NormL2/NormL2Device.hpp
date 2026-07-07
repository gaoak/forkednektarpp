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

#include "NormL2DeviceKernels.hpp"
#include "Operators/Common/DataWarehouse/BasisDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/GeometricDataWarehouse.hpp"
#include "Operators/ElmtOps/ElmtBlockOp.hpp"
#include "Operators/Field/MemoryRegion.hpp"
#include "Operators/Norm/NormL2/NormL2BlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NormL2BlockOpImpl : public NormL2BlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormL2BlockOpImpl(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      NekDataWarehouseSharedPtr dataWarehouse)
        : NormL2BlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID  = block_idx + 1;
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_nqTot     = exp->GetTotPoints();
        m_dimension = exp->GetShapeDimension();
        m_coordim   = exp->GetCoordim();

        // Get basis keys for fetching matrices
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();

            // Fetch element size.
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(basisKeys[d], eWeights)));
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<NormL2BlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<NormL2BlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    // Note we assume SumFac ie one thread per element
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::Device::warpSize;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed        = false;
    unsigned int m_nqTot     = 0;
    unsigned int m_dimension = 0;
    unsigned int m_coordim   = 0;
    const TData *m_jacptr    = nullptr;
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_nm;
    std::vector<const TData *> m_W;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 MemoryRegion<TData> &data) override
    {
        if (inblock.GetNumElements() == 0)
        {
            return;
        }

        switch (m_dimension)
        {
            case 1:
            {
                Operator1D(inblock, data);
                break;
            }
            case 2:
            {
                Operator2D(inblock, data);
                break;
            }
            case 3:
            {
                Operator3D(inblock, data);
                break;
            }
            default:
                ASSERTL0(
                    false,
                    "NormL2 Device only implemented for dimension 1, 2 or 3.");
        }
    }

    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    MemoryRegion<TData> &data)
    {
        auto sizeParam1D = NonTemplated1DPhysSizeParameters(m_coordim, m_nq[0]);

        const auto nelmt       = inblock.GetNumElements();
        const auto paddedNelmt = inblock.GetNumElementsWithPadding();

        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto redptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize = 0;
        const unsigned int blocksize =
            GetDeviceBlockSize<SumFac>(sizeParam1D.nq0());
        const unsigned int gridsize =
            GetDeviceGridSize<SumFac>(nelmt, blocksize, shmemsize);

        const unsigned int nComp = inblock.GetNumComponents();
        const unsigned int nHomo = inblock.GetNumHomoModes();

        // Compute volume
        if (this->m_normalised)
        {
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume1DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam1D, nelmt, m_W[0], m_jacptr,
                    redptr + nComp);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume1DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam1D, nelmt, m_W[0], m_jacptr,
                    redptr + nComp);
            }
        }

        // Compute norm
        for (unsigned int nc = 0; nc < nComp * nHomo; ++nc)
        {
            // SumFac kernels expect point-major, warp-lane-minor storage.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm1DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam1D, nelmt, m_W[0], m_jacptr, inptr,
                    redptr + nc);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm1DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam1D, nelmt, m_W[0], m_jacptr, inptr,
                    redptr + nc);
            }

            // Restore the original field layout for downstream operators.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize();
        }
    }

    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    MemoryRegion<TData> &data)
    {
        auto sizeParam2D =
            NonTemplated2DPhysSizeParameters(m_coordim, m_nq[0], m_nq[1]);
        const auto nqTot       = m_nq[0] * m_nq[1];
        const auto nelmt       = inblock.GetNumElements();
        const auto paddedNelmt = inblock.GetNumElementsWithPadding();

        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto redptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        const auto interleaveWidth = inblock.GetInterleaveWidth();

        const unsigned int shmemsize = 0;
        const unsigned int blocksize = GetDeviceBlockSize<SumFac>(nqTot);
        const unsigned int gridsize =
            GetDeviceGridSize<SumFac>(nelmt, blocksize, shmemsize);

        const unsigned int nComp = inblock.GetNumComponents();
        const unsigned int nHomo = inblock.GetNumHomoModes();

        // Compute volume
        if (this->m_normalised)
        {
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume2DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam2D, nelmt, m_W[0], m_W[1], m_jacptr,
                    redptr + nComp);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume2DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam2D, nelmt, m_W[0], m_W[1], m_jacptr,
                    redptr + nComp);
            }
        }

        // Compute norm
        for (unsigned int nc = 0; nc < nComp * nHomo; ++nc)
        {
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm2DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam2D, nelmt, m_W[0], m_W[1], m_jacptr,
                    inptr, redptr + nc);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm2DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam2D, nelmt, m_W[0], m_W[1], m_jacptr,
                    inptr, redptr + nc);
            }

            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            inptr += inblock.CompSize();
        }
    }

    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    MemoryRegion<TData> &data)
    {
        auto sizeParam3D =
            NonTemplated3DPhysSizeParameters(m_nq[0], m_nq[1], m_nq[2]);
        const auto nqTot       = m_nq[0] * m_nq[1] * m_nq[2];
        const auto nelmt       = inblock.GetNumElements();
        const auto paddedNelmt = inblock.GetNumElementsWithPadding();

        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto redptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        const auto interleaveWidth = inblock.GetInterleaveWidth();

        const unsigned int shmemsize = 0;
        const unsigned int blocksize = GetDeviceBlockSize<SumFac>(nqTot);
        const unsigned int gridsize =
            GetDeviceGridSize<SumFac>(nelmt, blocksize, shmemsize);

        const unsigned int nComp = inblock.GetNumComponents();
        const unsigned int nHomo = inblock.GetNumHomoModes();

        // Compute volume
        if (this->m_normalised)
        {
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume3DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam3D, nelmt, m_W[0], m_W[1], m_W[2],
                    m_jacptr, redptr + nComp);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Volume3DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam3D, nelmt, m_W[0], m_W[1], m_W[2],
                    m_jacptr, redptr + nComp);
            }
        }

        // Compute norm
        for (unsigned int nc = 0; nc < nComp * nHomo; ++nc)
        {
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm3DKernelLauncher<true>), gridsize, blocksize,
                    m_streamID, sizeParam3D, nelmt, m_W[0], m_W[1], m_W[2],
                    m_jacptr, inptr, redptr + nc);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Norm3DKernelLauncher<false>), gridsize, blocksize,
                    m_streamID, sizeParam3D, nelmt, m_W[0], m_W[1], m_W[2],
                    m_jacptr, inptr, redptr + nc);
            }

            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      paddedNelmt, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            inptr += inblock.CompSize();
        }
    }
};

} // namespace Nektar::Operators::detail
