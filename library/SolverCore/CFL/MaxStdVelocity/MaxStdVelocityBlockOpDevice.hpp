///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocityBlockOpDevice.hpp
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
// Description: MaxStdVelocity, device implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/ElmtBlockOp.hpp"
#include "SolverCore/CFL/MaxStdVelocity/MaxStdVelocityBlockOp.hpp"

#include "SolverCore/CFL/MaxStdVelocity/MaxStdVelocityBlockOpDeviceKernels.hpp"

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp"

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MaxStdVelocityBlockOpImpl : public MaxStdVelocityBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MaxStdVelocityBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MaxStdVelocityBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_nqTot     = exp->GetTotPoints();
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Fetch derivative factors. The SumFacTOP kernel reads the TOP
        // (interleave width 1) array component-major, so it is fetched
        // transposed - as PhysDeriv does for its own SumFacTOP kernel.
        m_dfptr1 = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx, 1, true));
        m_dfptr2 = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(
                block_idx, NektarSpaces::Device::warpSize, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<MaxStdVelocityBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<MaxStdVelocityBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    unsigned int m_streamID;
    bool m_isDeformed;
    unsigned int m_nqTot;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    const TData *m_dfptr1;
    const TData *m_dfptr2;

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        const TData soundSpeedFactor = this->m_soundSpeedFactor;
        const auto nelmt             = inblock.GetNumElements();
        const auto nelmtPad          = inblock.GetNumElementsWithPadding();
        const auto compSize          = inblock.CompSize();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Initialize pointers.
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        // Set Kernel parameters.
        const unsigned int shmemsize = 0;
        const unsigned int blocksize =
            (interleaveWidth == 1)
                ? Operators::GetDeviceBlockSize<Operators::SumFacTOP>(m_nqTot)
                : Operators::GetDeviceBlockSize<Operators::SumFac>(m_nqTot);
        const unsigned int gridsize =
            (interleaveWidth == 1)
                ? Operators::GetDeviceGridSize<Operators::SumFacTOP>(
                      nelmtPad, blocksize, shmemsize)
                : Operators::GetDeviceGridSize<Operators::SumFac>(
                      nelmtPad, blocksize, shmemsize);

        if (m_isDeformed)
        {
            if (interleaveWidth == 1)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MaxStdVelocityKernelLauncher<Operators::SumFacTOP, true>),
                    gridsize, blocksize, m_streamID, nelmt, compSize,
                    m_dimension, m_coordDim, m_nqTot, soundSpeedFactor,
                    m_dfptr1, inptr, dataptr + this->m_block_idx);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MaxStdVelocityKernelLauncher<Operators::SumFac, true>),
                    gridsize, blocksize, m_streamID, nelmt, compSize,
                    m_dimension, m_coordDim, m_nqTot, soundSpeedFactor,
                    m_dfptr2, inptr, dataptr + this->m_block_idx);
            }
        }
        else
        {
            if (interleaveWidth == 1)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MaxStdVelocityKernelLauncher<Operators::SumFacTOP, false>),
                    gridsize, blocksize, m_streamID, nelmt, compSize,
                    m_dimension, m_coordDim, m_nqTot, soundSpeedFactor,
                    m_dfptr1, inptr, dataptr + this->m_block_idx);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MaxStdVelocityKernelLauncher<Operators::SumFac, false>),
                    gridsize, blocksize, m_streamID, nelmt, compSize,
                    m_dimension, m_coordDim, m_nqTot, soundSpeedFactor,
                    m_dfptr2, inptr, dataptr + this->m_block_idx);
            }
        }
    }
};

} // namespace Nektar::SolverCore::detail
