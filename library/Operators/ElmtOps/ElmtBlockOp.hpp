///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtBlockOp.hpp
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

#include <algorithm>

#include "LibUtilities/Backends/DeviceProperties.hpp"
#include "Operators/Common/BlockOperator.hpp"

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/DataWarehouse/ModeIndexDataWarehouse.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <StdRegions/DataWarehouse/StdMatDataWarehouse.hpp>

namespace Nektar::Operators
{

#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
static constexpr unsigned int defaultElmtsPerBlock = 8u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#elif defined(SYCL_ENABLE_INTEL)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#else
static constexpr unsigned int defaultElmtsPerBlock = 1u;
#endif

// Core implementation types.
struct StdMat
{
    static inline const std::string name = "StdMat";
};

struct SumFac
{
    static inline const std::string name = "SumFac";
};

struct SumFacTOP
{
    static inline const std::string name = "SumFacTOP";
};

struct Generic
{
    static inline const std::string name = "Generic";
};

template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtBlockOp : public BlockOperator<TData>
{
public:
    // Accessor types for this operator's input and output fields. Derived
    // classes and the generated ShapeBlock definitions use these rather than
    // restating the field states, so each operator states them in exactly
    // one place: its ElmtBlockOp base clause in <Op>BlockOp.hpp.
    using InBlock  = LibUtilities::BlockAccessor<TData, TFieldIn>;
    using OutBlock = LibUtilities::BlockAccessor<TData, TFieldOut>;

    ~ElmtBlockOp() override = default;

    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, std::string implStr)
    {
        std::string requestedKey = TOperator<TData>::name + execStr + implStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            // See if there is a Generic implementation.
            auto requestedKey0 = requestedKey;
            requestedKey       = TOperator<TData>::name + execStr + "Generic";

            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey0 << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, block_idx, exp,
                                   dataWarehouse));
    }

    void Apply(LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
               LibUtilities::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
                    LibUtilities::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

protected:
    bool m_warnOnceTemplate = false; /// boolean flag to allow one warning

    ElmtBlockOp(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    virtual void v_Apply(
        LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) = 0;
};

#if defined(NEKTAR_ENABLE_DEVICE)
template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int elmtBlockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return std::min(((elmtBlockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
    else
    {
        return 0;
    }
}

template <typename Implementation,
          unsigned int elmtsPerBlock = defaultElmtsPerBlock>
NEK_FORCE_INLINE static unsigned int GetDeviceGridSize(
    const size_t nelmt, const unsigned int blockSize,
    [[maybe_unused]] const unsigned int shmemsize)
{
    constexpr size_t maxGridSize                 = 2147483647;
    constexpr unsigned int persistentQueueFactor = 2u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        return std::min((nelmt + blockSize - 1u) / blockSize, maxGridSize);
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        const auto smCount =
            static_cast<size_t>(GetDeviceProperties::NumMultiProcessors());
        // Number of blocks that can achieve full occupancy on each SM
        const size_t blocksByThreads =
            GetDeviceProperties::MaxThreadsPerMultiprocessor() / blockSize;
        // Number of blocks that can fully utilize shared memory on each SM
        const size_t blocksByShared =
            (shmemsize == 0)
                ? blocksByThreads
                : GetDeviceProperties::SharedMemoryPerMultiprocessor() /
                      shmemsize;
        const size_t residentBlocks = std::min(blocksByThreads, blocksByShared);
        // Number of blocks to launch to keep all SMs busy, controlled by
        // a customizable QueueFactor. This serves as a lower bound on grid size
        const size_t launchBlocks =
            smCount * residentBlocks * persistentQueueFactor;
        const size_t cappedLaunchBlocks =
            std::max<size_t>(1u, std::min(launchBlocks, maxGridSize));

        // Each block processes elmtsPerBlock elements per queue
        // iteration.
        const size_t ngroups = (nelmt + elmtsPerBlock - 1u) / elmtsPerBlock;

        // 1. If nelmt is big, then grid size is determined by ngroups
        // to ensure each block has enough work to do;
        // 2. If nelmt is small, then we try to ensure SMs are busy by
        // launching enough blocks;
        // 3. If nelmt is very small, then each block only process
        // one element to avoid unnecessary idling.
        const unsigned int gridsize = static_cast<unsigned int>(
            std::min(nelmt, std::max(ngroups, cappedLaunchBlocks)));

        return gridsize;
    }
    else
    {
        return 0;
    }
}
#endif

} // namespace Nektar::Operators
