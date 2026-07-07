///////////////////////////////////////////////////////////////////////////////
//
// File: BlockOperator.hpp
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

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LocalRegions/Expansion.h>

#include "Operators/Common/DataWarehouse/NekDataWarehouse.hpp"
#include "Operators/Field/Block.hpp"
#include "Operators/Field/Field.hpp"

namespace Nektar::Operators
{

// Forward-declare the BlockOperator base class so we can define the factory
template <typename TData> class BlockOperator;

// BlockOperator factory singleton
template <typename TData>
using BlockOperatorFactory = Nektar::LibUtilities::NekFactory<
    std::string, BlockOperator<TData>, const unsigned int,
    const LocalRegions::ExpansionSharedPtr &, NekDataWarehouseSharedPtr>;

// BlockOperator factory singleton
template <typename TData>
BlockOperatorFactory<TData> &GetBlockOperatorFactory();

template <typename TData> class BlockOperator
{
public:
    virtual ~BlockOperator() = default;

    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, const std::string &execStr)
    {
        std::string requestedKey = TOperator<TData>::name + execStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        // No suitible operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, block_idx, exp,
                                   dataWarehouse));
    }

protected:
    unsigned int m_block_idx;
    LocalRegions::ExpansionSharedPtr m_exp;
    NekDataWarehouseSharedPtr m_dataWarehouse;

    BlockOperator(const unsigned int block_idx,
                  const LocalRegions::ExpansionSharedPtr &exp,
                  NekDataWarehouseSharedPtr dataWarehouse)
        : m_block_idx(block_idx), m_exp(exp), m_dataWarehouse(dataWarehouse)
    {
    }

    // Static host and device storages are use by all instances of
    // BlockOperator to avoid repeated allocation and deallocation. Repeated
    // device memory allocation and deallocation can be very innefficient and
    // cause memory fragmentation while ownership of large device memory blocks
    // by all instances of BlockOperator is a waste of resources an  can results
    // in insufficient memory. The current implementation assumes that all
    // BlockOperator instances execute on the default stream/queue.
    template <typename MemSpace>
    static TData *GetStaticWorkSpace(const size_t size,
                                     const unsigned int streamID = 0)
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            SetHostWorkSpace(size, streamID);
            return m_hostWsp[streamID];
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            SetDeviceWorkSpace(size, streamID);
            return m_deviceWsp[streamID];
        }
    }

private:
    static void SetHostWorkSpace(const size_t size, const unsigned int streamID)
    {
        if (m_hostWspSize.find(streamID) == m_hostWspSize.end())
        {
            TData *hostptr = nullptr;
            hostMalloc(&hostptr, sizeof(TData) * size,
                       NektarSpaces::host_memory_alignment);
            memset(hostptr, 0, sizeof(TData) * size);
            m_hostWsp[streamID]     = hostptr;
            m_hostWspSize[streamID] = size;
        }
        else if (m_hostWspSize[streamID] < size)
        {
            TData *hostptr = m_hostWsp[streamID];
            hostFree(hostptr, NektarSpaces::host_memory_alignment);
            hostMalloc(&hostptr, sizeof(TData) * size,
                       NektarSpaces::host_memory_alignment);
            memset(hostptr, 0, sizeof(TData) * size);
            m_hostWsp[streamID]     = hostptr;
            m_hostWspSize[streamID] = size;
        }
    }

    static void SetDeviceWorkSpace(const size_t size,
                                   const unsigned int streamID)
    {
        if (m_deviceWspSize.find(streamID) == m_deviceWspSize.end())
        {
            TData *deviceptr = nullptr;
            deviceMalloc(&deviceptr, sizeof(TData) * size, streamID);
            m_deviceWsp[streamID]     = deviceptr;
            m_deviceWspSize[streamID] = size;
        }
        else if (m_deviceWspSize[streamID] < size)
        {
            TData *deviceptr = m_deviceWsp[streamID];
            deviceFree(deviceptr, sizeof(TData) * m_deviceWspSize[streamID],
                       streamID);
            deviceMalloc(&deviceptr, sizeof(TData) * size, streamID);
            m_deviceWsp[streamID]     = deviceptr;
            m_deviceWspSize[streamID] = size;
        }
    }

    static inline std::unordered_map<unsigned int, TData *> m_deviceWsp;
    static inline std::unordered_map<unsigned int, size_t> m_deviceWspSize;
    static inline std::unordered_map<unsigned int, TData *> m_hostWsp;
    static inline std::unordered_map<unsigned int, size_t> m_hostWspSize;
};

} // namespace Nektar::Operators
