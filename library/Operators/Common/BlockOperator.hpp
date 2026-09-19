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

/**
 * @file BlockOperator.hpp
 * @brief Block-level operator base class and factory machinery shared
 * by every operator family in the Operators library, plus the static
 * per-stream workspace they share.
 *
 * @details
 * BlockOperator fixes no input or output block type; it holds the
 * block index, representative expansion and data warehouse a block
 * operator was built with, provides the family-agnostic Create(), and
 * gives every block operator access to a per-stream scratch buffer
 * through GetStaticWorkSpace() so that repeated device and host
 * allocation is avoided. Families that act element by element derive
 * their block-operator base from ElmtBlockOp (ElmtOps/ElmtBlockOp.hpp)
 * instead, which adds the implementation-strategy tags and device
 * launch helpers on top of this class.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LocalRegions/Expansion.h>

#include "LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp"
#include "LibUtilities/BasicUtils/Field/Block.hpp"
#include "LibUtilities/BasicUtils/Field/Field.hpp"

namespace Nektar::Operators
{

// Forward-declare the BlockOperator base class so we can define the factory
template <typename TData> class BlockOperator;

/// @brief Factory of BlockOperator<TData> interface objects, keyed by
/// `TOperator::name + execStr`.
template <typename TData>
using BlockOperatorFactory =
    LibUtilities::NekFactory<std::string, BlockOperator<TData>,
                             const unsigned int,
                             const LocalRegions::ExpansionSharedPtr &,
                             LibUtilities::NekDataWarehouseSharedPtr>;

/// @brief Return the process-wide singleton BlockOperatorFactory<TData>.
template <typename TData>
BlockOperatorFactory<TData> &GetBlockOperatorFactory();

/**
 * @brief Common base class of the block-operator family interfaces:
 * holds the block index, representative expansion and data warehouse a
 * block operator was built with, provides the family-agnostic
 * Create(), and gives derived classes a per-stream scratch buffer
 * through GetStaticWorkSpace().
 *
 * @tparam TData  Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the element-by-element specialisation that adds
 * the implementation-strategy tags and device launch helpers;
 * Operator for the whole-field counterpart this class is created
 * alongside.
 */
template <typename TData> class BlockOperator
{
public:
    virtual ~BlockOperator() = default;

    /**
     * @brief Create a concrete block-operator instance through the
     * block-operator factory.
     *
     * The factory key is `TOperator::name + execStr`. The product is
     * handed back through a static_pointer_cast to @p TOperator, so
     * the creator registered under the key must construct that class
     * or one derived from it.
     *
     * @tparam TOperator  Family block-operator class supplying the
     *                    static `name` string.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block
     *                          (its first element).
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the factory
     *                          key.
     *
     * @return The newly created block operator. Creation raises a
     * fatal error (throws ErrorUtil::NekError) if no implementation is
     * registered under the requested key.
     */
    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr)
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
    /// Index of the block within the expansion list's Collections.
    unsigned int m_block_idx;
    /// Representative expansion of the block (its first element).
    LocalRegions::ExpansionSharedPtr m_exp;
    /// Data warehouse shared with the other operators on the expansion
    /// list.
    LibUtilities::NekDataWarehouseSharedPtr m_dataWarehouse;

    /**
     * @brief Construct the interface part of a concrete block
     * operator; called by the factory-registered creator functions.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    BlockOperator(const unsigned int block_idx,
                  const LocalRegions::ExpansionSharedPtr &exp,
                  LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : m_block_idx(block_idx), m_exp(exp), m_dataWarehouse(dataWarehouse)
    {
    }

    /**
     * @brief Return a per-stream scratch buffer in @p MemSpace, sized
     * to hold at least @p size elements of @p TData, growing and
     * reallocating it as needed.
     *
     * The buffer is static: every BlockOperator instance in the
     * process shares one buffer per stream, so repeated allocation and
     * deallocation is avoided. Repeated device allocation in
     * particular can be slow and fragment device memory, and one
     * buffer per instance would hold onto large device blocks that are
     * wasted while that instance is not using them, risking exhausting
     * device memory. The current implementation assumes every
     * BlockOperator instance executes on the default stream/queue,
     * i.e. @p streamID is always 0.
     *
     * @tparam MemSpace  NektarSpaces::HostSpace or
     *                   NektarSpaces::DeviceSpace; selects which
     *                   buffer and allocator are used.
     *
     * @param   size      Minimum number of @p TData elements the
     *                     buffer must hold.
     * @param   streamID  Stream/queue the buffer is kept for; each
     *                     value gets its own buffer.
     *
     * @return Pointer to the (possibly just grown) buffer for
     * @p streamID.
     */
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
    /**
     * @brief Ensure the host scratch buffer for @p streamID exists and
     * holds at least @p size elements, (re)allocating it if it is
     * missing or too small.
     *
     * @param   size      Minimum number of @p TData elements required.
     * @param   streamID  Stream the buffer is kept for.
     */
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

    /**
     * @brief Ensure the device scratch buffer for @p streamID exists
     * and holds at least @p size elements, (re)allocating it if it is
     * missing or too small.
     *
     * @param   size      Minimum number of @p TData elements required.
     * @param   streamID  Stream the buffer is kept for.
     */
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

    /// Device scratch buffer for each stream, keyed by stream ID.
    static inline std::unordered_map<unsigned int, TData *> m_deviceWsp;
    /// Element capacity of the buffer in #m_deviceWsp for each stream.
    static inline std::unordered_map<unsigned int, size_t> m_deviceWspSize;
    /// Host scratch buffer for each stream, keyed by stream ID.
    static inline std::unordered_map<unsigned int, TData *> m_hostWsp;
    /// Element capacity of the buffer in #m_hostWsp for each stream.
    static inline std::unordered_map<unsigned int, size_t> m_hostWspSize;
};

} // namespace Nektar::Operators
