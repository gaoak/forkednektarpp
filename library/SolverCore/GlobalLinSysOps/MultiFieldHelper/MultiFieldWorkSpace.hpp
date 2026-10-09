///////////////////////////////////////////////////////////////////////////////
//
// File: MultiFieldWorkSpace.hpp
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
// Description: Scratch memory of the MultiField products.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include <cstring>
#include <unordered_map>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Scratch memory of the MultiField products, one buffer per memory
 * space and stream.
 *
 * Buffers are kept for the life of the program and grown on demand, so that
 * repeated products do not allocate. A product uses one buffer per block, on
 * the block's stream, and one on stream 0 to combine the blocks.
 *
 * @tparam TData  The floating-point representation of the scratch.
 */
template <typename TData> class MultiFieldWorkSpace
{
public:
    /**
     * @brief Buffer of at least @p size values in @p MemSpace for
     * @p streamID.
     */
    template <typename MemSpace>
    static TData *Get(const size_t size, const unsigned int streamID)
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            auto &size0 = m_hostSize[streamID];
            auto &ptr   = m_host[streamID];
            if (size0 < size)
            {
                if (ptr)
                {
                    hostFree(ptr, NektarSpaces::host_memory_alignment);
                }
                hostMalloc(&ptr, sizeof(TData) * size,
                           NektarSpaces::host_memory_alignment);
                std::memset(ptr, 0, sizeof(TData) * size);
                size0 = size;
            }
            return ptr;
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            auto &size0 = m_deviceSize[streamID];
            auto &ptr   = m_device[streamID];
            if (size0 < size)
            {
                if (ptr)
                {
                    deviceFree(ptr, sizeof(TData) * size0, streamID);
                }
                deviceMalloc(&ptr, sizeof(TData) * size, streamID);
                size0 = size;
            }
            return ptr;
        }
    }

private:
    static inline std::unordered_map<unsigned int, TData *> m_host;
    static inline std::unordered_map<unsigned int, size_t> m_hostSize;
    static inline std::unordered_map<unsigned int, TData *> m_device;
    static inline std::unordered_map<unsigned int, size_t> m_deviceSize;
};

} // namespace Nektar::SolverCore::detail
