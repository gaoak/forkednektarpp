///////////////////////////////////////////////////////////////////////////////
//
// File: SharedFieldStorage.hpp
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
// Description: Contiguous host and device storage shared by several Field
// objects.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

namespace Nektar::LibUtilities
{

/**
 * @brief Contiguous host and device storage shared by several Field objects.
 *
 * Each Field built on it points into it at its own offset instead of
 * allocating storage of its own, so the fields lie one after the other in
 * memory. Host and device storage are each allocated, zero-initialised, on
 * first request, and freed with the last Field holding the storage.
 *
 * @tparam TData  The floating-point representation of the stored data.
 */
template <typename TData> class SharedFieldStorage
{
public:
    /**
     * @brief Construct the storage. Nothing is allocated until a space is
     * first requested.
     *
     * @param size         - Number of values.
     * @param memAllocType - [ eHostPageable,  eHostPinned, eDeviceMemoryPool,
     * eHostPinnedDeviceMemoryPool]
     * @param alignment    - Memory alignment of the host storage.
     */
    SharedFieldStorage(
        const size_t size, const MemAllocType &memAllocType = eHostPageable,
        const size_t alignment = NektarSpaces::host_memory_alignment)
        : m_size(size), m_memAllocType(memAllocType), m_alignment(alignment)
    {
    }

    ~SharedFieldStorage()
    {
        if (m_device)
        {
            const unsigned int streamID = 0;
            deviceFree(m_device, m_size * sizeof(TData), streamID,
                       m_memAllocType);
            nekStreamSynchronize(streamID);
        }

        if (m_host)
        {
            if (m_memAllocType == eHostPageable)
            {
                hostFree(m_host, m_alignment);
            }
            else if (m_memAllocType == eHostPinned)
            {
                hostFreePinned(m_host);
            }
        }
    }

    SharedFieldStorage(const SharedFieldStorage &)            = delete;
    SharedFieldStorage &operator=(const SharedFieldStorage &) = delete;

    /**
     * @brief Pointer to the start of the storage in @p MemSpace, allocating
     * it on first request.
     */
    template <typename MemSpace> TData *GetPtr()
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            if (!m_host)
            {
                size_t alignment_offset = 0;

                if (m_memAllocType == eHostPageable)
                {
                    hostMalloc(&m_host, m_size * sizeof(TData), m_alignment);
                }
                else if (m_memAllocType == eHostPinned)
                {
                    // Add extra bytes for alignment provision
                    hostMallocPinned(&m_host,
                                     m_size * sizeof(TData) + m_alignment);
                    // Compute offset in bytes for non-aligned memory.
                    if ((size_t)m_host % m_alignment)
                    {
                        alignment_offset =
                            m_alignment - (size_t)m_host % m_alignment;
                    }
                }
                m_host_aligned = (TData *)((size_t)m_host + alignment_offset);
                std::memset((void *)m_host_aligned, 0, m_size * sizeof(TData));
            }

            return m_host_aligned;
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            if (!m_device)
            {
                const unsigned int streamID = 0;
                deviceMalloc(&m_device, m_size * sizeof(TData), streamID,
                             m_memAllocType);
                deviceMemset(m_device, 0, m_size * sizeof(TData), streamID);
                nekStreamSynchronize(streamID);
            }

            return m_device;
        }
    }

    /**
     * @brief Number of values held.
     */
    size_t size() const
    {
        return m_size;
    }

    MemAllocType GetMemAllocType() const
    {
        return m_memAllocType;
    }

    size_t GetAlignment() const
    {
        return m_alignment;
    }

private:
    size_t m_size;
    MemAllocType m_memAllocType;
    size_t m_alignment;
    TData *m_host         = nullptr; ///< Host storage as allocated.
    TData *m_host_aligned = nullptr; ///< Aligned start of the host storage.
    TData *m_device       = nullptr; ///< Device storage.
};

} // namespace Nektar::LibUtilities
