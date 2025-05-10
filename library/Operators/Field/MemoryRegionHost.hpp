//////////////////////////////////////////////////////////////////////////////
//
// File: MemoryRegionHost.hpp
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

#include "MemoryAlloc.hpp"

enum MemAllocType
{
    eHostDevice,
    eDeviceOnly,
    ePinned
};

using namespace Nektar;

template <typename TData> class MemoryRegion;

/**
 * @brief Acts as a holder for a contiguous block of memory, allocated on the
 * host system.
 *
 * This class also acts as a base class for device-aware builds.
 */
template <typename TData> class MemoryRegionHost
{
    friend class MemoryRegion<TData>;

public:
    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param name         - name
     * @param size         - size of memory
     * @param alignment    - memory alignment
     * @param device_rank  - device (GPU) rank id
     * @param memAllocType - [eHostDevice, ePinned]
     */
    MemoryRegionHost(const std::string name, const size_t size,
                     const size_t alignment, const unsigned int device_rank,
                     const MemAllocType &memAllocType)
    {
        m_owned        = true;
        m_initialize   = true;
        m_name         = name;
        m_size         = size;
        m_alignment    = alignment;
        m_device_rank  = device_rank;
        m_memAllocType = memAllocType;
    }

    /**
     * @brief Constructor methods - create a new memory region from an
     * existing host pointer. Specialized constructor method used by
     * Field.hpp to allocate a contiguous host memory coupled with
     * distributed device memory.
     *
     * @param name        - name
     * @param h_src       - host src pointer
     * @param size        - size of memory
     * @param alignment   - memory alignment
     * @param device_rank - device (GPU) rank id
     */
    MemoryRegionHost(const std::string name, TData *h_src, const size_t size,
                     const size_t alignment, const unsigned int device_rank)
    {
        m_owned       = false;
        m_initialize  = false;
        m_name        = name;
        m_size        = size;
        m_alignment   = alignment;
        m_device_rank = device_rank;

        m_host       = h_src;
        m_host_valid = true;
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionHost
     *
     * @param rhs - MemoryRegionHost to move from
     */
    MemoryRegionHost(MemoryRegionHost &&rhs)
        : m_owned(rhs.m_owned), m_host(rhs.m_host), m_size(rhs.m_size),
          m_alignment(rhs.m_alignment), m_host_valid(rhs.m_host_valid),
          m_initialize(rhs.m_initialize), m_device_rank(rhs.m_device_rank),
          m_name(rhs.m_name)
    {
        rhs.m_owned        = true;
        rhs.m_host         = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid   = false;
        rhs.m_initialize   = true;
        rhs.m_device_rank  = 0;
        rhs.m_name         = "";
        rhs.m_memAllocType = eHostDevice;
    }

    /**
     * @brief Destructor method.
     *
     */
    virtual ~MemoryRegionHost()
    {
        if (m_host && m_owned)
        {
            if (m_memAllocType == ePinned)
            {
                hostFreePinned(m_host, m_alignment);
            }
            else
            {
                operator delete[](m_host, std::align_val_t(m_alignment));
            }
        }

        m_owned        = true;
        m_host         = nullptr;
        m_size         = 0;
        m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        m_host_valid   = false;
        m_initialize   = true;
        m_device_rank  = 0;
        m_name         = "";
        m_memAllocType = eHostDevice;
    }

protected:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionHost()                                       = delete;
    MemoryRegionHost(const MemoryRegionHost &rhs)            = delete;
    MemoryRegionHost &operator=(const MemoryRegionHost &rhs) = delete;

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryRegionHost to move from
     *
     * @return    - MemoryRegionHost<TData>&
     */
    MemoryRegionHost &operator=(MemoryRegionHost &&rhs)
    {
        m_owned        = rhs.m_owned;
        m_host         = rhs.m_host;
        m_size         = rhs.m_size;
        m_alignment    = rhs.m_alignment;
        m_host_valid   = rhs.m_host_valid;
        m_initialize   = rhs.m_initialize;
        m_device_rank  = rhs.m_device_rank;
        m_name         = rhs.m_name;
        m_memAllocType = rhs.m_memAllocType;

        rhs.m_owned        = true;
        rhs.m_host         = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid   = false;
        rhs.m_initialize   = true;
        rhs.m_device_rank  = 0;
        rhs.m_name         = "";
        rhs.m_memAllocType = eHostDevice;

        return *this;
    }

    /**
     * @brief Get ReadOnly pointer to the host memory
     *
     * @return - TData*
     *
     */
    virtual const TData *GetReadOnlyHostPtr()
    {
        if (m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetReadOnlyHostPtr - "
                     "attempt to access host memory (" +
                         m_name + ") without it being allocated.");
        }

        if (m_initialize)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetReadOnlyHostPtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        return m_host;
    }

    /**
     * @brief Get WriteOnly pointer to the host memory
     *
     * @return - TData*
     *
     */
    virtual TData *GetWriteOnlyHostPtr()
    {
        if (m_host == nullptr)
        {
            if (m_memAllocType == ePinned)
            {
                hostMallocPinned(m_host, m_size, m_alignment);
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
            else
            {
                m_host = static_cast<TData *>(::operator new[](
                    m_size * sizeof(TData), std::align_val_t(m_alignment)));
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
        }

        m_host_valid = true;
        m_initialize = false;

        return m_host;
    }

    /**
     * @brief Get ReadWrite pointer to the host memory
     *
     * @return - TData*
     *
     */
    virtual TData *GetReadWriteHostPtr()
    {
        if (m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetReadWriteHostPtr - "
                     "attempt to access host data (" +
                         m_name + ") without it being allocated.");
        }

        if (m_initialize)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetReadWriteHostPtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        m_host_valid = true;
        m_initialize = false;

        return m_host;
    }

    /**
     * @brief Get ReadOnly pointer to the Device memory
     *
     * @return - TData*
     *
     */
    virtual const TData *GetReadOnlyDevicePtr()
    {
        // Throw an error.
        NEKERROR(Nektar::ErrorUtil::efatal,
                 "MemoryRegionHost::GetReadOnlyDevicePtr - "
                 "attempt to access device data (" +
                     m_name + ") from host only MemoryRegion.");

        return nullptr;
    }

    /**
     * @brief Get WriteOnly pointer to the device memory
     *
     * @return - TData*
     *
     */
    virtual TData *GetWriteOnlyDevicePtr()
    {
        // Throw an error.
        NEKERROR(Nektar::ErrorUtil::efatal,
                 "MemoryRegionHost::GetWriteOnlyDevicePtr - "
                 "attempt to access device data (" +
                     m_name + ") from host only MemoryRegion.");

        return nullptr;
    }

    /**
     * @brief Get ReadWrite pointer to the device memory
     *
     * @return - TData*
     *
     */
    virtual TData *GetReadWriteDevicePtr()
    {
        // Throw an error.
        NEKERROR(Nektar::ErrorUtil::efatal,
                 "MemoryRegionHost::GetReadWriteDevicePtr - "
                 "attempt to access device data (" +
                     m_name + ") from host only MemoryRegion.");

        return nullptr;
    }

    /**
     * @brief Initialize the storage memory.
     *
     * @param val    - value to set
     * @param count  - number of values
     * @param offset - offset to m_host pointer
     *
     */
    virtual void Initialize(const TData val, const size_t count = 0,
                            const size_t offset = 0)
    {
        if (this->m_host == nullptr)
        {
            if (m_memAllocType == ePinned)
            {
                hostMallocPinned(m_host, m_size, m_alignment);
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
            else
            {
                m_host = static_cast<TData *>(::operator new[](
                    m_size * sizeof(TData), std::align_val_t(m_alignment)));
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
        }

        auto size = (count == 0) ? m_size : count;

        TData *dst = m_host + offset;

        // If the value is zero, memset is the most efficent.
        if constexpr (std::is_floating_point_v<TData> ||
                      std::is_integral_v<TData>)
        {
            if (val == TData(0))
            {
                std::memset(dst, 0, size * sizeof(TData));
            }
            // Nonzero value
            else
            {
                std::fill(dst, dst + size, val);
            }
        }
        else
        {
            std::fill(dst, dst + size, val);
        }

        m_host_valid = true;
        m_initialize = false;
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_host pointer
     */
    virtual void CopyFromHostPtr(const TData *src, const size_t size,
                                 const size_t offset = 0)
    {
        if (m_host == nullptr)
        {
            if (m_memAllocType == ePinned)
            {
                hostMallocPinned(m_host, m_size, m_alignment);
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
            else
            {
                m_host = static_cast<TData *>(::operator new[](
                    m_size * sizeof(TData), std::align_val_t(m_alignment)));
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
        }

        TData *dst = m_host + offset;

        std::memcpy(dst, src, size * sizeof(TData));

        m_host_valid = true;
        m_initialize = false;
    }

    // Member variables:
    bool m_owned = true; // Flag indicating if the host pointer is owned
                         // by the current object.
    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    bool m_host_valid = false; // Flag indicating that the host data is valid
    bool m_initialize = true;  // Flag indicating that the data needs
                               // to be initialize and is needed for
                               // host device transfers.
    size_t m_device_rank = 0;  // Index indicating device ID.
    std::string m_name{""};
    MemAllocType m_memAllocType{eHostDevice};
};
