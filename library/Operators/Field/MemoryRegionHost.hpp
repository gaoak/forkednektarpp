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

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <cstring>
#include <iostream>
#include <new>
#include <sstream>

template <typename TData> class MemoryRegion;

/**
 * @brief Stores underlying data for a Field on the host.
 *
 * It acts as a holder for a contiguous block of memory, allocated on the
 * host system.
 *
 * This class also acts as a base class for device-aware builds.
 */
template <typename TData> class MemoryRegionHost
{
    friend class MemoryRegion<TData>;

public:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionHost()                                       = delete;
    MemoryRegionHost(const MemoryRegionHost &rhs)            = delete;
    MemoryRegionHost &operator=(const MemoryRegionHost &rhs) = delete;

    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param size      - size of memory
     * @param alignment - memory alignment
     */
    MemoryRegionHost(std::string name, size_t size, size_t alignment,
                     bool device_only)
    {
        m_size        = size;
        m_device_only = device_only;

        createMemory(name, alignment);
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionHost
     *
     * @param rhs - MemoryRegionHost to move from
     */
    MemoryRegionHost(MemoryRegionHost &&rhs)
    {
        m_host        = rhs.m_host;
        m_size        = rhs.m_size;
        m_alignment   = rhs.m_alignment;
        m_host_valid  = rhs.m_host_valid;
        m_initialize  = rhs.m_initialize;
        m_device_only = rhs.m_device_only;
        m_name        = rhs.m_name;

        rhs.m_host        = nullptr;
        rhs.m_size        = 0;
        rhs.m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid  = false;
        rhs.m_initialize  = true;
        rhs.m_device_only = false;
        rhs.m_name        = "";
    }

    /**
     * @brief Templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        src pointer
     *
     * @param src       - pointer data type TDataIn to copy from
     * @param size      - number of element of type TDataIn to copy
     * @param alignment - memory alignment
     */
    template <typename TDataIn>
    MemoryRegionHost(std::string name, const TDataIn *src, const size_t size,
                     size_t alignment, bool device_only)
    {
        m_size        = size;
        m_device_only = device_only;

        createMemory(name, alignment);

        if (!m_device_only)
        {
            if constexpr (std::is_same_v<TDataIn, TData>)
            {
                std::memcpy(m_host, src, m_size * sizeof(TData));
            }
            else
            {
                std::copy(src, src + size, m_host);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Destructor method.
     *
     */
    virtual ~MemoryRegionHost()
    {
        if (m_host)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
        }

        m_host        = nullptr;
        m_size        = 0;
        m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        m_host_valid  = false;
        m_initialize  = true;
        m_device_only = false;
        m_name        = "";
    }

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryRegionHost to move from
     *
     * @return    - MemoryRegionHost<TData>&
     */
    MemoryRegionHost &operator=(MemoryRegionHost &&rhs)
    {
        if (m_host)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
        }

        m_host        = rhs.m_host;
        m_size        = rhs.m_size;
        m_alignment   = rhs.m_alignment;
        m_host_valid  = rhs.m_host_valid;
        m_initialize  = rhs.m_initialize;
        m_device_only = rhs.m_device_only;
        m_name        = rhs.m_name;

        rhs.m_host        = nullptr;
        rhs.m_size        = 0;
        rhs.m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid  = false;
        rhs.m_initialize  = true;
        rhs.m_device_only = false;
        rhs.m_name        = "";

        return *this;
    }

    /**
     * @brief osstream operator.
     *
     * @param rhs - MemoryRegion to stream
     *
     * @return    - stream
     */
    friend auto operator<<(std::ostream &os, MemoryRegionHost const &mr)
        -> std::ostream &
    {
        std::stringstream msg;
        msg << "Name: '" << mr.m_name << "' size: " << mr.m_size << " "
            << " initialize: " << mr.m_initialize << " ";

        if (mr.m_device_only)
        {
            msg << "MemoryRegion is only allocated on device! ";
        }
        else
        {
            msg << " host_valid: " << mr.m_host_valid << " ";
        }

        return os << msg.str();
    }

    /**
     * @brief Get the storage size
     *
     * @return - size_t
     */
    size_t size() const
    {
        return m_size;
    }

    /**
     * @brief Get the storage alignment
     *
     * @return - size_t
     */
    size_t getAlignment() const
    {
        return m_alignment;
    }

    /**
     * @brief Get the name
     *
     * @return - std::string
     */
    std::string getName() const
    {
        return m_name;
    }

protected:
    /**
     * @brief Create hostmemory
     *
     * @param alignment - memory alignment
     */
    void createMemory(std::string name, size_t alignment)
    {
        // C++17 aligned new
        m_alignment = alignment;

        if (!m_device_only)
        {
            m_host = static_cast<TData *>(::operator new[](
                m_size * sizeof(TData), std::align_val_t(m_alignment)));

            m_host_valid = false;
        }

        m_initialize = true;
        m_name       = name;
    }

    /**
     * @brief Get the const pointer to the host memory - assumes the data
     *        will not be modified.
     *
     * @return - TData*
     *
     * This is a virtual function so that subclasses can get const host memory
     */
    virtual const TData *GetHostConstPtr()
    {
        if (m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetHostConstPtr - "
                     "attempt to access host memory (" +
                         m_name + ") without it being allocated.");
        }

        if (m_initialize)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetHostConstPtr - "
                     "attempt to get a const host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        return m_host;
    }

    /**
     * @brief Get the pointer to the host memory - assumes the data
     *        will be modified.
     *
     * @return - TData*
     *
     * This is a virtual function so that subclasses can get host memory
     */
    virtual TData *GetHostPtr([[maybe_unused]] bool write_only = false)
    {
        if (m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetHostPtr - "
                     "attempt to access host data (" +
                         m_name + ") without it being allocated.");
        }

        m_host_valid = true;
        m_initialize = false;

        return m_host;
    }

    /**
     * @brief Initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     *
     */
    void initialize(TData val, size_t count = 0, size_t offset = 0)
    {
        if (m_host)
        {
            if (count == 0)
            {
                count = m_size;
            }

            TData *dst = m_host + offset;

            // Special handling for the simd_t.
            using simd_t = tinysimd::simd<TData>;

            if constexpr (std::is_same_v<simd_t, TData>)
            {
                std::fill(dst, dst + count, val);
            }
            else
            {
                // If the value is zero, memset is the most efficent.
                if (val == TData(0))
                {
                    std::memset(dst, 0, count * sizeof(TData));
                }
                // Otherwuse use the fill function.
                else
                {
                    std::fill(dst, dst + count, val);
                }
            }

            m_host_valid = true;
            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method.
     *
     * @param src       - pointer data type TDataIn to copy from
     * @param size      - number of element of type TDataIn to copy
     * @param offset    - offset to m_host pointer
     */
    template <typename TDataIn>
    void copyFrom(const TDataIn *src, const size_t size,
                  const size_t offset = 0)
    {
        if (!m_device_only)
        {
            TData *dst = m_host + offset;

            if constexpr (std::is_same_v<TDataIn, TData>)
            {
                std::memcpy(dst, src, size * sizeof(TData));
            }
            else
            {
                std::copy(src, src + size, dst);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Perform a host to device copy.
     *
     * @param force - copy regardless of status
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void HostToDeviceCopy()
    {
    }

    /**
     * @brief Perform a device to host copy.
     *
     * @param force - copy regardless of status
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void DeviceToHostCopy()
    {
    }

    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    bool m_host_valid = false;  // Flag indicating that the host data is valid
    bool m_initialize = true;   // Flag indicating that the data needs
                                // to be initialize and is needed for
                                // host device transfers.
    bool m_device_only = false; // Flag indicating that the data is only
                                // initialized on the device.
    std::string m_name{""};
};
