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
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <cstring>
#include <iostream>
#include <new>

using vec_t = tinysimd::simd<double>;

template <typename TData> class MemoryRegion;

// If the host alignment is set to this macro value then no host memory
// will be allocated.
#define __EXECSPACE_MEMORY_REGION_ONLY__ 0

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
    MemoryRegionHost(std::string name, size_t size,
                     size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        m_size = size;

        createMemory(name, alignment);
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionHost
     *
     * @param rhs - MemoryRegionHost to move from
     */
    MemoryRegionHost(MemoryRegionHost &&rhs)
    {
        m_host       = rhs.m_host;
        m_size       = rhs.m_size;
        m_alignment  = rhs.m_alignment;
        m_host_valid = rhs.m_host_valid;
        m_initialize = rhs.m_initialize;
        m_name       = rhs.m_name;

        rhs.m_host       = nullptr;
        rhs.m_host_valid = false;
        rhs.m_initialize = true;
        rhs.m_name       = "";
    }

    /**
     * @brief Templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param array     - std::vector to copy from
     * @param alignment - memory alignment
     */
    template <typename TDataIn = TData>
    MemoryRegionHost(std::string name, std::vector<TDataIn> const &array,
                     size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        m_size = array.size();

        createMemory(name, alignment);

        if (alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            if constexpr (std::is_same<TDataIn, TData>::value)
            {
                std::memcpy(m_host, array.data(), m_size * sizeof(TData));
            }
            else
            {
                std::copy(array.begin(), array.end(), m_host);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array     - Nektar::Array to copy from
     * @param alignment - memory alignment
     */
    template <typename TDataIn = TData>
    MemoryRegionHost(std::string name,
                     Nektar::Array<Nektar::OneD, TDataIn> const &array,
                     size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        m_size = array.size();

        createMemory(name, alignment);

        if (alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            if constexpr (std::is_same<TDataIn, TData>::value)
            {
                std::memcpy(m_host, array.data(), m_size * sizeof(TData));
            }
            else
            {
                std::copy(array.begin(), array.end(), m_host);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array     - Nektar::Array to copy from
     * @param alignment - memory alignment
     */
    template <typename TDataIn = TData>
    MemoryRegionHost(
        std::string name,
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        m_size = 0;

        for (auto i = 0; i < array.size(); ++i)
        {
            m_size += array[i].size();
        }

        createMemory(name, alignment);

        if (alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            TData *hostPtr = m_host;

            for (auto i = 0; i < array.size(); ++i)
            {
                if constexpr (std::is_same<TDataIn, TData>::value)
                {
                    std::memcpy(hostPtr, array[i].data(),
                                array[i].size() * sizeof(TData));
                }
                else
                {
                    std::copy(array[i].begin(), array[i].end(), hostPtr);
                }

                hostPtr += array[i].size();
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

        m_host       = nullptr;
        m_host_valid = false;
        m_initialize = true;
        m_name       = "";
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

        m_host       = rhs.m_host;
        m_size       = rhs.m_size;
        m_alignment  = rhs.m_alignment;
        m_host_valid = rhs.m_host_valid;
        m_initialize = rhs.m_initialize;
        m_name       = rhs.m_name;

        rhs.m_host       = nullptr;
        rhs.m_host_valid = false;
        rhs.m_initialize = true;
        rhs.m_name       = "";

        return *this;
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
    virtual TData *GetHostPtr()
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
     * @brief Perform a host to device copy.
     *
     * @param force - copy regardless of status
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void HostToDevice([[maybe_unused]] bool force = false)
    {
    }

    /**
     * @brief Perform a device to host copy.
     *
     * @param force - copy regardless of status
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void DeviceToHost([[maybe_unused]] bool force = false)
    {
    }

    /**
     * @brief Copy data from one host to another host.
     *
     * @param rhs - MemoryRegionHost to copy from
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void HostToHost(MemoryRegionHost<TData> &rhs)
    {
        if (m_host == nullptr || rhs.m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "HostToHost::HostToHost - "
                     "attempt to access host memory (" +
                         m_name + ") without it being allocated.");
        }

        size_t size = m_size < rhs.m_size ? m_size : rhs.m_size;

        std::memcpy(m_host, rhs.m_host, size * sizeof(TData));
    }

    /**
     * @brief Copy data from one device to another device.
     *
     * @param rhs - MemoryRegionHost to copy from
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void DeviceToDevice(MemoryRegionHost<TData> &rhs)
    {
        HostToHost(rhs);
    }

    /**
     * @brief Initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     *
     * This is a virtual function so that subclasses can copy memory.
     */
    virtual void initialize(TData val, size_t count = 0)
    {
        if (m_host)
        {
            if (count == 0)
            {
                count = m_size;
            }

            // Special handling for the vec_t.
            if constexpr (std::is_same<vec_t, TData>::value)
            {
                std::fill(m_host, m_host + count, val);
            }
            else
            {
                // If the value is zero, memset is the most efficent.
                if (val == TData(0))
                {
                    std::memset(m_host, 0, count * sizeof(TData));
                }
                // Otherwuse use the fill function.
                else
                {
                    std::fill(m_host, m_host + count, val);
                }
            }

            m_host_valid = true;
            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     */
    template <typename TDataIn = TData>
    void copyVector(std::vector<TDataIn> const &array)
    {
        if (m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            if constexpr (std::is_same<TDataIn, TData>::value)
            {
                std::memcpy(m_host, array.get(), m_size * sizeof(TData));
            }
            else
            {
                std::copy(array.begin(), array.end(), m_host);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyArray(Nektar::Array<Nektar::OneD, TDataIn> const &array)
    {
        if (m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            if constexpr (std::is_same<TDataIn, TData>::value)
            {
                std::memcpy(m_host, array.get(), m_size * sizeof(TData));
            }
            else
            {
                std::copy(array.begin(), array.end(), m_host);
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyArray(
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array)
    {
        if (m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            TData *hostPtr = m_host;

            for (auto i = 0; i < array.size(); ++i)
            {
                if constexpr (std::is_same<TDataIn, TData>::value)
                {
                    std::memcpy(hostPtr, array[i].data(),
                                array[i].size() * sizeof(TData));
                }
                else
                {
                    std::copy(array[i].begin(), array[i].end(), hostPtr);
                }

                hostPtr += array[i].size();
            }

            m_host_valid = true;

            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyRaw(TData *dest, TDataIn *src, size_t size)
    {
        if (m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            if constexpr (std::is_same<TDataIn, TData>::value)
            {
                std::memcpy(dest, src, size * sizeof(TData));
            }
            else
            {
                for (size_t i = 0; i < size; ++i)
                {
                    dest[i] = src[i];
                }
            }

            m_host_valid = true;

            m_initialize = false;
        }
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
     * @brief Set all storage data as being valid.
     *
     * This is a virtual function so that subclasses can set values.
     */
    virtual void setValid()
    {
        if (m_host)
        {
            m_host_valid = true;
            m_initialize = false;
        }
    }

    /**
     * @brief Get the storage alignment
     *
     * @return - size_t
     */
    size_t GetAlignment() const
    {
        return m_alignment;
    }

    /**
     * @brief Get the name
     *
     * @return - std::string
     */
    std::string GetName() const
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

        if (m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            m_host = static_cast<TData *>(::operator new[](
                m_size * sizeof(TData), std::align_val_t(m_alignment)));

            m_host_valid = false;
        }

        m_initialize = true;
        m_name       = name;
    }

    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    bool m_host_valid = false; ///< Flag indicating the host data is valid
    bool m_initialize = true;  ///< Flag indicating the data is being initialize
                               ///< and is neeeded for host device transfers.
    std::string m_name{""};
};
