//////////////////////////////////////////////////////////////////////////////
//
// File: MemoryStorage.hpp
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
template <typename TData> class MemoryStorage
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
    MemoryStorage(const std::string name, const size_t size,
                  const size_t alignment, const unsigned int device_rank,
                  const MemAllocType &memAllocType)
    {
        m_owned        = true;
        m_name         = name;
        m_size         = size;
        m_alignment    = alignment;
        m_device_rank  = device_rank;
        m_memAllocType = memAllocType;

        m_host         = nullptr;
        m_device       = nullptr;
        m_host_valid   = false;
        m_device_valid = false;
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
    MemoryStorage(const std::string name, TData *h_src, const size_t size,
                  const size_t alignment, const unsigned int device_rank)
    {
        m_owned       = false;
        m_name        = name;
        m_size        = size;
        m_alignment   = alignment;
        m_device_rank = device_rank;

        m_host         = h_src;
        m_device       = nullptr;
        m_host_valid   = true;
        m_device_valid = false;
    }

    /**
     * @brief Constructor methods - move from another MemoryStorage
     *
     * @param rhs - MemoryStorage to move from
     */
    MemoryStorage(MemoryStorage &&rhs)
        : m_owned(rhs.m_owned), m_host(rhs.m_host), m_size(rhs.m_size),
          m_alignment(rhs.m_alignment), m_host_valid(rhs.m_host_valid),
          m_device_rank(rhs.m_device_rank), m_name(rhs.m_name),
          m_memAllocType(rhs.m_memAllocType)
    {
        rhs.m_owned        = true;
        rhs.m_host         = nullptr;
        rhs.m_device       = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid   = false;
        rhs.m_device_valid = false;
        rhs.m_device_rank  = 0;
        rhs.m_name         = "";
        rhs.m_memAllocType = eHostDevice;
    }

    /**
     * @brief Destructor method.
     *
     */
    ~MemoryStorage()
    {
        if (m_device)
        {
            deviceFree(m_device, m_size * sizeof(TData), m_alignment,
                       m_device_rank);
        }

        if (m_host && m_owned)
        {
            if (m_memAllocType == ePinned)
            {
                hostFreePinned(m_host, m_alignment);
            }
            else
            {
                hostFree(m_host, m_alignment);
            }
        }

        m_owned        = true;
        m_host         = nullptr;
        m_device       = nullptr;
        m_size         = 0;
        m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        m_host_valid   = false;
        m_device_valid = false;
        m_device_rank  = 0;
        m_name         = "";
        m_memAllocType = eHostDevice;
    }

protected:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryStorage()                                    = delete;
    MemoryStorage(const MemoryStorage &rhs)            = delete;
    MemoryStorage &operator=(const MemoryStorage &rhs) = delete;

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryStorage to move from
     *
     * @return    - MemoryStorage<TData>&
     */
    MemoryStorage &operator=(MemoryStorage &&rhs)
    {
        m_owned        = rhs.m_owned;
        m_host         = rhs.m_host;
        m_device       = rhs.m_device;
        m_size         = rhs.m_size;
        m_alignment    = rhs.m_alignment;
        m_host_valid   = rhs.m_host_valid;
        m_device_valid = rhs.m_device_valid;
        m_device_rank  = rhs.m_device_rank;
        m_name         = rhs.m_name;
        m_memAllocType = rhs.m_memAllocType;

        rhs.m_owned        = true;
        rhs.m_host         = nullptr;
        rhs.m_device       = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid   = false;
        rhs.m_device_valid = false;
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
    const TData *GetReadOnlyHostPtr()
    {
        if (!m_host && !m_device && m_size > 0)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadOnlyHostPtr - "
                     "attempt to access host memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadOnlyHostPtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        DeviceToHostCopy(); // Move to host if necessary

        m_host_valid = true;

        return m_host;
    }

    /**
     * @brief Get WriteOnly pointer to the host memory
     *
     * @return - TData*
     *
     */
    TData *GetWriteOnlyHostPtr()
    {
        if (!m_host)
        {
            if (m_memAllocType == ePinned)
            {
                hostMallocPinned(&m_host, m_size * sizeof(TData), m_alignment);
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
            else
            {
                hostMalloc(&m_host, m_size * sizeof(TData), m_alignment);
                std::memset((void *)m_host, 0, m_size * sizeof(TData));
            }
        }

        m_host_valid   = true;
        m_device_valid = false;

        return m_host;
    }

    /**
     * @brief Get ReadWrite pointer to the host memory
     *
     * @return - TData*
     *
     */
    TData *GetReadWriteHostPtr()
    {
        if (!m_host && !m_device && m_size > 0)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadWriteHostPtr - "
                     "attempt to access host data (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadWriteHostPtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        DeviceToHostCopy(); // Move to host if necessary

        m_host_valid   = true;
        m_device_valid = false;

        return m_host;
    }

    /**
     * @brief Get ReadOnly pointer to the Device memory
     *
     * @return - TData*
     *
     */
    const TData *GetReadOnlyDevicePtr()
    {
        if (!m_host && !m_device && m_size > 0)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadOnlyDevicePtr - "
                     "attempt to access device memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadOnlyDevicePtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        HostToDeviceCopy(); // Move to device if necessary

        m_device_valid = true;

        return m_device;
    }

    /**
     * @brief Get WriteOnly pointer to the device memory
     *
     * @return - TData*
     *
     */
    TData *GetWriteOnlyDevicePtr()
    {
        if (!m_device)
        {
            deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment,
                         m_device_rank);
            deviceMemset(m_device, 0, m_size * sizeof(TData), m_device_rank);
        }

        m_host_valid   = false;
        m_device_valid = true;

        return m_device;
    }

    /**
     * @brief Get ReadWrite pointer to the device memory
     *
     * @return - TData*
     *
     */
    TData *GetReadWriteDevicePtr()
    {
        if (!m_host && !m_device && m_size > 0)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadWriteDevicePtr - "
                     "attempt to access device memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryStorage::GetReadWriteDevicePtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        HostToDeviceCopy(); // Move to device if necessary

        m_host_valid   = false;
        m_device_valid = true;

        return m_device;
    }

    /**
     * @brief Initialize the storage memory.
     *
     * @param val    - value to set
     * @param count  - number of values
     * @param offset - offset to m_host pointer
     *
     */
    template <typename MemSpace>
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0)
    {
        if (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            if (!m_host)
            {
                if (m_memAllocType == ePinned)
                {
                    hostMallocPinned(&m_host, m_size * sizeof(TData),
                                     m_alignment);
                    std::memset((void *)m_host, 0, m_size * sizeof(TData));
                }
                else
                {
                    hostMalloc(&m_host, m_size * sizeof(TData), m_alignment);
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

            m_host_valid   = true;
            m_device_valid = false;
        }
        else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            if (!m_device)
            {
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment,
                             m_device_rank);
                deviceMemset(m_device, 0, m_size * sizeof(TData),
                             m_device_rank);
            }

            auto size = (count == 0) ? m_size : count;

            TData *dst = m_device + offset;

            // If the value is zero, memset is the most efficent.
            if constexpr (std::is_floating_point_v<TData> ||
                          std::is_integral_v<TData>)
            {
                if (val == TData(0))
                {
                    deviceMemset(dst, 0, size * sizeof(TData), m_device_rank);
                }
                // Nonzero value
                else
                {
                    deviceFill(dst, val, size, m_device_rank);
                }
            }
            else
            {
                deviceFill(dst, val, size, m_device_rank);
            }

            m_host_valid   = false;
            m_device_valid = true;
        }
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_host pointer
     */
    template <typename MemSpace>
    void CopyFromHostPtr(const TData *src, const size_t size,
                         const size_t offset = 0)
    {
        if (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            if (!m_host)
            {
                if (m_memAllocType == ePinned)
                {
                    hostMallocPinned(&m_host, m_size * sizeof(TData),
                                     m_alignment);
                    std::memset((void *)m_host, 0, m_size * sizeof(TData));
                }
                else
                {
                    hostMalloc(&m_host, m_size * sizeof(TData), m_alignment);
                    std::memset((void *)m_host, 0, m_size * sizeof(TData));
                }
            }

            TData *dst = m_host + offset;

            std::memcpy(dst, src, size * sizeof(TData));

            m_host_valid   = true;
            m_device_valid = false;
        }
        else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            if (!m_device)
            {
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment,
                             m_device_rank);
                deviceMemset(m_device, 0, m_size * sizeof(TData),
                             m_device_rank);
            }

            TData *dst = m_device + offset;

            deviceMemcpy<HostToDevice>(dst, src, size * sizeof(TData),
                                       m_device_rank);

            m_host_valid   = false;
            m_device_valid = true;
        }
    }

    /**
     * @brief Perform a host to device copy.
     *
     */
    void HostToDeviceCopy(void)
    {
        if (!m_device_valid)
        {
            if (!m_host && m_size > 0)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryStorage::HostToDeviceCopy - attempt to "
                         "transfer data from the host (" +
                             m_name +
                             ") without any "
                             "valid host memory allocated.");
            }

            if (!m_device)
            {
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment,
                             m_device_rank);
            }

            // Make sure the host data is valid. It might not be.
            if (m_host_valid)
            {
                deviceMemcpy<HostToDevice>(
                    m_device, m_host, m_size * sizeof(TData), m_device_rank);
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryStorage::HostToDeviceCopy - attempt to "
                         "transfer data (" +
                             m_name +
                             ") to the device without any "
                             "valid host data.");
            }
        }
    }

    /**
     * @brief Perform a device to host copy.
     *
     */
    void DeviceToHostCopy(void)
    {
        if (!m_host_valid)
        {
            if (!m_device && m_size > 0)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryStorage::DeviceToHostCopy - attempt to "
                         "transfer data from the device (" +
                             m_name +
                             ") without any "
                             "valid device memory allocated.");
            }

            if (!m_host)
            {
                if (m_memAllocType == ePinned)
                {
                    hostMallocPinned(&m_host, m_size * sizeof(TData),
                                     m_alignment);
                }
                else
                {
                    hostMalloc(&m_host, m_size * sizeof(TData), m_alignment);
                }
            }

            // Make sure the device data is valid. It might not be.
            if (m_device_valid)
            {
                deviceMemcpy<DeviceToHost>(
                    m_host, m_device, m_size * sizeof(TData), m_device_rank);
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryStorage::DeviceToHostCopy - attempt to "
                         "transfer data (" +
                             m_name +
                             ") to the host without any "
                             "valid device data.");
            }
        }
    }

    // Member variables:
    bool m_owned = true; // Flag indicating if the host pointer is owned
                         // by the current object.
    TData *m_host      = nullptr; /// < Host memory pointer
    TData *m_device    = nullptr; ///< Device memory pointer
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    bool m_host_valid    = false; // Flag indicating that the host data is valid
    bool m_device_valid  = false; ///< Flag indicating the device data is valid
    size_t m_device_rank = 0;     // Index indicating device ID.
    std::string m_name{""};
    MemAllocType m_memAllocType{eHostDevice};
};
