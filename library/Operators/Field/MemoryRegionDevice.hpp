///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryRegionDevice.hpp
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

#include "DeviceMemory.hpp"

#include "Operators/Field/MemoryRegionHost.hpp"

using namespace Nektar;

// If this macro is set when calling the device side creation and copy
// methods will put the data on the host and the device. If not set,
// the device side creation and copy methods will put the data on the
// device only.

template <typename TData> class MemoryRegion;

/**
 * @brief Memory backend for device devices
 * @tparam TData Floating point datatype
 *
 * MemoryRegionDevice represents and manages the memory stored on a device
 * device.  This class also manages access to the host part of the
 * memory by inheriting from MemoryRegionHost.
 */
template <typename TData>
class MemoryRegionDevice : public MemoryRegionHost<TData>
{
    friend class MemoryRegion<TData>;

public:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionDevice()                                         = delete;
    MemoryRegionDevice(const MemoryRegionDevice &rhs)            = delete;
    MemoryRegionDevice &operator=(const MemoryRegionDevice &rhs) = delete;

    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param size      - size of memory
     * @param alignment - memory alignment
     */
    MemoryRegionDevice(const std::string name, const size_t size,
                       const size_t alignment, const size_t device_rank,
                       const bool device_only)
        : MemoryRegionHost<TData>(name, size, alignment, device_rank,
                                  device_only)
    {
        CreateMemory();
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionDevice
     *
     * @param rhs - MemoryRegionHost to move from
     */
    MemoryRegionDevice(MemoryRegionDevice<TData> &&rhs)
        : MemoryRegionHost<TData>(std::move(rhs))
    {
        m_device       = rhs.m_device;
        m_device_valid = rhs.m_device_valid;

        rhs.m_device       = nullptr;
        rhs.m_device_valid = false;
    }

    /**
     * @brief Constructor methods - move from a MemoryRegionHost while
     * creating device storage.
     *
     * @param host - MemoryRegionHost to move from
     */
    MemoryRegionDevice<TData>(MemoryRegionHost<TData> &&host)
        : MemoryRegionHost<TData>(std::move(host))
    {
        CreateMemory();
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
    MemoryRegionDevice(const std::string name, const TDataIn *src,
                       const size_t size, const size_t alignment,
                       const size_t device_rank, const bool device_only)
        : MemoryRegionHost<TData>(name, size, alignment, device_rank,
                                  device_only)
    {
        CreateMemory();

        if constexpr (std::is_same_v<TDataIn, TData>)
        {
            deviceMemcpy<HostToDevice>(this->m_device, src, this->m_size,
                                       this->m_device_rank);
        }
        else
        {
            std::vector<TData> tmp(this->m_size);
            std::copy(src, src + size, tmp.begin());
            deviceMemcpy<HostToDevice>(this->m_device, tmp.data(), this->m_size,
                                       this->m_device_rank);
        }

        this->m_device_valid = true;
        this->m_initialize   = false;
    }

    /**
     * @brief Destructor method.
     *
     */
    ~MemoryRegionDevice() override
    {
        if (this->m_device != nullptr)
        {
            deviceFree(this->m_device, this->m_device_rank);
            this->m_device       = nullptr;
            this->m_device_valid = false;
        }
    }

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryRegionHost to move from
     *
     * @return    - MemoryRegion<TData>&
     */
    void operator=(MemoryRegionDevice &&rhs)
    {
        MemoryRegionHost<TData>::operator=(std::move(rhs));
        m_device       = rhs.m_device;
        m_device_valid = rhs.m_device_valid;

        rhs.m_device       = nullptr;
        rhs.m_device_valid = false;
    }

    /**
     * @brief osstream operator.
     *
     * @param rhs - MemoryRegion to stream
     *
     * @return    - stream
     */
    friend auto operator<<(std::ostream &os, const MemoryRegionDevice &mr)
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

        msg << " device_valid: " << mr.m_device_valid << " ";

        return os << msg.str();
    }

protected:
    /**
     * @brief Create device memory
     *
     */
    void CreateMemory()
    {
        deviceMalloc(this->m_device, this->m_size, this->m_device_rank);
    }

    /**
     * @brief Get the const pointer to the host memory - assumes the data
     * will not be modified.
     *
     * @return - TData*
     */
    const TData *GetHostConstPtr() override
    {
        if (this->m_initialize)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionDevice::GetHostConstPtr - attempt to get a "
                     "const host pointer (" +
                         this->m_name +
                         ") before the data is "
                         "initialized.");
        }

        DeviceToHostCopy(); // Move to host if necessary

        return this->m_host;
    }

    /**
     * @brief Get the pointer to the host memory - assumes the data
     * will be modified.
     *
     * @return - TData*
     */
    TData *GetHostPtr(const bool write_only) override
    {
        if (write_only)
        {
            this->m_host_valid = true;
            this->m_initialize = false;
        }
        else
        {
            DeviceToHostCopy(); // Move to host if necessary
        }

        this->m_device_valid = false;

        return this->m_host;
    }

    /**
     * @brief Get the const pointer to the Device memory - assumes the data
     * will not be modified.
     *
     * @return - TData*
     */
    const TData *GetDeviceConstPtr()
    {
        if (this->m_initialize)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionDevice::GetDeviceConstPtr - attempt to get a "
                     "const device pointer (" +
                         this->m_name +
                         ") before the data is "
                         "initialized.");
        }

        HostToDeviceCopy(); // Move to device if necessary

        return this->m_device;
    }

    /**
     * @brief Get the pointer to the device memory - assumes the data
     * will be modified.
     *
     * @return - TData*
     */
    TData *GetDevicePtr(const bool write_only = false)
    {
        if (write_only)
        {
            this->m_device_valid = true;
            this->m_initialize   = false;
        }
        else
        {
            HostToDeviceCopy(); // Move to device if necessary
        }

        this->m_host_valid = false;

        return this->m_device;
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val    - value to set
     * @param count  - number of values
     * @param offset - offset to m_device pointer
     */
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0)
    {
        this->m_host_valid   = false;
        this->m_initialize   = false;
        this->m_device_valid = true;

        auto size = (count == 0) ? this->m_size : count;

        TData *dst = this->m_device + offset;

        // If the value is zero, memset is the most efficent.
        if (val == TData(0))
        {
            deviceMemset(dst, 0, size, this->m_device_rank);
        }
        // Nonzero value
        else
        {
            deviceFill(dst, val, size, this->m_device_rank);
        }
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_device pointer
     */
    template <typename MemCopy, typename TDataIn>
    void CopySRC(const TDataIn *src, const size_t size, const size_t offset = 0)
    {
        if constexpr (!std::is_same_v<TDataIn, TData>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionDevice::CopySRC - non-homogeneous datatype "
                     "not supported on MemoryRegionDevice");
        }

        if constexpr (std::is_same_v<MemCopy, HostToHost>)
        {
            MemoryRegionHost<TData>::template CopySRC<MemCopy, TData>(src, size,
                                                                      offset);
        }
        else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
        {
            TData *dst = this->m_host + offset;
            deviceMemcpy<DeviceToHost>(dst, src, size, this->m_device_rank);
            this->m_device_valid = false;
            this->m_host_valid   = true;
            this->m_initialize   = false;
        }
        else if constexpr (std::is_same_v<MemCopy, HostToDevice>)
        {
            TData *dst = this->m_device + offset;
            deviceMemcpy<HostToDevice>(dst, src, size, this->m_device_rank);
            this->m_device_valid = true;
            this->m_host_valid   = false;
            this->m_initialize   = false;
        }
        else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
        {
            TData *dst = this->m_device + offset;
            deviceMemcpy<DeviceToDevice>(dst, src, size, this->m_device_rank);
            this->m_device_valid = true;
            this->m_host_valid   = false;
            this->m_initialize   = false;
        }
    }

    /**
     * @brief Perform a host to device copy.
     *
     */
    void HostToDeviceCopy(void) override
    {
        if (!this->m_device_valid)
        {
            if (this->m_host == nullptr)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegionDevice::HostToDeviceCopy - attempt to "
                         "transfer data from the host (" +
                             this->m_name +
                             ") without any "
                             "valid host memory allocated.");
            }

            // Make sure the host data is valid. It might not be.
            if (this->m_host_valid)
            {
                deviceMemcpy<HostToDevice>(this->m_device, this->m_host,
                                           this->m_size, this->m_device_rank);
                this->m_device_valid = true;
            }
            // No data on the host so assume the device data is being
            // initialized.
            else if (this->m_initialize)
            {
                this->m_device_valid = true;
                this->m_initialize   = false;
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegionDevice::HostToDeviceCopy - attempt to "
                         "transfer data (" +
                             this->m_name +
                             ") to the device without any "
                             "valid host data.");
            }
        }
    }

    /**
     * @brief Perform a device to host copy.
     *
     */
    void DeviceToHostCopy(void) override
    {
        if (!this->m_host_valid)
        {
            if (this->m_host == nullptr)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegionDevice::DeviceToHostCopy - attempt to "
                         "transfer data (" +
                             this->m_name +
                             ") to the host without any "
                             "valid host memory allocated.");
            }

            // Make sure the device data is valid. It might not be.
            if (this->m_device_valid)
            {
                deviceMemcpy<DeviceToHost>(this->m_host, this->m_device,
                                           this->m_size, this->m_device_rank);
                this->m_host_valid = true;
            }
            // No data on the device so assume the host data is being
            // initialized.
            else if (this->m_initialize)
            {
                this->m_host_valid = true;
                this->m_initialize = false;
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegionDevice::DeviceToHostCopy - attempt to "
                         "transfer data (" +
                             this->m_name +
                             ") to the host without any "
                             "valid device data.");
            }
        }
    }

    // Member variables:
    TData *m_device     = nullptr; ///< Device memory pointer
    bool m_device_valid = false;   ///< Flag indicating the device data is valid
};
