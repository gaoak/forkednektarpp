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

#include "Operators/Field/MemoryRegionHost.hpp"

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
     * @brief Constructor methods - create a new memory region.
     *
     * @param name         - name
     * @param size         - size of memory
     * @param alignment    - memory alignment
     * @param device_rank  - device (GPU) rank id
     * @param memAllocType - [eHostDevice, ePinned]
     */
    MemoryRegionDevice(const std::string name, const size_t size,
                       const size_t alignment, const unsigned int device_rank,
                       const MemAllocType &memAllocType)
        : MemoryRegionHost<TData>(name, size, alignment, device_rank,
                                  memAllocType)
    {
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
    MemoryRegionDevice(const std::string name, TData *h_src, const size_t size,
                       const size_t alignment, const unsigned int device_rank)
        : MemoryRegionHost<TData>(name, h_src, size, alignment, device_rank)
    {
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionDevice
     *
     * @param rhs - MemoryRegionDevice to move from
     */
    MemoryRegionDevice(MemoryRegionDevice<TData> &&rhs)
        : MemoryRegionHost<TData>(std::move(rhs)), m_device(rhs.m_device),
          m_device_valid(rhs.m_device_valid)
    {
        rhs.m_device       = nullptr;
        rhs.m_device_valid = false;
    }

    /**
     * @brief Constructor methods - move from a MemoryRegionHost while
     * creating device storage.
     *
     * @param host - MemoryRegionHost to move from
     */
    MemoryRegionDevice(MemoryRegionHost<TData> &&host)
        : MemoryRegionHost<TData>(std::move(host)), m_device(nullptr),
          m_device_valid(false)
    {
    }

    /**
     * @brief Destructor method.
     *
     */
    ~MemoryRegionDevice() override
    {
        if (this->m_device != nullptr)
        {
            deviceFree(this->m_device, this->m_size * sizeof(TData),
                       this->m_alignment, this->m_device_rank);
            this->m_device       = nullptr;
            this->m_device_valid = false;
        }
    }

protected:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionDevice()                                         = delete;
    MemoryRegionDevice(const MemoryRegionDevice &rhs)            = delete;
    MemoryRegionDevice &operator=(const MemoryRegionDevice &rhs) = delete;

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryRegionDevice to move from
     *
     * @return    - MemoryRegionDevice<TData>&
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
     * @brief Get ReadOnly pointer to the host memory
     *
     * @return - TData*
     */
    const TData *GetReadOnlyHostPtr() override
    {
        DeviceToHostCopy(); // Move to host if necessary

        return this->m_host;
    }

    /**
     * @brief Get WriteOnly pointer to the host memory
     *
     * @return - TData*
     */
    TData *GetWriteOnlyHostPtr() override
    {
        if (this->m_host == nullptr)
        {
            if (this->m_memAllocType == ePinned)
            {
                hostMallocPinned(this->m_host, this->m_size * sizeof(TData),
                                 this->m_alignment);
                std::memset((void *)this->m_host, 0,
                            this->m_size * sizeof(TData));
            }
            else
            {
                hostMalloc(this->m_host, this->m_size * sizeof(TData),
                           this->m_alignment);
                std::memset((void *)this->m_host, 0,
                            this->m_size * sizeof(TData));
            }
        }

        this->m_host_valid   = true;
        this->m_initialize   = false;
        this->m_device_valid = false;

        return this->m_host;
    }

    /**
     * @brief Get ReadWrite pointer to the host memory
     *
     * @return - TData*
     *
     */
    TData *GetReadWriteHostPtr() override
    {
        DeviceToHostCopy(); // Move to host if necessary
        this->m_device_valid = false;

        return this->m_host;
    }

    /**
     * @brief Get ReadOnly pointer to the Device memory
     *
     * @return - TData*
     */
    const TData *GetReadOnlyDevicePtr() override
    {
        HostToDeviceCopy(); // Move to device if necessary

        return this->m_device;
    }

    /**
     * @brief Get WriteOnly pointer to the device memory
     *
     * @return - TData*
     */
    TData *GetWriteOnlyDevicePtr() override
    {
        if (this->m_device == nullptr)
        {
            deviceMalloc(this->m_device, this->m_size * sizeof(TData),
                         this->m_alignment, this->m_device_rank);
            deviceMemset(this->m_device, 0, this->m_size * sizeof(TData),
                         this->m_device_rank);
        }

        this->m_device_valid = true;
        this->m_initialize   = false;
        this->m_host_valid   = false;

        return this->m_device;
    }

    /**
     * @brief Get ReadWrite pointer to the device memory
     *
     * @return - TData*
     */
    TData *GetReadWriteDevicePtr() override
    {
        HostToDeviceCopy(); // Move to device if necessary
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
                    const size_t offset = 0) override
    {
        if (this->m_device == nullptr)
        {
            deviceMalloc(this->m_device, this->m_size * sizeof(TData),
                         this->m_alignment, this->m_device_rank);
            deviceMemset(this->m_device, 0, this->m_size * sizeof(TData),
                         this->m_device_rank);
        }

        auto size = (count == 0) ? this->m_size : count;

        TData *dst = this->m_device + offset;

        // If the value is zero, memset is the most efficent.
        if constexpr (std::is_floating_point_v<TData> ||
                      std::is_integral_v<TData>)
        {
            if (val == TData(0))
            {
                deviceMemset(dst, 0, size * sizeof(TData), this->m_device_rank);
            }
            // Nonzero value
            else
            {
                deviceFill(dst, val, size, this->m_device_rank);
            }
        }
        else
        {
            deviceFill(dst, val, size, this->m_device_rank);
        }

        this->m_host_valid   = false;
        this->m_device_valid = true;
        this->m_initialize   = false;
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_device pointer
     */
    void CopyFromHostPtr(const TData *src, const size_t size,
                         const size_t offset = 0) override
    {
        if (this->m_device == nullptr)
        {
            deviceMalloc(this->m_device, this->m_size * sizeof(TData),
                         this->m_alignment, this->m_device_rank);
            deviceMemset(this->m_device, 0, this->m_size * sizeof(TData),
                         this->m_device_rank);
        }

        TData *dst = this->m_device + offset;

        deviceMemcpy<HostToDevice>(dst, src, size * sizeof(TData),
                                   this->m_device_rank);

        this->m_host_valid   = false;
        this->m_device_valid = true;
        this->m_initialize   = false;
    }

    /**
     * @brief Perform a host to device copy.
     *
     */
    void HostToDeviceCopy(void)
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

            if (this->m_device == nullptr)
            {
                deviceMalloc(this->m_device, this->m_size * sizeof(TData),
                             this->m_alignment, this->m_device_rank);
            }

            // Make sure the host data is valid. It might not be.
            if (this->m_host_valid)
            {
                deviceMemcpy<HostToDevice>(this->m_device, this->m_host,
                                           this->m_size * sizeof(TData),
                                           this->m_device_rank);
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
    void DeviceToHostCopy(void)
    {
        if (!this->m_host_valid)
        {
            if (this->m_host == nullptr)
            {
                if (this->m_memAllocType == ePinned)
                {
                    hostMallocPinned(this->m_host, this->m_size * sizeof(TData),
                                     this->m_alignment);
                }
                else
                {
                    hostMalloc(this->m_host, this->m_size * sizeof(TData),
                               this->m_alignment);
                }
            }

            // Make sure the device data is valid. It might not be.
            if (this->m_device_valid)
            {
                deviceMemcpy<DeviceToHost>(this->m_host, this->m_device,
                                           this->m_size * sizeof(TData),
                                           this->m_device_rank);
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
