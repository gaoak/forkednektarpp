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

#include "Operators/Common/Spaces.hpp"
#include "Operators/Field/MemoryRegionHost.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#include <utility>

#ifdef NEKTAR_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/fill.h>
#elif defined(NEKTAR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/SYCLQueue.hpp"
#elif defined(NEKTAR_ENABLE_KOKKOS)
#endif

namespace Nektar
{
// MemoryCopy
struct HostToHost
{
};
struct DeviceToHost
{
};
struct HostToDevice
{
};
struct DeviceToDevice
{
};

// MemoryWrite
struct HostOnly
{
};
struct DeviceOnly
{
};
struct HostDevice
{
};
} // namespace Nektar

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
    MemoryRegionDevice(std::string name, size_t size, size_t alignment,
                       bool device_only)
        : MemoryRegionHost<TData>(name, size, alignment, device_only)
    {
        createMemory();
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
        createMemory();
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
    MemoryRegionDevice(std::string name, [[maybe_unused]] const TDataIn *src,
                       const size_t size, size_t alignment, bool device_only)
        : MemoryRegionHost<TData>(name, size, alignment, device_only)
    {
        createMemory();

        if constexpr (std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, src, this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, src, this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
            SYCLQueue::GetInstance()
                .memcpy(m_device, src, this->m_size * sizeof(TData))
                .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // Create unmanage Kokkos views from the raw pointers.
            TData *v_src = const_cast<TData *>(src);
            Kokkos::View<TData *, Kokkos::HostSpace> hostView(v_src,
                                                              this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, hostView);
#endif
        }
        else // if constexpr (!std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            std::vector<TData> tmp(this->m_size);

            std::copy(src, src + size, tmp.begin());

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            std::copy(src, src + size, tmp.begin());

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
            std::vector<TData> tmp(this->m_size);

            std::copy(src, src + size, tmp.begin());

            SYCLQueue::GetInstance()
                .memcpy(m_device, tmp.data(), this->m_size * sizeof(TData))
                .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // Create unmanage Kokkos views from the raw pointers.
            TData *v_src = const_cast<TData *>(src);
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(v_src,
                                                                this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, hostView);
#endif
        }

        m_device_valid = true;

        this->m_initialize = false;
    }

    /**
     * @brief Destructor method.
     *
     */
    ~MemoryRegionDevice() override
    {
        if (m_device != nullptr)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaFree(m_device);
#elif defined(NEKTAR_ENABLE_HIP)
            hipFree(m_device);
#elif defined(NEKTAR_ENABLE_SYCL)
            sycl::free(m_device, SYCLQueue::GetInstance());
#elif defined(NEKTAR_ENABLE_KOKKOS)
            Kokkos::kokkos_free(m_device);
#endif
            m_device       = nullptr;
            m_device_valid = false;
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
    friend auto operator<<(std::ostream &os, MemoryRegionDevice const &mr)
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
     * @brief Create hostmemory
     *
     */
    void createMemory()
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMalloc((void **)&m_device, this->m_size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_HIP)
        hipMalloc((void **)&m_device, this->m_size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_SYCL)
        m_device =
            sycl::malloc_device<TData>(this->m_size, SYCLQueue::GetInstance());
#elif defined(NEKTAR_ENABLE_KOKKOS)
        m_device = (TData *)
            Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
                this->m_name, this->m_size * sizeof(TData));
#endif
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
                     "attempt to get a const host pointer (" + this->m_name +
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
    TData *GetHostPtr(bool write_only) override
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

        m_device_valid = false;

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
                     "attempt to get a const device pointer (" + this->m_name +
                         ") before the data is "
                         "initialized.");
        }

        HostToDeviceCopy(); // Move to device if necessary

        return m_device;
    }

    /**
     * @brief Get the pointer to the device memory - assumes the data
     * will be modified.
     *
     * @return - TData*
     */
    TData *GetDevicePtr(bool write_only = false)
    {
        if (write_only)
        {
            m_device_valid     = true;
            this->m_initialize = false;
        }
        else
        {
            HostToDeviceCopy(); // Move to device if necessary
        }

        this->m_host_valid = false;

        return m_device;
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     */
    template <typename MemWrite = DeviceOnly>
    void initialize(TData val, size_t count = 0, size_t offset = 0)
    {
        if constexpr (std::is_same<MemWrite, HostOnly>::value ||
                      std::is_same<MemWrite, HostDevice>::value)
        {
            MemoryRegionHost<TData>::initialize(val, count, offset);
            if constexpr (std::is_same<MemWrite, HostOnly>::value)
            {
                m_device_valid = false;
            }
        }

        if constexpr (std::is_same<MemWrite, DeviceOnly>::value ||
                      std::is_same<MemWrite, HostDevice>::value)
        {
            if constexpr (std::is_same<MemWrite, DeviceOnly>::value)
            {
                this->m_host_valid = false;
                this->m_initialize = false;

                if (count == 0)
                {
                    count = this->m_size;
                }
            }

            m_device_valid = true;

            [[maybe_unused]] TData *dst = m_device + offset;

            // If the value is zero, memset is the most efficent.
            if (val == TData(0))
            {
#if defined(NEKTAR_ENABLE_CUDA)
                cudaMemset(dst, 0, count * sizeof(TData));
#elif defined(NEKTAR_ENABLE_HIP)
                hipMemset(dst, 0, count * sizeof(TData));
#elif defined(NEKTAR_ENABLE_SYCL)
                SYCLQueue::GetInstance()
                    .memset(dst, 0, count * sizeof(TData))
                    .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // Create an unmanage Kokkos view from the raw pointer.
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    dst, count);

                // Deep copy the val to the device view.
                Kokkos::deep_copy(deviceView, val);
#endif
            }
            // Nonzero value
            else
            {
#if defined(NEKTAR_ENABLE_CUDA)
                thrust::fill(dst, dst + count, val);
#elif defined(NEKTAR_ENABLE_HIP)
                hipLaunchKernelGGL(fill_, blocks, threads, 0, 0, count, dst,
                                   val); // TODO: implement fill_ kernel
#elif defined(NEKTAR_ENABLE_SYCL)
                SYCLQueue::GetInstance().fill(dst, val, count).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // Create an unmanage Kokkos view from the raw pointer.
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    dst, count);

                // Deep copy the val to the device view.
                Kokkos::deep_copy(deviceView, val);
#endif
            }
        }
    }

    /**
     * @brief Templated copy method.
     *
     * @param src       - pointer data type TDataIn to copy from
     * @param size      - number of element of type TDataIn to copy
     * @param offset    - offset to m_device pointer
     */
    template <typename TDataIn, typename MemCopy>
    void copyFrom(const TDataIn *src, const size_t size,
                  const size_t offset = 0)
    {
        if constexpr (!std::is_same<TDataIn, TData>::value)
        {
            NEKERROR(
                Nektar::ErrorUtil::efatal,
                "non-homogeneous datatype not supported on MemoryRegionDevice");
        }

        if constexpr (std::is_same<MemCopy, HostToHost>::value)
        {
            MemoryRegionHost<TData>::template copyFrom<TData>(src, size,
                                                              offset);
        }
        else if constexpr (std::is_same<MemCopy, DeviceToHost>::value)
        {
            TData *dst = this->m_host + offset;

#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_SYCL)
            SYCLQueue::GetInstance()
                .memcpy(dst, src, size * sizeof(TData))
                .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // Create unmanage Kokkos views from the raw pointers.
            TData *v_src = const_cast<TData *>(src);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> srcView(v_src,
                                                                         size);
            Kokkos::View<TData *, Kokkos::HostSpace> hostView(dst, size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(hostView, srcView);
#endif
            m_device_valid = false;

            this->m_host_valid = true;
            this->m_initialize = false;
        }
        else if constexpr (std::is_same<MemCopy, HostToDevice>::value)
        {
            TData *dst = m_device + offset;

#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
            SYCLQueue::GetInstance()
                .memcpy(dst, src, size * sizeof(TData))
                .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // Create unmanage Kokkos views from the raw pointers.
            TData *v_src = const_cast<TData *>(src);
            Kokkos::View<TData *, Kokkos::HostSpace> srcView(v_src, size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                dst, size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, srcView);
#endif

            m_device_valid = true;

            this->m_host_valid = false;
            this->m_initialize = false;
        }
        else if constexpr (std::is_same<MemCopy, DeviceToDevice>::value)
        {
            TData *dst = m_device + offset;

#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(dst, src, size * sizeof(TData),
                       cudaMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
            SYCLQueue::GetInstance()
                .memcpy(dst, src, size * sizeof(TData))
                .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // Create unmanage Kokkos views from the raw pointers.
            TData *v_src = const_cast<TData *>(src);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> srcView(v_src,
                                                                         size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                dst, size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, srcView);
#endif

            m_device_valid = true;

            this->m_host_valid = false;
            this->m_initialize = false;
        }
    }

    /**
     * @brief Perform a host to device copy.
     *
     * @param force - copy regardless of status
     */
    void HostToDeviceCopy(bool force = false) override
    {
        // Because the host pointer is used directly the device data may
        // be marked as valid as such the force can be used to assure
        // the copy occurs regardles.
        if (!this->m_device_only && (force || !m_device_valid))
        {
            if (this->m_host == nullptr)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "attempt to transfer data from the host (" +
                             this->m_name +
                             ") without any "
                             "valid host memory allocated.");
            }

            // Make sure the host data is valid. It might not be.
            if (force || this->m_host_valid)
            {
#if defined(NEKTAR_ENABLE_CUDA)
                cudaMemcpy(m_device, this->m_host, this->m_size * sizeof(TData),
                           cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
                hipMemcpy(m_device, this->m_host, this->m_size * sizeof(TData),
                          hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
                SYCLQueue::GetInstance()
                    .memcpy(m_device, this->m_host,
                            this->m_size * sizeof(TData))
                    .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TData *, Kokkos::HostSpace> hostView(this->m_host,
                                                                  this->m_size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    m_device, this->m_size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(deviceView, hostView);
#endif
                m_device_valid = true;
            }
            // No data on the host so assume the device data is being
            // initialized.
            else if (this->m_initialize)
            {
                m_device_valid     = true;
                this->m_initialize = false;
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "attempt to transfer data (" + this->m_name +
                             ") to the device without any "
                             "valid host data.");
            }
        }
    }

    /**
     * @brief Perform a device to host copy.
     *
     * @param force - copy regardless of status
     */
    void DeviceToHostCopy(bool force = false) override
    {
        // Because the device pointer is used directly the host data may
        // be marked as valid as such the force can be used to assure
        // the copy occurs regardles.
        if (!this->m_device_only && (force || !this->m_host_valid))
        {
            if (this->m_host == nullptr)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "attempt to transfer data (" + this->m_name +
                             ") to the host without any "
                             "valid host memory allocated.");
            }

            // Make sure the device data is valid. It might not be.
            if (force || m_device_valid)
            {
#if defined(NEKTAR_ENABLE_CUDA)
                cudaMemcpy(this->m_host, m_device, this->m_size * sizeof(TData),
                           cudaMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_HIP)
                hipMemcpy(this->m_host, m_device, this->m_size * sizeof(TData),
                          hipMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_SYCL)
                SYCLQueue::GetInstance()
                    .memcpy(this->m_host, m_device,
                            this->m_size * sizeof(TData))
                    .wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TData *, Kokkos::HostSpace> hostView(this->m_host,
                                                                  this->m_size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    m_device, this->m_size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(hostView, deviceView);
#endif
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
                         "attempt to transfer data (" + this->m_name +
                             ") to the host without any "
                             "valid device data.");
            }
        }
    }

    TData *m_device = nullptr; ///< Device memory pointer

    bool m_device_valid = false; ///< Flag indicating the device data is valid
};
