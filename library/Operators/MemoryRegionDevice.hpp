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

#include "MemoryRegionHost.hpp"

#include "Spaces.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#include <utility>

#ifdef NEKTAR_ENABLE_CUDA
#include <cuda_runtime.h>
#elif defined(NEKTAR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(NEKTAR_ENABLE_KOKKOS)
#endif

// If this macro is set when calling the device side creation and copy
// methods will put the data on the host and the device. If not set,
// the device side creation and copy methods will put the data on the
// device only.

// #define SYNC_WITH_HOST

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
    friend class MemoryRegionHost<TData>;

public:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionDevice()                              = delete;
    MemoryRegionDevice(const MemoryRegionDevice &rhs) = delete;
    MemoryRegionDevice &operator=(const MemoryRegionDevice &rhs) = delete;

    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param size      - size of memory
     * @param alignment - memory alignment
     */
    MemoryRegionDevice(std::string name, size_t size,
                       size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
        : MemoryRegionHost<TData>(name, size, alignment)
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
        rhs.m_host_valid   = false;
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
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param array     - std::vector to copy from
     * @param alignment - memory alignment
     */
    template <typename TDataIn = TData, class Alloc = std::allocator<TDataIn>>
    MemoryRegionDevice(std::string name,
                       std::vector<TDataIn, Alloc> const &array,
                       size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
#ifdef SYNC_WITH_HOST
        : MemoryRegionHost<TData>(name, array, alignment)
#else
        : MemoryRegionHost<TData>(name, array.size(), alignment)
#endif
    {
        createMemory();

        if constexpr (std::is_same<TDataIn, TData>::value &&
                      std::is_same<Alloc, std::allocator<TDataIn>>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, array.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, array.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            TData *arrayPtr = const_cast<TDataIn *>(array.data());

            // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
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

            std::copy(array.begin(), array.end(), tmp.begin());

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            std::copy(array.begin(), array.end(), tmp.begin());

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(array.data(),
                                                                this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, hostView);
#endif
        }

        m_device_valid     = true;
        this->m_initialize = false;
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
    MemoryRegionDevice(std::string name,
                       Nektar::Array<Nektar::OneD, TDataIn> const &array,
                       size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
#ifdef SYNC_WITH_HOST
        : MemoryRegionHost<TData>(name, array, alignment)
#else
        : MemoryRegionHost<TData>(name, array.size(), alignment)
#endif
    {
        createMemory();

        if constexpr (std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, array.get(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, array.get(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            TData *arrayPtr = const_cast<TDataIn *>(array.get());

            // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
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

            std::copy(array.begin(), array.end(), tmp.begin());

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            std::copy(array.begin(), array.end(), tmp.begin());

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(array.get(),
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
     * @brief Templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array     - Nektar::Array to copy from
     * @param alignment - memory alignment
     */
    template <typename TDataIn = TData>
    MemoryRegionDevice(
        std::string name,
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
#ifdef SYNC_WITH_HOST
        : MemoryRegionHost<TData>(name, array, alignment)
#else
        : MemoryRegionHost<TData>(name, array, __EXECSPACE_MEMORY_REGION_ONLY__)
#endif
    {
        // A bit backasswards - A constructor call is needed (the
        // default is deleted). With a 2D array the size is needed
        // which is determined on the host side.  Initially, allocate
        // as device only so to get the size in the constructor, then
        // create the host memory.
#ifdef SYNC_WITH_HOST
#else
        MemoryRegionHost<TData>::createMemory(name, alignment);
#endif
        createMemory();

        if constexpr (std::is_same<TDataIn, TData>::value)
        {
            TData *devicePtr = m_device;

            for (auto i = 0; i < array.size(); ++i)
            {
                size_t size = array[i].size();

#if defined(NEKTAR_ENABLE_CUDA)
                cudaMemcpy(devicePtr, array[i].get(), size * sizeof(TData),
                           cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
                hipMemcpy(devicePtr, array[i].get(), size * sizeof(TData),
                          hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
                TData *arrayPtr = const_cast<TDataIn *>(array[i].get());

                // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
                // char *dstPtr = reinterpret_cast<char *>(devicePtr);

                // Create an unmanage Kokkos view from the raw pointers.
                // Kokkos::View<char *, Kokkos::HostSpace> hostView(
                //     srcPtr, size * sizeof(TData));
                // Kokkos::View<char *, Kokkos::DefaultExecutionSpace>
                // deviceView(
                //     dstPtr, size * sizeof(TData));

                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
                                                                    size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    devicePtr, size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(deviceView, hostView);
#endif
                devicePtr += size;
            }
        }
        else // if constexpr (!std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            std::vector<TData> tmp(this->m_size);

            TData *tmpPtr = tmp.data();

            for (auto i = 0; i < array.size(); ++i)
            {
                std::copy(array[i].begin(), array[i].end(), tmpPtr);

                tmpPtr += array[i].size();
            }

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            TData *tmpPtr = tmp.data();

            for (auto i = 0; i < array.size(); ++i)
            {
                std::copy(array[i].begin(), array[i].end(), tmpPtr);

                tmpPtr += array[i].size();
            }

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            TData *devicePtr = m_device;

            for (auto i = 0; i < array.size(); ++i)
            {
                size_t size = array[i].size();

                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(
                    array[i].get(), size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    devicePtr, size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(deviceView, hostView);

                devicePtr += size;
            }
#endif
        }

        m_device_valid     = true;
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
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::operator=(std::move(rhs));
#endif
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

        if (mr.m_alignment == __EXECSPACE_MEMORY_REGION_ONLY__)
        {
            msg << "EXECSPACE_MEMORY_REGION_ONLY ";
        }
        else
        {
            msg << " host_valid: " << mr.m_host_valid << " ";
        }

        msg << " device_valid: " << mr.m_device_valid << " ";

        return os << msg.str();
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

        DeviceToHost(); // Move to host if necessary

        return this->m_host;
    }

    /**
     * @brief Get the pointer to the host memory - assumes the data
     * will be modified.
     *
     * @return - TData*
     */
    TData *GetHostPtr() override
    {
        DeviceToHost(); // Move to host if necessary

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

        HostToDevice(); // Move to device if necessary

        return m_device;
    }

    /**
     * @brief Get the pointer to the device memory - assumes the data
     * will be modified.
     *
     * @return - TData*
     */
    TData *GetDevicePtr()
    {
        HostToDevice(); // Move to device if necessary

        this->m_host_valid = false;

        return m_device;
    }

    /**
     * @brief Perform a host to device copy.
     *
     * @param force - copy regardless of status
     */
    void HostToDevice(bool force = false) override
    {
        // Because the host pointer is used directly the device data may
        // be marked as valid as such the force can be used to assure
        // the copy occurs regardles.
        if (this->m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__ &&
            (force || !m_device_valid))
        {
            if (this->m_host == nullptr)
            {
                // Two options - create host memory but using the
                // default alignment or toss an error.

                // MemoryRegionHost<TData>::createMemory();

                // WARNINGL0(false,
                //           "Attempt to transfer data the host without any "
                //           "valid host memory allocated. Creating host
                //           memory.");
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
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // char *srcPtr = reinterpret_cast<char *>(this->m_host);
                // char *dstPtr = reinterpret_cast<char *>(m_device);

                // Create an unmanage Kokkos view from the raw pointers.
                // Kokkos::View<char *, Kokkos::HostSpace> hostView(
                //     srcPtr, this->m_size * sizeof(TData));
                // Kokkos::View<char *, Kokkos::DefaultExecutionSpace>
                // deviceView(
                //     dstPtr, this->m_size * sizeof(TData));

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
    void DeviceToHost(bool force = false) override
    {
        // Because the device pointer is used directly the host data may
        // be marked as valid as such the force can be used to assure
        // the copy occurs regardles.
        if ( // this->m_alignment != __EXECSPACE_MEMORY_REGION_ONLY__ &&
            (force || !this->m_host_valid))
        {
            if (this->m_host == nullptr)
            {
                // Two options - create host memory but using the
                // default alignment or toss an error.

                // MemoryRegionHost<TData>::createMemory();

                // WARNINGL0(false,
                //           "Attempt to transfer data the host without any "
                //           "valid host memory allocated. Creating host
                //           memory.");
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
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // char *srcPtr = reinterpret_cast<char *>(m_device);
                // char *dstPtr = reinterpret_cast<char *>(this->m_host);

                // Create an unmanage Kokkos view from the raw pointers.
                // Kokkos::View<char *, Kokkos::DefaultExecutionSpace>
                // deviceView(
                //     srcPtr, this->m_size * sizeof(TData));
                // Kokkos::View<char *, Kokkos::HostSpace> hostView(
                //     dstPtr, this->m_size * sizeof(TData));

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

    /**
     * @brief Copy data from one host to another host.
     *
     * @param rhs - MemoryRegionHost to copy from
     */
    void HostToHost(MemoryRegionHost<TData> &rhs) override
    {
        MemoryRegionHost<TData>::HostToHost(rhs);
    }

    /**
     * @brief Copy data from one device to another device.
     *
     * @param rhs - MemoryRegionHost to copy from
     */
    void DeviceToDevice(MemoryRegionHost<TData> &rhs) override
    {
        // Make sure the source region is a device memory region.
        try
        {
            auto &rhsRet = dynamic_cast<MemoryRegionDevice<TData> &>(rhs);

            [[maybe_unused]] size_t size =
                this->m_size < rhs.size() ? this->m_size : rhs.size();

#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, rhsRet.m_device, size * sizeof(TData),
                       cudaMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, rhsRet.m_device, size * sizeof(TData),
                      hipMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(rhsRet.m_device);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> srcView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> dstView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> srcView(
                rhsRet.m_device, this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(dstView, srcView);
#endif
            m_device_valid = true;

            this->m_host_valid = false;
            this->m_initialize = false;
        }

        catch (const std::bad_cast &e)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     std::string("Calling DeviceToDevice but the memory "
                                 "region is not a MemoryRegionDevice but a ") +
                         typeid(rhs).name());
        }
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     */
    void initialize(TData val, size_t count = 0) override
    {
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::initialize(val, count);
#endif
        if (count == 0)
        {
            count = this->m_size;
        }

        // If the value is zero, memset is the most efficent.
        if (val == TData(0))
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemset(m_device, 0, count * sizeof(TData));
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemset(m_device, 0, count * sizeof(TData));
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // TData *host = new TData[count * sizeof(TData)];

            // std::fill(host, host + count, val);

            // char *srcPtr = reinterpret_cast<char *>(host);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create an unmanage Kokkos view from the raw pointer.
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, count);

            // Deep copy the val to the device view.
            Kokkos::deep_copy(deviceView, val);

            // delete[] (host);
#endif
        }
        // Nonzero value
        else
        {
            // If set on the host do a copy to the device.
            if (this->m_host)
            {
                HostToDevice(true);
            }
            // Create temporary memory, fill, copy to the device, then
            // delete the temporary memory.
            else
            {
#if defined(NEKTAR_ENABLE_CUDA)
                TData *host = new TData[count * sizeof(TData)];

                std::fill(host, host + count, val);

                cudaMemcpy(m_device, host, count * sizeof(TData),
                           cudaMemcpyHostToDevice);

                delete[](host);
#elif defined(NEKTAR_ENABLE_HIP)
                TData *host = new TData[count * sizeof(TData)];

                std::fill(host, host + count, val);

                hipMemcpy(m_device, host, count * sizeof(TData),
                          hipMemcpyHostToDevice);

                delete[](host);
#elif defined(NEKTAR_ENABLE_KOKKOS)
                // char *srcPtr = reinterpret_cast<char *>(host);
                // char *dstPtr = reinterpret_cast<char *>(m_device);

                // Create an unmanage Kokkos view from the raw pointers.
                // Kokkos::View<char *, Kokkos::HostSpace> hostView(
                //     srcPtr, this->m_size * sizeof(TData));
                // Kokkos::View<char *, Kokkos::DefaultExecutionSpace>
                // deviceView(
                //     dstPtr, this->m_size * sizeof(TData));

                // Create an unmanage Kokkos view from the raw pointer.
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    m_device, count);

                // Deep copy the val to the device view.
                Kokkos::deep_copy(deviceView, val);

                // delete[] (host);
#endif
            }
        }

        m_device_valid     = true;
        this->m_initialize = false;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     */
    template <typename TDataIn = TData, class Alloc = std::allocator<TDataIn>>
    void copyVector([[maybe_unused]] std::vector<TDataIn, Alloc> const &array)
    {
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::template copyVector<TDataIn>(array);
#endif
        if constexpr (std::is_same<TDataIn, TData>::value &&
                      std::is_same<Alloc, std::allocator<TDataIn>>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, array.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, array.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            TDataIn *arrayPtr = const_cast<TDataIn *>(array.data());

            // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
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

            std::copy(array.begin(), array.end(), tmp.begin());

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            std::copy(array.begin(), array.end(), tmp.begin());

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(array.get(),
                                                                this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, hostView);
#endif
        }

        m_device_valid     = true;
        this->m_initialize = false;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyArray(
        [[maybe_unused]] Nektar::Array<Nektar::OneD, TDataIn> const &array)
    {
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::template copyArray<TDataIn>(array);
#endif
        if constexpr (std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(m_device, array.get(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(m_device, array.get(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)

            TDataIn *arrayPtr = const_cast<TDataIn *>(array.get());

            // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
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

            std::copy(array.begin(), array.end(), tmp.begin());

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            std::copy(array.begin(), array.end(), tmp.begin());

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(array.get(),
                                                                this->m_size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                m_device, this->m_size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(deviceView, hostView);
#endif
        }

        m_device_valid     = true;
        this->m_initialize = false;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyArray(
        [[maybe_unused]] Nektar::Array<
            Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const &array)
    {
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::template copyArray<TDataIn>(array);
#endif
        if constexpr (std::is_same<TDataIn, TData>::value)
        {
            TData *devicePtr = m_device;

            for (auto i = 0; i < array.size(); ++i)
            {
                size_t size = array[i].size();

#if defined(NEKTAR_ENABLE_CUDA)
                cudaMemcpy(devicePtr, array[i].get(), size * sizeof(TData),
                           cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
                hipMemcpy(devicePtr, array[i].get(), size * sizeof(TData),
                          hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
                TDataIn *arrayPtr = const_cast<TDataIn *>(array[i].get());

                // char *srcPtr = reinterpret_cast<char *>(arrayPtr);
                // char *dstPtr = reinterpret_cast<char *>(devicePtr);

                // Create an unmanage Kokkos view from the raw pointers.
                // Kokkos::View<char *, Kokkos::HostSpace> hostView(
                //     srcPtr, size * sizeof(TData));
                // Kokkos::View<char *, Kokkos::DefaultExecutionSpace>
                // deviceView(
                //     dstPtr, size * sizeof(TData));

                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(arrayPtr,
                                                                    size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    devicePtr, size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(deviceView, hostView);
#endif

                devicePtr += size;
            }
        }
        else // if constexpr (!std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            std::vector<TData> tmp(this->m_size);

            TData *tmpPtr = tmp.data();

            for (auto i = 0; i < array.size(); ++i)
            {
                std::copy(array[i].begin(), array[i].end(), tmpPtr);

                tmpPtr += array[i].size();
            }

            cudaMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(this->m_size);

            TData *tmpPtr = tmp.data();

            for (auto i = 0; i < array.size(); ++i)
            {
                std::copy(array[i].begin(), array[i].end(), tmpPtr);

                tmpPtr += array[i].size();
            }

            hipMemcpy(m_device, tmp.data(), this->m_size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(m_device);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, this->m_size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, this->m_size * sizeof(TData));

            TData *devicePtr = m_device;

            for (auto i = 0; i < array.size(); ++i)
            {
                size_t size = array[i].size();

                // Create unmanage Kokkos views from the raw pointers.
                Kokkos::View<TDataIn *, Kokkos::HostSpace> hostView(
                    array[i].get(), size);
                Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> deviceView(
                    devicePtr, size);

                // Deep copy the host view to the device view.
                Kokkos::deep_copy(deviceView, hostView);

                devicePtr += size;
            }
#endif
        }

        m_device_valid     = true;
        this->m_initialize = false;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     */
    template <typename TDataIn = TData>
    void copyRaw([[maybe_unused]] TData *dest, [[maybe_unused]] TDataIn *src,
                 [[maybe_unused]] size_t size)
    {
#ifdef SYNC_WITH_HOST
        MemoryRegionHost<TData>::template copyRaw<TDataIn>(dest, src, size);
#endif
        if constexpr (std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            cudaMemcpy(dest, src, size * sizeof(TData), cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            hipMemcpy(dest, src, size * sizeof(TData), hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(src);
            // char *dstPtr = reinterpret_cast<char *>(dest);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> srcView(src, size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dest,
                                                                         size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(dstView, srcView);
#endif
        }
        else // if constexpr (!std::is_same<TDataIn, TData>::value)
        {
#if defined(NEKTAR_ENABLE_CUDA)
            std::vector<TData> tmp(size);

            for (size_t i = 0; i < size; ++i)
            {
                tmp[i] = src[i];
            }

            cudaMemcpy(dest, tmp.data(), size * sizeof(TData),
                       cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
            std::vector<TData> tmp(size);

            for (size_t i = 0; i < size; ++i)
            {
                tmp[i] = src[i];
            }

            hipMemcpy(dest, tmp.data(), size * sizeof(TData),
                      hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_KOKKOS)
            // char *srcPtr = reinterpret_cast<char *>(tmp.data());
            // char *dstPtr = reinterpret_cast<char *>(dest);

            // Create an unmanage Kokkos view from the raw pointers.
            // Kokkos::View<char *, Kokkos::HostSpace> hostView(
            //     srcPtr, size * sizeof(TData));
            // Kokkos::View<char *, Kokkos::DefaultExecutionSpace> deviceView(
            //     dstPtr, size * sizeof(TData));

            // Create unmanage Kokkos views from the raw pointers.
            Kokkos::View<TDataIn *, Kokkos::HostSpace> srcView(src, size);
            Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dest,
                                                                         size);

            // Deep copy the host view to the device view.
            Kokkos::deep_copy(dstView, srcView);
#endif
        }

        m_device_valid     = true;
        this->m_initialize = false;
    }

    /**
     * @brief Set all storage data as being valid.
     *
     */
    void setValid(bool valid) override
    {
        m_device_valid     = valid;
        this->m_host_valid = !valid;
        this->m_initialize = false;
    }

    /**
     * @brief Get the storage valid.
     *
     * This is a virtual function so that subclasses can set values.
     */
    bool getValid() const override
    {
        return m_device_valid;
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
#elif defined(NEKTAR_ENABLE_KOKKOS)
        m_device = (TData *)
            Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
                this->m_name, this->m_size * sizeof(TData));
#endif
    }

    TData *m_device = nullptr; ///< Device memory pointer

    bool m_device_valid = false; ///< Flag indicating the device data is valid
};
