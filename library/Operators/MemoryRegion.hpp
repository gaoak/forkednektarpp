///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryRegion.hpp
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

#include "MemoryRegionDevice.hpp"

#include "Spaces.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>

using default_fp_type = double;

/**
 * @brief This function will return an alignment based on the memory type.
 *
 * When running on a device temporary memeory may be need that is only
 * on the device. Passing __EXECSPACE_MEMORY_REGION_ONLY__ will create
 * such memory. However, when code is common to both the host and the
 * device and when running on the host and requiring memeory
 * __STDCPP_DEFAULT_NEW_ALIGNMENT__ should be passed.
 *
 * @return - alignment
 */
template <typename MemSpace> inline size_t EXECSPACE_MEMORY_REGION_ONLY()
{
    if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                  std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
    {
        return __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    }
    else if constexpr (std::is_same<MemSpace, Kokkos::DefaultExecutionSpace::
                                                  memory_space>::value ||
                       std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
    {
        return __EXECSPACE_MEMORY_REGION_ONLY__;
    }
    else
    {
        std::string msg("EXECSPACE_MEMORY_REGION_ONLY - "
                        "invalid memory space: ");
        msg += Nektar::demangleTypeName(typeid(MemSpace));

        NEKERROR(Nektar::ErrorUtil::efatal, msg);
    }

    return __STDCPP_DEFAULT_NEW_ALIGNMENT__;
}

/**
 * @brief A MemoryRegion represents a memory region
 * @tparam TData  The floating-point representation used by the MemoryRegion.
 */
template <typename TData = default_fp_type> class MemoryRegion
{
public:
    /**
     * @brief Construct a new MemoryRegion object.
     */
    MemoryRegion() = default; // Default removes implicit moves

    /**
     * @brief No copy method
     */
    MemoryRegion(const MemoryRegion &) = delete;

    /**
     * @brief Destructor for a MemoryRegion object.
     */
    virtual ~MemoryRegion() = default; // Default removes implicit moves

    /**
     * @brief Construct a new MemoryRegion object by moving storage
     *        from an existing MemoryRegion object.
     *
     * @param rhs - MemoryRegion to move from
     */
    MemoryRegion(MemoryRegion &&rhs) : m_storage(std::move(rhs.m_storage))
    {
    }

    /**
     * @brief Move assignment operator.
     *
     * @param rhs - MemoryRegion to move from
     *
     * @return    - MemoryRegion&
     */
    MemoryRegion &operator=(MemoryRegion &&rhs)
    {
        m_storage = std::move(rhs.m_storage);

        return *this;
    }

    /**
     * @brief osstream operator.
     *
     * @param rhs - MemoryRegion to stream
     *
     * @return    - stream
     */
    friend auto operator<<(std::ostream &os, MemoryRegion const &mr)
        -> std::ostream &
    {
        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested
            // from a MemoryRegionHost storage.
            auto &ret =
                dynamic_cast<MemoryRegionDevice<TData> &>(*(mr.m_storage));

            return os << ret;
        }

        catch (const std::bad_cast &e)
        {
            auto &ret =
                dynamic_cast<MemoryRegionHost<TData> &>(*(mr.m_storage));

            return os << ret;
        }
    }

    /**
     * @brief Get the const pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace> const TData *GetConstPtr()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            return m_storage->GetHostConstPtr();
        }

        else if constexpr (
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                return ret.GetDeviceConstPtr();
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                return ret.GetDeviceConstPtr();
            }
        }

        return nullptr;
    }

    /**
     * @brief Get the pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace> TData *GetPtr()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            return m_storage->GetHostPtr();
        }

        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                return ret.GetDevicePtr();
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                return ret.GetDevicePtr();
            }
        }

        return nullptr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion.
     *
     * @param name      - name of the memory region
     * @param size      - size of memory
     * @param alignment - memory alignment
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace>
    static MemoryRegion<TData> create(
        std::string name, size_t size,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(name, size,
                                                                     alignment);
        }
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment);
        }
        else
        {
            std::string msg("MemoryRegion::create - "
                            "invaid memory space (");
            msg += name + "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion.
     *
     * @param size      - size of memory
     * @param alignment - memory alignment
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace>
    static MemoryRegion<TData> create(
        size_t size, size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        return MemoryRegion<TData>::template create<MemSpace>("", size,
                                                              alignment);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param name      - name of the memory region
     * @param array     - std::vector to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> fromVector(
        std::string name, std::vector<TDataIn, Alloc> const &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, array, alignment);
        }
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, array, alignment);
        }
        else
        {
            std::string msg("MemoryRegion::fromVector - "
                            "invaid memory space (");
            msg += name + "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param array     - std::vector to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> fromVector(
        std::vector<TDataIn, Alloc> const &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        return MemoryRegion<TData>::template fromVector<MemSpace, TDataIn>(
            "", array, alignment);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param name      - name of the memory region
     * @param array     - Nektar::Array to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData>
    static MemoryRegion<TData> fromArray(
        std::string name, Nektar::Array<Nektar::OneD, TDataIn> const &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, array, alignment);
        }
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, array, alignment);
        }
        else
        {
            std::string msg("MemoryRegion::fromArray1D - "
                            "invaid memory space (");
            msg += name + "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from an Nektar::Array
     *
     * @param array     - Nektar::Array to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData>
    static MemoryRegion<TData> fromArray(
        Nektar::Array<Nektar::OneD, TDataIn> const &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        return MemoryRegion<TData>::template fromArray<MemSpace, TDataIn>(
            "", array, alignment);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param name      - name of the memory region
     * @param array     - Nektar::Array to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData>
    static MemoryRegion<TData> fromArray(
        std::string name,
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, array, alignment);
        }
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, array, alignment);
        }
        else
        {
            std::string msg("MemoryRegion::fromArray2D - "
                            "invaid memory space (");
            msg += name + "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array     - Nektar::Array to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn = TData>
    static MemoryRegion<TData> fromArray(
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        return MemoryRegion<TData>::template fromArray<MemSpace, TDataIn>(
            "", array, alignment);
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     *
     */
    template <typename MemSpace, typename TDataIn = TData,
              class Alloc = std::allocator<TDataIn>>
    void copyVector(std::vector<TDataIn, Alloc> const &array)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->copyVector(array);
        }

        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyVector(array);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyVector(array);
            }
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn = TData>
    void copyArray(Nektar::Array<Nektar::OneD, TDataIn> const &array)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->copyArray(array);
        }

        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyArray(array);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyArray(array);
            }
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn = TData>
    void copyArray(
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->copyArray(array);
        }

        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyArray(array);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.copyArray(array);
            }
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn = TData>
    void copyRaw(TData *dest, TDataIn *src, size_t size)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->copyRaw(dest, src, size);
        }

        else if constexpr (
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                [[maybe_unused]] auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                m_storage->copyRaw(dest, src, size);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                [[maybe_unused]] auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                m_storage->copyRaw(dest, src, size);
            }
        }
        else
        {
            std::string msg("MemoryRegion::copyRaw - "
                            "invaid memory space (");
            msg += m_storage->getName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }
    }

    /**
     * @brief Copy data to a std:vector.
     *
     * @return Array<Nektar::OneD, TData>
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TData>>
    std::vector<TData, Alloc> toVector() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<TDataOut, TData>::value)
        {
            return std::vector<TDataOut, Alloc>(m_storage->size(),
                                                m_storage->GetHostPtr());
        }
        else
        {
            std::vector<TDataOut, Alloc> vector(m_storage->size());

            // Copy the data from the input field
            auto *ptr    = m_storage->GetHostConstPtr();
            auto *vecPtr = vector.data();

            std::copy(ptr, ptr + m_storage->size(), vecPtr);

            return vector;
        }
    }

    /**
     * @brief Copy data to a Nektar::Array
     *
     * @return Array<Nektar::OneD, TDataOut>
     */
    template <typename TDataOut = TData>
    Nektar::Array<Nektar::OneD, TDataOut> toArray() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same<TDataOut, TData>::value)
        {
            return Nektar::Array<Nektar::OneD, TDataOut>(
                m_storage->size(), m_storage->GetHostPtr());
        }
        else
        {
            Nektar::Array<Nektar::OneD, TDataOut> array(m_storage->size());

            // Copy the data from the input field
            auto *ptr    = m_storage->GetHostConstPtr();
            auto *arrPtr = array.data();

            std::copy(ptr, ptr + m_storage->size(), arrPtr);

            return array;
        }
    }

    /**
     * @brief Get the storage size
     *
     * @return - size_t
     */
    size_t size() const
    {
        return m_storage->size();
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     */
    void initialize(TData val, size_t count = 0)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        m_storage->initialize(val, count);
    }

    /**
     * @brief Get the storage name.
     *
     */
    std::string getName() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        return m_storage->getName();
    }

    /**
     * @brief Set this memory as being valid on the selected MemSpace
     * and the other (if it exists) as invlaid.
     *
     */
    template <typename MemSpace> void setValid()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Set the Host memory as being valid.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            // If the memory region is a MemoryRegionDevice then
            // invalidate the device side which validates the host
            // side.
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.setValid(false);
            }

            // If the memory region is a MemoryRegionHost then
            // just validate the host side.
            catch (const std::bad_cast &e)
            {
                m_storage->setValid(true);
            }
        }

        // Set the Device memory as being valid.
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            // If the memory region is a MemoryRegionDevice then
            // validate the device side which invalidates the host
            // side.
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.setValid(true);
            }

            // If the memory region is a MemoryRegionHost then
            // just validate the host side.
            catch (const std::bad_cast &e)
            {
                m_storage->setValid(true);
            }
        }
    }

    /**
     * @brief Force a host to device copy.
     *
     */
    template <typename MemSpace> void HostToDevice(bool force = true)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Check the memory space type so to not do any more checks as
        // necessary.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->HostToDevice(force);
        }
        else if constexpr (
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device.
                GetStorage<MemoryRegionDevice>();
            }

            m_storage->HostToDevice(force);
        }
    }

    /**
     * @brief Force a device to host copy.
     *
     */
    template <typename MemSpace> void DeviceToHost(bool force = true)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Check the memory space type so to not do any more checks as
        // necessary.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            m_storage->DeviceToHost(force);
        }
        else if constexpr (
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                m_storage->DeviceToHost(force);
            }

            catch (const std::bad_cast &e)
            {
                std::string msg("MemoryRegion::DeviceToHost - "
                                "the memory region (");
                msg += m_storage->getName() +
                       ") is not a MemoryRegionDevice but a " +
                       Nektar::demangleTypeName(typeid(*m_storage));

                NEKERROR(Nektar::ErrorUtil::efatal, msg);

                // Convert the storage to device.
                // GetStorage<MemoryRegionDevice>();
            }
        }
        else
        {
            std::string msg("MemoryRegion::DeviceToHost - "
                            "invaid memory space (");
            msg += m_storage->getName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }
    }

    /**
     * @brief Copy data from one region to another region.
     *
     * @param rhs - MemoryRegion to copy from
     */
    template <typename MemSpace> void RegionToRegion(MemoryRegion &rhs)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Check the memory space type to determine the copy.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            this->HostToHost(rhs);
        }
        else if constexpr (
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            this->DeviceToDevice(rhs);
        }
    }

    /**
     * @brief Copy data from one host to another host.
     *
     * @param rhs - MemoryRegion to copy from
     */
    void HostToHost(const MemoryRegion &rhs)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested
            // from a MemoryRegionHost storage
            auto &rhsRet =
                dynamic_cast<MemoryRegionDevice<TData> &>(*(rhs.m_storage));

            // Make sure the data is on the host side.
            rhsRet.DeviceToHost();
        }

        catch (const std::bad_cast &e)
        {
        }

        m_storage->HostToHost(*(rhs.m_storage));
    }

    /**
     * @brief Copy data from one device to another device.
     *
     * @param rhs - MemoryRegion to copy from
     */
    void DeviceToDevice(MemoryRegion &rhs)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Make sure the source region has a device memory region.
        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested
            // from a MemoryRegionHost storage
            auto &rhsRet =
                dynamic_cast<MemoryRegionDevice<TData> &>(*(rhs.m_storage));

            // Make sure the data is on the device side.
            rhsRet.HostToDevice();
        }

        catch (const std::bad_cast &e)
        {
            // Convert the storage to device
            rhs.template GetStorage<MemoryRegionDevice>();

            auto &rhsRet =
                dynamic_cast<MemoryRegionDevice<TData> &>(*(rhs.m_storage));

            // Make sure the data is on the device side.
            rhsRet.HostToDevice();
        }

        // Make sure the destination region has a device memory region.
        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested
            // from a MemoryRegionHost storage
            auto &ret =
                dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

            ret.DeviceToDevice(*(rhs.m_storage));
        }

        catch (const std::bad_cast &e)
        {
            // Convert the storage to device.
            GetStorage<MemoryRegionDevice>();

            auto &ret =
                dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

            ret.DeviceToDevice(*(rhs.m_storage));
        }
    }

protected:
    /**
     * @brief Get the underlying storage of the memory region as the
     *        requested type.
     *
     * @return MemoryRegion<TData> storage converted to the requested type
     *
     * This routine performs MemoryRegion conversions if necessary to
     * enable casting of, for example a host memory region to a device
     * memory region to support the use of a device operator if
     * necessary.
     *
     * A runtime warning is provided if a transfer of data from device
     * to host is required to achieve the conversion.
     */
    template <template <typename> class TMemoryRegion = MemoryRegionHost>
    TMemoryRegion<TData> &GetStorage()
    {
        using T = TMemoryRegion<TData>;

        static_assert(std::is_base_of<MemoryRegionHost<TData>, T>::value,
                      "TMemoryRegion must derive MemoryRegionHost<TData>");

        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested from a
            // MemoryRegionHost object.
            auto &ret = dynamic_cast<T &>(*m_storage);

            std::string name  = Nektar::demangleTypeName(typeid(T));
            std::string sname = Nektar::demangleTypeName(typeid(*m_storage));

            // Debug warning, a (possibly) undesired conversion occured.
            std::string msg("MemoryRegion::GetStorage - "
                            "Requested backing storage (");
            msg += m_storage->getName() + ") of type " + name +
                   " != actual storage type " + sname;

            WARNINGL0(typeid(*m_storage) == typeid(T), msg);

            return ret;
        }

        // Dynamic cast threw an exception, attempt to allocate the
        // requested TMemoryRegion from old data.
        catch (const std::bad_cast &e)
        {
            std::string name  = Nektar::demangleTypeName(typeid(T));
            std::string sname = Nektar::demangleTypeName(typeid(*m_storage));

            std::string msg("MemoryRegion::GetStorage - "
                            "Converting backing storage (");
            msg += m_storage->getName() + ") from " + sname + " to " + name;

            WARNINGL0(false, msg);

            // Make sure the memory is on the host.
            m_storage->DeviceToHost();

            // Cast to the MemoryRegionHost base class.
            auto &ret = dynamic_cast<MemoryRegionHost<TData> &>(*m_storage);

            // Create new TMemoryRegion from the MemoryRegionHost base
            // class.
            m_storage = std::make_unique<T>(T(std::move(ret)));

            return dynamic_cast<T &>(*m_storage);
        }
    }

    std::unique_ptr<MemoryRegionHost<TData>> m_storage = nullptr;
};
