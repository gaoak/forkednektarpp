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

#include "Operators/Common/Spaces.hpp"
#include "Operators/Field/MemoryRegionDevice.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>

using default_fp_type = double;

// MemoryQualifier
struct ReadOnly
{
};
struct WriteOnly
{
};
struct ReadWrite
{
};

// const_if metafunction return "const T" type if B = true and "T" type
// otherwise.
template <bool B, class T = void> struct const_if
{
    typedef T type;
};

template <class T> struct const_if<true, T>
{
    typedef const T type;
};

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
     * @brief Get the pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace, typename MemQualifier>
    typename const_if<std::is_same_v<MemQualifier, ReadOnly>, TData>::type *GetPtr()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            if constexpr (std::is_same_v<MemQualifier, ReadOnly>)
            {
                return m_storage->GetHostConstPtr();
            }
            else if constexpr (std::is_same_v<MemQualifier, WriteOnly>)
            {
                return m_storage->GetHostPtr(true);
            }
            else if constexpr (std::is_same_v<MemQualifier, ReadWrite>)
            {
                return m_storage->GetHostPtr();
            }
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                if constexpr (std::is_same_v<MemQualifier, ReadOnly>)
                {
                    return ret.GetDeviceConstPtr();
                }
                else if constexpr (std::is_same_v<MemQualifier, WriteOnly>)
                {
                    return ret.GetDevicePtr(true);
                }
                else if constexpr (std::is_same_v<MemQualifier, ReadWrite>)
                {
                    return ret.GetDevicePtr();
                }
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                if constexpr (std::is_same_v<MemQualifier, ReadOnly>)
                {
                    return ret.GetDeviceConstPtr();
                }
                else if constexpr (std::is_same_v<MemQualifier, WriteOnly>)
                {
                    return ret.GetDevicePtr(true);
                }
                else if constexpr (std::is_same_v<MemQualifier, ReadWrite>)
                {
                    return ret.GetDevicePtr();
                }
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
    static MemoryRegion<TData> create(std::string name, size_t size,
                                      size_t alignment,
                                      [[maybe_unused]] bool device_only = false)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment, false);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment, device_only);
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
    static MemoryRegion<TData> create(size_t size, size_t alignment,
                                      bool device_only = false)
    {
        return MemoryRegion<TData>::template create<MemSpace>(
            "", size, alignment, device_only);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a pointer
     *
     * @param name      - name of the memory region
     * @param src       - src pointer to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> fromData(std::string name, const TDataIn *src,
                                        const size_t size, size_t alignment,
                                        [[maybe_unused]] bool device_only)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, src, size, alignment, false);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, src, size, alignment, device_only);
        }
        else
        {
            std::string msg("MemoryRegion::fromData - "
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
     * @param name      - name of the memory region
     * @param array     - std::vector to copy from
     * @param alignment - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> fromVector(
        std::string name, std::vector<TDataIn, Alloc> const &array,
        size_t alignment, bool device_only = false)
    {
        return MemoryRegion<TData>::template fromData<MemSpace, TDataIn>(
            name, array.data(), array.size(), alignment, device_only);
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
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> fromVector(
        std::vector<TDataIn, Alloc> const &array, size_t alignment,
        bool device_only = false)
    {
        return MemoryRegion<TData>::template fromVector<MemSpace, TDataIn>(
            "", array, alignment, device_only);
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
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> fromArray(
        std::string name, Nektar::Array<Nektar::OneD, TDataIn> const &array,
        size_t alignment, bool device_only = false)
    {
        return MemoryRegion<TData>::template fromData<MemSpace, TDataIn>(
            name, array.data(), array.size(), alignment, device_only);
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
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> fromArray(
        Nektar::Array<Nektar::OneD, TDataIn> const &array, size_t alignment,
        bool device_only = false)
    {
        return MemoryRegion<TData>::template fromArray<MemSpace, TDataIn>(
            "", array, alignment, device_only);
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
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> fromArray(
        std::string name,
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment, bool device_only = false)
    {
        size_t size = 0;
        for (auto i = 0; i < array.size(); ++i)
        {
            size += array[i].size();
        }

        std::vector<TData> tmp(size);

        TData *tmpPtr = tmp.data();

        for (auto i = 0; i < array.size(); ++i)
        {
            std::copy(array[i].begin(), array[i].end(), tmpPtr);

            tmpPtr += array[i].size();
        }

        return MemoryRegion<TData>::template fromData<MemSpace, TDataIn>(
            name, tmp.data(), tmp.size(), alignment, device_only);
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
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> fromArray(
        Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, TDataIn>> const
            &array,
        size_t alignment, bool device_only = false)
    {
        return MemoryRegion<TData>::template fromArray<MemSpace, TDataIn>(
            "", array, alignment, device_only);
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
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        try
        {
            // This cast fails if e.g. a MemoryRegionDevice is requested
            // from a MemoryRegionHost storage.
            auto &ret = dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

            ret.template initialize<MemWrite>(val, count, offset);
        }
        catch (const std::bad_cast &e)
        {
            // MemWrite is ignored for host-only memory region.
            m_storage->initialize(val, count, offset);
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        src pointer
     *
     * @param src - pointer to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice>
    void copyFrom(const TDataIn *src, const size_t size,
                  const size_t offset = 0)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            // MemCopy is ignored for host-only memory region.
            m_storage->copyFrom(src, size, offset);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.template copyFrom<TDataIn, MemCopy>(src, size, offset);
            }

            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.template copyFrom<TDataIn, MemCopy>(src, size, offset);
            }
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice,
              class Alloc      = std::allocator<TDataIn>>
    void copyVector(std::vector<TDataIn, Alloc> const &array)
    {
        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::copyVector - Can only copy std::vector "
                     "from HostToHost or from "
                     "HostToDevice.");
        }

        if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::copyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        this->template copyFrom<MemSpace, TDataIn, MemCopy>(array.data(),
                                                            array.size());
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice>
    void copyArray(Nektar::Array<Nektar::OneD, TDataIn> const &array)
    {
        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::copyArray - Can only copy Nektar::Array "
                     "from HostToHost or from "
                     "HostToDevice.");
        }

        if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::copyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        this->template copyFrom<MemSpace, TDataIn, MemCopy>(array.data(),
                                                            array.size());
    }

    /**
     * @brief Copy data from one region to another region.
     *
     * @param rhs - MemoryRegion to copy from
     */
    template <typename MemSpace, typename MemCopy = DeviceToDevice>
    void copyMemoryRegion(MemoryRegion &rhs)
    {
        if (this->size() != rhs.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::copyMemoryRegion - "
                << "Memory size mismatch between (" << rhs.getName()
                << ") and (" << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        this->template copyFrom<MemSpace, TData, MemCopy>(
            rhs.template GetPtr<MemSpace, ReadOnly>(), rhs.size());
    }

    /**
     * @brief Copy data to a std:vector.
     *
     * @return std::vector<TDataOut>
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TData>>
    std::vector<TDataOut, Alloc> toVector() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return std::vector<TDataOut, Alloc>(m_storage->size(),
                                                m_storage->GetHostConstPtr());
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

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return Nektar::Array<Nektar::OneD, TDataOut>(
                m_storage->size(), m_storage->GetHostConstPtr());
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
     * @brief Force a host to device copy.
     *
     */
    template <typename MemSpace> void HostToDeviceCopy(bool force = true)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal, "Storage has not allocated.");
        }

        // Check the memory space type so to not do any more checks as
        // necessary.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            m_storage->HostToDeviceCopy(force);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
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

            m_storage->HostToDeviceCopy(force);
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
            m_storage->DeviceToHostCopy();

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
