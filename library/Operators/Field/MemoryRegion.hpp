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

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "Operators/Field/MemoryRegionDevice.hpp"

// Memory access qualifier
struct ReadOnly
{
};
struct WriteOnly
{
};
struct ReadWrite
{
};

/**
 * @brief Possible states for Field data.
 *
 * These identify the mathematical representation of the field data. The two
 * main states are *Phys*, representing the field at the quadrature points,
 * and *Coeff*, representing the field in terms of its spectral/hp element
 * basis coefficients.
 */
enum class FieldState
{
    Phys,
    Coeff
};

// const_if metafunction return "const T" type if B = true and "T" type
// otherwise.
template <bool B, typename TData = void> struct const_if
{
    typedef TData type;
};

template <class TData> struct const_if<true, TData>
{
    typedef const TData type;
};

/**
 * @brief A MemoryRegion represents a memory region
 * @tparam TData  The floating-point representation used by the MemoryRegion.
 */
template <typename TData> class MemoryRegion
{
    template <typename TDataField, FieldState TState> friend class Field;

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
    friend auto operator<<(std::ostream &os, const MemoryRegion &mr)
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
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetPtr - Storage has not allocated.");
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
     * @param name        - name of the memory region
     * @param size        - size of memory
     * @param alignment   - memory alignment
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace>
    static MemoryRegion<TData> Create(
        const std::string name, const size_t size, const size_t alignment,
        [[maybe_unused]] const bool device_only = false,
        const size_t device_rank                = 0)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment, device_rank, false);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment, device_rank, device_only);
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
     * @param size        - size of memory
     * @param alignment   - memory alignment
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace>
    static MemoryRegion<TData> Create(const size_t size, const size_t alignment,
                                      const bool device_only   = false,
                                      const size_t device_rank = 0)
    {
        return MemoryRegion<TData>::template Create<MemSpace>(
            "", size, alignment, device_only, device_rank);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param name        - name of the memory region
     * @param array       - std::vector to copy from
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> FromVector(
        const std::string name, std::vector<TDataIn, Alloc> const &array,
        const size_t alignment, const bool device_only = false,
        const size_t device_rank = 0)
    {
        return MemoryRegion<TData>::template FromSRC<MemSpace, TDataIn>(
            name, array.data(), array.size(), alignment, device_only,
            device_rank);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param array       - std::vector to copy from
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> FromVector(
        std::vector<TDataIn, Alloc> const &array, const size_t alignment,
        const bool device_only = false, const size_t device_rank = 0)
    {
        return MemoryRegion<TData>::template FromVector<MemSpace, TDataIn>(
            "", array, alignment, device_only, device_rank);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param name        - name of the memory region
     * @param array       - Nektar::Array to copy from
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> FromArray(
        const std::string name,
        Nektar::Array<Nektar::OneD, TDataIn> const &array,
        const size_t alignment, const bool device_only = false,
        const size_t device_rank = 0)
    {
        return MemoryRegion<TData>::template FromSRC<MemSpace, TDataIn>(
            name, array.data(), array.size(), alignment, device_only,
            device_rank);
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
    static MemoryRegion<TData> FromArray(
        Nektar::Array<Nektar::OneD, TDataIn> const &array,
        const size_t alignment, const bool device_only = false,
        const size_t device_rank = 0)
    {
        return MemoryRegion<TData>::template FromArray<MemSpace, TDataIn>(
            "", array, alignment, device_only, device_rank);
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     */
    template <typename MemSpace>
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::Initialize - Storage has not allocated.");
        }

        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            // MemSpace is ignored for host-only memory region.
            m_storage->Initialize(val, count, offset);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage.
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.Initialize(val, count, offset);
            }
            catch (const std::bad_cast &e)
            {
                // MemSpace is ignored for host-only memory region.
                m_storage->Initialize(val, count, offset);
            }
        }
    }

    /**
     * @brief Copy data from one region to another region.
     *
     * @param rhs - MemoryRegion to copy from
     */
    template <typename MemSpace, typename MemCopy> void Copy(MemoryRegion &rhs)
    {
        if (this->size() != rhs.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::Copy - "
                << "Memory size mismatch between (" << rhs.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            // Copy from device.
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                [[maybe_unused]] auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*rhs.m_storage);

                this->template CopySRC<MemSpace, MemCopy>(
                    rhs.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>(),
                    rhs.size());
            }
            catch (const std::bad_cast &e)
            {
                // rhs does not have device allocation, copy from host instead.
                if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
                {
                    this->template CopySRC<MemSpace, HostToDevice>(
                        rhs.template GetPtr<NektarSpaces::HostSpace,
                                            ReadOnly>(),
                        rhs.size());
                }
                else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
                {
                    this->template CopySRC<MemSpace, HostToHost>(
                        rhs.template GetPtr<NektarSpaces::HostSpace,
                                            ReadOnly>(),
                        rhs.size());
                }
            }
        }
        else
        {
            // Copy from host.
            this->template CopySRC<MemSpace, MemCopy>(
                rhs.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
                rhs.size());
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
    void CopyVector(const std::vector<TDataIn, Alloc> &array)
    {
        if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::CopyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::CopyVector - Can only copy std::vector "
                     "from HostToHost or from "
                     "HostToDevice.");
        }
        else
        {
            this->template CopySRC<MemSpace, MemCopy>(array.data(),
                                                      array.size());
        }
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
    void CopyArray(const Nektar::Array<Nektar::OneD, TDataIn> &array)
    {
        if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::CopyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::CopyArray - Can only copy Nektar::Array "
                     "from HostToHost or from "
                     "HostToDevice.");
        }
        else
        {
            this->template CopySRC<MemSpace, MemCopy>(array.data(),
                                                      array.size());
        }
    }

    /**
     * @brief Copy data to a std:vector.
     *
     * @return std::vector<TDataOut>
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TData>>
    std::vector<TDataOut, Alloc> ToVector()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::ToVector - Storage has not allocated.");
        }

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return std::vector<TDataOut, Alloc>(
                m_storage->m_size,
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>());
        }
        else
        {
            std::vector<TDataOut, Alloc> vector(m_storage->m_size);

            // Copy the data from the input field
            auto ptr =
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto vecPtr = vector.data();

            std::copy(ptr, ptr + m_storage->m_size, vecPtr);

            return vector;
        }
    }

    /**
     * @brief Copy data to a Nektar::Array
     *
     * @return Array<Nektar::OneD, TDataOut>
     */
    template <typename TDataOut = TData>
    Nektar::Array<Nektar::OneD, TDataOut> ToArray()
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::ToArray - Storage has not allocated.");
        }

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return Nektar::Array<Nektar::OneD, TDataOut>(
                m_storage->m_size,
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>());
        }
        else
        {
            Nektar::Array<Nektar::OneD, TDataOut> array(m_storage->m_size);

            // Copy the data from the input field
            auto ptr =
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto arrPtr = array.data();

            std::copy(ptr, ptr + m_storage->m_size, arrPtr);

            return array;
        }
    }

    size_t GetAlignment() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetAlignment - Storage has not allocated.");
        }

        return m_storage->m_alignment;
    }

    /**
     * @brief Get the storage device rank.
     *
     */
    size_t GetDeviceRank() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(
                Nektar::ErrorUtil::efatal,
                "MemoryRegion::GetDeviceRank - Storage has not allocated.");
        }

        return m_storage->m_device_rank;
    }

    /**
     * @brief Get the storage size
     *
     * @return - size_t
     */
    size_t size() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::size - Storage has not allocated.");
        }

        return m_storage->m_size;
    }

    /**
     * @brief Get the storage name.
     *
     */
    std::string GetName() const
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetName - Storage has not allocated.");
        }

        return m_storage->m_name;
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
                      "MemoryRegion::GetStorage - TMemoryRegion must derive "
                      "MemoryRegionHost<TData>");

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
            msg += m_storage->m_name + ") of type " + name +
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
            msg += m_storage->m_name + ") from " + sname + " to " + name;

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

    /**
     * @brief Force a host to device copy.
     *
     */
    template <typename MemSpace> void HostToDeviceCopy(const bool force = true)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(
                Nektar::ErrorUtil::efatal,
                "MemoryRegion::HostToDeviceCopy - Storage has not allocated.");
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

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a pointer
     *
     * @param name        - name of the memory region
     * @param src         - src pointer to copy from
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> FromSRC(const std::string name,
                                       const TDataIn *src, const size_t size,
                                       const size_t alignment,
                                       [[maybe_unused]] const bool device_only,
                                       const size_t device_rank = 0)
    {
        auto mr = MemoryRegion();

        // Create a new MemoryRegion and polymorphically store as
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment, device_rank, false);
            mr.m_storage->template CopySRC<HostToHost>(src, size);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            mr.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment, device_rank, device_only);
            auto &ret =
                dynamic_cast<MemoryRegionDevice<TData> &>(*mr.m_storage);
            ret.template CopySRC<HostToDevice>(src, size);
        }
        else
        {
            std::string msg("MemoryRegion::FromSRC - "
                            "invalid memory space (");
            msg += name + "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        return mr;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        src pointer
     *
     * @param src - pointer to copy from
     * @param size- size of memory
     *
     */
    template <typename MemSpace, typename MemCopy, typename TDataIn>
    void CopySRC(const TDataIn *src, const size_t size, const size_t offset = 0)
    {
        if (m_storage == nullptr)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::CopySRC - Storage has not allocated.");
        }

        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            // MemCopy is ignored for host-only memory region.
            m_storage->template CopySRC<MemCopy>(src, size, offset);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            try
            {
                // This cast fails if e.g. a MemoryRegionDevice is requested
                // from a MemoryRegionHost storage
                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.template CopySRC<MemCopy>(src, size, offset);
            }
            catch (const std::bad_cast &e)
            {
                // Convert the storage to device
                GetStorage<MemoryRegionDevice>();

                auto &ret =
                    dynamic_cast<MemoryRegionDevice<TData> &>(*m_storage);

                ret.template CopySRC<MemCopy>(src, size, offset);
            }
        }
    }

    // Member variables:
    std::unique_ptr<MemoryRegionHost<TData>> m_storage = nullptr;
};
