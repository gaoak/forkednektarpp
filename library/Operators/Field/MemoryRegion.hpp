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
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "Operators/Field/MemoryAlloc.hpp"

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

namespace Nektar::Operators
{

template <typename TData> class FieldBase;

class MemoryRegionBase
{
};

/**
 * @brief A MemoryRegion represents a memory region
 * @tparam TData  The floating-point representation used by the MemoryRegion.
 */
template <typename TData> class MemoryRegion : public MemoryRegionBase
{
    template <typename TDataField> friend class FieldBase;
    template <typename TDataField, FieldState TState> friend class Field;
    template <typename MemSpace, typename TDataField>
    friend void AllocateFieldStorage(FieldBase<TDataField> *field);

public:
    MemoryRegion() = default;

    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param name         - name
     * @param size         - size of memory
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - memory alignment
     */
    MemoryRegion(const std::string name, const size_t size,
                 const MemAllocType &memAllocType = ePageable,
                 const size_t alignment = NektarSpaces::memory_alignment::value)
    {
        m_allocated = true;

        m_host_owned   = true;
        m_device_owned = true;
        m_name         = name;
        m_size         = size;
        m_memAllocType = memAllocType;
        m_alignment    = alignment;

        m_host         = nullptr;
        m_device       = nullptr;
        m_host_valid   = false;
        m_device_valid = false;
    }

    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param size         - size of memory
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - memory alignment
     */
    MemoryRegion(const size_t size,
                 const MemAllocType &memAllocType = ePageable,
                 const size_t alignment = NektarSpaces::memory_alignment::value)
        : MemoryRegion<TData>("", size, memAllocType, alignment)
    {
    }

    /**
     * @brief Constructor methods - move from another MemoryRegion
     *
     * @param rhs - MemoryRegion to move from
     */
    MemoryRegion(MemoryRegion &&rhs)
        : m_allocated(rhs.m_allocated), m_host_owned(rhs.m_host_owned),
          m_device_owned(rhs.m_device_owned), m_host(rhs.m_host),
          m_device(rhs.m_device), m_size(rhs.m_size),
          m_alignment(rhs.m_alignment), m_host_valid(rhs.m_host_valid),
          m_device_valid(rhs.m_device_valid), m_name(rhs.m_name),
          m_memAllocType(rhs.m_memAllocType)
    {
        rhs.m_allocated    = false;
        rhs.m_host_owned   = true;
        rhs.m_device_owned = true;
        rhs.m_host         = nullptr;
        rhs.m_device       = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = NektarSpaces::memory_alignment::value;
        rhs.m_host_valid   = false;
        rhs.m_device_valid = false;
        rhs.m_name         = "";
        rhs.m_memAllocType = ePageable;
    }

    /**
     * @brief Destructor method.
     *
     */
    ~MemoryRegion()
    {
        if (m_device && m_device_owned)
        {
            deviceFree(m_device, m_size * sizeof(TData), m_alignment);
        }

        if (m_host && m_host_owned)
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

        m_allocated    = false;
        m_host_owned   = true;
        m_device_owned = true;
        m_host         = nullptr;
        m_device       = nullptr;
        m_size         = 0;
        m_alignment    = NektarSpaces::memory_alignment::value;
        m_host_valid   = false;
        m_device_valid = false;
        m_name         = "";
        m_memAllocType = ePageable;
    }

    /**
     * @brief Constructor methods - No copy methods
     *
     */
    MemoryRegion(const MemoryRegion &)            = delete;
    MemoryRegion &operator=(const MemoryRegion &) = delete;

    /**
     * @brief Move assignment operator.
     *
     * @param rhs - MemoryRegion to move from
     *
     * @return    - MemoryRegion&
     */
    MemoryRegion &operator=(MemoryRegion &&rhs)
    {
        m_allocated    = rhs.m_allocated;
        m_host_owned   = rhs.m_host_owned;
        m_device_owned = rhs.m_device_owned;
        m_host         = rhs.m_host;
        m_device       = rhs.m_device;
        m_size         = rhs.m_size;
        m_alignment    = rhs.m_alignment;
        m_host_valid   = rhs.m_host_valid;
        m_device_valid = rhs.m_device_valid;
        m_name         = rhs.m_name;
        m_memAllocType = rhs.m_memAllocType;

        rhs.m_allocated    = false;
        rhs.m_host_owned   = true;
        rhs.m_device_owned = true;
        rhs.m_host         = nullptr;
        rhs.m_device       = nullptr;
        rhs.m_size         = 0;
        rhs.m_alignment    = NektarSpaces::memory_alignment::value;
        rhs.m_host_valid   = false;
        rhs.m_device_valid = false;
        rhs.m_name         = "";
        rhs.m_memAllocType = ePageable;

        return *this;
    }

    /**
     * @brief Get the pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace, typename MemAccess>
    typename const_if<std::is_same_v<MemAccess, ReadOnly>, TData>::type *GetPtr()
    {
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetPtr - Storage is not allocated.");
        }

        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            if constexpr (std::is_same_v<MemAccess, ReadOnly>)
            {
                return GetReadOnlyHostPtr();
            }
            else if constexpr (std::is_same_v<MemAccess, WriteOnly>)
            {
                return GetWriteOnlyHostPtr();
            }
            else if constexpr (std::is_same_v<MemAccess, ReadWrite>)
            {
                return GetReadWriteHostPtr();
            }
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            if constexpr (std::is_same_v<MemAccess, ReadOnly>)
            {
                return GetReadOnlyDevicePtr();
            }
            else if constexpr (std::is_same_v<MemAccess, WriteOnly>)
            {
                return GetWriteOnlyDevicePtr();
            }
            else if constexpr (std::is_same_v<MemAccess, ReadWrite>)
            {
                return GetReadWriteDevicePtr();
            }
        }

        return nullptr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param name         - name of the memory region
     * @param array        - std::vector to copy from
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> FromVector(
        const std::string name, std::vector<TDataIn, Alloc> const &array,
        const MemAllocType &memAllocType = ePageable,
        const size_t alignment = NektarSpaces::memory_alignment::value)
    {
        auto mr =
            MemoryRegion<TData>(name, array.size(), memAllocType, alignment);
        mr.template CopyFromHostPtr<MemSpace>(array.data(), array.size());
        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a std::vector
     *
     * @param array        - std::vector to copy from
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    static MemoryRegion<TData> FromVector(
        std::vector<TDataIn, Alloc> const &array,
        const MemAllocType &memAllocType = ePageable,
        const size_t alignment = NektarSpaces::memory_alignment::value)
    {
        return MemoryRegion<TData>::template FromVector<MemSpace, TDataIn>(
            "", array, memAllocType, alignment);
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param name         - name of the memory region
     * @param array        - Nektar::Array to copy from
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> FromArray(
        const std::string name,
        Nektar::Array<Nektar::OneD, TDataIn> const &array,
        const MemAllocType &memAllocType = ePageable,
        const size_t alignment = NektarSpaces::memory_alignment::value)
    {
        auto mr =
            MemoryRegion<TData>(name, array.size(), memAllocType, alignment);
        mr.template CopyFromHostPtr<MemSpace>(array.data(), array.size());
        return mr;
    }

    /**
     * @brief Static templated creation method. This method creates a
     *        new MemoryRegion that copies data from an Nektar::Array
     *
     * @param array        - Nektar::Array to copy from
     * @param memAllocType - [ePageable, ePinned]
     * @param alignment    - Memory alignment to use.
     *
     * @return MemoryRegion<TData>
     */
    template <typename MemSpace, typename TDataIn>
    static MemoryRegion<TData> FromArray(
        Nektar::Array<Nektar::OneD, TDataIn> const &array,
        const MemAllocType &memAllocType = ePageable,
        const size_t alignment = NektarSpaces::memory_alignment::value)
    {
        return MemoryRegion<TData>::template FromArray<MemSpace, TDataIn>(
            "", array, memAllocType, alignment);
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val    - value to set
     * @param count  - number of values
     * @param offset - offset to m_host pointer
     */
    template <typename MemSpace>
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0)
    {
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::Initialize - Storage is not allocated.");
        }

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
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment);
                deviceMemset(m_device, 0, m_size * sizeof(TData));
            }

            auto size = (count == 0) ? m_size : count;

            TData *dst = m_device + offset;

            // If the value is zero, memset is the most efficent.
            if constexpr (std::is_floating_point_v<TData> ||
                          std::is_integral_v<TData>)
            {
                if (val == TData(0))
                {
                    deviceMemset(dst, 0, size * sizeof(TData));
                }
                // Nonzero value
                else
                {
                    deviceFill(dst, val, size);
                }
            }
            else
            {
                deviceFill(dst, val, size);
            }

            m_host_valid   = false;
            m_device_valid = true;
        }
    }

    /**
     * @brief Copy data from one region to another region.
     *
     * @param rhs - MemoryRegion to copy from
     */
    template <typename MemSpace> void Copy(MemoryRegion &rhs)
    {
        if (this->size() != rhs.size())
        {
            std::stringstream msg;

            msg << "MemoryRegion::Copy - "
                << "Memory size mismatch between (" << rhs.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        auto dst = this->template GetPtr<MemSpace, WriteOnly>();
        auto src = rhs.template GetPtr<MemSpace, ReadOnly>();
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            std::copy(src, src + this->size(), dst);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            deviceMemcpy<DeviceToDevice>(dst, src,
                                         this->size() * sizeof(TData));
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
              class Alloc = std::allocator<TDataIn>>
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

        this->CopyFromHostPtr<MemSpace>(array.data(), array.size());
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn>
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

        this->CopyFromHostPtr<MemSpace>(array.data(), array.size());
    }

    /**
     * @brief Copy data to a std:vector.
     *
     * @return std::vector<TDataOut>
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TDataOut>>
    std::vector<TDataOut, Alloc> ToVector()
    {
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::ToVector - Storage is not allocated.");
        }

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return std::vector<TDataOut, Alloc>(
                m_size,
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>());
        }
        else
        {
            std::vector<TDataOut, Alloc> vector(m_size);

            // Copy the data from the input field
            auto ptr =
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto vecPtr = vector.data();

            std::copy(ptr, ptr + m_size, vecPtr);

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
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::ToArray - Storage is not allocated.");
        }

        if constexpr (std::is_same_v<TDataOut, TData>)
        {
            return Nektar::Array<Nektar::OneD, TDataOut>(
                m_size,
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>());
        }
        else
        {
            Nektar::Array<Nektar::OneD, TDataOut> array(m_size);

            // Copy the data from the input field
            auto ptr =
                this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto arrPtr = array.data();

            std::copy(ptr, ptr + m_size, arrPtr);

            return array;
        }
    }

    /**
     * @brief Get the storage memory alignment.
     *
     * @return - size_t
     */
    size_t GetAlignment() const
    {
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetAlignment - Storage is not allocated.");
        }

        return m_alignment;
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
     * @brief Get the storage name.
     *
     */
    std::string GetName() const
    {
        if (!m_allocated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetName - Storage is not allocated.");
        }

        return m_name;
    }

    typedef TData value_type;

private:
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
                     "MemoryRegion::GetReadOnlyHostPtr - "
                     "attempt to access host memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetReadOnlyHostPtr - "
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
                     "MemoryRegion::GetReadWriteHostPtr - "
                     "attempt to access host data (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetReadWriteHostPtr - "
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
                     "MemoryRegion::GetReadOnlyDevicePtr - "
                     "attempt to access device memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetReadOnlyDevicePtr - "
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
            deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment);
            deviceMemset(m_device, 0, m_size * sizeof(TData));
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
                     "MemoryRegion::GetReadWriteDevicePtr - "
                     "attempt to access device memory (" +
                         m_name + ") without it being allocated.");
        }

        if (!m_host_valid && !m_device_valid)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::GetReadWriteDevicePtr - "
                     "attempt to get a host pointer (" +
                         m_name + ") before the data is initialized.");
        }

        HostToDeviceCopy(); // Move to device if necessary

        m_host_valid   = false;
        m_device_valid = true;

        return m_device;
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_host pointer
     */
    template <typename MemSpace, typename TDataIn>
    void CopyFromHostPtr(const TDataIn *src, const size_t size,
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

            if constexpr (std::is_same_v<TDataIn, TData>)
            {
                std::memcpy(dst, src, size * sizeof(TData));
            }
            else
            {
                std::vector<TData> tmp(size);
                std::copy(src, src + size, tmp.data());
                std::memcpy(dst, tmp.data(), size * sizeof(TData));
            }

            m_host_valid   = true;
            m_device_valid = false;
        }
        else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            if (!m_device)
            {
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment);
                deviceMemset(m_device, 0, m_size * sizeof(TData));
            }

            TData *dst = m_device + offset;

            if constexpr (std::is_same_v<TDataIn, TData>)
            {
                deviceMemcpy<HostToDevice>(dst, src, size * sizeof(TData));
            }
            else
            {
                std::vector<TData> tmp(size);
                std::copy(src, src + size, tmp.data());
                deviceMemcpy<HostToDevice>(dst, tmp.data(),
                                           size * sizeof(TData));
            }

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
                         "MemoryRegion::HostToDeviceCopy - attempt to "
                         "transfer data from the host (" +
                             m_name +
                             ") without any "
                             "valid host memory allocated.");
            }

            if (!m_device)
            {
                deviceMalloc(&m_device, m_size * sizeof(TData), m_alignment);
            }

            // Make sure the host data is valid. It might not be.
            if (m_host_valid)
            {
                deviceMemcpy<HostToDevice>(
                    m_device, m_host, m_size * sizeof(TData), m_memAllocType);
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegion::HostToDeviceCopy - attempt to "
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
                         "MemoryRegion::DeviceToHostCopy - attempt to "
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
                    m_host, m_device, m_size * sizeof(TData), m_memAllocType);
            }
            else
            {
                // Throw an error.
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegion::DeviceToHostCopy - attempt to "
                         "transfer data (" +
                             m_name +
                             ") to the host without any "
                             "valid device data.");
            }
        }
    }
    /**
     * @brief Initialize host pointer for an existing (external) pointer.
     * Specialized function used in Field.h to allocate a contiguous host memory
     * accross all MemoryRegion objects from a Field object. This function
     * should not be otherwise used.
     *
     * @param src - source pointer from Field
     *
     */
    void SetHostStorage(TData *src)
    {
        if (m_host)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::SetHostStorage - Host storage has already "
                     "been allocated");
        }

        m_host       = src;
        m_host_owned = false;
    }

    /**
     * @brief Initialize device pointer for an existing (external) pointer.
     * Specialized function used in Field.h to allocate a contiguous device
     * memory accross all MemoryRegion objects from a Field object. This
     * function should not be otherwise used.
     *
     * @param src - source pointer from Field
     *
     */
    void SetDeviceStorage(TData *src)
    {
        if (m_device)
        {
            // Throw an error.
            NEKERROR(
                Nektar::ErrorUtil::efatal,
                "MemoryRegion::SetDeviceStorage - Device storage has already "
                "been allocated");
        }

        m_device       = src;
        m_device_owned = false;
    }

    // Member variables:
    bool m_allocated  = false;
    bool m_host_owned = true;   // Flag indicating if the host pointer is owned
                                // by the current object.
    bool m_device_owned = true; // Flag indicating if the device pointer is
                                // owned by the current object.
    TData *m_host      = nullptr; /// < Host memory pointer
    TData *m_device    = nullptr; ///< Device memory pointer
    size_t m_size      = 0;
    size_t m_alignment = NektarSpaces::memory_alignment ::value;

    bool m_host_valid   = false; // Flag indicating that the host data is valid
    bool m_device_valid = false; ///< Flag indicating the device data is valid
    std::string m_name{""};
    MemAllocType m_memAllocType{ePageable};
};

} // namespace Nektar::Operators
