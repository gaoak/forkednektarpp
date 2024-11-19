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

template <typename TData> class MemoryRegion;

/**
 * @brief Acts as a holder for a contiguous block of memory, allocated on the
 * host system.
 *
 * This class also acts as a base class for device-aware builds.
 */
template <typename TData> class MemoryRegionHost
{
    friend class MemoryRegion<TData>;

public:
    /**
     * @brief Constructor methods - create a new memory region.
     *
     * @param size        - size of memory
     * @param alignment   - memory alignment
     * @param device_only - flag to only allocated memory on device
     */
    MemoryRegionHost(const std::string name, const size_t size,
                     const size_t alignment, const size_t device_rank,
                     const bool device_only)
    {
        m_owned       = true;
        m_size        = size;
        m_alignment   = alignment;
        m_initialize  = true;
        m_device_rank = device_rank;
        m_device_only = device_only;
        m_name        = name;

        if (!m_device_only)
        {
            m_host = static_cast<TData *>(::operator new[](
                m_size * sizeof(TData), std::align_val_t(m_alignment)));

            m_host_valid = false;
        }
    }

    /**
     * @brief Constructor methods - create a new memory region from an
     * existing host pointer. Specialized constructor method used by
     * Field.hpp to allocate a contiguous host memory coupled with
     * distributed device memory.
     *
     * @param h_src       - host src pointer
     * @param size        - size of memory
     * @param alignment   - memory alignment
     */
    MemoryRegionHost(const std::string name, TData *h_src, const size_t size,
                     const size_t alignment, const size_t device_rank)
    {
        m_owned       = false;
        m_size        = size;
        m_alignment   = alignment;
        m_initialize  = true;
        m_device_rank = device_rank;
        m_device_only = false;
        m_name        = name;

        m_host       = h_src;
        m_host_valid = false;
    }

    /**
     * @brief Constructor methods - move from another MemoryRegionHost
     *
     * @param rhs - MemoryRegionHost to move from
     */
    MemoryRegionHost(MemoryRegionHost &&rhs)
        : m_owned(rhs.m_owned), m_host(rhs.m_host), m_size(rhs.m_size),
          m_alignment(rhs.m_alignment), m_host_valid(rhs.m_host_valid),
          m_initialize(rhs.m_initialize), m_device_rank(rhs.m_device_rank),
          m_device_only(rhs.m_device_only), m_name(rhs.m_name)
    {
        rhs.m_owned       = true;
        rhs.m_host        = nullptr;
        rhs.m_size        = 0;
        rhs.m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid  = false;
        rhs.m_initialize  = true;
        rhs.m_device_rank = 0;
        rhs.m_device_only = false;
        rhs.m_name        = "";
    }

    /**
     * @brief Destructor method.
     *
     */
    virtual ~MemoryRegionHost()
    {
        if (m_host && m_owned)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
        }

        m_owned       = true;
        m_host        = nullptr;
        m_size        = 0;
        m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        m_host_valid  = false;
        m_initialize  = true;
        m_device_rank = 0;
        m_device_only = false;
        m_name        = "";
    }

protected:
    /**
     * @brief Constructor methods - no base, copy methods
     *
     */
    MemoryRegionHost()                                       = delete;
    MemoryRegionHost(const MemoryRegionHost &rhs)            = delete;
    MemoryRegionHost &operator=(const MemoryRegionHost &rhs) = delete;

    /**
     * @brief Move operator
     *
     * @param rhs - MemoryRegionHost to move from
     *
     * @return    - MemoryRegionHost<TData>&
     */
    MemoryRegionHost &operator=(MemoryRegionHost &&rhs)
    {
        m_owned       = rhs.m_owned;
        m_host        = rhs.m_host;
        m_size        = rhs.m_size;
        m_alignment   = rhs.m_alignment;
        m_host_valid  = rhs.m_host_valid;
        m_initialize  = rhs.m_initialize;
        m_device_rank = rhs.m_device_rank;
        m_device_only = rhs.m_device_only;
        m_name        = rhs.m_name;

        rhs.m_owned       = true;
        rhs.m_host        = nullptr;
        rhs.m_size        = 0;
        rhs.m_alignment   = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_host_valid  = false;
        rhs.m_initialize  = true;
        rhs.m_device_rank = 0;
        rhs.m_device_only = false;
        rhs.m_name        = "";

        return *this;
    }

    /**
     * @brief osstream operator.
     *
     * @param rhs - MemoryRegion to stream
     *
     * @return    - stream
     */
    friend auto operator<<(std::ostream &os, const MemoryRegionHost &mr)
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

        return os << msg.str();
    }

    /**
     * @brief Get the pointer to the host memory
     *
     * @return - TData*
     *
     */
    template <typename MemAccess>
    typename const_if<std::is_same_v<MemAccess, ReadOnly>, TData>::type *
    GetHostPtr()
    {
        if (m_host == nullptr)
        {
            // Throw an error.
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegionHost::GetHostPtr - "
                     "attempt to access host data (" +
                         m_name + ") without it being allocated.");
        }

        if constexpr (std::is_same_v<MemAccess, ReadOnly> ||
                      std::is_same_v<MemAccess, ReadWrite>)
        {
            if (m_initialize)
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "MemoryRegionHost::GetHostPtr - "
                         "attempt to get a const host pointer (" +
                             m_name + ") before the data is initialized.");
            }
        }

        if constexpr (std::is_same_v<MemAccess, WriteOnly> ||
                      std::is_same_v<MemAccess, ReadWrite>)
        {
            m_host_valid = true;
            m_initialize = false;
        }

        return m_host;
    }

    /**
     * @brief Initialize the storage memory.
     *
     * @param val    - value to set
     * @param count  - number of values
     * @param offset - offset to m_host pointer
     *
     */
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0)
    {
        if (m_host)
        {
            auto size = (count == 0) ? m_size : count;

            TData *dst = m_host + offset;

            // If the value is zero, memset is the most efficent.
            if (val == TData(0))
            {
                std::memset(dst, 0, size * sizeof(TData));
            }
            // Otherwise use the fill function.
            else
            {
                std::fill(dst, dst + size, val);
            }

            m_host_valid = true;
            m_initialize = false;
        }
    }

    /**
     * @brief Templated copy method.
     *
     * @param src    - pointer data type TDataIn to copy from
     * @param size   - number of element of type TDataIn to copy
     * @param offset - offset to m_host pointer
     */
    template <typename TDataIn>
    void CopyFromHostPtr(const TDataIn *src, const size_t size,
                         const size_t offset = 0)
    {
        if (!m_device_only)
        {
            TData *dst = m_host + offset;

            if constexpr (std::is_same_v<TDataIn, TData>)
            {
                std::memcpy(dst, src, size * sizeof(TData));
            }
            else
            {
                std::copy(src, src + size, dst);
            }

            m_host_valid = true;
            m_initialize = false;
        }
    }

    // Member variables:
    bool m_owned = true; // Flag indicating if the host pointer is owned
                         // by the current object.
    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    bool m_host_valid = false;    // Flag indicating that the host data is valid
    bool m_initialize = true;     // Flag indicating that the data needs
                                  // to be initialize and is needed for
                                  // host device transfers.
    size_t m_device_rank = 0;     // Index indicating device ID.
    bool m_device_only   = false; // Flag indicating that the data is only
                                  // initialized on the device.
    std::string m_name{""};
};
