#pragma once

#include <new>

/**
 * @brief Stores underlying data for a Field on the CPU.
 *
 * It acts as a holder for a contiguous block of memory, allocated on the
 * host system.
 *
 * This class also acts as a base class for device-aware builds.
 */
template <typename TData> class MemoryRegionCPU
{
public:
    MemoryRegionCPU()                           = delete;
    MemoryRegionCPU(const MemoryRegionCPU &rhs) = delete;
    MemoryRegionCPU &operator=(const MemoryRegionCPU &rhs) = delete;

    MemoryRegionCPU(MemoryRegionCPU &&rhs)
    {
        m_host      = rhs.m_host;
        m_size      = rhs.m_size;
        m_alignment = rhs.m_alignment;
        rhs.m_host  = nullptr;
    }

    MemoryRegionCPU(size_t n,
                    size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        // C++17 aligned new
        m_host      = new (std::align_val_t(alignment)) TData[n];
        m_alignment = alignment;
        m_size      = n;
    }

    virtual ~MemoryRegionCPU()
    {
        if (m_host != nullptr)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
            m_host = nullptr;
        }
    }

    MemoryRegionCPU &operator=(MemoryRegionCPU &&rhs)
    {
        if (m_host)
            operator delete[](m_host, std::align_val_t(m_alignment));

        m_host      = rhs.m_host;
        m_size      = rhs.m_size;
        m_alignment = rhs.m_alignment;
        rhs.m_host  = nullptr;

        return *this;
    }

    /**
     * @brief Get the pointer to the CPU memory.
     *
     * This is a virtual function so that subclasses can move memory to the
     * CPU from a device if needed.
     */
    virtual TData *GetCPUPtr()
    {
        return m_host;
    }

    /**
     * @brief Move memory to the CPU.
     *
     * This is a virtual function so that subclasses can move memory to the
     * CPU from a device if needed.
     */
    virtual void ToCPU()
    {
    }

    size_t size()
    {
        return m_size;
    }

protected:
    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};
