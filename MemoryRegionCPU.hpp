#pragma once

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

    MemoryRegionCPU(MemoryRegionCPU &&rhs)
    {
        m_host     = rhs.m_host;
        m_size     = rhs.m_size;
        rhs.m_host = nullptr;
    }

    MemoryRegionCPU(size_t n)
    {
        m_host = new TData[n];
        m_size = n;
    }

    virtual ~MemoryRegionCPU()
    {
        if (m_host != nullptr)
        {
            delete[] m_host;
            m_host = nullptr;
        }
    }

    void operator=(MemoryRegionCPU &&rhs)
    {
        m_host     = rhs.m_host;
        m_size     = rhs.m_size;
        rhs.m_host = nullptr;
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
    TData *m_host = nullptr;
    size_t m_size = 0;
};
