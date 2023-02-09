#pragma once

#include <cstdio>

/**
 * @brief MemoryRegionCPU stores underlying data for Field on the CPU
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
     * @brief Get the pointer to the CPU memory. Virtual so that subclasses can
     * move memory to CPU from the device
     */
    virtual TData *GetCPUPtr()
    {
        return m_host;
    }

    /**
     * @brief Move memory to the CPU. Necessary for subclasses to implement.
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
