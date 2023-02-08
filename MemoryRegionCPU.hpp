#pragma once

#include <cstdio>

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

    virtual TData *GetCPUPtr()
    {
        return m_host;
    }

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
