#pragma once
#include "MemoryRegion.hpp"

template <typename tData> class MemoryRegion<tData, BackendCPU>
{
public:
    MemoryRegion()                        = default;
    MemoryRegion(const MemoryRegion &rhs) = delete;

    MemoryRegion(MemoryRegion &&rhs)
    {
        m_host     = rhs.m_host;
        m_size     = rhs.m_size;
        rhs.m_host = nullptr;
    }

    MemoryRegion(size_t n)
    {
        m_host = new tData[n];
        m_size = n;
    }

    virtual ~MemoryRegion()
    {
        if (m_host != nullptr)
        {
            delete[] m_host;
            m_host = nullptr;
        }
    }

    void operator=(MemoryRegion &&rhs)
    {
        m_host     = rhs.m_host;
        m_size     = rhs.m_size;
        rhs.m_host = nullptr;
    }

    virtual double *GetPtr()
    {
        return m_host;
    }

    size_t size() {
        return m_size;
    }

protected:
    double *m_host = nullptr;
    size_t m_size  = 0;
};
