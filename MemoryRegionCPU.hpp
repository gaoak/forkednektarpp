#pragma once
#include "MemoryRegion.hpp"

template<typename tData>
class MemoryRegion<tData, BackendCPU>
{
public:
    MemoryRegion() = default;
    MemoryRegion(const MemoryRegion &rhs) = delete;
    MemoryRegion(MemoryRegion &&rhs) {
        m_host = rhs.m_host;
        rhs.m_host = nullptr;
    }
    MemoryRegion(size_t n)
    {
        m_host = new tData[n];
    }
    ~MemoryRegion()
    {
        if (m_host != nullptr)
        {
            delete [] m_host;
            m_host = nullptr;
        }
    }

    void operator=(MemoryRegion &&rhs) {
        m_host = rhs.m_host;
        rhs.m_host = nullptr;
    }

    double *m_host = nullptr;

    double *GetPtr()
    {
        return m_host;
    }
};
