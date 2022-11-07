#pragma once
#include "MemoryRegion.hpp"

template<typename tData>
class MemoryRegion<tData, BackendCUDA>
{
public:
    MemoryRegion() = default;
    MemoryRegion(const MemoryRegion &rhs) = delete;
    MemoryRegion(MemoryRegion &&rhs) {
        m_host = rhs.m_host;
        m_device = rhs.m_device;
        m_size = rhs.m_size;
        rhs.m_host = nullptr;
        rhs.m_device = nullptr;
        rhs.m_size = 0;
    }
    MemoryRegion(size_t n);
    ~MemoryRegion()
    {
/*
        if (m_host != nullptr)
        {
            delete [] m_host;
            m_host = nullptr;
        }
*/
    }

    void operator=(MemoryRegion &&rhs) {
        m_host = rhs.m_host;
        m_device = rhs.m_device;
        m_size = rhs.m_size;
        rhs.m_host = nullptr;
        rhs.m_device = nullptr;
        rhs.m_size = 0;
    }

    double *m_host = nullptr;
    double *m_device = nullptr;
    size_t m_size = 0;
    bool m_ondevice = false;

    double *GetPtr()
    {
        return m_ondevice ? m_device : m_host;
    }

    void HostToDevice();
    void DeviceToHost();

    bool GetOnDevice() const
    {
        return m_ondevice;
    }
};

