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
        rhs.m_host = nullptr;
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
        rhs.m_host = nullptr;
    }

    double *m_host = nullptr;
    double *m_device = nullptr;
    bool m_ondevice = false;

    double *GetPtr()
    {
        return m_ondevice ? m_device : m_host;
    }

    void HostToDevice()
    {
        // whatever
        m_ondevice = true;
    }

    void DeviceToHost()
    {
        // inverse whatever
        m_ondevice = false;
    }

    bool GetOnDevice() const
    {
        return m_ondevice;
    }
};

