#pragma once
#include "MemoryRegion.hpp"
#include "MemoryRegionCPU.hpp"

template<typename tData>
class MemoryRegion<tData, BackendCUDA> : 
        public MemoryRegion<tData, BackendCPU>
{
public:
    MemoryRegion() : MemoryRegion<tData, BackendCPU>() {}
    MemoryRegion(const MemoryRegion &rhs) = delete;
    MemoryRegion(MemoryRegion &&rhs) : MemoryRegion<tData, BackendCPU>(rhs) {
        // C++ is less retarded
        m_device = rhs.m_device;
        m_size = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size = 0;
    }
    MemoryRegion(size_t n);
    virtual ~MemoryRegion() override;

    void operator=(MemoryRegion &&rhs) {
        MemoryRegion<tData, BackendCPU>::operator=(std::move(rhs));
        m_device = rhs.m_device;
        m_size = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size = 0;
    }

    virtual double *GetPtr() override
    {
        return m_ondevice ? m_device : this->m_host;
    }

    void HostToDevice();
    void DeviceToHost();

    bool GetOnDevice() const
    {
        return m_ondevice;
    }

protected:
    // double *m_host = nullptr;
    double *m_device = nullptr;
    size_t m_size = 0;
    bool m_ondevice = false;

};

