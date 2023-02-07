#pragma once

#include <utility>

#include "MemoryRegionCPU.hpp"

template <typename TData> class MemoryRegionCUDA : public MemoryRegionCPU<TData>
{
public:
    MemoryRegionCUDA(const MemoryRegionCUDA<TData> &rhs) = delete;
    MemoryRegionCUDA(MemoryRegionCUDA &&rhs)
        : MemoryRegionCPU<TData>(std::move(rhs))
    {
        // C++ is less retarded
        m_device     = rhs.m_device;
        m_size       = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size   = 0;
    }
    MemoryRegionCUDA(size_t n);
    virtual ~MemoryRegionCUDA() override;

    void operator=(MemoryRegionCUDA &&rhs)
    {
        MemoryRegionCPU<TData>::operator=(std::move(rhs));
        m_device     = rhs.m_device;
        m_size       = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size   = 0;
    }

    virtual TData *GetCPUPtr() override
    {
        if (m_ondevice)
        {
            DeviceToHost();
        }

        return this->m_host;
    }

    TData *GetGPUPtr()
    {
        if (!m_ondevice)
        {
            HostToDevice();
        }

        return m_device;
    }

    void HostToDevice();
    void DeviceToHost();

    bool GetOnDevice() const
    {
        return m_ondevice;
    }

protected:
    TData *m_device = nullptr;
    size_t m_size   = 0;
    bool m_ondevice = false;
};
