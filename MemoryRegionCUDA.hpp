#pragma once

#include <utility>

#include "MemoryRegionCPU.hpp"

/**
 * @brief Memory backend for CUDA devices
 * @tparam TData Floating point datatype
 *
 * MemoryRegionCUDA represents and manages the memory stored on a CUDA device.
 * This class also manages access to the CPU part
 * of the memory by inheriting from MemoryRegionCPU.
 */
template <typename TData> class MemoryRegionCUDA : public MemoryRegionCPU<TData>
{
public:
    MemoryRegionCUDA(const MemoryRegionCUDA<TData> &rhs) = delete;
    MemoryRegionCUDA(MemoryRegionCUDA &&rhs)
        : MemoryRegionCPU<TData>(std::move(rhs))
    {
        m_device     = rhs.m_device;
        m_size       = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size   = 0;
    }

    MemoryRegionCUDA(size_t n);
    virtual ~MemoryRegionCUDA() override;

    /**
     * @brief Create MemoryRegionCUDA from MemoryRegionCPU r-value
     *
     * This method allows for a Field to construct a new MemoryRegionCUDA from a
     * MemoryRegion of any other type, through the MemoryRegionCPU base class
     */
    static MemoryRegionCUDA<TData> fromCPU(MemoryRegionCPU<TData> &&cpu)
    {
        return MemoryRegionCUDA<TData>(std::move(cpu));
    }

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
            DeviceToHost(); // Move to CPU if necessary
        }

        return this->m_host;
    }

    virtual void ToCPU() override
    {
        DeviceToHost();
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
    MemoryRegionCUDA<TData>(MemoryRegionCPU<TData> &&cpu)
        : MemoryRegionCPU<TData>(std::move(cpu))
    {
        initFromSize(cpu.size());
    }

private:
    void initFromSize(size_t n);

    TData *m_device = nullptr;      ///< Device memory pointer
    size_t m_size   = 0;            ///< Device storage size
    bool m_ondevice = false;        ///< Flag indicating if data is on device
};
