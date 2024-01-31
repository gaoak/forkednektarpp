#include <cuda.h>
#include <iostream>

#include "MemoryRegionCUDA.hpp"

template <typename TData>
MemoryRegionCUDA<TData>::MemoryRegionCUDA(size_t n, size_t alignment)
    : MemoryRegionCPU<TData>(n, alignment)
{
    initFromSize(n);
}

template <typename TData> void MemoryRegionCUDA<TData>::initFromSize(size_t n)
{
    cudaMalloc((void **)&m_device, sizeof(TData) * n);
    m_size = n;
}

template <typename TData> MemoryRegionCUDA<TData>::~MemoryRegionCUDA()
{
    if (m_device != nullptr)
    {
        cudaFree(m_device);
        m_device = nullptr;
    }
}

template <typename TData> void MemoryRegionCUDA<TData>::HostToDevice()
{
    cudaMemcpy(m_device, this->m_host, m_size * sizeof(TData),
               cudaMemcpyHostToDevice);
    m_ondevice = true;
}

template <typename TData> void MemoryRegionCUDA<TData>::DeviceToHost()
{
    cudaMemcpy(this->m_host, m_device, m_size * sizeof(TData),
               cudaMemcpyDeviceToHost);
    m_ondevice = false;
}

template class MemoryRegionCUDA<double>;
