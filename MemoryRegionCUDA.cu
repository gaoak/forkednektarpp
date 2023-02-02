#include <cuda.h>
#include <iostream>

#include "MemoryRegionCUDA.hpp"

template<typename tData>
MemoryRegion<tData, BackendCUDA>::MemoryRegion(size_t n)
    : MemoryRegion<tData, BackendCPU>(n)
{
    //m_host = new tData[n];
    cudaMalloc((void **)&m_device, sizeof(tData) * n);
    m_size = n;
    std::cout << "HEllo" << std::endl;
}

template<typename tData>
MemoryRegion<tData, BackendCUDA>::~MemoryRegion() {
    if (m_device != nullptr) {
        cudaFree(m_device);
        m_device = nullptr; // lol
    }
}

template<typename tData>
void MemoryRegion<tData, BackendCUDA>::HostToDevice()
{
    cudaMemcpy(m_device, this->m_host, m_size*sizeof(tData), cudaMemcpyHostToDevice);
    m_ondevice = true;
}

template<typename tData>
void MemoryRegion<tData, BackendCUDA>::DeviceToHost()
{
    cudaMemcpy(this->m_host, m_device, m_size*sizeof(tData), cudaMemcpyDeviceToHost);
    m_ondevice = false;
}

template MemoryRegion<double, BackendCUDA>::MemoryRegion(size_t n);
template MemoryRegion<double, BackendCUDA>::~MemoryRegion();
template void MemoryRegion<double, BackendCUDA>::HostToDevice();
template void MemoryRegion<double, BackendCUDA>::DeviceToHost();

