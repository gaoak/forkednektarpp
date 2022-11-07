#include "MemoryRegionCUDA.hpp"
#include <cuda.h>
#include <iostream>
template<typename tData>
MemoryRegion<tData, BackendCUDA>::MemoryRegion(size_t n)
{
    std::cout << "malloc " << n << std::endl;
    m_host = new tData[n];
    cudaMalloc((void **)&m_device, sizeof(tData) * n);
    m_size = n;
}

template<typename tData>
void MemoryRegion<tData, BackendCUDA>::HostToDevice()
{
    std::cout << "Lol" << std::endl;
    cudaMemcpy(m_device, m_host, m_size*sizeof(tData), cudaMemcpyHostToDevice);
    m_ondevice = true;
}

template<typename tData>
void MemoryRegion<tData, BackendCUDA>::DeviceToHost()
{
    std::cout << "Lol2" << std::endl;
    cudaMemcpy(m_host, m_device, m_size*sizeof(tData), cudaMemcpyDeviceToHost);
    m_ondevice = false;
}

template MemoryRegion<double, BackendCUDA>::MemoryRegion(size_t n);
template void MemoryRegion<double, BackendCUDA>::HostToDevice();
template void MemoryRegion<double, BackendCUDA>::DeviceToHost();

