#include "MemoryRegionCUDA.hpp"
#include <cuda.h>

template<typename tData>
MemoryRegion<tData, BackendCUDA>::MemoryRegion(size_t n)
{
    m_host = new tData[n];
    cudaMalloc(&m_device, sizeof(tData) * n);
}

template MemoryRegion<double, BackendCUDA>::MemoryRegion(size_t n);

