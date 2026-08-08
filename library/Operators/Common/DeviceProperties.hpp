///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceProperties.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Operators/Common/Spaces.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <limits>
#include <unordered_map>

#if defined(NEKTAR_ENABLE_CUDA)
class GetDeviceProperties
{
public:
    static const size_t &SharedMemoryPerBlock(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].sharedMemPerBlock;
    }

    static const size_t &SharedMemoryPerMultiprocessor(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].sharedMemPerMultiprocessor;
    }

    static size_t &TotalGlobalMemory(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].totalGlobalMem;
    }

    static const int &NumMultiProcessors(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].multiProcessorCount;
    }

    static const int &MaxThreadsPerMultiprocessor(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].maxThreadsPerMultiProcessor;
    }

    static void CheckSharedMemoryUsage(const size_t shmemsize)
    {
        ASSERTL0(
            shmemsize == 0 ||
                shmemsize <= GetDeviceProperties::SharedMemoryPerBlock(),
            "Shared memory available is " +
                std::to_string(GetDeviceProperties::SharedMemoryPerBlock()) +
                " bytes, requested " + std::to_string(shmemsize) + " bytes");
    }

    static void CheckGlobalMemoryUsage(const size_t memsize)
    {
        ASSERTL0(memsize <= GetDeviceProperties::TotalGlobalMemory(),
                 "Insufficient global memory, requested " +
                     std::to_string(memsize) + " bytes, remains " +
                     std::to_string(GetDeviceProperties::TotalGlobalMemory()) +
                     " bytes");
    }

private:
    static std::unordered_map<unsigned int, cudaDeviceProp> *prop;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (prop == nullptr)
        {
            prop = new std::unordered_map<unsigned int, cudaDeviceProp>;
        }

        if (prop->find(id) == prop->end())
        {
            prop->emplace(id, cudaDeviceProp{});
            CHECK_HIPCUDA_ERROR(cudaGetDeviceProperties(&(*prop)[id], id));
        }
    }
};

[[maybe_unused]] static void PrintDeviceProperties()
{
    cudaDeviceProp prop;
    auto id = Nektar::nekGetDevice();
    CHECK_HIPCUDA_ERROR(cudaGetDeviceProperties(&prop, id));
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
#if CUDART_VERSION >= 13000
    int memoryClockRate;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, id);
    printf("  Memory Clock Rate (KHz): %d\n", memoryClockRate);
#else
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
#endif
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total Global Memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("  Shared Memory per Block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("  Shared Memory per Multiprocessor (bytes): %ld\n",
           prop.sharedMemPerMultiprocessor);
#if CUDART_VERSION >= 13000
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
#else
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
#endif
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per Multiprocessor: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
}

#elif defined(NEKTAR_ENABLE_HIP)
class GetDeviceProperties
{
public:
    static const size_t &SharedMemoryPerBlock(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].sharedMemPerBlock;
    }

    static const size_t &SharedMemoryPerMultiprocessor(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].sharedMemPerMultiprocessor;
    }

    static size_t &TotalGlobalMemory(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].totalGlobalMem;
    }

    static const int &NumMultiProcessors(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].multiProcessorCount;
    }

    static const int &MaxThreadsPerMultiprocessor(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return (*prop)[id].maxThreadsPerMultiProcessor;
    }

    static void CheckSharedMemoryUsage(const size_t shmemsize)
    {
        ASSERTL0(
            shmemsize == 0 ||
                shmemsize <= GetDeviceProperties::SharedMemoryPerBlock(),
            "Shared memory available is " +
                std::to_string(GetDeviceProperties::SharedMemoryPerBlock()) +
                " bytes, requested " + std::to_string(shmemsize) + " bytes");
    }

    static void CheckGlobalMemoryUsage(const size_t memsize)
    {
        ASSERTL0(memsize <= GetDeviceProperties::TotalGlobalMemory(),
                 "Insufficient global memory, requested " +
                     std::to_string(memsize) + " bytes, remains " +
                     std::to_string(GetDeviceProperties::TotalGlobalMemory()) +
                     " bytes");
    }

private:
    static std::unordered_map<unsigned int, hipDeviceProp_t> *prop;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (prop == nullptr)
        {
            prop = new std::unordered_map<unsigned int, cudaDeviceProp>;
        }

        if (prop->find(id) == prop->end())
        {
            prop->emplace(id, hipDeviceProp_t{});
            CHECK_HIPCUDA_ERROR(hipGetDeviceProperties(&(*prop)[id], id));
        }
    }
};

[[maybe_unused]] static void PrintDeviceProperties()
{
    hipDeviceProp_t prop;
    auto id = Nektar::nekGetDevice();
    CHECK_HIPCUDA_ERROR(hipGetDeviceProperties(&prop, id));
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total Global Memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("  Shared Memory per Block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("  Shared Memory per Multiprocessor (bytes): %ld\n",
           prop.sharedMemPerMultiprocessor);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per Multiprocessor: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
}

#elif defined(NEKTAR_ENABLE_SYCL)
class GetDeviceProperties
{
public:
    static const size_t &SharedMemoryPerBlock(void)
    {
        FetchDeviceProperties(internalSYCLDeviceId);
        return m_sharedMemoryPerBlock[internalSYCLDeviceId];
    }

    static const size_t &SharedMemoryPerMultiprocessor(void)
    {
        FetchDeviceProperties(internalSYCLDeviceId);
        return m_sharedMemoryPerMultiprocessor[internalSYCLDeviceId];
    }

    static size_t &TotalGlobalMemory(void)
    {
        FetchDeviceProperties(internalSYCLDeviceId);
        return m_totalGlobalMemory[internalSYCLDeviceId];
    }

    static const unsigned &NumMultiProcessors(void)
    {
        FetchDeviceProperties(internalSYCLDeviceId);
        return m_numMultiProcessors[internalSYCLDeviceId];
    }

    static const size_t &MaxThreadsPerMultiprocessor(void)
    {
        FetchDeviceProperties(internalSYCLDeviceId);
        return m_maxThreadsPerMultiprocessor[internalSYCLDeviceId];
    }

    static void CheckSharedMemoryUsage(const size_t shmemsize)
    {
        ASSERTL0(
            shmemsize == 0 ||
                shmemsize <= GetDeviceProperties::SharedMemoryPerBlock(),
            "Shared memory available is " +
                std::to_string(GetDeviceProperties::SharedMemoryPerBlock()) +
                " bytes, requested " + std::to_string(shmemsize) + " bytes");
    }

    static void CheckGlobalMemoryUsage(const size_t memsize)
    {
        ASSERTL0(memsize <= GetDeviceProperties::TotalGlobalMemory(),
                 "Insufficient global memory, requested " +
                     std::to_string(memsize) + " bytes, remains " +
                     std::to_string(GetDeviceProperties::TotalGlobalMemory()) +
                     " bytes");
    }

private:
    static std::unordered_map<unsigned int, size_t> m_sharedMemoryPerBlock;
    static std::unordered_map<unsigned int, size_t>
        m_sharedMemoryPerMultiprocessor;
    static std::unordered_map<unsigned int, size_t> m_totalGlobalMemory;
    static std::unordered_map<unsigned int, unsigned int> m_numMultiProcessors;
    static std::unordered_map<unsigned int, size_t>
        m_maxThreadsPerMultiprocessor;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (m_totalGlobalMemory.find(id) == m_totalGlobalMemory.end())
        {
            auto device =
                SYCLQueue::GetInstance(0).get_info<sycl::info::queue::device>();
            m_totalGlobalMemory.emplace(
                id, device.get_info<sycl::info::device::global_mem_size>());
            m_sharedMemoryPerBlock.emplace(
                id, device.get_info<sycl::info::device::local_mem_size>());
            m_sharedMemoryPerMultiprocessor.emplace(
                id, device.get_info<sycl::info::device::local_mem_size>());
            m_numMultiProcessors.emplace(
                id, device.get_info<sycl::info::device::max_compute_units>());
            m_maxThreadsPerMultiprocessor.emplace(
                id, device.get_info<sycl::info::device::max_work_group_size>());
        }
    }
};

[[maybe_unused]] static void PrintDeviceProperties()
{
#if defined(NEKTAR_ENABLE_SYCL) && !defined(SYCL_ENABLE_CPU) &&                \
    defined(__INTEL_LLVM_COMPILER)
    auto device =
        SYCLQueue::GetInstance(0).get_info<sycl::info::queue::device>();
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n",
           device.get_info<sycl::info::device::name>().c_str());
    printf(
        "  Memory Clock Rate (KHz): %d\n",
        1000 *
            device
                .get_info<sycl::ext::intel::info::device::memory_clock_rate>());
    printf("  Memory Bus Width (bits): %d\n",
           device.get_info<sycl::ext::intel::info::device::memory_bus_width>());
    printf("  Total Global Memory (bytes): %ld\n",
           device.get_info<sycl::info::device::global_mem_size>());
    printf("  Shared Memory per Block (bytes): %ld\n",
           device.get_info<sycl::info::device::local_mem_size>());
    printf(
        "  Peak Memory Bandwidth (GB/s): %f\n",
        2.0 *
            device
                .get_info<sycl::ext::intel::info::device::memory_clock_rate>() *
            (device
                 .get_info<sycl::ext::intel::info::device::memory_bus_width>() /
             8) /
            1.0e3);
    printf("  Number of multiprocessors: %d\n",
           device.get_info<sycl::info::device::max_compute_units>());
#elif defined(NEKTAR_ENABLE_SYCL)
    auto device =
        SYCLQueue::GetInstance(0).get_info<sycl::info::queue::device>();
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n",
           device.get_info<sycl::info::device::name>().c_str());
    printf(
        "  Total Global Memory (bytes): %ld\n",
        (long unsigned)device.get_info<sycl::info::device::global_mem_size>());
    printf(
        "  Shared Memory per Block (bytes): %ld\n",
        (long unsigned)device.get_info<sycl::info::device::local_mem_size>());
    printf("  Number of multiprocessors: %d\n",
           device.get_info<sycl::info::device::max_compute_units>());
#endif
}

#elif defined(NEKTAR_ENABLE_DEVICEONHOST)
class GetDeviceProperties
{
public:
    static size_t SharedMemoryPerBlock(void)
    {
        return std::numeric_limits<size_t>::max();
    }

    static size_t SharedMemoryPerMultiprocessor(void)
    {
        return std::numeric_limits<size_t>::max();
    }

    static size_t TotalGlobalMemory(void)
    {
        return std::numeric_limits<size_t>::max();
    }

    static int NumMultiProcessors(void)
    {
        return 0;
    }

    static size_t MaxThreadsPerMultiprocessor(void)
    {
        return std::numeric_limits<size_t>::max();
    }

    static void CheckSharedMemoryUsage([[maybe_unused]] const size_t shmemsize)
    {
    }

    static void CheckGlobalMemoryUsage([[maybe_unused]] const size_t memsize)
    {
    }

private:
};

[[maybe_unused]] static void PrintDeviceProperties()
{
    std::cout << "PrintDeviceProperties: No device found" << std::endl;
}

#else

[[maybe_unused]] static void PrintDeviceProperties()
{
    std::cout << "PrintDeviceProperties: No device found" << std::endl;
}

#endif
