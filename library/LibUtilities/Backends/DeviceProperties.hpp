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

#include <limits>
#include <string>
#include <unordered_map>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

namespace Nektar
{

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

    /// Architecture id of the current device, "sm_" and its compute
    /// capability (e.g. "sm_90").
    static std::string Architecture(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return "sm_" + std::to_string((*prop)[id].major) +
               std::to_string((*prop)[id].minor);
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

    /// Architecture id of the current device, its gcnArchName without the
    /// feature flags (e.g. "gfx90a").
    static std::string Architecture(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        const std::string name = (*prop)[id].gcnArchName;
        return name.substr(0, name.find(':'));
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
            prop = new std::unordered_map<unsigned int, hipDeviceProp_t>;
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

#else

[[maybe_unused]] static void PrintDeviceProperties()
{
    std::cout << "PrintDeviceProperties: No device found" << std::endl;
}

#endif

/**
 * @brief Return the architecture id of the device the calling rank uses,
 * as the default implementation table (MultiRegions DefaultOpImplTable.hpp)
 * names it: "sm_" and the compute capability on CUDA (e.g. "sm_90"), the
 * gcnArchName without its feature flags on HIP (e.g. "gfx90a").
 *
 * It is empty without a CUDA or HIP device: SYCL reports no portable
 * architecture id, so a SYCL build, like a host-only one, takes the table's
 * rows for every architecture.
 */
[[maybe_unused]] static std::string GetDeviceArchitecture()
{
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
    return GetDeviceProperties::Architecture();
#else
    return "";
#endif
}

/**
 * @brief Limits of a device architecture that the launch heuristics read,
 * in the units GetDeviceProperties reports them.
 */
struct DeviceArchProperties
{
    /// Whether the build's target architecture is listed.
    bool known                           = false;
    size_t sharedMemoryPerBlock          = 0;
    size_t sharedMemoryPerMultiprocessor = 0;
    size_t maxThreadsPerMultiprocessor   = 0;
};

/**
 * @brief Limits of the build's target architecture, known at compile time.
 *
 * CUDA and HIP builds, and SYCL-CUDA and SYCL-HIP builds given one, target
 * the single architecture NEKTAR_DEVICE_ARCH, whose limits CMake
 * (NektarDeviceArchLimits.cmake) defines for host and device code alike. An
 * architecture it does not list, and the other back-ends, are not known.
 * CheckDeviceArchProperties() verifies the limits against the device.
 */
constexpr DeviceArchProperties GetDeviceArchProperties()
{
#if defined(NEKTAR_DEVICE_SHARED_MEMORY_PER_BLOCK)
    return DeviceArchProperties{true, NEKTAR_DEVICE_SHARED_MEMORY_PER_BLOCK,
                                NEKTAR_DEVICE_SHARED_MEMORY_PER_MP,
                                NEKTAR_DEVICE_MAX_THREADS_PER_MP};
#else
    return DeviceArchProperties{};
#endif
}

/**
 * @brief Check, once, that the device has the limits
 * GetDeviceArchProperties() lists for the build's target architecture.
 *
 * A launch that relies on more shared memory per block than the device has
 * fails, so less of it is fatal; the per-multiprocessor limits only steer the
 * SumFac occupancy heuristic, so a mismatch there is a warning. SYCL reports
 * no per-multiprocessor limits of its own (GetDeviceProperties stands in
 * work-group ones), so those are not compared there.
 */
[[maybe_unused]] static void CheckDeviceArchProperties()
{
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL)
    [[maybe_unused]] static const bool checked = [] {
#if defined(NEKTAR_DEVICE_ARCH_NAME)
        constexpr DeviceArchProperties arch = GetDeviceArchProperties();
        if constexpr (arch.known)
        {
            const std::string msg =
                " the one NektarDeviceArchLimits.cmake defines for " +
                std::string(NEKTAR_DEVICE_ARCH_NAME) + ".";
            ASSERTL0(GetDeviceProperties::SharedMemoryPerBlock() >=
                         arch.sharedMemoryPerBlock,
                     "Shared memory per block is less than" + msg);
#if !defined(NEKTAR_ENABLE_SYCL)
            WARNINGL0(GetDeviceProperties::SharedMemoryPerMultiprocessor() ==
                          arch.sharedMemoryPerMultiprocessor,
                      "Shared memory per multiprocessor differs from" + msg);
            WARNINGL0(static_cast<size_t>(
                          GetDeviceProperties::MaxThreadsPerMultiprocessor()) ==
                          arch.maxThreadsPerMultiprocessor,
                      "Threads per multiprocessor differ from" + msg);
#endif
        }
#endif
        return true;
    }();
#endif
}

} // namespace Nektar
