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
        return prop[id].sharedMemPerBlock;
    }

    static size_t &TotalGlobalMemory(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(cudaGetDevice(&id));
        FetchDeviceProperties(id);
        return prop[id].totalGlobalMem;
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
    static std::unordered_map<unsigned int, cudaDeviceProp> prop;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (prop.find(id) == prop.end())
        {
            prop.emplace(id, cudaDeviceProp{});
            CHECK_HIPCUDA_ERROR(cudaGetDeviceProperties(&prop[id], id));
        }
    }
}; // namespace GetDeviceProperties

#elif defined(NEKTAR_ENABLE_HIP)
class GetDeviceProperties
{
public:
    static const size_t &SharedMemoryPerBlock(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return prop[id].sharedMemPerBlock;
    }

    static size_t &TotalGlobalMemory(void)
    {
        int id = -1;
        CHECK_HIPCUDA_ERROR(hipGetDevice(&id));
        FetchDeviceProperties(id);
        return prop[id].totalGlobalMem;
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
    static std::unordered_map<unsigned int, hipDeviceProp_t> prop;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (prop.find(id) == prop.end())
        {
            prop.emplace(id, hipDeviceProp_t{});
            CHECK_HIPCUDA_ERROR(hipGetDeviceProperties(&prop[id], id));
        }
    }
}; // namespace GetDeviceProperties

#elif defined(NEKTAR_ENABLE_SYCL)
class GetDeviceProperties
{
public:
    static const size_t &SharedMemoryPerBlock(void)
    {
        FetchDeviceProperties(0);
        return m_sharedMemoryPerBlock[0];
    }

    static size_t &TotalGlobalMemory(void)
    {
        FetchDeviceProperties(0);
        return m_totalGlobalMemory[0];
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
    static std::unordered_map<unsigned int, size_t> m_totalGlobalMemory;

    static void FetchDeviceProperties(const unsigned int id)
    {
        if (m_totalGlobalMemory.find(id) == m_totalGlobalMemory.end())
        {
            auto device =
                SYCLQueue::GetInstance().get_info<sycl::info::queue::device>();
            m_totalGlobalMemory.emplace(
                id, device.get_info<sycl::info::device::global_mem_size>());
            m_sharedMemoryPerBlock.emplace(
                id, device.get_info<sycl::info::device::local_mem_size>());
        }
    }
}; // namespace GetDeviceProperties

#endif
