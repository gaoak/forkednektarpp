///////////////////////////////////////////////////////////////////////////////
//
// File: Math.cpp
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

#include "Operators/MathKernels/Math.hpp"
#include "Operators/MathKernels/MathKernels.hpp"

namespace Nektar
{

#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
static void *internal_device_buffer = nullptr;
static void *internal_host_buffer   = nullptr;
#endif

template <typename TData, FieldState TFieldState>
void Math::neg(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
               const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::neg<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::neg<NektarSpaces::AVX>(x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::neg<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::neg(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
               const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::neg<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::neg<NektarSpaces::AVX>(x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::neg<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::add(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
               Field<TData, TFieldState> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::add<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::add<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::add<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::add(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
               MemoryRegion<TData> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::add<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::add<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::add<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::sub(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
               Field<TData, TFieldState> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::sub<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::sub<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::sub<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::sub(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
               MemoryRegion<TData> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::sub<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::sub<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::sub<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::mul(const TData alpha, Field<TData, TFieldState> &x,
               Field<TData, TFieldState> &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::mul<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::mul<NektarSpaces::AVX>(alpha, x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::mul<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::mul(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
               Field<TData, TFieldState> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::mul<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::mul<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::mul<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::mul(const TData alpha, MemoryRegion<TData> &x,
               MemoryRegion<TData> &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::mul<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::mul<NektarSpaces::AVX>(alpha, x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::mul<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::mul(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
               MemoryRegion<TData> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::mul<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::mul<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::mul<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::div(const TData alpha, Field<TData, TFieldState> &x,
               Field<TData, TFieldState> &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::div<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::div<NektarSpaces::AVX>(alpha, x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::div<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::div(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
               Field<TData, TFieldState> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::div<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::div<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::div<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::div(const TData alpha, MemoryRegion<TData> &x,
               MemoryRegion<TData> &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::div<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::div<NektarSpaces::AVX>(alpha, x, y);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::div<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::div(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
               MemoryRegion<TData> &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::div<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::div<NektarSpaces::AVX>(x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::div<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
void Math::daxpy(const TData alpha, Field<TData, TFieldState> &x,
                 Field<TData, TFieldState> &y, Field<TData, TFieldState> &z,
                 const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::daxpy<NektarSpaces::Serial>(alpha, x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::daxpy<NektarSpaces::AVX>(alpha, x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::daxpy<NektarSpaces::Device>(alpha, x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData>
void Math::daxpy(const TData alpha, MemoryRegion<TData> &x,
                 MemoryRegion<TData> &y, MemoryRegion<TData> &z,
                 const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::daxpy<NektarSpaces::Serial>(alpha, x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::daxpy<NektarSpaces::AVX>(alpha, x, y, z);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        Nektar::daxpy<NektarSpaces::Device>(alpha, x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, FieldState TFieldState>
TData Math::reduceSum(Field<TData, TFieldState> &x,
                      const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceSum<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceSum<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceSum<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::reduceSum(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceSum<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceSum<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceSum<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::reduceMax(Field<TData, TFieldState> &x,
                      const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceMax<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceMax<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceMax<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::reduceMax(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceMax<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceMax<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceMax<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::reduceMin(Field<TData, TFieldState> &x,
                      const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceMin<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceMin<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceMin<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::reduceMin(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::reduceMin<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::reduceMin<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::reduceMin<NektarSpaces::Device>(
            x, (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::ddot(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
                 const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::ddot<NektarSpaces::Serial>(x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::ddot<NektarSpaces::AVX>(x, y, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::ddot<NektarSpaces::Device>(x, y,
                                           (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::ddot(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                 const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::ddot<NektarSpaces::Serial>(x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::ddot<NektarSpaces::AVX>(x, y, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::ddot<NektarSpaces::Device>(x, y,
                                           (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::l1norm(Field<TData, TFieldState> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::l1norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::l1norm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::l1norm<NektarSpaces::Device>(x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::l1norm(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::l1norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::l1norm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::l1norm<NektarSpaces::Device>(x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::l2norm(Field<TData, TFieldState> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::l2norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::l2norm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::l2norm<NektarSpaces::Device>(x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::l2norm(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::l2norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::l2norm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::l2norm<NektarSpaces::Device>(x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::lpnorm(const unsigned int p, Field<TData, TFieldState> &x,
                   const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::lpnorm<NektarSpaces::Serial>(p, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::lpnorm<NektarSpaces::AVX>(p, x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::lpnorm<NektarSpaces::Device>(p, x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::lpnorm(const unsigned int p, MemoryRegion<TData> &x,
                   const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::lpnorm<NektarSpaces::Serial>(p, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::lpnorm<NektarSpaces::AVX>(p, x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::lpnorm<NektarSpaces::Device>(p, x,
                                             (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData, FieldState TFieldState>
TData Math::linfnorm(Field<TData, TFieldState> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::linfnorm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::linfnorm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::linfnorm<NektarSpaces::Device>(x,
                                               (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename TData>
TData Math::linfnorm(MemoryRegion<TData> &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    TData out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::linfnorm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::linfnorm<NektarSpaces::AVX>(x, &out);
    }
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    else if (execSpace0 == "Device")
    {
        if (internal_device_buffer == nullptr)
        {
            Nektar::deviceMalloc(&internal_device_buffer, sizeof(TData),
                                 NektarSpaces::Device::alignment, 0);
            Nektar::hostMallocPinned(&internal_host_buffer, sizeof(TData),
                                     NektarSpaces::Device::alignment);
        }
        Nektar::linfnorm<NektarSpaces::Device>(x,
                                               (TData *)internal_device_buffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internal_host_buffer, internal_device_buffer, sizeof(TData), 0);
        out = *(TData *)internal_host_buffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

// neg template specialization.
template void Math::neg<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<double>(MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                const std::string &execSpace);
template void Math::neg<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                               const std::string &execSpace);

// add template specialization.
template void Math::add<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::add<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::add<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::add<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::add<double>(MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                MemoryRegion<double> &z,
                                const std::string &execSpace);
template void Math::add<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                               MemoryRegion<float> &z,
                               const std::string &execSpace);

// sub template specialization.
template void Math::sub<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::sub<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::sub<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::sub<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::sub<double>(MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                MemoryRegion<double> &z,
                                const std::string &execSpace);
template void Math::sub<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                               MemoryRegion<float> &z,
                               const std::string &execSpace);

// mul template specialization.
template void Math::mul<double, FieldState::Phys>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, const std::string &execSpace);
template void Math::mul<float, FieldState::Phys>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, const std::string &execSpace);
template void Math::mul<double, FieldState::Coeff>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::mul<float, FieldState::Coeff>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::mul<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::mul<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::mul<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::mul<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::mul<double>(const double alpha, MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                const std::string &execSpace);
template void Math::mul<float>(const float alpha, MemoryRegion<float> &x,
                               MemoryRegion<float> &y,
                               const std::string &execSpace);
template void Math::mul<double>(MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                MemoryRegion<double> &z,
                                const std::string &execSpace);
template void Math::mul<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                               MemoryRegion<float> &z,
                               const std::string &execSpace);

// div template specialization.
template void Math::div<double, FieldState::Phys>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, const std::string &execSpace);
template void Math::div<float, FieldState::Phys>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, const std::string &execSpace);
template void Math::div<double, FieldState::Coeff>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::div<float, FieldState::Coeff>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::div<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::div<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::div<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::div<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::div<double>(const double alpha, MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                const std::string &execSpace);
template void Math::div<float>(const float alpha, MemoryRegion<float> &x,
                               MemoryRegion<float> &y,
                               const std::string &execSpace);
template void Math::div<double>(MemoryRegion<double> &x,
                                MemoryRegion<double> &y,
                                MemoryRegion<double> &z,
                                const std::string &execSpace);
template void Math::div<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                               MemoryRegion<float> &z,
                               const std::string &execSpace);

// daxpy template specialization.
template void Math::daxpy<double, FieldState::Phys>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<float, FieldState::Phys>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<double, FieldState::Coeff>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<float, FieldState::Coeff>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<double>(const double alpha, MemoryRegion<double> &x,
                                  MemoryRegion<double> &y,
                                  MemoryRegion<double> &z,
                                  const std::string &execSpace);
template void Math::daxpy<float>(const float alpha, MemoryRegion<float> &x,
                                 MemoryRegion<float> &y, MemoryRegion<float> &z,
                                 const std::string &execSpace);

// reduceSum template specialization.
template double Math::reduceSum<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceSum<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceSum<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceSum<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceSum<double>(MemoryRegion<double> &x,
                                        const std::string &execSpace);
template float Math::reduceSum<float>(MemoryRegion<float> &x,
                                      const std::string &execSpace);

// reduceMax template specialization.
template double Math::reduceMax<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceMax<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceMax<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMax<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceMax<double>(MemoryRegion<double> &x,
                                        const std::string &execSpace);
template float Math::reduceMax<float>(MemoryRegion<float> &x,
                                      const std::string &execSpace);

// reduceMin template specialization.
template double Math::reduceMin<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceMin<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceMin<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMin<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceMin<double>(MemoryRegion<double> &x,
                                        const std::string &execSpace);
template float Math::reduceMin<float>(MemoryRegion<float> &x,
                                      const std::string &execSpace);

// ddot template specialization.
template double Math::ddot<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float Math::ddot<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double Math::ddot<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float Math::ddot<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template double Math::ddot<double>(MemoryRegion<double> &x,
                                   MemoryRegion<double> &y,
                                   const std::string &execSpace);
template float Math::ddot<float>(MemoryRegion<float> &x, MemoryRegion<float> &y,
                                 const std::string &execSpace);

// l1norm template specialization.
template double Math::l1norm<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::l1norm<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::l1norm<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l1norm<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::l1norm<double>(MemoryRegion<double> &x,
                                     const std::string &execSpace);
template float Math::l1norm<float>(MemoryRegion<float> &x,
                                   const std::string &execSpace);

// l2norm template specialization.
template double Math::l2norm<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::l2norm<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::l2norm<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l2norm<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::l2norm<double>(MemoryRegion<double> &x,
                                     const std::string &execSpace);
template float Math::l2norm<float>(MemoryRegion<float> &x,
                                   const std::string &execSpace);

// lpnorm template specialization.
template double Math::lpnorm<double, FieldState::Phys>(
    const unsigned int p, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::lpnorm<float, FieldState::Phys>(
    const unsigned int p, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::lpnorm<double, FieldState::Coeff>(
    const unsigned int p, Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::lpnorm<float, FieldState::Coeff>(
    const unsigned int p, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::lpnorm<double>(const unsigned int p,
                                     MemoryRegion<double> &x,
                                     const std::string &execSpace);
template float Math::lpnorm<float>(const unsigned int p, MemoryRegion<float> &x,
                                   const std::string &execSpace);

// linfnorm template specialization.
template double Math::linfnorm<double, FieldState::Phys>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::linfnorm<float, FieldState::Phys>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::linfnorm<double, FieldState::Coeff>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::linfnorm<float, FieldState::Coeff>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::linfnorm<double>(MemoryRegion<double> &x,
                                       const std::string &execSpace);
template float Math::linfnorm<float>(MemoryRegion<float> &x,
                                     const std::string &execSpace);
} // namespace Nektar
