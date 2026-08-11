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

#include "MultiRegions/Field/Field.hpp"

#include "Operators/Math/Math.hpp"
#include "Operators/Math/MathKernels.hpp"

namespace Nektar
{
#if defined(NEKTAR_ENABLE_DEVICE)
std::unordered_map<unsigned int, void *> internalMemoryBufferMap;
void *internalDeviceBuffer           = nullptr;
void *internalHostBuffer             = nullptr;
unsigned int internalMaxDataSizeByte = 16;
#endif
} // namespace Nektar

namespace Nektar::Operators
{

template <typename T> void Math::zero(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::zero<NektarSpaces::Serial>(x);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::zero<NektarSpaces::AVX>(x);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::zero<NektarSpaces::Device>(x);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, typename T>
void Math::fill(const TData &val, T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::fill<NektarSpaces::Serial>(val, x);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::fill<NektarSpaces::AVX>(val, x);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::fill<NektarSpaces::Device>(val, x);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T> void Math::copy(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::copy<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::copy<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::copy<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T> void Math::abs(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::abs<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::abs<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::abs<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T> void Math::neg(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::neg<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::neg<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::neg<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T> void Math::sqrt(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::sqrt<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::sqrt<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::sqrt<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::add(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::add<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::add<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::add<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::sub(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::sub<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::sub<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::sub<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::mul(const typename T::value_type alpha, T &x, T &y,
               const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::mul<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::mul<NektarSpaces::AVX>(alpha, x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::mul<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::mul(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::mul<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::mul<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::mul<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::div(const typename T::value_type alpha, T &x, T &y,
               const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::div<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::div<NektarSpaces::AVX>(alpha, x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::div<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::div(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::div<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::div<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::div<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void Math::daxpy(const typename T::value_type alpha, T &x, T &y, T &z,
                 const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Operators::daxpy<NektarSpaces::Serial>(alpha, x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::daxpy<NektarSpaces::AVX>(alpha, x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Operators::daxpy<NektarSpaces::Device>(alpha, x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
typename T::value_type Math::reduceSum(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceSum<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceSum<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceSum<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::reduceSum(M &mask, T &x,
                                       const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceSum<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceSum<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceSum<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::reduceMax(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceMax<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceMax<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceMax<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::reduceMax(M &mask, T &x,
                                       const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceMax<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceMax<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceMax<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::reduceMin(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceMin<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceMin<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceMin<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::reduceMin(M &mask, T &x,
                                       const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::reduceMin<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::reduceMin<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::reduceMin<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::ddot(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::ddot<NektarSpaces::Serial>(x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::ddot<NektarSpaces::AVX>(x, y, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::ddot<NektarSpaces::Device>(
            x, y, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::ddot(M &mask, T &x, T &y,
                                  const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::ddot<NektarSpaces::Serial>(mask, x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::ddot<NektarSpaces::AVX>(mask, x, y, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::ddot<NektarSpaces::Device>(
            mask, x, y, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::l1norm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::l1norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::l1norm<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::l1norm<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::l1norm(M &mask, T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::l1norm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::l1norm<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::l1norm<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::l2norm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::l2norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::l2norm<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::l2norm<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::l2norm(M &mask, T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::l2norm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::l2norm<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::l2norm<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::lpnorm(const unsigned int p, T &x,
                                    const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::lpnorm<NektarSpaces::Serial>(p, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::lpnorm<NektarSpaces::AVX>(p, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::lpnorm<NektarSpaces::Device>(
            p, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::lpnorm(const unsigned int p, M &mask, T &x,
                                    const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::lpnorm<NektarSpaces::Serial>(p, mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::lpnorm<NektarSpaces::AVX>(p, mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::lpnorm<NektarSpaces::Device>(
            p, mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename T>
typename T::value_type Math::linfnorm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::linfnorm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::linfnorm<NektarSpaces::AVX>(x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::linfnorm<NektarSpaces::Device>(
            x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

template <typename M, typename T>
typename T::value_type Math::linfnorm(M &mask, T &x,
                                      const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Operators::linfnorm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Operators::linfnorm<NektarSpaces::AVX>(mask, x, &out);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        const unsigned int streamID = 0;
        if (internalDeviceBuffer == nullptr)
        {
            Nektar::deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte,
                                 streamID);
            Nektar::hostMallocPinned(&internalHostBuffer,
                                     internalMaxDataSizeByte);
            Nektar::nekStreamSynchronize(streamID);
        }
        Nektar::Operators::linfnorm<NektarSpaces::Device>(
            mask, x, (typename T::value_type *)internalDeviceBuffer);
        Nektar::deviceMemcpy<DeviceToHost>(
            internalHostBuffer, internalDeviceBuffer,
            sizeof(typename T::value_type), streamID);
        out = *(typename T::value_type *)internalHostBuffer;
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }

    return out;
}

// zero template specialization.
template void Math::zero<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::zero<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::zero<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::zero<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::zero<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template void Math::zero<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// fill template specialization.
template void Math::fill<double, MultiRegions::Field<double, FieldState::Phys>>(
    const double &val, MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::fill<float, MultiRegions::Field<float, FieldState::Phys>>(
    const float &val, MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::fill<double,
                         MultiRegions::Field<double, FieldState::Coeff>>(
    const double &val, MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::fill<float, MultiRegions::Field<float, FieldState::Coeff>>(
    const float &val, MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::fill<double, LibUtilities::MemoryRegion<double>>(
    const double &val, LibUtilities::MemoryRegion<double> &x,
    const std::string &execSpace);
template void Math::fill<float, LibUtilities::MemoryRegion<float>>(
    const float &val, LibUtilities::MemoryRegion<float> &x,
    const std::string &execSpace);

// copy template specialization.
template void Math::copy<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::copy<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::copy<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::copy<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::copy<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::copy<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

// abs template specialization.
template void Math::abs<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::abs<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::abs<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::abs<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::abs<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::abs<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

// neg template specialization.
template void Math::neg<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::neg<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

// sqrt template specialization.
template void Math::sqrt<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::sqrt<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::sqrt<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::sqrt<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::sqrt<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::sqrt<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

// add template specialization.
template void Math::add<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::add<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::add<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::add<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::add<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void Math::add<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);

// sub template specialization.
template void Math::sub<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::sub<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::sub<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::sub<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::sub<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void Math::sub<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);

// mul template specialization.
template void Math::mul<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::mul<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::mul<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::mul<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, const std::string &execSpace);
template void Math::mul<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void Math::mul<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);

// div template specialization.
template void Math::div<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::div<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::div<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void Math::div<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, const std::string &execSpace);
template void Math::div<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void Math::div<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);

// daxpy template specialization.
template void Math::daxpy<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void Math::daxpy<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, LibUtilities::MemoryRegion<float> &z,
    const std::string &execSpace);

// reduceSum template specialization.
template double Math::reduceSum<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceSum<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceSum<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceSum<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceSum<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceSum<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceSum<MultiRegions::Field<uint8_t, FieldState::Phys>,
                                MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceSum<MultiRegions::Field<uint8_t, FieldState::Phys>,
                               MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceSum<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                                MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceSum<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                               MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceSum<LibUtilities::MemoryRegion<uint8_t>,
                                LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceSum<LibUtilities::MemoryRegion<uint8_t>,
                               LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// reduceMax template specialization.
template double Math::reduceMax<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMax<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMax<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceMax<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMax<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMax<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceMax<MultiRegions::Field<uint8_t, FieldState::Phys>,
                                MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMax<MultiRegions::Field<uint8_t, FieldState::Phys>,
                               MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMax<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                                MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceMax<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                               MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMax<LibUtilities::MemoryRegion<uint8_t>,
                                LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMax<LibUtilities::MemoryRegion<uint8_t>,
                               LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// reduceMin template specialization.
template double Math::reduceMin<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMin<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMin<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceMin<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMin<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMin<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceMin<MultiRegions::Field<uint8_t, FieldState::Phys>,
                                MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMin<MultiRegions::Field<uint8_t, FieldState::Phys>,
                               MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMin<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                                MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::reduceMin<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                               MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMin<LibUtilities::MemoryRegion<uint8_t>,
                                LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMin<LibUtilities::MemoryRegion<uint8_t>,
                               LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// ddot template specialization.
template double Math::ddot<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float Math::ddot<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double Math::ddot<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float Math::ddot<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template double Math::ddot<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template float Math::ddot<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

template double Math::ddot<MultiRegions::Field<uint8_t, FieldState::Phys>,
                           MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float Math::ddot<MultiRegions::Field<uint8_t, FieldState::Phys>,
                          MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double Math::ddot<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                           MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float Math::ddot<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                          MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template double Math::ddot<LibUtilities::MemoryRegion<uint8_t>,
                           LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template float Math::ddot<LibUtilities::MemoryRegion<uint8_t>,
                          LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);

// l1norm template specialization.
template double Math::l1norm<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l1norm<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l1norm<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::l1norm<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l1norm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l1norm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::l1norm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                             MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l1norm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                            MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l1norm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                             MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::l1norm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                            MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l1norm<LibUtilities::MemoryRegion<uint8_t>,
                             LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l1norm<LibUtilities::MemoryRegion<uint8_t>,
                            LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// l2norm template specialization.
template double Math::l2norm<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l2norm<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l2norm<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::l2norm<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l2norm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l2norm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::l2norm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                             MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l2norm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                            MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l2norm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                             MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::l2norm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                            MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l2norm<LibUtilities::MemoryRegion<uint8_t>,
                             LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l2norm<LibUtilities::MemoryRegion<uint8_t>,
                            LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// lpnorm template specialization.
template double Math::lpnorm<MultiRegions::Field<double, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::lpnorm<MultiRegions::Field<float, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::lpnorm<MultiRegions::Field<double, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::lpnorm<MultiRegions::Field<float, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::lpnorm<LibUtilities::MemoryRegion<double>>(
    const unsigned int p, LibUtilities::MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::lpnorm<LibUtilities::MemoryRegion<float>>(
    const unsigned int p, LibUtilities::MemoryRegion<float> &x,
    const std::string &execSpace);
template double Math::lpnorm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                             MultiRegions::Field<double, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::lpnorm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                            MultiRegions::Field<float, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::lpnorm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                             MultiRegions::Field<double, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::lpnorm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                            MultiRegions::Field<float, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::lpnorm<LibUtilities::MemoryRegion<uint8_t>,
                             LibUtilities::MemoryRegion<double>>(
    const unsigned int p, LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::lpnorm<LibUtilities::MemoryRegion<uint8_t>,
                            LibUtilities::MemoryRegion<float>>(
    const unsigned int p, LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

// linfnorm template specialization.
template double Math::linfnorm<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::linfnorm<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::linfnorm<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::linfnorm<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::linfnorm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::linfnorm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double Math::linfnorm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                               MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::linfnorm<MultiRegions::Field<uint8_t, FieldState::Phys>,
                              MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::linfnorm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                               MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::linfnorm<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                              MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::linfnorm<LibUtilities::MemoryRegion<uint8_t>,
                               LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float Math::linfnorm<LibUtilities::MemoryRegion<uint8_t>,
                              LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);

} // namespace Nektar::Operators
