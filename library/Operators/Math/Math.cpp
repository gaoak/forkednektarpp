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

#include "Operators/Field/Field.hpp"

#include "Operators/Math/Math.hpp"
#include "Operators/Math/MathKernels.hpp"

#if defined(NEKTAR_ENABLE_DEVICE)
namespace Nektar
{
void *internalMemoryBuffer           = nullptr;
void *internalDeviceBuffer           = nullptr;
void *internalHostBuffer             = nullptr;
unsigned int internalMaxDataSizeByte = 16;
} // namespace Nektar
#endif

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
template void Math::zero<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template void Math::zero<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template void Math::zero<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template void Math::zero<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template void Math::zero<MemoryRegion<double>>(MemoryRegion<double> &x,
                                               const std::string &execSpace);
template void Math::zero<MemoryRegion<float>>(MemoryRegion<float> &x,
                                              const std::string &execSpace);

// fill template specialization.
template void Math::fill<double, Field<double, FieldState::Phys>>(
    const double &val, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::fill<float, Field<float, FieldState::Phys>>(
    const float &val, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void Math::fill<double, Field<double, FieldState::Coeff>>(
    const double &val, Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::fill<float, Field<float, FieldState::Coeff>>(
    const float &val, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template void Math::fill<double, MemoryRegion<double>>(
    const double &val, MemoryRegion<double> &x, const std::string &execSpace);
template void Math::fill<float, MemoryRegion<float>>(
    const float &val, MemoryRegion<float> &x, const std::string &execSpace);

// copy template specialization.
template void Math::copy<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::copy<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::copy<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::copy<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::copy<MemoryRegion<double>>(MemoryRegion<double> &x,
                                               MemoryRegion<double> &y,
                                               const std::string &execSpace);
template void Math::copy<MemoryRegion<float>>(MemoryRegion<float> &x,
                                              MemoryRegion<float> &y,
                                              const std::string &execSpace);

// abs template specialization.
template void Math::abs<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::abs<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::abs<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::abs<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::abs<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              const std::string &execSpace);
template void Math::abs<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             const std::string &execSpace);

// neg template specialization.
template void Math::neg<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::neg<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::neg<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              const std::string &execSpace);
template void Math::neg<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             const std::string &execSpace);

// sqrt template specialization.
template void Math::sqrt<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::sqrt<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void Math::sqrt<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::sqrt<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void Math::sqrt<MemoryRegion<double>>(MemoryRegion<double> &x,
                                               MemoryRegion<double> &y,
                                               const std::string &execSpace);
template void Math::sqrt<MemoryRegion<float>>(MemoryRegion<float> &x,
                                              MemoryRegion<float> &y,
                                              const std::string &execSpace);

// add template specialization.
template void Math::add<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::add<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::add<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::add<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::add<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              MemoryRegion<double> &z,
                                              const std::string &execSpace);
template void Math::add<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             MemoryRegion<float> &z,
                                             const std::string &execSpace);

// sub template specialization.
template void Math::sub<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::sub<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::sub<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::sub<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::sub<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              MemoryRegion<double> &z,
                                              const std::string &execSpace);
template void Math::sub<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             MemoryRegion<float> &z,
                                             const std::string &execSpace);

// mul template specialization.
template void Math::mul<Field<double, FieldState::Phys>>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, const std::string &execSpace);
template void Math::mul<Field<float, FieldState::Phys>>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, const std::string &execSpace);
template void Math::mul<Field<double, FieldState::Coeff>>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::mul<Field<float, FieldState::Coeff>>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::mul<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::mul<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::mul<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::mul<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::mul<MemoryRegion<double>>(const double alpha,
                                              MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              const std::string &execSpace);
template void Math::mul<MemoryRegion<float>>(const float alpha,
                                             MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             const std::string &execSpace);
template void Math::mul<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              MemoryRegion<double> &z,
                                              const std::string &execSpace);
template void Math::mul<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             MemoryRegion<float> &z,
                                             const std::string &execSpace);

// div template specialization.
template void Math::div<Field<double, FieldState::Phys>>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, const std::string &execSpace);
template void Math::div<Field<float, FieldState::Phys>>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, const std::string &execSpace);
template void Math::div<Field<double, FieldState::Coeff>>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::div<Field<float, FieldState::Coeff>>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, const std::string &execSpace);
template void Math::div<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    Field<double, FieldState::Phys> &z, const std::string &execSpace);
template void Math::div<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    Field<float, FieldState::Phys> &z, const std::string &execSpace);
template void Math::div<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    Field<double, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::div<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    Field<float, FieldState::Coeff> &z, const std::string &execSpace);
template void Math::div<MemoryRegion<double>>(const double alpha,
                                              MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              const std::string &execSpace);
template void Math::div<MemoryRegion<float>>(const float alpha,
                                             MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             const std::string &execSpace);
template void Math::div<MemoryRegion<double>>(MemoryRegion<double> &x,
                                              MemoryRegion<double> &y,
                                              MemoryRegion<double> &z,
                                              const std::string &execSpace);
template void Math::div<MemoryRegion<float>>(MemoryRegion<float> &x,
                                             MemoryRegion<float> &y,
                                             MemoryRegion<float> &z,
                                             const std::string &execSpace);

// daxpy template specialization.
template void Math::daxpy<Field<double, FieldState::Phys>>(
    const double alpha, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<Field<float, FieldState::Phys>>(
    const float alpha, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void Math::daxpy<Field<double, FieldState::Coeff>>(
    const double alpha, Field<double, FieldState::Coeff> &x,
    Field<double, FieldState::Coeff> &y, Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<Field<float, FieldState::Coeff>>(
    const float alpha, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);
template void Math::daxpy<MemoryRegion<double>>(const double alpha,
                                                MemoryRegion<double> &x,
                                                MemoryRegion<double> &y,
                                                MemoryRegion<double> &z,
                                                const std::string &execSpace);
template void Math::daxpy<MemoryRegion<float>>(const float alpha,
                                               MemoryRegion<float> &x,
                                               MemoryRegion<float> &y,
                                               MemoryRegion<float> &z,
                                               const std::string &execSpace);

// reduceSum template specialization.
template double Math::reduceSum<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceSum<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceSum<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceSum<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceSum<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceSum<MemoryRegion<float>>(
    MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceSum<Field<uint8_t, FieldState::Phys>,
                                Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceSum<Field<uint8_t, FieldState::Phys>,
                               Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceSum<Field<uint8_t, FieldState::Coeff>,
                                Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceSum<Field<uint8_t, FieldState::Coeff>,
                               Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceSum<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::reduceSum<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// reduceMax template specialization.
template double Math::reduceMax<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceMax<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceMax<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMax<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceMax<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMax<MemoryRegion<float>>(
    MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceMax<Field<uint8_t, FieldState::Phys>,
                                Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMax<Field<uint8_t, FieldState::Phys>,
                               Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMax<Field<uint8_t, FieldState::Coeff>,
                                Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMax<Field<uint8_t, FieldState::Coeff>,
                               Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMax<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::reduceMax<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// reduceMin template specialization.
template double Math::reduceMin<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::reduceMin<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::reduceMin<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMin<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::reduceMin<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::reduceMin<MemoryRegion<float>>(
    MemoryRegion<float> &x, const std::string &execSpace);
template double Math::reduceMin<Field<uint8_t, FieldState::Phys>,
                                Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::reduceMin<Field<uint8_t, FieldState::Phys>,
                               Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::reduceMin<Field<uint8_t, FieldState::Coeff>,
                                Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::reduceMin<Field<uint8_t, FieldState::Coeff>,
                               Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::reduceMin<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::reduceMin<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// ddot template specialization.
template double Math::ddot<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float Math::ddot<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double Math::ddot<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float Math::ddot<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template double Math::ddot<MemoryRegion<double>>(MemoryRegion<double> &x,
                                                 MemoryRegion<double> &y,
                                                 const std::string &execSpace);
template float Math::ddot<MemoryRegion<float>>(MemoryRegion<float> &x,
                                               MemoryRegion<float> &y,
                                               const std::string &execSpace);

template double Math::ddot<Field<uint8_t, FieldState::Phys>,
                           Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    Field<double, FieldState::Phys> &y, const std::string &execSpace);
template float Math::ddot<Field<uint8_t, FieldState::Phys>,
                          Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    Field<float, FieldState::Phys> &y, const std::string &execSpace);
template double Math::ddot<Field<uint8_t, FieldState::Coeff>,
                           Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float Math::ddot<Field<uint8_t, FieldState::Coeff>,
                          Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    Field<float, FieldState::Coeff> &y, const std::string &execSpace);
template double Math::ddot<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    MemoryRegion<double> &y, const std::string &execSpace);
template float Math::ddot<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x, MemoryRegion<float> &y,
    const std::string &execSpace);

// l1norm template specialization.
template double Math::l1norm<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::l1norm<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::l1norm<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l1norm<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::l1norm<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l1norm<MemoryRegion<float>>(MemoryRegion<float> &x,
                                                 const std::string &execSpace);
template double Math::l1norm<Field<uint8_t, FieldState::Phys>,
                             Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l1norm<Field<uint8_t, FieldState::Phys>,
                            Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l1norm<Field<uint8_t, FieldState::Coeff>,
                             Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l1norm<Field<uint8_t, FieldState::Coeff>,
                            Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l1norm<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::l1norm<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// l2norm template specialization.
template double Math::l2norm<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::l2norm<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::l2norm<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l2norm<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::l2norm<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::l2norm<MemoryRegion<float>>(MemoryRegion<float> &x,
                                                 const std::string &execSpace);
template double Math::l2norm<Field<uint8_t, FieldState::Phys>,
                             Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::l2norm<Field<uint8_t, FieldState::Phys>,
                            Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::l2norm<Field<uint8_t, FieldState::Coeff>,
                             Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::l2norm<Field<uint8_t, FieldState::Coeff>,
                            Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::l2norm<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::l2norm<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// lpnorm template specialization.
template double Math::lpnorm<Field<double, FieldState::Phys>>(
    const unsigned int p, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::lpnorm<Field<float, FieldState::Phys>>(
    const unsigned int p, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::lpnorm<Field<double, FieldState::Coeff>>(
    const unsigned int p, Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float Math::lpnorm<Field<float, FieldState::Coeff>>(
    const unsigned int p, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::lpnorm<MemoryRegion<double>>(
    const unsigned int p, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::lpnorm<MemoryRegion<float>>(const unsigned int p,
                                                 MemoryRegion<float> &x,
                                                 const std::string &execSpace);
template double Math::lpnorm<Field<uint8_t, FieldState::Phys>,
                             Field<double, FieldState::Phys>>(
    const unsigned int p, Field<uint8_t, FieldState::Phys> &mask,
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::lpnorm<Field<uint8_t, FieldState::Phys>,
                            Field<float, FieldState::Phys>>(
    const unsigned int p, Field<uint8_t, FieldState::Phys> &mask,
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::lpnorm<Field<uint8_t, FieldState::Coeff>,
                             Field<double, FieldState::Coeff>>(
    const unsigned int p, Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::lpnorm<Field<uint8_t, FieldState::Coeff>,
                            Field<float, FieldState::Coeff>>(
    const unsigned int p, Field<uint8_t, FieldState::Coeff> &mask,
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::lpnorm<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    const unsigned int p, MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::lpnorm<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    const unsigned int p, MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

// linfnorm template specialization.
template double Math::linfnorm<Field<double, FieldState::Phys>>(
    Field<double, FieldState::Phys> &x, const std::string &execSpace);
template float Math::linfnorm<Field<float, FieldState::Phys>>(
    Field<float, FieldState::Phys> &x, const std::string &execSpace);
template double Math::linfnorm<Field<double, FieldState::Coeff>>(
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::linfnorm<Field<float, FieldState::Coeff>>(
    Field<float, FieldState::Coeff> &x, const std::string &execSpace);
template double Math::linfnorm<MemoryRegion<double>>(
    MemoryRegion<double> &x, const std::string &execSpace);
template float Math::linfnorm<MemoryRegion<float>>(
    MemoryRegion<float> &x, const std::string &execSpace);
template double Math::linfnorm<Field<uint8_t, FieldState::Phys>,
                               Field<double, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float Math::linfnorm<Field<uint8_t, FieldState::Phys>,
                              Field<float, FieldState::Phys>>(
    Field<uint8_t, FieldState::Phys> &mask, Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double Math::linfnorm<Field<uint8_t, FieldState::Coeff>,
                               Field<double, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask,
    Field<double, FieldState::Coeff> &x, const std::string &execSpace);
template float Math::linfnorm<Field<uint8_t, FieldState::Coeff>,
                              Field<float, FieldState::Coeff>>(
    Field<uint8_t, FieldState::Coeff> &mask, Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double Math::linfnorm<MemoryRegion<uint8_t>, MemoryRegion<double>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<double> &x,
    const std::string &execSpace);
template float Math::linfnorm<MemoryRegion<uint8_t>, MemoryRegion<float>>(
    MemoryRegion<uint8_t> &mask, MemoryRegion<float> &x,
    const std::string &execSpace);

} // namespace Nektar::Operators
