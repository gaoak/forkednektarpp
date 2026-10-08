///////////////////////////////////////////////////////////////////////////////
//
// File: MathHelperDef.hpp
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

#include "LibUtilities/BasicUtils/Math/Math.hpp"

#include "LibUtilities/BasicUtils/Math/MathHelper.hpp"

// Provide definitions for the templated function of the abstract class
// MathHelper. This should NOT by included by users in a *.cpp file. This
// file notably allows deferred instantation for a Field object, defined in
// MultiRegions.
namespace Nektar::Math
{

template <typename T> void MathHelper::zero(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::zero<NektarSpaces::Serial>(x);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::zero<NektarSpaces::AVX>(x);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::zero<NektarSpaces::Device>(x);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename TData, typename T>
void MathHelper::fill(const TData &val, T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::fill<NektarSpaces::Serial>(val, x);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::fill<NektarSpaces::AVX>(val, x);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::fill<NektarSpaces::Device>(val, x);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::copy(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::copy<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::copy<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::copy<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::abs(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::abs<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::abs<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::abs<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::neg(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::neg<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::neg<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::neg<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::sqrt(T &x, T &y, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::sqrt<NektarSpaces::Serial>(x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::sqrt<NektarSpaces::AVX>(x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::sqrt<NektarSpaces::Device>(x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::add(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::add<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::add<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::add<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::sub(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::sub<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::sub<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::sub<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::mul(const typename T::value_type alpha, T &x, T &y,
                     const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::mul<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::mul<NektarSpaces::AVX>(alpha, x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::mul<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::mul(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::mul<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::mul<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::mul<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::div(const typename T::value_type alpha, T &x, T &y,
                     const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::div<NektarSpaces::Serial>(alpha, x, y);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::div<NektarSpaces::AVX>(alpha, x, y);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::div<NektarSpaces::Device>(alpha, x, y);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::div(T &x, T &y, T &z, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::div<NektarSpaces::Serial>(x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::div<NektarSpaces::AVX>(x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::div<NektarSpaces::Device>(x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
void MathHelper::daxpy(const typename T::value_type alpha, T &x, T &y, T &z,
                       const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    if (execSpace0 == "Serial")
    {
        Nektar::Math::daxpy<NektarSpaces::Serial>(alpha, x, y, z);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::daxpy<NektarSpaces::AVX>(alpha, x, y, z);
    }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
    else if (execSpace0 == "Device")
    {
        Nektar::Math::daxpy<NektarSpaces::Device>(alpha, x, y, z);
    }
#endif
    else
    {
        ASSERTL0(false, "Unknown Execution space: " + execSpace)
    }
}

template <typename T>
typename T::value_type MathHelper::reduceSum(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceSum<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceSum<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::reduceSum<NektarSpaces::Device>(
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
typename T::value_type MathHelper::reduceSum(M &mask, T &x,
                                             const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceSum<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceSum<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::reduceSum<NektarSpaces::Device>(
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
typename T::value_type MathHelper::reduceMax(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceMax<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceMax<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::reduceMax<NektarSpaces::Device>(
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
typename T::value_type MathHelper::reduceMax(M &mask, T &x,
                                             const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceMax<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceMax<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::reduceMax<NektarSpaces::Device>(
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
typename T::value_type MathHelper::reduceMin(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceMin<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceMin<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::reduceMin<NektarSpaces::Device>(
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
typename T::value_type MathHelper::reduceMin(M &mask, T &x,
                                             const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::reduceMin<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::reduceMin<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::reduceMin<NektarSpaces::Device>(
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
typename T::value_type MathHelper::ddot(T &x, T &y,
                                        const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::ddot<NektarSpaces::Serial>(x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::ddot<NektarSpaces::AVX>(x, y, &out);
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
        Nektar::Math::ddot<NektarSpaces::Device>(
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
typename T::value_type MathHelper::ddot(M &mask, T &x, T &y,
                                        const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::ddot<NektarSpaces::Serial>(mask, x, y, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::ddot<NektarSpaces::AVX>(mask, x, y, &out);
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
        Nektar::Math::ddot<NektarSpaces::Device>(
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
typename T::value_type MathHelper::l1norm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::l1norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::l1norm<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::l1norm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::l1norm(M &mask, T &x,
                                          const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::l1norm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::l1norm<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::l1norm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::l2norm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::l2norm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::l2norm<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::l2norm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::l2norm(M &mask, T &x,
                                          const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::l2norm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::l2norm<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::l2norm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::lpnorm(const unsigned int p, T &x,
                                          const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::lpnorm<NektarSpaces::Serial>(p, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::lpnorm<NektarSpaces::AVX>(p, x, &out);
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
        Nektar::Math::lpnorm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::lpnorm(const unsigned int p, M &mask, T &x,
                                          const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::lpnorm<NektarSpaces::Serial>(p, mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::lpnorm<NektarSpaces::AVX>(p, mask, x, &out);
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
        Nektar::Math::lpnorm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::linfnorm(T &x, const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::linfnorm<NektarSpaces::Serial>(x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::linfnorm<NektarSpaces::AVX>(x, &out);
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
        Nektar::Math::linfnorm<NektarSpaces::Device>(
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
typename T::value_type MathHelper::linfnorm(M &mask, T &x,
                                            const std::string &execSpace)
{
    auto execSpace0 = execSpace == "" ? m_defaultExecSpace : execSpace;

    typename T::value_type out = 0.0;
    if (execSpace0 == "Serial")
    {
        Nektar::Math::linfnorm<NektarSpaces::Serial>(mask, x, &out);
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execSpace0 == "AVX")
    {
        Nektar::Math::linfnorm<NektarSpaces::AVX>(mask, x, &out);
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
        Nektar::Math::linfnorm<NektarSpaces::Device>(
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

} // namespace Nektar::Math
