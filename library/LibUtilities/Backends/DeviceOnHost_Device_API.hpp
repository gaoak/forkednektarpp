///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceOnHost_Device_API.hpp
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

#if defined(NEKTAR_ENABLE_DEVICEONHOST)
namespace Nektar
{

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostLocalIdxX;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostLocalIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostLocalIdxY;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostLocalIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostLocalIdxY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostLocalIdxZ;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostBlockDimX;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockDimY;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockDimY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostBlockDimZ;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostBlockDimX * deviceOnHostBlockIdxX +
               deviceOnHostLocalIdxX;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockDimX * deviceOnHostBlockIdxX +
               deviceOnHostLocalIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockDimY * deviceOnHostBlockIdxY +
               deviceOnHostLocalIdxY;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockDimX * deviceOnHostBlockIdxX +
               deviceOnHostLocalIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockDimY * deviceOnHostBlockIdxY +
               deviceOnHostLocalIdxY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostBlockDimZ * deviceOnHostBlockIdxZ +
               deviceOnHostLocalIdxZ;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostGridDimX * deviceOnHostBlockDimX;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostGridDimX * deviceOnHostBlockDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostGridDimY * deviceOnHostBlockDimY;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static size_t getGlobalRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostGridDimX * deviceOnHostBlockDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostGridDimY * deviceOnHostBlockDimY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostGridDimZ * deviceOnHostBlockDimZ;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostBlockIdxX;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockIdxY;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockIdx(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        // Fastest moving.
        return deviceOnHostBlockIdxX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostBlockIdxY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostBlockIdxZ;
    }
    else
    {
        return 0;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<1> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostGridDimX;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<2> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostGridDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostGridDimY;
    }
    else
    {
        return 1;
    }
}

template <unsigned int dim = 0>
NEK_DEVICE_INLINE static unsigned int getBlockRange(
    [[maybe_unused]] const deviceOnHostBlock<3> &threadBlock)
{
    if constexpr (dim == 0)
    {
        return deviceOnHostGridDimX;
    }
    else if constexpr (dim == 1)
    {
        return deviceOnHostGridDimY;
    }
    else if constexpr (dim == 2)
    {
        return deviceOnHostGridDimZ;
    }
    else
    {
        return 1;
    }
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getWarpIdx(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return getGlobalIdx(threadBlock);
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return 0;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_add(TData *const dest, const TData val)
{
    *dest += val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_sub(TData *const dest, const TData val)
{
    *dest -= val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_max(TData *const dest, const TData val)
{
    *dest = std::max(*dest, val);
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_min(TData *const dest, const TData val)
{
    *dest = std::min(*dest, val);
}

// Keep whichever of *dest and val has the larger magnitude, preserving its
// sign; ties keep *dest. Matches the CUDA/HIP and SYCL atomic_absmax
// semantics.
template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_absmax(TData *const dest, const TData val)
{
    if (std::abs(val) > std::abs(*dest))
    {
        *dest = val;
    }
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_or(TData *const dest, const TData val)
{
    *dest |= val;
}

template <typename Scope, typename TData>
NEK_DEVICE_INLINE static void atomic_and(TData *const dest, const TData val)
{
    *dest &= val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMax(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceMin(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceOr(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static TData warpReduceAnd(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red += val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMax(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red = std::max(*red, val);
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceMin(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red = std::min(*red, val);
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceOr(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red |= val;
}

template <unsigned int ndim, typename TData>
NEK_DEVICE_INLINE static void blockReduceAnd(
    const TData val,
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock, TData *red)
{
    *red &= val;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAll(
    int predictate, [[maybe_unused]] deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpVoteAny(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE static int warpBallot(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE void localBarrier(
    [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_and(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_or(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

template <unsigned int ndim>
NEK_DEVICE_INLINE int localBarrier_count(
    int predictate, [[maybe_unused]] const deviceOnHostBlock<ndim> &threadBlock)
{
    return predictate ? 1 : 0;
}

} // namespace Nektar
#endif
