///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionSYCL.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

namespace Nektar
{

// Parallel for launchers.
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const size_t begin, const size_t end, const Functor &functor)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(end - begin),
                         [=](sycl::id<1> indx) { functor(begin + indx); });
    });
}

// Reduction kernels.
template <bool init, typename TData, typename Functor>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (init && lid == 0)
                {
                    buffer[indx.get_group(0)] = 0.0;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = 0.0;
                while (gid < end)
                {
                    tmp += functor(gid);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] += scratch[lid + n];
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] += scratch[0];
                }
            });
    });
}

template <bool init, typename TData, typename Functor>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (init && lid == 0)
                {
                    buffer[indx.get_group(0)] = min;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = min;
                while (gid < end)
                {
                    tmp = sycl::fmax(tmp, functor(gid));
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] =
                        sycl::fmax(buffer[indx.get_group(0)], scratch[0]);
                }
            });
    });
}

template <bool init, typename TData, typename Functor>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (init && lid == 0)
                {
                    buffer[indx.get_group(0)] = max;
                }

                indx.barrier(sycl::access::fence_space::local_space);

                TData tmp = max;
                while (gid < end)
                {
                    tmp = sycl::fmin(tmp, functor(gid));
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

                indx.barrier(sycl::access::fence_space::local_space);

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        scratch[lid] =
                            sycl::fmin(scratch[lid], scratch[lid + n]);
                    }
                    indx.barrier(sycl::access::fence_space::local_space);
                    n /= 2;
                }

                if (lid == 0)
                {
                    buffer[indx.get_group(0)] =
                        sycl::fmin(buffer[indx.get_group(0)], scratch[0]);
                }
            });
    });
}

// Parallel reduction launchers without device-to-host copy.
template <typename ExecSpace, bool init, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

#if defined(USE_SYCL_BUILTIN_REDUCER)
    const unsigned int gridSize = NektarSpaces::Device::maximumBlockSize;
#else
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;
#endif

    if (internalMemoryBuffer == nullptr)
    {
        const unsigned int internalMemoryBufferSize =
            internalMaxDataSizeByte * gridSize;
        deviceMalloc(&internalMemoryBuffer, internalMemoryBufferSize);
    }

    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
#if defined(USE_SYCL_BUILTIN_REDUCER)
        sycl::queue &Q = SYCLQueue::GetInstance();
        sycl::property_list initializer =
            init ? sycl::property_list{sycl::property::reduction::
                                           initialize_to_identity{}}
                 : sycl::property_list{};
        Q.parallel_for(sycl::range<1>(end - begin),
                       sycl::reduction(out, sycl::plus<>(), initializer),
                       [=](sycl::id<1> indx, auto &reducer) {
                           reducer.combine(functor(begin + indx));
                       });
#else
        TData *buffer = (TData *)internalMemoryBuffer;
        reduceSumKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceSumKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
#endif
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
#if defined(USE_SYCL_BUILTIN_REDUCER)
        sycl::queue &Q = SYCLQueue::GetInstance();
        sycl::property_list initializer =
            init ? sycl::property_list{sycl::property::reduction::
                                           initialize_to_identity{}}
                 : sycl::property_list{};
        Q.parallel_for(sycl::range<1>(end - begin),
                       sycl::reduction(out, sycl::maximum<>(), initializer),
                       [=](sycl::id<1> indx, auto &reducer) {
                           reducer.combine(functor(begin + indx));
                       });
#else
        TData *buffer = (TData *)internalMemoryBuffer;
        reduceMaxKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceMaxKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
#endif
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
#if defined(USE_SYCL_BUILTIN_REDUCER)
        sycl::queue &Q = SYCLQueue::GetInstance();
        sycl::property_list initializer =
            init ? sycl::property_list{sycl::property::reduction::
                                           initialize_to_identity{}}
                 : sycl::property_list{};
        Q.parallel_for(sycl::range<1>(end - begin),
                       sycl::reduction(out, sycl::minimum<>(), initializer),
                       [=](sycl::id<1> indx, auto &reducer) {
                           reducer.combine(functor(begin + indx));
                       });
#else
        TData *buffer = (TData *)internalMemoryBuffer;
        reduceMinKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceMinKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
#endif
    }
}

// Parallel reduction launchers with device-to-host copy.
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type &out)
{
    using TData = typename Reduction::value_type;

    if (internalHostBuffer == nullptr)
    {
        hostMallocPinned(&internalHostBuffer, internalMaxDataSizeByte);
        deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte);
    }

    parallel_reduce<ExecSpace, true, Reduction>(begin, end, functor,
                                                (TData *)internalDeviceBuffer);

    deviceMemcpy<DeviceToHost>(internalHostBuffer, internalDeviceBuffer,
                               sizeof(TData));

    out = *(TData *)internalHostBuffer;
}

} // namespace Nektar

#endif
