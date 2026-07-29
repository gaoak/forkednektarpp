///////////////////////////////////////////////////////////////////////////////
//
// File: SYCL_Host_API.hpp
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
#include "Operators/Common/Backends/SYCLQueue.hpp"

#if defined(SYCL_ENABLE_CUDA) && defined(__ADAPTIVECPP__)
#define sycl_backend sycl::backend::cuda
#elif defined(SYCL_ENABLE_CUDA) && defined(__DPCPP_COMPILER)
#define sycl_backend sycl::backend::ext_oneapi_cuda
#elif defined(SYCL_ENABLE_HIP) && defined(__ADAPTIVECPP__)
#define sycl_backend sycl::backend::hip
#elif defined(SYCL_ENABLE_HIP) && defined(__DPCPP_COMPILER)
#define sycl_backend sycl::backend::ext_oneapi_hip
#endif

static void inline setSYCLDefaultExecutionDependency(
    const unsigned int streamID, sycl::handler &cgh)
{
    // Set SYCL depedencies to reproduce CUDA/HIP default stream behavior.
    if (streamID == 0)
    {
        // Tasks in the "default" queue depend on all "non-default" queue.
        for (auto &item : SYCLQueue::GetAllEvents())
        {
            if (item.first != 0)
            {
                cgh.depends_on(item.second);
            }
        }
    }
    else
    {
        // Tasks in "non-default" queue depend on the "default" queue.
        cgh.depends_on(SYCLQueue::GetEvent(0));
    }
}

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
// clang-format off
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAMID, ...)                           \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSHMEMSIZE = SHMEMSIZE;                                  \
        const auto syclSTREAMID  = STREAMID;                                   \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(syclSHMEMSIZE), cgh);                           \
            cgh.parallel_for(                                                  \
                sycl::nd_range<1>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<1> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    auto shmemptr = shmem                                      \
                        .template get_multi_ptr<sycl::access::decorated::yes>()\
                        .get();                                                \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }
// clang-format on

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
// clang-format off
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY,            \
                                      BLOCKSIZEX, BLOCKSIZEY, SHMEMSIZE,       \
                                      STREAMID, ...)                           \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSHMEMSIZE = SHMEMSIZE;                                  \
        const auto syclSTREAMID  = STREAMID;                                   \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::range<2> GRIDSIZE(GRIDSIZEY, GRIDSIZEX);                         \
        sycl::range<2> BLOCKSIZE(BLOCKSIZEY, BLOCKSIZEX);                      \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(syclSHMEMSIZE), cgh);                           \
            cgh.parallel_for(                                                  \
                sycl::nd_range<2>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<2> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    auto shmemptr = shmem                                      \
                        .template get_multi_ptr<sycl::access::decorated::yes>()\
                        .get();                                                \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// sycl::nd_item<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, \
                                      BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ,      \
                                      SHMEMSIZE, STREAMID, ...)                \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSHMEMSIZE = SHMEMSIZE;                                  \
        const auto syclSTREAMID  = STREAMID;                                   \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::range<3> GRIDSIZE(GRIDSIZEZ, GRIDSIZEY, GRIDSIZEX);              \
        sycl::range<3> BLOCKSIZE(BLOCKSIZEZ, BLOCKSIZEY, BLOCKSIZEX);          \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            sycl::local_accessor<unsigned char, 1> shmem(                      \
                sycl::range<1>(syclSHMEMSIZE), cgh);                           \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<3> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    auto shmemptr = shmem                                      \
                        .template get_multi_ptr<sycl::access::decorated::yes>()\
                        .get();                                                \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   shmemptr, item_ct1);                        \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }
// clang-format on

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<1>.
// clang-format off
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAMID, ...)                   \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSTREAMID = STREAMID;                                    \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            cgh.parallel_for(                                                  \
                sycl::nd_range<1>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<1> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }
// clang-format on

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<2>.
// clang-format off
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, BLOCKSIZEX, BLOCKSIZEY, STREAMID, ...)       \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSTREAMID = STREAMID;                                    \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::range<2> GRIDSIZE(GRIDSIZEY, GRIDSIZEX);                         \
        sycl::range<2> BLOCKSIZE(BLOCKSIZEY, BLOCKSIZEX);                      \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            cgh.parallel_for(                                                  \
                sycl::nd_range<2>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<2> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }
// clang-format on

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type sycl::nd_item<3>.
// clang-format off
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, BLOCKSIZEX, BLOCKSIZEY,           \
    BLOCKSIZEZ, STREAMID, ...)                                                 \
    {                                                                          \
        sycl::queue &Q = SYCLQueue::GetInstance(STREAMID);                     \
        const auto syclSTREAMID = STREAMID;                                    \
        const auto args = std::make_tuple(__VA_ARGS__);                        \
        sycl::range<3> GRIDSIZE(GRIDSIZEZ, GRIDSIZEY, GRIDSIZEX);              \
        sycl::range<3> BLOCKSIZE(BLOCKSIZEZ, BLOCKSIZEY, BLOCKSIZEX);          \
        sycl::event e = Q.submit([=](sycl::handler &cgh) {                     \
            setSYCLDefaultExecutionDependency(syclSTREAMID, cgh);              \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(GRIDSIZE * BLOCKSIZE, BLOCKSIZE),            \
                [=](sycl::nd_item<3> item_ct1) SYCL_SUBGROUP_SIZE(             \
                    NektarSpaces::Device::warpSize) {                          \
                    std::apply(                                                \
                        [&](auto &&...args) {                                  \
                            KERNEL(std::forward<decltype(args)>(args)...,      \
                                   item_ct1);                                  \
                        },                                                     \
                        args);                                                 \
                });                                                            \
        });                                                                    \
        SYCLQueue::SetEvent(STREAMID, e);                                      \
    }
// clang-format on
#endif
