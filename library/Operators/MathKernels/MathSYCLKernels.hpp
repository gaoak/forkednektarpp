///////////////////////////////////////////////////////////////////////////////
//
// File: MathSYCLKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Nektar
{

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    negKernel(const unsigned int nsize, const TData *x, TData *y)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  y[i] = -x[i];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    addKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  z[i] = x[i] + y[i];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    subKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  z[i] = x[i] - y[i];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    daxpyKernel(const unsigned int nsize, const TData alpha, const TData *x,
                const TData *y, TData *z)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  z[i] = alpha * x[i] + y[i];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    divKernel(const unsigned int nsize, const TData *x, const TData *y,
              TData *z)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  z[i] = x[i] / y[i];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename TData>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = 0.0;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = 0.0;
                 while (gid < nsize)
                 {
                     tmp += x[gid];
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int nsize, const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = min;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = min;
                 while (gid < nsize)
                 {
                     tmp = sycl::fmax(tmp, x[gid]);
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const unsigned int nsize, const TData *x, TData *out)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = max;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = max;
                 while (gid < nsize)
                 {
                     tmp = sycl::fmin(tmp, x[gid]);
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void ddotKernel(const unsigned int gridSize, const unsigned int blockSize,
                const unsigned int nsize, const TData *x, const TData *y,
                TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = 0.0;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = 0.0;
                 while (gid < nsize)
                 {
                     tmp += x[gid] * y[gid];
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void l1normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = 0.0;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = 0.0;
                 while (gid < nsize)
                 {
                     tmp += sycl::fabs(x[gid]);
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void l2normKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int nsize, const TData *x, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = 0.0;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = 0.0;
                 while (gid < nsize)
                 {
                     tmp += x[gid] * x[gid];
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void lpnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                  const unsigned int nsize, const int p, const TData *x,
                  TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = 0.0;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = 0.0;
                 while (gid < nsize)
                 {
                     tmp += sycl::pown(sycl::fabs(x[gid]), p);
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename TData>
void linfnormKernel(const unsigned int gridSize, const unsigned int blockSize,
                    const unsigned int nsize, const TData *x, TData *out)
{
    constexpr TData min = std::numeric_limits<TData>::min();

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> scratch(sycl::range<1>(blockSize), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 const unsigned int lid = indx.get_local_id(0);
                 unsigned int gid       = indx.get_global_id(0);

                 if (lid == 0)
                 {
                     out[indx.get_group(0)] = min;
                 }

                 indx.barrier(sycl::access::fence_space::local_space);

                 TData tmp = min;
                 while (gid < nsize)
                 {
                     tmp = sycl::fmax(tmp, sycl::fabs(x[gid]));
                     gid += indx.get_global_range(0);
                 }
                 scratch[lid] = tmp;

                 indx.barrier(sycl::access::fence_space::local_space);

                 unsigned int n = 512;
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
                     out[indx.get_group(0)] = scratch[0];
                 }
             });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    reduceSumKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    reduceSumKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceSumKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    reduceMaxKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    reduceMaxKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceMaxKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    reduceMinKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    reduceMinKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceMinKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    ddotKernel(const unsigned int nsize, const TData *x, const TData *y,
               TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    ddotKernel<TData>(gridSize, blockSize, nsize, x, y, (TData *)syclBuffer);
    reduceSumKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    l1normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    l1normKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceSumKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    l2normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    l2normKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceSumKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    lpnormKernel(const unsigned int nsize, const unsigned int p, const TData *x,
                 TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer =
            sycl::malloc_device<TData>(gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    lpnormKernel<TData>(gridSize, blockSize, nsize, p, x, (TData *)syclBuffer);
    reduceSumKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    linfnormKernel(const unsigned int nsize, const TData *x, TData *out)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::SYCL::defaultGridSize;

    if (syclBuffer == nullptr)
    {
        syclBuffer = (void *)sycl::malloc_device<TData>(
            gridSize, SYCLQueue::GetInstance());
    }
    TData *d_out = sycl::malloc_device<TData>(1, SYCLQueue::GetInstance());
    linfnormKernel<TData>(gridSize, blockSize, nsize, x, (TData *)syclBuffer);
    reduceMaxKernel<TData>(1, gridSize, gridSize, (TData *)syclBuffer, d_out);
    SYCLQueue::GetInstance().memcpy(out, d_out, sizeof(TData)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

} // namespace Nektar

#endif
