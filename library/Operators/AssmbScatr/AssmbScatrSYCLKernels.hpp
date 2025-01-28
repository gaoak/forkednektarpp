///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrSYCLKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_FORCE_INLINE static void AssembleKernel(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData *__restrict__ signPtr,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
            outptr + assmbPtr[i], signPtr[i] * inptr[i]);
        i += stride;
    }
}

template <typename TData>
NEK_FORCE_INLINE static void AssembleKernel(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData sign,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
            outptr + assmbPtr[i], sign * inptr[i]);
        i += stride;
    }
}

template <typename TData>
NEK_FORCE_INLINE static void AssembleKernel(const unsigned int nsize,
                                            const int *__restrict__ assmbPtr,
                                            const TData *__restrict__ inptr,
                                            TData *__restrict__ outptr,
                                            const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
            outptr + assmbPtr[i], inptr[i]);
        i += stride;
    }
}

template <typename TData>
NEK_FORCE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData *__restrict__ signPtr, const TData *__restrict__ inptr,
    TData *__restrict__ outptr, const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        outptr[i] = signPtr[i] * inptr[assmbPtr[i]];
        i += stride;
    }
}

template <typename TData>
NEK_FORCE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData sign, const TData *__restrict__ inptr,
    TData *__restrict__ outptr, const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        outptr[i] = sign * inptr[assmbPtr[i]];
        i += stride;
    }
}

template <typename TData>
NEK_FORCE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData *__restrict__ inptr, TData *__restrict__ outptr,
    const sycl::nd_item<1> item_ct1)
{
    unsigned int i            = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    while (i < nsize)
    {
        outptr[i] = inptr[assmbPtr[i]];
        i += stride;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr,
               const TData *signPtr, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              AssembleKernel<TData>(nsize, assmbPtr, signPtr,
                                                    inptr, outptr, item_ct1);
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr, const TData sign,
               const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              AssembleKernel<TData>(nsize, assmbPtr, sign,
                                                    inptr, outptr, item_ct1);
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr,
               const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              AssembleKernel<TData>(nsize, assmbPtr, inptr,
                                                    outptr, item_ct1);
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData *signPtr, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              GlobalToLocalKernel<TData>(nsize, assmbPtr,
                                                         signPtr, inptr, outptr,
                                                         item_ct1);
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData sign, const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              GlobalToLocalKernel<TData>(nsize, assmbPtr, sign,
                                                         inptr, outptr,
                                                         item_ct1);
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData *inptr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                              GlobalToLocalKernel<TData>(nsize, assmbPtr, inptr,
                                                         outptr, item_ct1);
                          });
     }).wait();
}

} // namespace Nektar::Operators::detail

#endif
