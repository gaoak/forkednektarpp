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
#include "Operators/SYCLQueue.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  TData *const ptr = outPtr + assmbPtr[i];
                                  const TData val  = signPtr[i] * inPtr[i];
                                  Nektar::atomic_add<ExecSpace>(ptr, val);
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData sign, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  TData *const ptr = outPtr + assmbPtr[i];
                                  const TData val  = sign * inPtr[i];
                                  Nektar::atomic_add<ExecSpace>(ptr, val);
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  TData *const ptr = outPtr + assmbPtr[i];
                                  const TData val  = inPtr[i];
                                  Nektar::atomic_add<ExecSpace>(ptr, val);
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  outPtr[i] = signPtr[i] * inPtr[assmbPtr[i]];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData sign, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  outPtr[i] = sign * inPtr[assmbPtr[i]];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  outPtr[i] = inPtr[assmbPtr[i]];
                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

} // namespace Nektar::Operators::detail

#endif
