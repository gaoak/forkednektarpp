///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondSYCLKernels.hpp
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

template <typename ExecSpace, bool negflag, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    RobBndCond1DKernel(const unsigned int nsize, const unsigned int *offsetPtr,
                       const TData *matPtr, const unsigned int *mapPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::width;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<1> indx) {
                              unsigned int i = indx.get_global_id(0);

                              while (i < nsize)
                              {
                                  const unsigned int offset = offsetPtr[i];
                                  const unsigned int map    = mapPtr[i];

                                  TData *const ptr = coeffPtr + offset + map;
                                  const TData val =
                                      matPtr[i] * incoeffPtr[offset + map];
                                  if constexpr (negflag)
                                  {
                                      Nektar::atomic_sub<ExecSpace>(ptr, val);
                                  }
                                  else
                                  {
                                      Nektar::atomic_add<ExecSpace>(ptr, val);
                                  }

                                  i += indx.get_global_range(0);
                              }
                          });
     }).wait();
}

template <typename ExecSpace, bool negflag, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    RobBndCond2DKernel(const unsigned int nmaxcoeff, const unsigned int nsize,
                       const unsigned int *ncoeffPtr,
                       const unsigned int *offsetPtr,
                       const unsigned int *matOffsetPtr,
                       const unsigned int *mapOffsetPtr, const TData *matPtr,
                       const unsigned int *mapPtr, const int *signPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::width;
    const unsigned int gridSize  = nsize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> vEdgeCoeffs(sycl::range<1>(nmaxcoeff),
                                                    cgh);
         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> indx) {
                 unsigned int j = indx.get_group(0);
                 TData *vEdgeCoeffsPtr =
                     vEdgeCoeffs
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();

                 while (j < nsize)
                 {
                     const unsigned int ncoeff    = ncoeffPtr[j];
                     const unsigned int offset    = offsetPtr[j];
                     const unsigned int matOffset = matOffsetPtr[j];
                     const unsigned int mapOffset = mapOffsetPtr[j];

                     for (unsigned int i = indx.get_local_id(0); i < ncoeff;
                          i += indx.get_local_range(0))
                     {
                         const unsigned int index = mapOffset + i;
                         vEdgeCoeffsPtr[i] =
                             incoeffPtr[offset + mapPtr[index]] *
                             signPtr[index];
                     }

                     indx.barrier(sycl::access::fence_space::local_space);

                     for (unsigned int i = indx.get_local_id(0); i < ncoeff;
                          i += indx.get_local_range(0))
                     {
                         TData tmp = 0.0;
                         for (unsigned int k = 0; k < ncoeff; k++)
                         {
                             tmp += matPtr[matOffset + ncoeff * k + i] *
                                    vEdgeCoeffsPtr[k];
                         }

                         const unsigned int index = mapOffset + i;
                         TData *const ptr = coeffPtr + offset + mapPtr[index];
                         const TData val  = tmp * signPtr[index];
                         if constexpr (negflag)
                         {
                             Nektar::atomic_sub<ExecSpace>(ptr, val);
                         }
                         else
                         {
                             Nektar::atomic_add<ExecSpace>(ptr, val);
                         }
                     }

                     indx.barrier(sycl::access::fence_space::local_space);

                     j += indx.get_group_range(0);
                 }
             });
     }).wait();
}

} // namespace Nektar::Operators::detail

#endif
