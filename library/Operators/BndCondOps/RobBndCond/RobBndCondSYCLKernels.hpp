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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
RobBndCond1DKernel(const unsigned int nsize, const unsigned int *offsetPtr,
                   const TData *matPtr, const unsigned int *mapPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                             RobBndCond1DKernel<negflag>(
                                 nsize, offsetPtr, matPtr, mapPtr, incoeffPtr,
                                 coeffPtr, item_ct1);
                         });
    });
}

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
RobBndCond2DKernel(const unsigned int nmaxcoeff, const unsigned int nsize,
                   const unsigned int *ncoeffPtr, const unsigned int *offsetPtr,
                   const unsigned int *matOffsetPtr,
                   const unsigned int *mapOffsetPtr, const TData *matPtr,
                   const unsigned int *mapPtr, const int *signPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;
    const unsigned int gridSize  = nsize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(nmaxcoeff), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr =
                    shmem.template get_multi_ptr<sycl::access::decorated::no>()
                        .get();
#pragma forceinline
                RobBndCond2DKernel<negflag>(nsize, ncoeffPtr, offsetPtr,
                                            matOffsetPtr, mapOffsetPtr, matPtr,
                                            mapPtr, signPtr, incoeffPtr,
                                            coeffPtr, shmemptr, item_ct1);
            });
    });
}

} // namespace Nektar::Operators::detail

#endif
