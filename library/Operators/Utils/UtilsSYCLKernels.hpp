///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsSYCLKernels.hpp
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

template <unsigned int VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int npts,
           TData *inout)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize      = VectorWidth * numMetaBlocks * npts;

    GetDeviceProperties::CheckGlobalMemoryUsage(sizeof(TData) * bufferSize);
    TData *buffer =
        sycl::malloc_device<TData>(bufferSize, SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> item_ct1) {
                             interleaveKernel(VectorWidth, numMetaBlocks, npts,
                                              buffer, inout, item_ct1);
                         });
    });

    sycl::free(buffer, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int npts, TData *inout)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize      = VectorWidth * numMetaBlocks * npts;

    GetDeviceProperties::CheckGlobalMemoryUsage(sizeof(TData) * bufferSize);
    TData *buffer =
        sycl::malloc_device<TData>(bufferSize, SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> item_ct1) {
                             deInterleaveKernel(VectorWidth, numMetaBlocks,
                                                npts, buffer, inout, item_ct1);
                         });
    });

    sycl::free(buffer, SYCLQueue::GetInstance());
}

template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
BuildInterleaveMap(const unsigned int numMetaBlocks, const unsigned int npts,
                   const unsigned int newVecWidth, const unsigned int offset,
                   int *deInterleaveMapPtr, int *interleaveMapPtr)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const size_t bufferSize      = newVecWidth * numMetaBlocks * npts;

    GetDeviceProperties::CheckGlobalMemoryUsage(sizeof(int) * bufferSize);
    int *buffer =
        sycl::malloc_device<int>(bufferSize, SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<1> item_ct1) {
                             BuildInterleaveMapKernel(
                                 npts, newVecWidth, offset, deInterleaveMapPtr,
                                 interleaveMapPtr, buffer, item_ct1);
                         });
    });

    sycl::free(buffer, SYCLQueue::GetInstance());
}

} // namespace Nektar

#endif
