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

#include "Operators/Common/Spaces.hpp"

namespace Nektar
{

template <size_t VectorWidth, typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    interleave(const unsigned int numMetaBlocks, const unsigned int npts,
               TData *inout)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;

    TData *buffer = sycl::malloc_device<TData>(
        VectorWidth * numMetaBlocks * npts, SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const unsigned int metaBlock = indx.get_group(0);
                const unsigned int offset    = npts * VectorWidth * metaBlock;

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * VectorWidth; idx += indx.get_local_range(0))
                {
                    buffer[offset + idx] = inout[offset + idx];
                }

                indx.barrier(sycl::access::fence_space::local_space);

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * VectorWidth; idx += indx.get_local_range(0))
                {
                    unsigned int vecElem = idx % VectorWidth;
                    unsigned int iElem   = idx / VectorWidth;
                    inout[offset + idx] =
                        buffer[offset + vecElem * npts + iElem];
                }
            });
    });

    sycl::free(buffer, SYCLQueue::GetInstance());
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    deInterleave(const unsigned int VectorWidth,
                 const unsigned int numMetaBlocks, const unsigned int npts,
                 TData *inout)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;

    TData *buffer = sycl::malloc_device<TData>(
        VectorWidth * numMetaBlocks * npts, SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const unsigned int metaBlock = indx.get_group(0);
                const unsigned int offset    = npts * VectorWidth * metaBlock;

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * VectorWidth; idx += indx.get_local_range(0))
                {
                    buffer[offset + idx] = inout[offset + idx];
                }

                indx.barrier(sycl::access::fence_space::local_space);

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * VectorWidth; idx += indx.get_local_range(0))
                {
                    unsigned int vecElem = idx / npts;
                    unsigned int iElem   = idx % npts;
                    inout[offset + idx] =
                        buffer[offset + iElem * VectorWidth + vecElem];
                }
            });
    });

    sycl::free(buffer, SYCLQueue::GetInstance());
}

template <typename ExecSpace>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BuildInterleaveMap(const unsigned int numMetaBlocks,
                       const unsigned int npts, const unsigned int newVecWidth,
                       const unsigned int offset, int *deInterleaveMapPtr,
                       int *interleaveMapPtr)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;

    int *buffer = sycl::malloc_device<int>(newVecWidth * numMetaBlocks * npts,
                                           SYCLQueue::GetInstance());

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) {
                const unsigned int metaBlock   = indx.get_group(0);
                const unsigned int groupOffset = npts * newVecWidth * metaBlock;

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * newVecWidth; idx += indx.get_local_range(0))
                {
                    for (unsigned int vecElem = 0; vecElem < newVecWidth;
                         ++vecElem)
                    {
                        buffer[groupOffset + idx] = offset + groupOffset + idx;
                    }
                }

                indx.barrier(sycl::access::fence_space::local_space);

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * newVecWidth; idx += indx.get_local_range(0))
                {
                    unsigned int vecElem = idx % newVecWidth;
                    unsigned int iElem   = idx / newVecWidth;
                    deInterleaveMapPtr[groupOffset + idx] =
                        buffer[groupOffset + vecElem * npts + iElem];
                }

                indx.barrier(sycl::access::fence_space::local_space);

                for (unsigned int idx = indx.get_local_id(0);
                     idx < npts * newVecWidth; idx += indx.get_local_range(0))
                {
                    interleaveMapPtr[deInterleaveMapPtr[groupOffset + idx]] =
                        offset + groupOffset + idx;
                }
            });
    });
}

} // namespace Nektar

#endif
