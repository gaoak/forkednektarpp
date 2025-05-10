///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsSerialAVXKernels.hpp
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

#include "Operators/Common/Spaces.hpp"

namespace Nektar
{

template <unsigned int VectorWidth, typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    interleave(const size_t numMetaBlocks, const unsigned int npts,
               TData *inout)
{
    const unsigned int metaBlockSize = npts * VectorWidth;
    std::vector<TData> wsp(metaBlockSize);

    for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        std::copy(inout, inout + metaBlockSize, wsp.data());

        for (unsigned int idx = 0; idx < npts; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < VectorWidth; ++vecElem)
            {
                inout[idx * VectorWidth + vecElem] = wsp[vecElem * npts + idx];
            }
        }
        inout += metaBlockSize;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    deInterleave(const unsigned int VectorWidth, const size_t numMetaBlocks,
                 const unsigned int npts, TData *inout)
{
    const unsigned int metaBlockSize = npts * VectorWidth;
    std::vector<TData> wsp(metaBlockSize);

    for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        std::copy(inout, inout + metaBlockSize, wsp.data());

        for (unsigned int idx = 0; idx < npts; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < VectorWidth; ++vecElem)
            {
                inout[vecElem * npts + idx] = wsp[idx * VectorWidth + vecElem];
            }
        }
        inout += metaBlockSize;
    }
}

/*template <typename ExecSpace>
inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    BuildInterleaveMap(const unsigned int numMetaBlocks,
                       const unsigned int npts, const unsigned int newVecWidth,
                       const unsigned int offset, int *deInterleaveMapPtr,
                       int *interleaveMapPtr)
{
    auto MetaBlockSize = newVecWidth * npts;
    std::vector<int> tmp(MetaBlockSize);
    unsigned int count = offset;

    for (unsigned int metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        // assign count+0, count+1, count+2, count+3, count+4, ....
        for (unsigned int i = 0; i < MetaBlockSize; i++)
        {
            tmp[i] = count + i;
        }
        // get the deinterleave map
        for (unsigned int n = 0; n < npts; n++)
        {
            for (unsigned int vecElem = 0; vecElem < newVecWidth; ++vecElem)
            {
                deInterleaveMapPtr[n * newVecWidth + vecElem] =
                    tmp[vecElem * npts + n];
            }
        }
        // get the interleave map
        for (unsigned int i = 0; i < MetaBlockSize; i++)
        {
            interleaveMapPtr[deInterleaveMapPtr[i]] = count + i;
        }
        deInterleaveMapPtr += MetaBlockSize;
        count += MetaBlockSize;
    }
}*/

template <typename simd_type>
NEK_FORCE_INLINE static void MatVecKernel(const unsigned int n,
                                          const simd_type *Mat,
                                          const simd_type *in, simd_type *out)
{
    for (unsigned int i = 0, cnt = 0; i < n; ++i)
    {
        simd_type i_sum = 0.0;

        for (unsigned int j = 0; j < n; ++j, ++cnt)
        {
            i_sum.fma(Mat[cnt], in[j]);
        }

        out[i] = i_sum; // Store 1x
    }
}
} // namespace Nektar
