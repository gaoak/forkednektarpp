///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsAVX.hpp
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

template <size_t VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int dataLen,
           TData *inout)
{
    const unsigned int metaBlockSize = dataLen * VectorWidth;
    Nektar::Array<Nektar::OneD, TData> wsp(metaBlockSize);

    for (unsigned int metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        std::copy(inout, inout + metaBlockSize, wsp.get());

        for (unsigned int idx = 0; idx < dataLen; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < VectorWidth; ++vecElem)
            {
                inout[idx * VectorWidth + vecElem] =
                    wsp[vecElem * dataLen + idx];
            }
        }
        inout += metaBlockSize;
    }
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int dataLen, TData *inout)
{
    const unsigned int metaBlockSize = dataLen * VectorWidth;
    Nektar::Array<Nektar::OneD, TData> wsp(metaBlockSize);

    for (unsigned int metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        std::copy(inout, inout + metaBlockSize, wsp.get());

        for (unsigned int idx = 0; idx < dataLen; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < VectorWidth; ++vecElem)
            {
                inout[vecElem * dataLen + idx] =
                    wsp[idx * VectorWidth + vecElem];
            }
        }
        inout += metaBlockSize;
    }
}

template <typename ExecSpace>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
BuildInterleaveMapKernel(const unsigned int numMetaBlocks,
                         const unsigned int ncoeff,
                         const unsigned int newVecWidth,
                         const unsigned int offset, int *deInterleaveMapPtr,
                         int *interleaveMapPtr)
{
    auto MetaBlockSize = newVecWidth * ncoeff;
    std::vector<int> tmp(MetaBlockSize);
    int count = offset;

    for (size_t metaBlock = 0; metaBlock < numMetaBlocks; ++metaBlock)
    {
        // assign count+0, count+1, count+2, count+3, count+4, ....
        for (unsigned int i = 0; i < MetaBlockSize; i++)
        {
            tmp[i] = count + i;
        }
        // get the deinterleave map
        for (unsigned int n = 0; n < ncoeff; n++)
        {
            for (unsigned int vecElem = 0; vecElem < newVecWidth; ++vecElem)
            {
                deInterleaveMapPtr[n * newVecWidth + vecElem] =
                    tmp[vecElem * ncoeff + n];
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
}

} // namespace Nektar
