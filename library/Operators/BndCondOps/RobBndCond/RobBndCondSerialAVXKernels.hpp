///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondSerialAVXKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
RobBndCond1DKernel(const unsigned int nsize, const unsigned int *offsetPtr,
                   const TData *matPtr, const unsigned int *mapPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    if constexpr (negflag)
    {
        for (unsigned int i = 0; i < nsize; i++)
        {
            unsigned int offset = offsetPtr[i];
            unsigned int map    = mapPtr[i];
            coeffPtr[offset + map] -= matPtr[i] * incoeffPtr[offset + map];
        }
    }
    else
    {
        for (unsigned int i = 0; i < nsize; i++)
        {
            unsigned int offset = offsetPtr[i];
            unsigned int map    = mapPtr[i];
            coeffPtr[offset + map] += matPtr[i] * incoeffPtr[offset + map];
        }
    }
}

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
RobBndCond2DKernel(const unsigned int nmaxcoeff, const unsigned int nsize,
                   const unsigned int *ncoeffPtr, const unsigned int *offsetPtr,
                   const unsigned int *matOffsetPtr,
                   const unsigned int *mapOffsetPtr, const TData *matPtr,
                   const unsigned int *mapPtr, const int *signPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    std::vector<TData> vEdgeCoeffs(nmaxcoeff);
    for (unsigned int j = 0; j < nsize; j++)
    {
        const unsigned int ncoeff    = ncoeffPtr[j];
        const unsigned int offset    = offsetPtr[j];
        const unsigned int matOffset = matOffsetPtr[j];
        const unsigned int mapOffset = mapOffsetPtr[j];

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            const unsigned int index = mapOffset + i;
            vEdgeCoeffs[i] =
                incoeffPtr[offset + mapPtr[index]] * signPtr[index];
        }

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            TData tmp = 0.0;
            for (unsigned int k = 0; k < ncoeff; k++)
            {
                tmp += matPtr[matOffset + ncoeff * k + i] * vEdgeCoeffs[k];
            }

            const unsigned int index = mapOffset + i;
            if constexpr (negflag)
            {
                coeffPtr[offset + mapPtr[index]] -= tmp * signPtr[index];
            }
            else
            {
                coeffPtr[offset + mapPtr[index]] += tmp * signPtr[index];
            }
        }
    }
}

} // namespace Nektar::Operators::detail
