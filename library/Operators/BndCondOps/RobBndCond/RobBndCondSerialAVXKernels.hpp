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
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    RobBndCond1DKernel(const size_t nsize, const size_t *offsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    if constexpr (negflag)
    {
        for (size_t i = 0; i < nsize; i++)
        {
            size_t offset = offsetPtr[i];
            size_t map    = mapPtr[i];
            coeffPtr[offset + map] -= matPtr[i] * incoeffPtr[offset + map];
        }
    }
    else
    {
        for (size_t i = 0; i < nsize; i++)
        {
            size_t offset = offsetPtr[i];
            size_t map    = mapPtr[i];
            coeffPtr[offset + map] += matPtr[i] * incoeffPtr[offset + map];
        }
    }
}

template <typename ExecSpace, bool negflag, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    RobBndCond2DKernel(const unsigned int nmaxcoeff, const size_t nsize,
                       const unsigned int *ncoeffPtr, const size_t *offsetPtr,
                       const size_t *matOffsetPtr, const size_t *mapOffsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const int *signPtr, const TData *incoeffPtr,
                       TData *coeffPtr)
{
    std::vector<TData> vEdgeCoeffs(nmaxcoeff);
    for (size_t j = 0; j < nsize; j++)
    {
        const unsigned int ncoeff = ncoeffPtr[j];
        const size_t offset       = offsetPtr[j];
        const size_t matOffset    = matOffsetPtr[j];
        const size_t mapOffset    = mapOffsetPtr[j];

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            const size_t index = mapOffset + i;
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

            const size_t index = mapOffset + i;
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
