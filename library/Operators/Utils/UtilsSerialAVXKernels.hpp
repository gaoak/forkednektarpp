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

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    interleave(const unsigned int interleaveWidth, const size_t numElmtGroups,
               const unsigned int npts, TData *inout)
{
    const unsigned int elmtGroupSize = npts * interleaveWidth;
    std::vector<TData> wsp(elmtGroupSize);

    for (size_t e = 0; e < numElmtGroups; ++e)
    {
        std::copy(inout, inout + elmtGroupSize, wsp.data());

        for (unsigned int idx = 0; idx < npts; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < interleaveWidth; ++vecElem)
            {
                inout[idx * interleaveWidth + vecElem] =
                    wsp[vecElem * npts + idx];
            }
        }
        inout += elmtGroupSize;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    deInterleave(const unsigned int interleaveWidth, const size_t numElmtGroups,
                 const unsigned int npts, TData *inout)
{
    const unsigned int elmtGroupSize = npts * interleaveWidth;
    std::vector<TData> wsp(elmtGroupSize);

    for (size_t e = 0; e < numElmtGroups; ++e)
    {
        std::copy(inout, inout + elmtGroupSize, wsp.data());

        for (unsigned int idx = 0; idx < npts; ++idx)
        {
            for (unsigned int vecElem = 0; vecElem < interleaveWidth; ++vecElem)
            {
                inout[vecElem * npts + idx] =
                    wsp[idx * interleaveWidth + vecElem];
            }
        }
        inout += elmtGroupSize;
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    MultiplyByJacobian(const size_t nelmt, const unsigned int nqTot,
                       const TData *jacptr, const TData *inptr, TData *outptr,
                       const TScalar scale)
{
    if constexpr (DEFORMED)
    {
        for (size_t i = 0; i < nelmt * nqTot; ++i)
        {
            outptr[i] = scale * jacptr[i] * inptr[i];
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; ++e)
        {
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                outptr[e * nqTot + i] =
                    scale * jacptr[e] * inptr[e * nqTot + i];
            }
        }
    }
}

template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    DivideByJacobian(const size_t nelmt, const unsigned int nqTot,
                     const TData *jacptr, const TData *inptr, TData *outptr)
{
    if constexpr (DEFORMED)
    {
        for (size_t i = 0; i < nelmt * nqTot; ++i)
        {
            outptr[i] = inptr[i] / jacptr[i];
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; ++e)
        {
            TData invjac = 1.0 / jacptr[e];
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                outptr[e * nqTot + i] = inptr[e * nqTot + i] * invjac;
            }
        }
    }
}

} // namespace Nektar
