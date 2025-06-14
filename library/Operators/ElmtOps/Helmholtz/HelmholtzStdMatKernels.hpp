///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzStdMatKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                            void>::type
    MultiplyByDiffusionCoeff(const size_t nelmt, const unsigned int nqTot,
                             const unsigned int ncoord, const size_t outsize,
                             const TData *diffCoeff, TData *inout)
{
    const auto nsize = nelmt * nqTot;

    // Multiply by diffusion coefficient.
    for (size_t idx = 0; idx < nsize; idx++)
    {
        TData tmp[3];
        for (unsigned int d = 0; d < ncoord; d++)
        {
            tmp[d] = diffCoeff[d * ncoord] * inout[idx];
            for (unsigned int l = 1; l < ncoord; l++)
            {
                tmp[d] += diffCoeff[d * ncoord + l] * inout[l * outsize + idx];
            }
        }

        for (unsigned int d = 0; d < ncoord; d++)
        {
            inout[d * outsize + idx] = tmp[d];
        }
    }
}

template <typename ExecSpace, typename TData, typename TScalar>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    MultiplyByDiffusionCoeff(const size_t nelmt, const unsigned int nqTot,
                             const unsigned int ncoord, const size_t outsize,
                             const TScalar *diffCoeff, TData *inout)
{
    const auto nsize = nelmt * nqTot;

    // Multiply by diffusion coefficient.
    for (size_t idx = 0; idx < nsize; idx++)
    {
        TData tmp[3];
        for (unsigned int d = 0; d < ncoord; d++)
        {
            tmp[d] = diffCoeff[d * ncoord] * inout[idx];
            for (unsigned int l = 1; l < ncoord; l++)
            {
                tmp[d].fma(diffCoeff[d * ncoord + l], inout[l * outsize + idx]);
            }
        }

        for (unsigned int d = 0; d < ncoord; d++)
        {
            inout[d * outsize + idx] = tmp[d];
        }
    }
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    MultiplyByDiffusionCoeff(const size_t nelmt, const unsigned int nqTot,
                             const unsigned int ncoord, const size_t outsize,
                             const TData *diffCoeff, TData *inout)
{
    const auto nsize = nelmt * nqTot;

    // Multiply by diffusion coefficient.
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            TData tmp[3];
            for (unsigned int d = 0; d < ncoord; d++)
            {
                tmp[d] = diffCoeff[d * ncoord] * inout[idx];
                for (unsigned int l = 1; l < ncoord; l++)
                {
                    tmp[d] +=
                        diffCoeff[d * ncoord + l] * inout[l * outsize + idx];
                }
            }

            for (unsigned int d = 0; d < ncoord; d++)
            {
                inout[d * outsize + idx] = tmp[d];
            }
        });
}
