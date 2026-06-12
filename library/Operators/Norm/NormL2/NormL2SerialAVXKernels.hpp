///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2SerialAVXKernels.hpp
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
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type L2Norm1DKernel(const unsigned int nq0,
                                                 const simd_type *w0,
                                                 const simd_type *jac,
                                                 const simd_type *in)
{
    simd_type acc = 0.0;
    for (unsigned int i = 0; i < nq0; ++i)
    {
        if constexpr (DEFORMED)
        {
            acc.fma(in[i] * in[i], w0[i] * jac[i]);
        }
        else
        {
            acc.fma(in[i] * in[i], w0[i] * jac[0]);
        }
    }
    return acc;
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type Volume1DKernel(const unsigned int nq0,
                                                 const simd_type *w0,
                                                 const simd_type *jac)
{
    simd_type vol = 0.0;
    for (unsigned int i = 0; i < nq0; ++i)
    {
        if constexpr (DEFORMED)
        {
            vol.fma(w0[i], jac[i]);
        }
        else
        {
            vol.fma(w0[i], jac[0]);
        }
    }
    return vol;
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type L2Norm2DKernel(
    const unsigned int nq0, const unsigned int nq1, const simd_type *w0,
    const simd_type *w1, const simd_type *jac, const simd_type *in)
{
    simd_type acc  = 0.0;
    unsigned int q = 0;

    for (unsigned int j = 0; j < nq1; ++j)
    {
        const auto w1j = w1[j];
        for (unsigned int i = 0; i < nq0; ++i, ++q)
        {
            const auto weight = w0[i] * w1j;
            if constexpr (DEFORMED)
            {
                acc.fma(in[q] * in[q], weight * jac[q]);
            }
            else
            {
                acc.fma(in[q] * in[q], weight * jac[0]);
            }
        }
    }

    return acc;
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type Volume2DKernel(const unsigned int nq0,
                                                 const unsigned int nq1,
                                                 const simd_type *w0,
                                                 const simd_type *w1,
                                                 const simd_type *jac)
{
    simd_type vol  = 0.0;
    unsigned int q = 0;

    for (unsigned int j = 0; j < nq1; ++j)
    {
        const auto w1j = w1[j];
        for (unsigned int i = 0; i < nq0; ++i, ++q)
        {
            const auto weight = w0[i] * w1j;
            if constexpr (DEFORMED)
            {
                vol.fma(weight, jac[q]);
            }
            else
            {
                vol.fma(weight, jac[0]);
            }
        }
    }

    return vol;
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type L2Norm3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *w0, const simd_type *w1, const simd_type *w2,
    const simd_type *jac, const simd_type *in)
{
    simd_type acc  = 0.0;
    unsigned int q = 0;

    for (unsigned int k = 0; k < nq2; ++k)
    {
        const auto w2k = w2[k];
        for (unsigned int j = 0; j < nq1; ++j)
        {
            const auto w12 = w1[j] * w2k;
            for (unsigned int i = 0; i < nq0; ++i, ++q)
            {
                const auto weight = w0[i] * w12;
                if constexpr (DEFORMED)
                {
                    acc.fma(in[q] * in[q], weight * jac[q]);
                }
                else
                {
                    acc.fma(in[q] * in[q], weight * jac[0]);
                }
            }
        }
    }

    return acc;
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type Volume3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *w0, const simd_type *w1, const simd_type *w2,
    const simd_type *jac)
{
    simd_type vol  = 0.0;
    unsigned int q = 0;

    for (unsigned int k = 0; k < nq2; ++k)
    {
        const auto w2k = w2[k];
        for (unsigned int j = 0; j < nq1; ++j)
        {
            const auto w12 = w1[j] * w2k;
            for (unsigned int i = 0; i < nq0; ++i, ++q)
            {
                const auto weight = w0[i] * w12;
                if constexpr (DEFORMED)
                {
                    vol.fma(weight, jac[q]);
                }
                else
                {
                    vol.fma(weight, jac[0]);
                }
            }
        }
    }

    return vol;
}

} // namespace Nektar::Operators::detail
