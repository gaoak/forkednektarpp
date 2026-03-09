///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceSumFacTOPKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    return nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(const unsigned int nq0,
                                                         const unsigned int nq1,
                                                         const unsigned int nm0,
                                                         const unsigned int nm1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1 + nmTot;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2 + nmTot;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nm0 * (nm0 + 1) * nm0 / 2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t inoffset,
    const TData *__restrict__ w0, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            sum += df[k * dfsize + dfindex] * in[k * inoffset + i];
        }

        if constexpr (DEFORMED)
        {
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
            out[i] = sum * jac[0] * w0[i];
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out0, TData *__restrict__ out1,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        TData tmp  = in[idx];
        TData sum1 = df[0u * dfsize + dfindex] * tmp;
        TData sum2 = df[1u * dfsize + dfindex] * tmp;
        tmp        = in[inoffset + idx];
        sum1 += df[2u * dfsize + dfindex] * tmp;
        sum2 += df[3u * dfsize + dfindex] * tmp;
        if (ncoord == 3u)
        {
            tmp = in[2u * inoffset + idx];
            sum1 += df[4u * dfsize + dfindex] * tmp;
            sum2 += df[5u * dfsize + dfindex] * tmp;
        }

        TData tmpQ = w0[i] * w1[j];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            out0[idx] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out0,
    TData *__restrict__ out1, TData *__restrict__ out2,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        TData tmp  = in[idx];
        TData sum1 = df[0u * dfsize + dfindex] * tmp;
        TData sum2 = df[1u * dfsize + dfindex] * tmp;
        TData sum3 = df[2u * dfsize + dfindex] * tmp;
        tmp        = in[inoffset + idx];
        sum1 += df[3u * dfsize + dfindex] * tmp;
        sum2 += df[4u * dfsize + dfindex] * tmp;
        sum3 += df[5u * dfsize + dfindex] * tmp;
        tmp = in[2u * inoffset + idx];
        sum1 += df[6u * dfsize + dfindex] * tmp;
        sum2 += df[7u * dfsize + dfindex] * tmp;
        sum3 += df[8u * dfsize + dfindex] * tmp;

        TData tmpQ = w0[i] * w1[j] * w2[k];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            out0[idx] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + sum3 * f0[i]) * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTDerivBase1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const unsigned int inoffset,
    const TData *__restrict__ dbasis0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *deriv = (TData *)shmemptr;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nq0 * e;
        TData *outptr       = out + nm0 * e;

        StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, inoffset, w0,
                                                     dfptr, jacptr, inptr,
                                                     deriv, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<false, false, DEFORMED>(
            nm0, nq0, dbasis0, deriv, outptr, (TData)1.0, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTDerivBase2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const unsigned int inoffset, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *deriv0    = (TData *)shmemptr;
    TData *deriv1    = deriv0 + nqTot;
    TData *deriv     = deriv1 + nqTot;
    TData *s_wsp0    = deriv + nqTot;
    TData *s_basis0  = s_wsp0 + offset;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_out1ptr = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * e;
        TData *outptr       = out + nmTot * e;

        StdAlignDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr, inptr,
            deriv0, deriv1, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<false, DEFORMED>(
            nq0, nq1, D0, D1, deriv0, deriv1, deriv, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, deriv,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, s_out1ptr, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, s_out1ptr,
                                               outptr, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTDerivBase3DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const unsigned int inoffset,
    const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *deriv0    = (TData *)shmemptr;
    TData *deriv1    = deriv0 + nqTot;
    TData *deriv2    = deriv1 + nqTot;
    TData *deriv     = deriv2 + nqTot;
    TData *s_wsp0    = deriv + nqTot;
    TData *s_wsp1    = s_wsp0 + offset0;
    TData *s_basis0  = s_wsp1 + offset1;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_basis2  = s_basis1 + nmode1 * nq1;
    TData *s_out1ptr = s_basis2 + nmode2 * nq2;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * e;
        TData *outptr       = out + nmTot * e;

        StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr, jacptr,
            inptr, deriv0, deriv1, deriv2, threadBlock);
        SumDerivTensor3DSumFacTOPKernel<false, DEFORMED>(
            nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, deriv,
            threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, deriv, outptr, s_wsp0, s_wsp1, (TData)1.0,
                threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, s_out1ptr,
                                               outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, s_out1ptr,
                                               outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, deriv, outptr, s_wsp0,
                s_wsp1, (TData)1.0, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTDerivBase1DKernelLauncher(
        NonTemplated1DSizeParameters sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const unsigned int inoffset,
        const TData *__restrict__ dbasis0, const TData *__restrict__ w0,
        const TData *__restrict__ df, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase1DSumFacTOPKernel<DEFORMED>(
        ncoord, sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt, inoffset, dbasis0,
        w0, df, jac, in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTDerivBase1DKernelLauncher(
        [[maybe_unused]] Templated1DSizeParameters<nm0, nq0> sizeParam1D,
        const unsigned int ncoord, const size_t nelmt,
        const unsigned int inoffset, const TData *__restrict__ dbasis0,
        const TData *__restrict__ w0, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase1DSumFacTOPKernel<DEFORMED>(
        ncoord, nm0, nq0, nelmt, inoffset, dbasis0, w0, df, jac, in, out, wsp,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTDerivBase2DKernelLauncher(
        NonTemplated2DSizeParameters sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const unsigned int inoffset, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, inoffset, isModified,
        index0, basis0, basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, in,
        out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TthreadBlock,
          typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTDerivBase2DKernelLauncher(
        [[maybe_unused]] Templated2DSizeParameters<nm0, nm1, nmTot, nq0, nq1>
            sizeParam2D,
        const unsigned int ncoord, const size_t nelmt,
        const unsigned int inoffset, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, inoffset, isModified, index0,
        basis0, basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, in, out, wsp,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTDerivBase3DKernelLauncher(
        NonTemplated3DSizeParameters sizeParam3D, const size_t nelmt,
        const unsigned int inoffset, const bool isModified,
        const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ D2,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ w2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
        const TData *__restrict__ df, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, inoffset, isModified, index0, index1, index2,
        basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2,
        nodToMod, df, jac, in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTDerivBase3DKernelLauncher(
        [[maybe_unused]] Templated3DSizeParameters<nm0, nm1, nm2, nmTot, nq0,
                                                   nq1, nq2>,
        const size_t nelmt, const unsigned int inoffset, const bool isModified,
        const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ D2,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ w2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
        const TData *__restrict__ df, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, inoffset, isModified,
        index0, index1, index2, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2,
        f0, f1, f1m, f2, nodToMod, df, jac, in, out, wsp, shmemptr,
        threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
