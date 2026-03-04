///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFacTOPKernels.hpp
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

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int MassSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    return nq0 + nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int MassSharedMemorySize(const unsigned int nq0,
                                         const unsigned int nq1,
                                         const unsigned int nm0,
                                         const unsigned int nm1)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + nmTot + nq0 * nq1 +
               std::max(nq0 * nm1, nm0 * nq1);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq0 + nmTot * nq1 + nmTot + nq0 * nq1 + nm0 * nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int MassSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) +
               std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void MassSumFacTOP1DKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *bwd = (TData *)shmemptr;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * e;
        TData *outptr       = out + nm0 * e;
        BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, inptr, bwd, threadBlock);

        for (unsigned int i = idx0; i < nq0; i += stride)
        {
            if constexpr (DEFORMED)
            {
                bwd[i] *= jacptr[i] * w0[i];
            }
            else
            {
                bwd[i] *= jacptr[0] * w0[i];
            }
        }

        localBarrier(threadBlock);

        IProductWRTBaseSegSumFacTOPKernel<false, false, DEFORMED>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void MassSumFacTOP2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset{0}, nmode0{0}, nmode1{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = std::max(nm0 * nq1, nm1 * nq0);
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

    TData *tmp      = (TData *)shmemptr;
    TData *bwd      = tmp + nmTot;
    TData *s_wsp0   = bwd + nqTot;
    TData *s_basis0 = s_wsp0 + offset;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

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
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }
            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                        s_basis1, tmp, bwd, s_wsp0,
                                        threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                       s_basis0, s_basis1, tmp, bwd, s_wsp0,
                                       threadBlock);
        }

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            if constexpr (DEFORMED)
            {
                bwd[idx] *= jacptr[idx] * w0[i] * w1[j];
            }
            else
            {
                bwd[idx] *= jacptr[0] * w0[i] * w1[j];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, tmp, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void MassSumFacTOP3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    [[maybe_unused]] const unsigned int *__restrict__ index3,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset0{0}, offset1{0}, nmode0{0}, nmode1{0}, nmode2{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
        offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *tmp      = (TData *)shmemptr;
    TData *bwd      = tmp + nmTot;
    TData *s_wsp0   = bwd + nqTot;
    TData *s_wsp1   = s_wsp0 + offset0;
    TData *s_basis0 = s_wsp1 + offset1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

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
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTet ||
                      SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       s_basis0, s_basis1, s_basis2, tmp, bwd,
                                       s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, index0, index3, s_basis0,
                                       s_basis1, s_basis2, tmp, bwd, s_wsp0,
                                       s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, s_basis0, s_basis1, s_basis2,
                                       tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            if constexpr (DEFORMED)
            {
                bwd[idx] *= jacptr[idx] * w0[i] * w1[j] * w2[k];
            }
            else
            {
                bwd[idx] *= jacptr[0] * w0[i] * w1[j] * w2[k];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, bwd, outptr, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Mass1DKernelLauncher(const unsigned int nm0, const unsigned int nq0,
                         const size_t nelmt, const TData *__restrict__ basis0,
                         const TData *__restrict__ w0,
                         const TData *__restrict__ jac,
                         const TData *__restrict__ in, TData *__restrict__ out,
                         TData *__restrict__ wsp, unsigned char *shmemptr,
                         const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP1DKernel<DEFORMED>(nm0, nq0, nelmt, basis0, w0, jac, in, out,
                                    wsp, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock)
        Mass1DKernelLauncher(const size_t nelmt,
                             const TData *__restrict__ basis0,
                             const TData *__restrict__ w0,
                             const TData *__restrict__ jac,
                             const TData *__restrict__ in,
                             TData *__restrict__ out, TData *__restrict__ wsp,
                             unsigned char *shmemptr,
                             const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP1DKernel<DEFORMED>(nm0, nq0, nelmt, basis0, w0, jac, in, out,
                                    wsp, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Mass2DKernelLauncher(
        const unsigned int nm0, const unsigned int nm1,
        const unsigned int nmTot, const unsigned int nq0,
        const unsigned int nq1, const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, nodToMod, jac, in, out, wsp, shmemptr, threadBlock);
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
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Mass2DKernelLauncher(
        const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP2DKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, nodToMod, jac, in, out, wsp, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Mass3DKernelLauncher(
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nmTot, const unsigned int nq0,
        const unsigned int nq1, const unsigned int nq2, const size_t nelmt,
        const bool isModified, const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const unsigned int *__restrict__ index3,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ w0,
        const TData *__restrict__ w1, const TData *__restrict__ w2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, w0, w1, w2, nodToMod, jac, in,
        out, wsp, shmemptr, threadBlock);
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
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Mass3DKernelLauncher(
        const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const unsigned int *__restrict__ index3,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ w0,
        const TData *__restrict__ w1, const TData *__restrict__ w2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    MassSumFacTOP3DKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, w0, w1, w2, nodToMod, jac, in,
        out, wsp, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
