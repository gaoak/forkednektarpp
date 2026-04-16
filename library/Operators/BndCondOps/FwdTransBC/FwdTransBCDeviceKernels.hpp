///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCDeviceKernels.hpp
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

#include "ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp"
#include "ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/Utils/UtilsDeviceKernels.hpp"

namespace Nektar::Operators::detail
{
#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
inline unsigned int FwdTransBCSharedMemorySize(
    [[maybe_unused]] const unsigned int nm0, const unsigned int nq0)
{
    return 2 * nq0;
}

inline unsigned int FwdTransBCSharedMemorySize(
    const LibUtilities::ShapeType shapeType,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1)
{
    if (shapeType == LibUtilities::Quad)
    {
        return 2 * nq0 * nq1 + std::max(nq0 * nq1, 2 * nq0 + 2 * nq1) +
               std::max(nq0, nq1);
    }
    else if (shapeType == LibUtilities::Tri)
    {
        return 3 * nq0 * nq1 + std::max(nq0, nq1);
    }
    else if (shapeType == LibUtilities::NodalTri)
    {
        return 0;
    }

    return 0;
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void FwdTransBCSegSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const unsigned int offset_seg, const TData *NEK_RESTRICT invintmass,
    [[maybe_unused]] const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // Get size of interior DoF
    // Note nmBndSeg is correct for modified and GLL_Lagrange basis
    const unsigned int nmBndSeg = 2;
    const unsigned int nmInt    = nm0 - nmBndSeg;

    /// Step 1: Evaluate vertex contriubtion
    /// i.e set vertex modes and evaluate mass matrix
    // TODO need different map for right vertex when using GLL_LAGRANGE
    for (unsigned int i = idx0; i < nmBndSeg; i += stride)
    {
        unsigned int idBnd = i == 0 ? 0 : nq0 - 1;
        out[i]             = in[idBnd];
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Do BwdTrans of vertex modes
    BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, out, wsp2, threadBlock);

    // Apply Jacobian and quadrature weights for IProduct
    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        if constexpr (DEFORMED)
        {
            wsp2[i] = (in[i] - wsp2[i]) * jac[i] * w0[i];
        }
        else
        {
            wsp2[i] = (in[i] - wsp2[i]) * w0[i];
        }
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Do IProduct to complete Mass matrix
    // use negative scale to subtract from Dirichlet condition
    IProductWRTBaseSegSumFacTOPKernel<false, false>(
        nm0, nq0, basis0, wsp2, wsp1, (TData)1.0, threadBlock);

    // Step 4: Project edge interior modes onto boundary
    // Matrix vector product with interior mass matrix
    TData tmp;
    for (unsigned int i = idx0; i < nmInt; i += stride)
    {
        tmp = 0.0;
        for (unsigned int j = 0; j < nmInt; j++)
        {
            tmp = tmp + wsp1[j + offset_seg] * invintmass[i * nmInt + j];
        }

        out[i + offset_seg] = tmp;
    }

    // synchronize threads.
    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void FwdTransBCQuadSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const unsigned int offset_seg, const TData *NEK_RESTRICT invintmass0,
    const TData *NEK_RESTRICT invintmass1, const TData *NEK_RESTRICT tJac,
    const unsigned int *NEK_RESTRICT tMap, const int *NEK_RESTRICT tSign,
    const unsigned int nmTotInt, const unsigned int *iMap,
    const TData *NEK_RESTRICT invintmass,
    [[maybe_unused]] const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    TData *NEK_RESTRICT wsp3, TData *NEK_RESTRICT wsp4,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // Get relevant sizes in phys and coeff space
    const unsigned int nqTot     = nq0 * nq1;
    const unsigned int nmEdgeTot = 2 * nm0 + 2 * nm1;

    /// Step 1: Extract edge modes
    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        wsp1[i]             = in[i];
        wsp1[i + nq0 + nq1] = in[nq0 * (nq1 - 1) + i];
    }
    for (unsigned int i = idx0; i < nq1; i += stride)
    {
        wsp1[i + nq0]           = in[nq0 - 1 + i * nq0];
        wsp1[i + 2 * nq0 + nq1] = in[i * nq0];
    }

    // Zero wsp2
    for (unsigned int i = idx0; i < nmEdgeTot; i += stride)
    {
        wsp2[i] = 0.0;
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Remove vertex contribution for every edge
    // TODO Do we need a separate check (vector of booleans) whether segment
    // is deformed?
    FwdTransBCSegSumFacTOPKernel<false>(nm0, nq0, basis0, w0, offset_seg,
                                        invintmass0, tJac, wsp1, wsp2, wsp3,
                                        wsp4, threadBlock);
    FwdTransBCSegSumFacTOPKernel<false>(nm1, nq1, basis1, w1, offset_seg,
                                        invintmass1, tJac + 1, wsp1 + nq0,
                                        wsp2 + nm0, wsp3, wsp4, threadBlock);
    FwdTransBCSegSumFacTOPKernel<false>(
        nm0, nq0, basis0, w0, offset_seg, invintmass0, tJac + 2,
        wsp1 + nq0 + nq1, wsp2 + nm0 + nm1, wsp3, wsp4, threadBlock);
    FwdTransBCSegSumFacTOPKernel<false>(
        nm1, nq1, basis1, w1, offset_seg, invintmass1, tJac + 3,
        wsp1 + 2 * nq0 + nq1, wsp2 + 2 * nm0 + nm1, wsp3, wsp4, threadBlock);

    // Map edge modes (without vertex contribution) back into face
    for (unsigned int j = idx0; j < nmEdgeTot; j += stride)
    {
        out[tMap[j]] = tSign[j] * wsp2[j];
    }

    // synchronize threads.
    localBarrier(threadBlock);

    /// Step 2: Evaluate edge contributions via mass matrix
    BwdTransQuadSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, basis0, basis1, out,
                                wsp1, wsp2, threadBlock);

    // Apply jacobian and quadrature weights
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        if constexpr (DEFORMED)
        {
            wsp1[idx] = (in[idx] - wsp1[idx]) * jac[idx] * w0[i] * w1[j];
        }
        else
        {
            wsp1[idx] = (in[idx] - wsp1[idx]) * w0[i] * w1[j];
        }
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Complete mass matrix with IProduct
    IProductWRTBaseQuadSumFacTOPKernel<false, false>(
        nm0, nm1, nmTot, nq0, nq1, nqTot, basis0, basis1, wsp1, wsp2, wsp3,
        (TData)1.0, threadBlock);

    /// Step 3: Project face/interior modes onto boundary
    // Map to interior coeffs
    for (unsigned int j = idx0; j < nmTotInt; j += stride)
    {
        wsp1[j] = wsp2[iMap[j]];
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Projection ie matrix vector product with inverse interior mass
    Nektar::MatVecSumFacTOPKernel<false, false>(nmTotInt, invintmass, wsp1,
                                                wsp2, threadBlock);

    // Map to volume
    for (unsigned int j = idx0; j < nmTotInt; j += stride)
    {
        out[iMap[j]] = wsp2[j];
    }

    // synchronize threads.
    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void FwdTransBCTriSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const unsigned int *NEK_RESTRICT index0, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT interp1to0,
    const unsigned int offset_seg, const TData *NEK_RESTRICT invintmass0,
    const TData *NEK_RESTRICT tJac, const unsigned int *NEK_RESTRICT tMap,
    const int *NEK_RESTRICT tSign, const unsigned int nmTotInt,
    const unsigned int *iMap, const TData *NEK_RESTRICT invintmass,
    [[maybe_unused]] const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    TData *NEK_RESTRICT wsp3, TData *NEK_RESTRICT wsp4,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // Get relevant sizes in phys and coeff space
    const unsigned int nqTot     = nq0 * nq1;
    const unsigned int nmEdgeTot = nm0 + 2 * nm1;

    /// Step 1: Extract edge modes
    TData tmp, tmp2;
    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        // Interpolate basis1 to basis0 for edge1 and edge2
        // Note interpolation only required if basis1 != GLL
        // Interpolate always to avoid branching
        tmp = 0.0, tmp2 = 0.0;
        for (unsigned int j = 0u; j < nq1; j++)
        {
            tmp += in[nq0 - 1 + j * nq0] * interp1to0[i * nq1 + j];
            tmp2 += in[j * nq0] * interp1to0[i * nq1 + j];
        }

        wsp1[i]           = in[i];
        wsp1[i + nq0]     = tmp;
        wsp1[i + 2 * nq0] = tmp2;
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Zero wsp2
    for (unsigned int i = idx0; i < nmEdgeTot; i += stride)
    {
        wsp2[i] = 0.0;
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Compute vertex contribution for every edge
    // TODO Do we need a separate check whether segment is deformed?
    FwdTransBCSegSumFacTOPKernel<false>(nm0, nq0, basis0, w0, offset_seg,
                                        invintmass0, tJac, wsp1, wsp2, wsp3,
                                        wsp4, threadBlock);
    FwdTransBCSegSumFacTOPKernel<false>(nm0, nq0, basis0, w0, offset_seg,
                                        invintmass0, tJac + 1, wsp1 + nq0,
                                        wsp2 + nm0, wsp3, wsp4, threadBlock);

    // Map edge modes (without vertex contribution) back into face
    for (unsigned int j = idx0; j < 2 * nm0; j += stride)
    {
        out[tMap[j]] = tSign[j] * wsp2[j];
    }

    FwdTransBCSegSumFacTOPKernel<false>(
        nm0, nq0, basis0, w0, offset_seg, invintmass0, tJac + 2,
        wsp1 + nq0 + nq0, wsp2 + nm0 + nm1, wsp3, wsp4, threadBlock);

    // Map edge modes (without vertex contribution) back into face
    for (unsigned int j = idx0; j < nm0; j += stride)
    {
        out[tMap[j + nm0 + nm1]] = tSign[j + nm0 + nm1] * wsp2[j + nm0 + nm1];
    }

    // synchronize threads.
    localBarrier(threadBlock);

    /// Step 2: Evaluate edge contributions via mass matrix
    BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, isModified, basis0,
                               basis1, out, wsp1, wsp2, threadBlock);

    // Apply jacobian and quadrature weights
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        if constexpr (DEFORMED)
        {
            wsp1[idx] = (in[idx] - wsp1[idx]) * jac[idx] * w0[i] * w1[j];
        }
        else
        {
            wsp1[idx] = (in[idx] - wsp1[idx]) * w0[i] * w1[j];
        }
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Complete mass matrix with IProduct
    IProductWRTBaseTriSumFacTOPKernel<false, false>(
        nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, basis0, basis1,
        wsp1, wsp2, wsp3, (TData)1.0, threadBlock);

    /// Step 3: Project face/interior modes onto boundary
    // Map to interior coeffs
    for (unsigned int j = idx0; j < nmTotInt; j += stride)
    {
        wsp1[j] = wsp2[iMap[j]];
    }

    // synchronize threads.
    localBarrier(threadBlock);

    // Projection ie matrix vector product with inverse interior mass
    Nektar::MatVecSumFacTOPKernel<false, false>(nmTotInt, invintmass, wsp1,
                                                wsp2, threadBlock);

    // Map to volume
    for (unsigned int j = idx0; j < nmTotInt; j += stride)
    {
        out[iMap[j]] = wsp2[j];
    }

    // synchronize threads.
    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void FwdTransBC1DKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const unsigned int offset_seg, const TData *NEK_RESTRICT invintmass,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out,
    [[maybe_unused]] unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    // Per Element
    size_t e = getBlockIdx(threadBlock);

    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nq0 * e;
        TData *outptr       = out + nm0 * e;

        // Get shared memory
        TData *s_wsp1 = (TData *)shmemptr;
        TData *s_wsp2 = s_wsp1 + nq0;

        FwdTransBCSegSumFacTOPKernel<DEFORMED>(
            nm0, nq0, basis0, w0, offset_seg, invintmass, jacptr, inptr, outptr,
            s_wsp1, s_wsp2, threadBlock);

        // Increment to next element
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void FwdTransBC2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const unsigned int *NEK_RESTRICT index0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT interp1to0, const unsigned int offset_seg,
    const TData *NEK_RESTRICT invintmass0,
    const TData *NEK_RESTRICT invintmass1, const TData *NEK_RESTRICT tJac,
    const unsigned int *NEK_RESTRICT tMap, const int *NEK_RESTRICT tSign,
    const unsigned int nmTotInt, const unsigned int *iMap,
    const TData *NEK_RESTRICT invintmass, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    [[maybe_unused]] unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int nEdges = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        nEdges = 4;
    }
    if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        nEdges = 3;
    }

    const unsigned int *tmapptr = tMap;
    const int *tsignptr         = tSign;
    const unsigned int *imapptr = iMap;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *tjacptr = tJac + nEdges * e;
        const TData *massintptr =
            (DEFORMED) ? invintmass + nmTotInt * nmTotInt * e : invintmass;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * e;
        TData *outptr       = out + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            // Get shared memory
            TData *s_wsp1 = (TData *)shmemptr;
            TData *s_wsp2 = s_wsp1 + std::max(nqTot, 2 * nq0 + 2 * nq1);
            TData *s_wsp3 = s_wsp2 + nqTot;
            TData *s_wsp4 = s_wsp3 + nqTot;

            FwdTransBCQuadSumFacTOPKernel<DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, basis0, basis1, w0, w1, offset_seg,
                invintmass0, invintmass1, tjacptr, tmapptr, tsignptr, nmTotInt,
                imapptr, invintmass, jacptr, inptr, outptr, s_wsp1, s_wsp2,
                s_wsp3, s_wsp4, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            // Get shared memory
            TData *s_wsp1 = (TData *)shmemptr;
            TData *s_wsp2 = s_wsp1 + nqTot;
            TData *s_wsp3 = s_wsp2 + nqTot;
            TData *s_wsp4 = s_wsp3 + nqTot;

            FwdTransBCTriSumFacTOPKernel<DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, isModified, index0, basis0, basis1,
                w0, w1, interp1to0, offset_seg, invintmass0, tjacptr, tmapptr,
                tsignptr, nmTotInt, imapptr, massintptr, jacptr, inptr, outptr,
                s_wsp1, s_wsp2, s_wsp3, s_wsp4, threadBlock);
        }

        // Increment to next element
        e += getBlockRange(threadBlock);
    }
}

// Kernel Launchers.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void FwdTransBC1DKernelLauncher(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const unsigned int offset_seg, const TData *NEK_RESTRICT invintmass,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    FwdTransBC1DKernel<DEFORMED>(nm0, nq0, nelmt, basis0, w0, offset_seg,
                                 invintmass, jac, in, out, shmemptr,
                                 threadBlock);
}

// Kernel Launchers.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void FwdTransBC2DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const unsigned int *NEK_RESTRICT index0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT interp1to0, const unsigned int offsetSeg,
    const TData *NEK_RESTRICT invintmass0,
    const TData *NEK_RESTRICT invintmass1, const TData *NEK_RESTRICT tJac,
    const unsigned int *NEK_RESTRICT tMap, const int *NEK_RESTRICT tSign,
    const unsigned int nmTotInt, const unsigned int *iMap,
    const TData *NEK_RESTRICT invintmass, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    FwdTransBC2DKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, interp1to0, offsetSeg, invintmass0, invintmass1, tJac, tMap,
        tSign, nmTotInt, iMap, invintmass, jac, in, out, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
