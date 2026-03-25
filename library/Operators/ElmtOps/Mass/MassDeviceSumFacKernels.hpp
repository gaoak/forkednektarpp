///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFacKernels.hpp
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

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline unsigned int MassSharedMemorySize(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nm0)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline unsigned int MassSharedMemorySize(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline unsigned int MassSharedMemorySize(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2)
{
    return 0;
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Mass1DSumFacKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
    [[maybe_unused]] unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nm0 * warpsize * iwarp;
        TData *wspptr      = wsp + nq0 * warpsize * iwarp;
        TData *outptr      = out + nm0 * warpsize * iwarp;
        BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, wspptr);
        IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
            ilane, nm0, nq0, basis0, w0, jacptr, wspptr, outptr, (TData)1.0);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Mass2DSumFacKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT nodToMod,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
    [[maybe_unused]] unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * warpsize * iwarp;
        TData *outptr      = out + nmTot * warpsize * iwarp;
        TData *bwd         = wsp + nqTot * warpsize * iwarp;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + nqTot * nelmt + nq1 * warpsize * iwarp;
            BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0, basis1,
                                     inptr, bwd, wsp0);
            IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr, bwd,
                outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 =
                wsp + nqTot * nelmt + std::max(nq1, nm0) * warpsize * iwarp;
            BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                    basis0, basis1, inptr, bwd, wsp0);
            IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, bwd, outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *modes = wsp + nqTot * nelmt + nmTot * warpsize * iwarp;
            TData *wsp0  = wsp + (nqTot + nmTot) * nelmt +
                          std::max(nq1, nm0) * warpsize * iwarp;
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                    basis0, basis1, modes, bwd, wsp0);
            IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, bwd, modes, wsp0, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Mass3DSumFacKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp,
    [[maybe_unused]] unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * warpsize * iwarp;
        TData *outptr      = out + nmTot * warpsize * iwarp;
        TData *bwd         = wsp + nqTot * warpsize * iwarp;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 =
                wsp + (nqTot + nq1 * nq2) * nelmt + nq2 * warpsize * iwarp;
            BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0,
                                    basis1, basis2, inptr, bwd, wsp0, wsp1);
            IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2, w0,
                w1, w2, jacptr, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + (nqTot + nq1 * nq2) * nelmt +
                          std::max(nq2, nm0) * warpsize * iwarp;
            BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                    isModified, basis0, basis1, basis2, inptr,
                                    bwd, wsp0, wsp1);
            IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *modes = wsp + nqTot * nelmt + nmTot * warpsize * iwarp;
            TData *wsp0 =
                wsp + (nqTot + nmTot) * nelmt + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + (nqTot + nq1 * nq2 + nmTot) * nelmt +
                          std::max(nq2, nm0) * warpsize * iwarp;
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                    isModified, basis0, basis1, basis2, modes,
                                    bwd, wsp0, wsp1);
            IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, modes, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nqTot * nelmt +
                          std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
            TData *wsp1 = wsp +
                          (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                          std::max(nq2, nm0) * warpsize * iwarp;
            BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                      isModified, basis0, basis1, basis2, inptr,
                                      bwd, wsp0, wsp1);
            IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *modes = wsp + nqTot * nelmt + nmTot * warpsize * iwarp;
            TData *wsp0  = wsp + (nqTot + nmTot) * nelmt +
                          std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
            TData *wsp1 =
                wsp + (nqTot + nmTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                std::max(nq2, nm0) * warpsize * iwarp;
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                      isModified, basis0, basis1, basis2, modes,
                                      bwd, wsp0, wsp1);
            IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, modes, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nqTot * nelmt +
                          std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
            TData *wsp1 = wsp +
                          (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                          std::max(nq2, nm0) * warpsize * iwarp;
            BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                    isModified, basis0, basis1, basis2, inptr,
                                    bwd, wsp0, wsp1);
            IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        e += getGlobalRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Mass1DKernelLauncher(NonTemplated1DSizeParameters sizeParam1D,
                         const size_t nelmt, const TData *NEK_RESTRICT basis0,
                         const TData *NEK_RESTRICT w0,
                         const TData *NEK_RESTRICT jac,
                         const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
                         TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
                         const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass1DSumFacKernel<DEFORMED>(sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt,
                                 basis0, w0, jac, in, out, wsp, shmemptr,
                                 threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Mass1DKernelLauncher(
        [[maybe_unused]] Templated1DSizeParameters<nm0, nq0> sizeParam1D,
        const size_t nelmt, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass1DSumFacKernel<DEFORMED>(nm0, nq0, nelmt, basis0, w0, jac, in, out, wsp,
                                 shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Mass2DKernelLauncher(
        NonTemplated2DSizeParameters sizeParam2D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, isModified, basis0, basis1,
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
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Mass2DKernelLauncher(
        [[maybe_unused]] Templated2DSizeParameters<nm0, nm1, nmTot, nq0, nq1>
            sizeParam2D,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, basis0, basis1, w0, w1,
        nodToMod, jac, in, out, wsp, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    Mass3DKernelLauncher(
        NonTemplated3DSizeParameters sizeParam3D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, isModified, basis0, basis1, basis2, w0, w1,
        w2, nodToMod, jac, in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Mass3DKernelLauncher(
        [[maybe_unused]] Templated3DSizeParameters<nm0, nm1, nm2, nmTot, nq0,
                                                   nq1, nq2>
            sizeParam3D,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Mass3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, basis0, basis1,
        basis2, w0, w1, w2, nodToMod, jac, in, out, wsp, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
