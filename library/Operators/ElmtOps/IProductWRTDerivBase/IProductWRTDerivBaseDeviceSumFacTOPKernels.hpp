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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (IsPhysSizeParameter1D_v<TSizeParameter1D>)
    {
        return nq0;
    }
    else
    {
        return 2 * nq0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        return 2 * nq0 * nq1 + nq0 * nq0 + nq1 * nq1;
    }
    else
    {
        const unsigned int nm0 = sizeParam2D.nm0();
        const unsigned int nm1 = sizeParam2D.nm1();

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return nm0 * nq0 + nm1 * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1 + nmTot;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        return 3 * nq0 * nq1 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2;
    }
    else
    {
        const unsigned int nm0 = sizeParam3D.nm0();
        const unsigned int nm1 = sizeParam3D.nm1();
        const unsigned int nm2 = sizeParam3D.nm2();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm01 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
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
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nmTot;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t inoffset,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT out2,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv  = deriv0 + nq0;

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
        TData *outptr = out + nm0 * nelmt * (nmode * c + m) + nm0 * e;

        StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, inoffset, w0,
                                                     dfptr, jacptr, inptr,
                                                     deriv0, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<false, false>(nq0, D0, deriv0, deriv,
                                                      scale, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<true, APPEND>(
            nm0, nq0, basis0, deriv, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter1D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *deriv = (TData *)shmemptr;

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
        TData *outptr = out + nq0 * nelmt * (nmode * c + m) + nq0 * e;

        StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, inoffset, w0,
                                                     dfptr, jacptr, inptr,
                                                     deriv, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<true, APPEND>(nq0, D0, deriv, outptr,
                                                      scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

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
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * nelmt * (inDim * nmode * c + m) + nqTot * e;
        TData *outptr = out + nmTot * nelmt * (nmode * c + m) + nmTot * e;

        StdAlignDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr, inptr,
            deriv0, deriv1, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<false, false>(
            nq0, nq1, D0, D1, deriv0, deriv1, deriv, scale, threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, deriv,
                outptr, s_wsp0, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, outptr, s_wsp0, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<true, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, s_out1ptr, s_wsp0, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter2D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv1 = deriv0 + nqTot;
    TData *s_D0   = deriv1 + nqTot;
    TData *s_D1   = s_D0 + nq0 * nq0;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
    {
        s_D0[idx] = D0[idx];
    }

    for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
    {
        s_D1[idx] = D1[idx];
    }

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * nelmt * (inDim * nmode * c + m) + nqTot * e;
        TData *outptr = out + nqTot * nelmt * (nmode * c + m) + nqTot * e;

        StdAlignDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr, inptr,
            deriv0, deriv1, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<true, APPEND>(
            nq0, nq1, s_D0, s_D1, deriv0, deriv1, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const size_t inoffset, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

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
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr =
            in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
        TData *outptr = out + nmTot * nelmt * (nmode * c + m) + nmTot * e;

        StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr, jacptr,
            inptr, deriv0, deriv1, deriv2, threadBlock);
        // Always overwrite the shared-memory deriv scratch (it is reused
        // across the element loop); the op-level append is handled by the
        // IProductWRTBase kernels writing the global output below.
        SumDerivTensor3DSumFacTOPKernel<false, false>(
            nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, deriv, scale,
            threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, deriv, outptr, s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<true, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<true, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, deriv, outptr, s_wsp0,
                s_wsp1, scale, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter3D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv1 = deriv0 + nqTot;
    TData *deriv2 = deriv1 + nqTot;
    TData *s_D0   = deriv2 + nqTot;
    TData *s_D1   = s_D0 + nq0 * nq0;
    TData *s_D2   = s_D1 + nq1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
    {
        s_D0[idx] = D0[idx];
    }

    for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
    {
        s_D1[idx] = D1[idx];
    }

    for (unsigned int idx = idx0; idx < nq2 * nq2; idx += stride)
    {
        s_D2[idx] = D2[idx];
    }

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr =
            in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
        TData *outptr = out + nqTot * nelmt * (nmode * c + m) + nqTot * e;

        StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr, jacptr,
            inptr, deriv0, deriv1, deriv2, threadBlock);
        SumDerivTensor3DSumFacTOPKernel<true, APPEND>(
            nq0, nq1, nq2, s_D0, s_D1, s_D2, deriv0, deriv1, deriv2, outptr,
            scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
