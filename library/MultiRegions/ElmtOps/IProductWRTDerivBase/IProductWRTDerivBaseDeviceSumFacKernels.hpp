///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceSumFacKernels.hpp
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

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (IsPhysSizeParameter1D_v<TSizeParameter1D>)
    {
        wspsize = nq0 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Seg)
        {
            wspsize = 2 * nq0 * nelmt;
        }
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        wspsize = 2 * nq0 * nq1 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            wspsize = (3 * nq0 * nq1 + nq1 + nmTot) * nelmt;
        }
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        wspsize = 3 * nq0 * nq1 * nq2 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 + nmTot) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 + nmTot) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nq0 = sizeParam2D.nq0();
        const unsigned int nq1 = sizeParam2D.nq1();

        return nq0 + nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
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

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nq0 + 2 * nq1 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nq0 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nq0 + nq1 + nq2;
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            sum += df[k * warpsize + dfindex] * in[k * inoffset + index];
        }

        if constexpr (DEFORMED)
        {
            out[index] = sum * jac[index] * w0[i];
        }
        else
        {
            out[index] = sum * jac[0] * w0[i];
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            TData tmp  = in[index];
            TData sum1 = df[0u * warpsize + dfindex] * tmp;
            TData sum2 = df[1u * warpsize + dfindex] * tmp;
            tmp        = in[inoffset + index];
            sum1 += df[2u * warpsize + dfindex] * tmp;
            sum2 += df[3u * warpsize + dfindex] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + index];
                sum1 += df[4u * warpsize + dfindex] * tmp;
                sum2 += df[5u * warpsize + dfindex] * tmp;
            }

            TData tmpQ = w0[i] * w1[j];
            if constexpr (DEFORMED)
            {
                tmpQ *= jac[index];
            }
            else
            {
                tmpQ *= jac[0];
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                out0[index] = sum1 * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                               SHAPE_TYPE == LibUtilities::NodalTri)
            {
                out0[index] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1,
    TData *NEK_RESTRICT out2)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                TData tmp  = in[index];
                TData sum1 = df[0u * warpsize + dfindex] * tmp;
                TData sum2 = df[1u * warpsize + dfindex] * tmp;
                TData sum3 = df[2u * warpsize + dfindex] * tmp;
                tmp        = in[inoffset + index];
                sum1 += df[3u * warpsize + dfindex] * tmp;
                sum2 += df[4u * warpsize + dfindex] * tmp;
                sum3 += df[5u * warpsize + dfindex] * tmp;
                tmp = in[2u * inoffset + index];
                sum1 += df[6u * warpsize + dfindex] * tmp;
                sum2 += df[7u * warpsize + dfindex] * tmp;
                sum3 += df[8u * warpsize + dfindex] * tmp;

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[index];
                }
                else
                {
                    tmpQ *= jac[0];
                }

                if constexpr (SHAPE_TYPE == LibUtilities::Hex)
                {
                    out0[index] = sum1 * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                                   SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    out0[index] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + sum3 * f0[i]) * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
        TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nm0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + nq0 * nelmt * nmode * ncomp +
                       nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase1DSumFacKernel<DEFORMED>(
            ilane, ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv0);
        SumDerivTensor1DKernel<false, false>(ilane, nq0, D0, deriv0, deriv,
                                             scale);
        IProductWRTBaseSegSumFacKernel<true, APPEND>(ilane, nm0, nq0, basis0,
                                                     deriv, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter1D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase1DSumFacKernel<DEFORMED>(
            ilane, ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv);
        SumDerivTensor1DKernel<true, APPEND>(ilane, nq0, D0, deriv, outptr,
                                             scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Pre-compute factor.
    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        const unsigned int idx0   = getLocalIdx<0>(threadBlock);
        const unsigned int stride = getLocalRange<0>(threadBlock);

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                       nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, w0, w1, s_f0, s_f1, dfptr,
            jacptr, inptr, deriv0, deriv1);
        SumDerivTensor2DKernel<false, false>(ilane, nq0, nq1, D0, D1, deriv0,
                                             deriv1, deriv, scale);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, deriv, outptr, wsp0,
                scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, deriv,
                outptr, wsp0, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr =
                wsp + 3 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 3 * nqTot) * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<true, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, deriv,
                out1ptr, wsp0, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Pre-compute factor.
    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        const unsigned int idx0   = getLocalIdx<0>(threadBlock);
        const unsigned int stride = getLocalRange<0>(threadBlock);

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, w0, w1, s_f0, s_f1, dfptr,
            jacptr, inptr, deriv0, deriv1);
        SumDerivTensor2DKernel<true, APPEND>(ilane, nq0, nq1, D0, D1, deriv0,
                                             deriv1, outptr, scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const size_t inoffset, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Pre-compute factor.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr =
            in + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv2 = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                       nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, w0, w1, w2, s_f0, s_f1, s_f1m, s_f2,
            dfptr, jacptr, inptr, deriv0, deriv1, deriv2);
        SumDerivTensor3DKernel<false, false>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                             deriv0, deriv1, deriv2, deriv,
                                             scale);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *out1ptr =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 =
                wsp + (4 * nqTot + nmTot) * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 =
                wsp + (4 * nqTot + nq1 * nq2 + nmTot) * nelmt * nmode * ncomp +
                nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<true, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 =
                wsp + (4 * nqTot + nmTot) * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 =
                wsp + (4 * nqTot + nq1 * nq2 + nmTot) * nelmt * nmode * ncomp +
                nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<true, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Pre-compute factor.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr =
            in + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv2 = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, w0, w1, w2, s_f0, s_f1, s_f1m, s_f2,
            dfptr, jacptr, inptr, deriv0, deriv1, deriv2);
        SumDerivTensor3DKernel<true, APPEND>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                             deriv0, deriv1, deriv2, outptr,
                                             scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
