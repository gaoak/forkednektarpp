///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzDeviceSumFacKernels.hpp
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

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord,
    const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (SHAPE_TYPE == LibUtilities::Seg)
    {
        wspsize = (1 + ncoord) * nq0 * nelmt;
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord,
    const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wspsize = ((1 + ncoord) * nq0 * nq1 + nq1) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        wspsize = ((1 + ncoord) * nq0 * nq1 + std::max(nq1, nm0)) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        wspsize =
            ((1 + ncoord) * nq0 * nq1 + std::max(nq1, nm0) + nmTot) * nelmt;
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    const size_t nelmt, [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

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
        wspsize = (4 * nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0)) *
                  nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (4 * nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0) + nmTot) *
                  nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wspsize = (4 * nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0)) *
                  nelmt;
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
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
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
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
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const TData *NEK_RESTRICT in,
    TData *out, TData *NEK_RESTRICT bwd, const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData metric[3] = {0.0};
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 1)
            {
                metric[0] = diffCoeff[0] * df[ilane];
            }
            else if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[warpsize + ilane];
                metric[1] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[warpsize + ilane];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[warpsize + ilane] +
                            diffCoeff[3] * df[2 * warpsize + ilane];
                metric[1] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[warpsize + ilane] +
                            diffCoeff[4] * df[2 * warpsize + ilane];
                metric[2] = diffCoeff[3] * df[ilane] +
                            diffCoeff[4] * df[warpsize + ilane] +
                            diffCoeff[5] * df[2 * warpsize + ilane];
            }
        }
    }

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[0] * df[dfindex];
                }
                else if (ncoord == 2)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[warpsize + dfindex];
                    metric[1] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[warpsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[warpsize + dfindex] +
                                diffCoeff[3] * df[2 * warpsize + dfindex];
                    metric[1] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[warpsize + dfindex] +
                                diffCoeff[4] * df[2 * warpsize + dfindex];
                    metric[2] = diffCoeff[3] * df[dfindex] +
                                diffCoeff[4] * df[warpsize + dfindex] +
                                diffCoeff[5] * df[2 * warpsize + dfindex];
                }
            }
        }

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            if (diffCoeff)
            {
                sum += metric[k] * in[k * inoffset + index];
            }
            else
            {
                sum += df[k * warpsize + dfindex] * in[k * inoffset + index];
            }
        }

        if constexpr (DEFORMED)
        {
            bwd[index] *= lambda * jac[index] * w0[i];
            out[index] = sum * jac[index] * w0[i];
        }
        else
        {
            bwd[index] *= lambda * jac[0] * w0[i];
            out[index] = sum * jac[0] * w0[i];
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT bwd, const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    TData metric[6] = {0.0};
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane];
                metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                            diffCoeff[1] * df[3 * warpsize + ilane];
                metric[2] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[2 * warpsize + ilane];
                metric[3] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane] +
                            diffCoeff[3] * df[4 * warpsize + ilane];
                metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                            diffCoeff[1] * df[3 * warpsize + ilane] +
                            diffCoeff[3] * df[5 * warpsize + ilane];
                metric[2] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[2 * warpsize + ilane] +
                            diffCoeff[4] * df[4 * warpsize + ilane];
                metric[3] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane] +
                            diffCoeff[4] * df[5 * warpsize + ilane];
                metric[4] = diffCoeff[3] * df[ilane] +
                            diffCoeff[4] * df[2 * warpsize + ilane] +
                            diffCoeff[5] * df[4 * warpsize + ilane];
                metric[5] = diffCoeff[3] * df[1 * warpsize + ilane] +
                            diffCoeff[4] * df[3 * warpsize + ilane] +
                            diffCoeff[5] * df[5 * warpsize + ilane];
            }
        }
    }

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            if constexpr (DEFORMED)
            {
                if (diffCoeff)
                {
                    if (ncoord == 2)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex];
                        metric[2] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[2 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex];
                    }
                    else if (ncoord == 3)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[3] * df[4 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex] +
                                    diffCoeff[3] * df[5 * warpsize + dfindex];
                        metric[2] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[2 * warpsize + dfindex] +
                                    diffCoeff[4] * df[4 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex];
                        metric[4] = diffCoeff[3] * df[dfindex] +
                                    diffCoeff[4] * df[2 * warpsize + dfindex] +
                                    diffCoeff[5] * df[4 * warpsize + dfindex];
                        metric[5] = diffCoeff[3] * df[1 * warpsize + dfindex] +
                                    diffCoeff[4] * df[3 * warpsize + dfindex] +
                                    diffCoeff[5] * df[5 * warpsize + dfindex];
                    }
                }
            }

            TData sum1 = 0.0, sum2 = 0.0;
            if (diffCoeff)
            {
                TData tmp = in[index];
                sum1 += metric[0u] * tmp;
                sum2 += metric[1u] * tmp;
                tmp = in[inoffset + index];
                sum1 += metric[2u] * tmp;
                sum2 += metric[3u] * tmp;
                if (ncoord == 3u)
                {
                    tmp = in[2u * inoffset + index];
                    sum1 += metric[4u] * tmp;
                    sum2 += metric[5u] * tmp;
                }
            }
            else
            {
                TData tmp = in[index];
                sum1 += df[0u * warpsize + dfindex] * tmp;
                sum2 += df[1u * warpsize + dfindex] * tmp;
                tmp = in[inoffset + index];
                sum1 += df[2u * warpsize + dfindex] * tmp;
                sum2 += df[3u * warpsize + dfindex] * tmp;
                if (ncoord == 3u)
                {
                    tmp = in[2u * inoffset + index];
                    sum1 += df[4u * warpsize + dfindex] * tmp;
                    sum2 += df[5u * warpsize + dfindex] * tmp;
                }
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

            bwd[index] *= lambda * tmpQ;

            // Moving from standard to collapsed coordinates.
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
NEK_DEVICE_INLINE static void ApplyMetric3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT out2, TData *NEK_RESTRICT bwd,
    const TData lambda)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    TData metric[9] = {0.0};
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            metric[0] = diffCoeff[0] * df[ilane] +
                        diffCoeff[1] * df[3 * warpsize + ilane] +
                        diffCoeff[3] * df[6 * warpsize + ilane];
            metric[1] = diffCoeff[0] * df[1 * warpsize + ilane] +
                        diffCoeff[1] * df[4 * warpsize + ilane] +
                        diffCoeff[3] * df[7 * warpsize + ilane];
            metric[2] = diffCoeff[0] * df[2 * warpsize + ilane] +
                        diffCoeff[1] * df[5 * warpsize + ilane] +
                        diffCoeff[3] * df[8 * warpsize + ilane];
            metric[3] = diffCoeff[1] * df[ilane] +
                        diffCoeff[2] * df[3 * warpsize + ilane] +
                        diffCoeff[4] * df[6 * warpsize + ilane];
            metric[4] = diffCoeff[1] * df[1 * warpsize + ilane] +
                        diffCoeff[2] * df[4 * warpsize + ilane] +
                        diffCoeff[4] * df[7 * warpsize + ilane];
            metric[5] = diffCoeff[1] * df[2 * warpsize + ilane] +
                        diffCoeff[2] * df[5 * warpsize + ilane] +
                        diffCoeff[4] * df[8 * warpsize + ilane];
            metric[6] = diffCoeff[3] * df[ilane] +
                        diffCoeff[4] * df[3 * warpsize + ilane] +
                        diffCoeff[5] * df[6 * warpsize + ilane];
            metric[7] = diffCoeff[3] * df[1 * warpsize + ilane] +
                        diffCoeff[4] * df[4 * warpsize + ilane] +
                        diffCoeff[5] * df[7 * warpsize + ilane];
            metric[8] = diffCoeff[3] * df[2 * warpsize + ilane] +
                        diffCoeff[4] * df[5 * warpsize + ilane] +
                        diffCoeff[5] * df[8 * warpsize + ilane];
        }
    }

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                if constexpr (DEFORMED)
                {
                    if (diffCoeff)
                    {
                        metric[0] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[3 * warpsize + dfindex] +
                                    diffCoeff[3] * df[6 * warpsize + dfindex];
                        metric[1] = diffCoeff[0] * df[1 * warpsize + dfindex] +
                                    diffCoeff[1] * df[4 * warpsize + dfindex] +
                                    diffCoeff[3] * df[7 * warpsize + dfindex];
                        metric[2] = diffCoeff[0] * df[2 * warpsize + dfindex] +
                                    diffCoeff[1] * df[5 * warpsize + dfindex] +
                                    diffCoeff[3] * df[8 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[6 * warpsize + dfindex];
                        metric[4] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[4 * warpsize + dfindex] +
                                    diffCoeff[4] * df[7 * warpsize + dfindex];
                        metric[5] = diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[2] * df[5 * warpsize + dfindex] +
                                    diffCoeff[4] * df[8 * warpsize + dfindex];
                        metric[6] = diffCoeff[3] * df[dfindex] +
                                    diffCoeff[4] * df[3 * warpsize + dfindex] +
                                    diffCoeff[5] * df[6 * warpsize + dfindex];
                        metric[7] = diffCoeff[3] * df[1 * warpsize + dfindex] +
                                    diffCoeff[4] * df[4 * warpsize + dfindex] +
                                    diffCoeff[5] * df[7 * warpsize + dfindex];
                        metric[8] = diffCoeff[3] * df[2 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex] +
                                    diffCoeff[5] * df[8 * warpsize + dfindex];
                    }
                }

                TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                if (diffCoeff)
                {
                    TData tmp = in[index];
                    sum1 += metric[0u] * tmp;
                    sum2 += metric[1u] * tmp;
                    sum3 += metric[2u] * tmp;
                    tmp = in[inoffset + index];
                    sum1 += metric[3u] * tmp;
                    sum2 += metric[4u] * tmp;
                    sum3 += metric[5u] * tmp;
                    tmp = in[2u * inoffset + index];
                    sum1 += metric[6u] * tmp;
                    sum2 += metric[7u] * tmp;
                    sum3 += metric[8u] * tmp;
                }
                else
                {
                    TData tmp = in[index];
                    sum1 += df[0u * warpsize + dfindex] * tmp;
                    sum2 += df[1u * warpsize + dfindex] * tmp;
                    sum3 += df[2u * warpsize + dfindex] * tmp;
                    tmp = in[inoffset + index];
                    sum1 += df[3u * warpsize + dfindex] * tmp;
                    sum2 += df[4u * warpsize + dfindex] * tmp;
                    sum3 += df[5u * warpsize + dfindex] * tmp;
                    tmp = in[2u * inoffset + index];
                    sum1 += df[6u * warpsize + dfindex] * tmp;
                    sum2 += df[7u * warpsize + dfindex] * tmp;
                    sum3 += df[8u * warpsize + dfindex] * tmp;
                }

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[index];
                }
                else
                {
                    tmpQ *= jac[0];
                }

                bwd[index] *= lambda * tmpQ;

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
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData lambda,
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
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nm0 * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nq0 * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nq0 * nelmt * ncomp +
                       nq0 * (ncoord * nelmt * c + warpsize * iwarp);
        BwdTransSegSumFacKernel<false>(ilane, nm0, nq0, basis0, inptr, bwd);
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0, D0,
                                          dfptr, bwd, deriv);
        ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0, w0,
                                            dfptr, jacptr, coeff, deriv, deriv,
                                            bwd, lambda);
        SumDerivTensor1DKernel<false, true>(ilane, nq0, D0, deriv, bwd);
        IProductWRTBaseSegSumFacKernel<false, false>(ilane, nm0, nq0, basis0,
                                                     bwd, outptr, (TData)1.0);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

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
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nqTot * nelmt * ncomp +
                       nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        TData *deriv0 = wsp + nqTot * nelmt * ncomp +
                        nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * (ncomp + 1) +
                        nqTot * (ncoord * nelmt * c + warpsize * iwarp);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          nq1 * (nelmt * c + warpsize * iwarp);
            BwdTransQuadSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1, basis0,
                                            basis1, inptr, bwd, wsp0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, inptr,
                                           bwd, wsp0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *in1ptr = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);

            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, in1ptr,
                                           bwd, wsp0);
        }

        PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, nelmt * nqTot, D0, D1, s_f0, s_f1, dfptr,
            bwd, deriv);
        ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, nelmt * nqTot, w0, w1, s_f0, s_f1, dfptr,
            jacptr, coeff, deriv, deriv0, deriv1, bwd, lambda);
        SumDerivTensor2DKernel<false, true>(ilane, nq0, nq1, D0, D1, deriv0,
                                            deriv1, bwd);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, bwd, outptr, wsp0,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                out1ptr, wsp0, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData lambda,
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
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nqTot * nelmt * ncomp +
                       nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv0 = wsp + nqTot * nelmt * ncomp +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * (ncomp + 1) +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv2 = wsp + nqTot * nelmt * (ncomp + 2) +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            BwdTransHexSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           basis0, basis1, basis2, inptr, bwd,
                                           wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *in1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           in1ptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, inptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *in1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 =
                wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                              nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, in1ptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPyrSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
        }
        PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, nelmt * nqTot, D0, D1, D2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, bwd, deriv);
        ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, nelmt * nqTot, w0, w1, w2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, jacptr, coeff, deriv, deriv0, deriv1, deriv2, bwd,
            lambda);
        SumDerivTensor3DKernel<true, true>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                           deriv0, deriv1, deriv2, bwd);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *out1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 =
                wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                              nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
