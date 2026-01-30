///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianDeviceSumFacKernels.hpp
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
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation>
inline unsigned int LaplacianSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        return 0;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        return 4 * nq0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int LaplacianSharedMemorySize(const unsigned int nq0,
                                              const unsigned int nq1,
                                              const unsigned int nm0,
                                              const unsigned int nm1)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return 0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            return nq0 + nq1;
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return nm0 * nq0 + nm1 * nq1 + 4 * nq0 * nq1 +
                   std::max(nq0 * nm1, nm0 * nq1) + 6;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            const unsigned int nmTot =
                LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
            return nm0 * nq0 + nmTot * nq1 + 4 * nq0 * nq1 + nm0 * nq1 + 6;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot =
                LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
            return nm0 * nq0 + nmTot * nq1 + 4 * nq0 * nq1 + nm0 * nq1 + 6 +
                   nmTot;
        }
    }

    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int LaplacianSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
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
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) +
                   std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2) + 9;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm01 * nq2 + 9;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 9;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 9;
        }
    }

    return 0;
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *out)
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
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            if (diffCoeff)
            {
                sum += metric[d] * in[d * inoffset + index];
            }
            else
            {
                sum += df[d * warpsize + dfindex] * in[d * inoffset + index];
            }
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
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *__restrict__ out0,
    TData *__restrict__ out1)
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
                metric[2] = diffCoeff[0] * df[ilane] +
                            diffCoeff[1] * df[2 * warpsize + ilane] +
                            diffCoeff[3] * df[4 * warpsize + ilane];
                metric[3] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane] +
                            diffCoeff[4] * df[5 * warpsize + ilane];
                metric[4] = diffCoeff[1] * df[ilane] +
                            diffCoeff[2] * df[2 * warpsize + ilane] +
                            diffCoeff[4] * df[4 * warpsize + ilane];
                metric[5] = diffCoeff[1] * df[1 * warpsize + ilane] +
                            diffCoeff[2] * df[3 * warpsize + ilane] +
                            diffCoeff[4] * df[5 * warpsize + ilane];
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
                        metric[2] = diffCoeff[0] * df[dfindex] +
                                    diffCoeff[1] * df[2 * warpsize + dfindex] +
                                    diffCoeff[3] * df[4 * warpsize + dfindex];
                        metric[3] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex];
                        metric[4] = diffCoeff[1] * df[dfindex] +
                                    diffCoeff[2] * df[2 * warpsize + dfindex] +
                                    diffCoeff[4] * df[4 * warpsize + dfindex];
                        metric[5] = diffCoeff[1] * df[1 * warpsize + dfindex] +
                                    diffCoeff[2] * df[3 * warpsize + dfindex] +
                                    diffCoeff[4] * df[5 * warpsize + dfindex];
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
    const unsigned int nq2, const size_t inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1,
    [[maybe_unused]] const TData *__restrict__ f1m,
    [[maybe_unused]] const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *__restrict__ out0,
    TData *__restrict__ out1, TData *__restrict__ out2)
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

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *out, const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    TData metric[3] = {0.0};
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 1)
            {
                metric[0] = diffCoeff[0] * df[0];
            }
            else if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1];
                metric[1] = diffCoeff[1] * df[0] + diffCoeff[2] * df[1];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1] +
                            diffCoeff[3] * df[2];
                metric[1] = diffCoeff[1] * df[0] + diffCoeff[2] * df[1] +
                            diffCoeff[4] * df[2];
                metric[2] = diffCoeff[3] * df[0] + diffCoeff[4] * df[1] +
                            diffCoeff[5] * df[2];
            }
        }
    }

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[0] * df[i];
                }
                else if (ncoord == 2)
                {
                    metric[0] =
                        diffCoeff[0] * df[i] + diffCoeff[1] * df[dfsize + i];
                    metric[1] =
                        diffCoeff[1] * df[i] + diffCoeff[2] * df[dfsize + i];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[i] +
                                diffCoeff[1] * df[dfsize + i] +
                                diffCoeff[3] * df[2 * dfsize + i];
                    metric[1] = diffCoeff[1] * df[i] +
                                diffCoeff[2] * df[dfsize + i] +
                                diffCoeff[4] * df[2 * dfsize + i];
                    metric[2] = diffCoeff[3] * df[i] +
                                diffCoeff[4] * df[dfsize + i] +
                                diffCoeff[5] * df[2 * dfsize + i];
                }
            }
        }

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            if (diffCoeff)
            {
                sum += metric[d] * in[d * inoffset + i];
            }
            else
            {
                sum += df[d * dfsize + dfindex] * in[d * inoffset + i];
            }
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
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ diffCoeff,
    const TData *__restrict__ in, TData *out0, TData *out1, TData *metric,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[2];
                metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[3];
                metric[2] = diffCoeff[1] * df[0] + diffCoeff[2] * df[2];
                metric[3] = diffCoeff[1] * df[1] + diffCoeff[2] * df[3];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[2] +
                            diffCoeff[3] * df[4];
                metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[3] +
                            diffCoeff[3] * df[5];
                metric[2] = diffCoeff[0] * df[0] + diffCoeff[1] * df[2] +
                            diffCoeff[3] * df[4];
                metric[3] = diffCoeff[1] * df[1] + diffCoeff[2] * df[3] +
                            diffCoeff[4] * df[5];
                metric[4] = diffCoeff[1] * df[0] + diffCoeff[2] * df[2] +
                            diffCoeff[4] * df[4];
                metric[5] = diffCoeff[1] * df[1] + diffCoeff[2] * df[3] +
                            diffCoeff[4] * df[5];
            }

            localBarrier(threadBlock);
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 2)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex];
                    metric[2] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[2 * dfsize + dfindex];
                    metric[3] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                                diffCoeff[2] * df[3 * dfsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex] +
                                diffCoeff[3] * df[4 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex] +
                                diffCoeff[3] * df[5 * dfsize + dfindex];
                    metric[2] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex] +
                                diffCoeff[2] * df[4 * dfsize + dfindex];
                    metric[3] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                                diffCoeff[2] * df[3 * dfsize + dfindex] +
                                diffCoeff[4] * df[5 * dfsize + dfindex];
                    metric[4] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[2 * dfsize + dfindex] +
                                diffCoeff[4] * df[4 * dfsize + dfindex];
                    metric[5] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                                diffCoeff[2] * df[3 * dfsize + dfindex] +
                                diffCoeff[4] * df[5 * dfsize + dfindex];
                }
            }
        }

        TData sum1 = 0.0, sum2 = 0.0;
        if (diffCoeff)
        {
            TData tmp = in[idx];
            sum1 += metric[0u] * tmp;
            sum2 += metric[1u] * tmp;
            tmp = in[inoffset + idx];
            sum1 += metric[2u] * tmp;
            sum2 += metric[3u] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + idx];
                sum1 += metric[4u] * tmp;
                sum2 += metric[5u] * tmp;
            }
        }
        else
        {
            TData tmp = in[idx];
            sum1 += df[0u * dfsize + dfindex] * tmp;
            sum2 += df[1u * dfsize + dfindex] * tmp;
            tmp = in[inoffset + idx];
            sum1 += df[2u * dfsize + dfindex] * tmp;
            sum2 += df[3u * dfsize + dfindex] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + idx];
                sum1 += df[4u * dfsize + dfindex] * tmp;
                sum2 += df[5u * dfsize + dfindex] * tmp;
            }
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
NEK_DEVICE_INLINE static void ApplyMetric3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int inoffset, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ diffCoeff, const TData *__restrict__ in,
    TData *out0, TData *out1, TData *out2, TData *metric,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[3] +
                        diffCoeff[3] * df[6];
            metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[4] +
                        diffCoeff[3] * df[7];
            metric[2] = diffCoeff[0] * df[2] + diffCoeff[1] * df[5] +
                        diffCoeff[3] * df[8];
            metric[3] = diffCoeff[1] * df[0] + diffCoeff[2] * df[3] +
                        diffCoeff[4] * df[6];
            metric[4] = diffCoeff[1] * df[1] + diffCoeff[2] * df[4] +
                        diffCoeff[4] * df[7];
            metric[5] = diffCoeff[1] * df[2] + diffCoeff[2] * df[5] +
                        diffCoeff[4] * df[8];
            metric[6] = diffCoeff[3] * df[0] + diffCoeff[4] * df[3] +
                        diffCoeff[5] * df[6];
            metric[7] = diffCoeff[3] * df[1] + diffCoeff[4] * df[4] +
                        diffCoeff[5] * df[7];
            metric[8] = diffCoeff[3] * df[2] + diffCoeff[4] * df[5] +
                        diffCoeff[5] * df[8];

            localBarrier(threadBlock);
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1 * nq2; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                metric[0] = diffCoeff[0] * df[dfindex] +
                            diffCoeff[1] * df[3 * dfsize + dfindex] +
                            diffCoeff[3] * df[6 * dfsize + dfindex];
                metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                            diffCoeff[1] * df[4 * dfsize + dfindex] +
                            diffCoeff[3] * df[7 * dfsize + dfindex];
                metric[2] = diffCoeff[0] * df[2 * dfsize + dfindex] +
                            diffCoeff[1] * df[5 * dfsize + dfindex] +
                            diffCoeff[3] * df[8 * dfsize + dfindex];
                metric[3] = diffCoeff[1] * df[dfindex] +
                            diffCoeff[2] * df[3 * dfsize + dfindex] +
                            diffCoeff[4] * df[6 * dfsize + dfindex];
                metric[4] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                            diffCoeff[2] * df[4 * dfsize + dfindex] +
                            diffCoeff[4] * df[7 * dfsize + dfindex];
                metric[5] = diffCoeff[1] * df[2 * dfsize + dfindex] +
                            diffCoeff[2] * df[5 * dfsize + dfindex] +
                            diffCoeff[4] * df[8 * dfsize + dfindex];
                metric[6] = diffCoeff[3] * df[dfindex] +
                            diffCoeff[4] * df[3 * dfsize + dfindex] +
                            diffCoeff[5] * df[6 * dfsize + dfindex];
                metric[7] = diffCoeff[3] * df[1 * dfsize + dfindex] +
                            diffCoeff[4] * df[4 * dfsize + dfindex] +
                            diffCoeff[5] * df[7 * dfsize + dfindex];
                metric[8] = diffCoeff[3] * df[2 * dfsize + dfindex] +
                            diffCoeff[4] * df[5 * dfsize + dfindex] +
                            diffCoeff[5] * df[8 * dfsize + dfindex];
            }
        }

        TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        if (diffCoeff)
        {
            TData tmp = in[idx];
            sum1 += metric[0u] * tmp;
            sum2 += metric[1u] * tmp;
            sum3 += metric[2u] * tmp;
            tmp = in[inoffset + idx];
            sum1 += metric[3u] * tmp;
            sum2 += metric[4u] * tmp;
            sum3 += metric[5u] * tmp;
            tmp = in[2u * inoffset + idx];
            sum1 += metric[6u] * tmp;
            sum2 += metric[7u] * tmp;
            sum3 += metric[8u] * tmp;
        }
        else
        {
            TData tmp = in[idx];
            sum1 += df[0u * dfsize + dfindex] * tmp;
            sum2 += df[1u * dfsize + dfindex] * tmp;
            sum3 += df[2u * dfsize + dfindex] * tmp;
            tmp = in[inoffset + idx];
            sum1 += df[3u * dfsize + dfindex] * tmp;
            sum2 += df[4u * dfsize + dfindex] * tmp;
            sum3 += df[5u * dfsize + dfindex] * tmp;
            tmp = in[2u * inoffset + idx];
            sum1 += df[6u * dfsize + dfindex] * tmp;
            sum2 += df[7u * dfsize + dfindex] * tmp;
            sum3 += df[8u * dfsize + dfindex] * tmp;
        }

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

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void Laplacian1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

        size_t e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nm0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            TData *bwd         = wsp + nq0 * warpsize * iwarp;
            TData *deriv       = wsp + nq0 * nelmt + nq0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, bwd);
            PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0,
                                              D0, dfptr, bwd, deriv);
            ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0,
                                                w0, dfptr, jacptr, coeff, deriv,
                                                deriv);
            SumDerivTensor1DKernel<false, DEFORMED>(ilane, nq0, D0, deriv, bwd);
            IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nq0, basis0, bwd, outptr, (TData)1.0);
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        TData *bwd   = (TData *)shmemptr;
        TData *deriv = bwd + nq0;

        size_t e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * e;
            TData *outptr       = out + nm0 * e;

            BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, inptr, bwd,
                                       threadBlock);
            PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr,
                                                 bwd, deriv, threadBlock);
            ApplyMetric1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, w0, dfptr,
                                                   jacptr, coeff, deriv, deriv,
                                                   threadBlock);
            SumDerivTensor1DQPKernel<false, DEFORMED>(nq0, D0, deriv, bwd,
                                                      threadBlock);
            IProductWRTBaseSegSumFacTOPKernel<false, false, DEFORMED>(
                nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);

            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Laplacian2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

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

        size_t e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, bwd, wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, bwd, wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                TData *in1ptr = wsp + (1 + ncoord) * nqTot * nelmt +
                                nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;

                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, in1ptr, bwd, wsp0);
            }

            PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt * nqTot, D0, D1, s_f0, s_f1,
                dfptr, bwd, deriv);
            ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt * nqTot, w0, w1, s_f0, s_f1,
                dfptr, jacptr, coeff, deriv, deriv0, deriv1);
            SumDerivTensor2DKernel<false, DEFORMED>(ilane, nq0, nq1, D0, D1,
                                                    deriv0, deriv1, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, bwd, outptr,
                    wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                    outptr, wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                TData *out1ptr = wsp + (1 + ncoord) * nqTot * nelmt +
                                 nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                    out1ptr, wsp0, (TData)1.0);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
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

        TData *metric   = (TData *)shmemptr;
        TData *bwd      = metric + 6;
        TData *deriv    = bwd + nqTot;
        TData *tmp      = deriv;
        TData *deriv0   = deriv;
        TData *deriv1   = deriv0 + nqTot;
        TData *s_wsp0   = deriv + ncoord * nqTot;
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
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * e;
            TData *outptr       = out + nmTot * e;

            if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                MatVecQPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
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
                BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot,
                                           isModified, s_basis0, s_basis1, tmp,
                                           bwd, s_wsp0, threadBlock);
            }

            PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, bwd, deriv,
                threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[6];
                ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr,
                    coeff, deriv, deriv0, deriv1, dmetric, threadBlock);
            }
            else
            {
                ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr,
                    coeff, deriv, deriv0, deriv1, metric, threadBlock);
            }
            SumDerivTensor2DQPKernel<false, DEFORMED>(nq0, nq1, D0, D1, deriv0,
                                                      deriv1, bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                    outptr, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, outptr, s_wsp0, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, tmp, s_wsp0, (TData)1.0,
                    threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Laplacian3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    [[maybe_unused]] const unsigned int *__restrict__ index3,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

        TData *s_f0  = nullptr;
        TData *s_f1  = nullptr;
        TData *s_f1m = nullptr;
        TData *s_f2  = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);
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

        size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv2 = wsp + 3 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, bwd,
                                        wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                TData *in1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nmTot + 4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        in1ptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                TData *in1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp +
                    (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                        nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          in1ptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt * nqTot, D0, D1, D2, s_f0, s_f1,
                s_f1m, s_f2, dfptr, bwd, deriv);
            ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt * nqTot, w0, w1, w2, s_f0, s_f1,
                s_f1m, s_f2, dfptr, jacptr, coeff, deriv, deriv0, deriv1,
                deriv2);
            SumDerivTensor3DKernel<false, DEFORMED>(
                ilane, nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                TData *out1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nmTot + 4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                TData *out1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp +
                    (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                        nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0,
                     nmode2 = 0;
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

        TData *metric   = (TData *)shmemptr;
        TData *bwd      = metric + 9;
        TData *deriv    = bwd + nqTot;
        TData *tmp      = deriv;
        TData *deriv0   = deriv;
        TData *deriv1   = deriv0 + nqTot;
        TData *deriv2   = deriv1 + nqTot;
        TData *s_wsp0   = deriv2 + nqTot;
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
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * e;
            TData *outptr       = out + nmTot * e;

            // Copy to shared memory.
            if constexpr (SHAPE_TYPE == LibUtilities::NodalTet ||
                          SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                MatVecQPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
            }
            else
            {
                // Copy to shared memory.
                for (unsigned int idx = idx0; idx < nmTot; idx += stride)
                {
                    tmp[idx] = inptr[idx];
                }
            }

            localBarrier(threadBlock);

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                BwdTransHexSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                           s_basis0, s_basis1, s_basis2, tmp,
                                           bwd, s_wsp0, s_wsp1, threadBlock);
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
                BwdTransPyrSumFacTOPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
            }
            PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
                deriv, threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[9];
                ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, deriv0, deriv1, deriv2, dmetric,
                    threadBlock);
            }
            else
            {
                ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, deriv0, deriv1, deriv2, metric,
                    threadBlock);
            }
            SumDerivTensor3DQPKernel<false, DEFORMED>(nq0, nq1, nq2, D0, D1, D2,
                                                      deriv0, deriv1, deriv2,
                                                      bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, bwd, outptr,
                    s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL void Laplacian1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian1DKernel<Implementation, DEFORMED>(ncoord, nm0, nq0, nelmt, basis0,
                                                D0, w0, df, jac, coeff, in, out,
                                                wsp, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    Laplacian1DKernelLauncher(
        const unsigned int ncoord, const size_t nelmt,
        const TData *__restrict__ basis0, const TData *__restrict__ D0,
        const TData *__restrict__ w0, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ coeff,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian1DKernel<Implementation, DEFORMED>(ncoord, nm0, nq0, nelmt, basis0,
                                                D0, w0, df, jac, coeff, in, out,
                                                wsp, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Laplacian2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, in, out, wsp,
        shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TthreadBlock,
          typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    Laplacian2DKernelLauncher(
        const unsigned int ncoord, const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ coeff,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, in, out, wsp,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void Laplacian3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    Laplacian3DKernelLauncher(
        const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const unsigned int *__restrict__ index3,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ D0,
        const TData *__restrict__ D1, const TData *__restrict__ D2,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ w2, const TData *__restrict__ f0,
        const TData *__restrict__ f1, const TData *__restrict__ f1m,
        const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
        const TData *__restrict__ df, const TData *__restrict__ jac,
        const TData *__restrict__ coeff, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Laplacian3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, in, out, wsp, shmemptr, threadBlock);
}

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void Laplacian1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *basis0, const TData *D0, const TData *w0,
    const TData *df, const TData *jac, const TData *coeff, const TData *in,
    TData *out, TData *wsp)
{
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian1DKernelLauncher<Implementation, DEFORMED>), gridsize,
        blocksize, shmemsize, 0, ncoord, nm0, nq0, nelmt, basis0, D0, w0, df,
        jac, coeff, in, out, wsp);
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void Laplacian1DKernel(
    const unsigned int ncoord, const size_t nelmt, const TData *basis0,
    const TData *D0, const TData *w0, const TData *df, const TData *jac,
    const TData *coeff, const TData *in, TData *out, TData *wsp)
{
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian1DKernelLauncher<Implementation, DEFORMED, nm0, nq0>),
        gridsize, blocksize, shmemsize, 0, ncoord, nelmt, basis0, D0, w0, df,
        jac, coeff, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Laplacian2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *D0, const TData *D1, const TData *w0,
    const TData *w1, const TData *f0, const TData *f1, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *in,
    TData *out, TData *wsp)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
        gridsize, blocksize, shmemsize, 0, ncoord, nm0, nm1, nmTot, nq0, nq1,
        nelmt, isModified, index0, basis0, basis1, D0, D1, w0, w1, f0, f1,
        nodToMod, df, jac, coeff, in, out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void Laplacian2DKernel(
    const unsigned int ncoord, const size_t nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *D0, const TData *D1, const TData *w0, const TData *w1,
    const TData *f0, const TData *f1, const TData *nodToMod, const TData *df,
    const TData *jac, const TData *coeff, const TData *in, TData *out,
    TData *wsp)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED, nm0,
                                   nm1, nmTot, nq0, nq1>),
        gridsize, blocksize, shmemsize, 0, ncoord, nelmt, isModified, index0,
        basis0, basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, in,
        out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Laplacian3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *in,
    TData *out, TData *wsp)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
        gridsize, blocksize, shmemsize, 0, nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
        nelmt, isModified, index0, index1, index2, index3, basis0, basis1,
        basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, nodToMod, df, jac,
        coeff, in, out, wsp);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void Laplacian3DKernel(
    const size_t nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *in,
    TData *out, TData *wsp)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * LaplacianSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    DEVICE_1DGRID_KERNEL_LAUNCHER(
        (Laplacian3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED, nm0,
                                   nm1, nm2, nmTot, nq0, nq1, nq2>),
        gridsize, blocksize, shmemsize, 0, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, in, out, wsp);
}
#endif

} // namespace Nektar::Operators::detail
