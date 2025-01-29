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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

// Helper function
template <typename Implementation>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        return 0;
    }
    else
    {
        return nq0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(const unsigned int nq0,
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
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return nq0 + nq1;
        }
    }
    else
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
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return 0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return nq0 + 2 * nq1 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return nq0 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return nq0 + nq1 + nq2;
        }
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm01 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int insize, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            sum += df[d * warpsize + dfindex] * in[d * insize * nq0 + index];
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
    const unsigned int nq1, const unsigned int insize,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out0, TData *__restrict__ out1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0; d < ncoord; ++d)
            {
                TData tmp = in[d * insize * nqTot + index];
                sum1 += df[(2u * d) * warpsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * warpsize + dfindex] * tmp;
            }

            TData tmpQ = w0[i] * w1[j];
            if constexpr (DEFORMED)
            {
                tmpQ *= jac[warpsize * cnt_ji + ilane];
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
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
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
    const unsigned int nq2, const unsigned int insize,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1,
    [[maybe_unused]] const TData *__restrict__ f1m,
    [[maybe_unused]] const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out0, TData *__restrict__ out1,
    TData *__restrict__ out2)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (unsigned int d = 0u; d < ncoord; ++d)
                {
                    TData tmp = in[d * insize * nqTot + index];
                    sum1 += df[(3u * d) * warpsize + dfindex] * tmp;
                    sum2 += df[(3u * d + 1u) * warpsize + dfindex] * tmp;
                    sum3 += df[(3u * d + 2u) * warpsize + dfindex] * tmp;
                }

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[warpsize * cnt_kji + ilane];
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
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
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

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSYCLSumFacKernels.hpp"
