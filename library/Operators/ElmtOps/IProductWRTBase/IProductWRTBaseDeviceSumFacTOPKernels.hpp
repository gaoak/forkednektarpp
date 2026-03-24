///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFacTOPKernels.hpp
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
#include "Operators/Utils/UtilsDeviceKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTBaseSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    return nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTBaseSharedMemorySize(const unsigned int nq0,
                                                    const unsigned int nq1,
                                                    const unsigned int nm0,
                                                    const unsigned int nm1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        return nm0 * nq0 + nmTot * nq1 + nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = nm0 * (nm0 + 1) / 2;
        return nm0 * nq0 + nmTot * nq1 + nq0 * nq1 + nm0 * nq1 + nmTot;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int IProductWRTBaseSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = nm0 * (nm0 + 1) * (nm0 + 2) / 6;
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2 + nmTot;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = nm0 * (nm0 + 1) * nm0 / 2;
        const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nmTot;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            const unsigned int index = warpsize * i + ilane;
            if constexpr (DEFORMED)
            {
                sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
            }
            else
            {
                sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
            }
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * p + ilane] += sum;
        }
        else
        {
            out[warpsize * p + ilane] = sum;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            sum += in[warpsize * i + ilane] * basis0[p * nq0 + i];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * p + ilane] += sum;
        }
        else
        {
            out[warpsize * p + ilane] = sum;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * cnt_ji + ilane;
                if constexpr (DEFORMED)
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
                }
                else
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                }
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[q * nq1 + j] * w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * (nm0 * q + p) + ilane] += sum;
            }
            else
            {
                out[warpsize * (nm0 * q + p) + ilane] = sum;
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += in[warpsize * cnt_ji + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[q * nq1 + j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * (nm0 * q + p) + ilane] += sum;
            }
            else
            {
                out[warpsize * (nm0 * q + p) + ilane] = sum;
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * cnt_ji + ilane;
                if constexpr (DEFORMED)
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
                }
                else
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                }
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[mode_pq * nq1 + j] *
                       w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * mode_pq + ilane] += sum;
            }
            else
            {
                out[warpsize * mode_pq + ilane] = sum;
            }
        }
    }

    // Correction for singular vertex in collpased coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * (nq0 * j + i) + ilane;

                if constexpr (DEFORMED)
                {
                    prod += in[index] * w0[i] * w1[j] * basis1[nq1 + j] *
                            basis0[nq0 + i] * jac[index];
                }
                else
                {
                    prod += in[index] * w0[i] * w1[j] * basis1[nq1 + j] *
                            basis0[nq0 + i] * jac[0];
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += in[warpsize * cnt_ji + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[mode_pq * nq1 + j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * mode_pq + ilane] += sum;
            }
            else
            {
                out[warpsize * mode_pq + ilane] = sum;
            }
        }
    }

    // Correction for singular vertex in collpased coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                prod += in[warpsize * cnt_ji + ilane] * basis1[nq1 + j] *
                        basis0[nq0 + i];
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2; ++r)
            {
                const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                TData sum = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum += wsp1[warpsize * k + ilane] * basis2[r * nq2 + k] *
                           w2[k];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_rqp + ilane] += sum;
                }
                else
                {
                    out[warpsize * cnt_rqp + ilane] = sum;
                }
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2; ++r)
            {
                const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                TData sum = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum += wsp1[warpsize * k + ilane] * basis2[r * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_rqp + ilane] += sum;
                }
                else
                {
                    out[warpsize * cnt_rqp + ilane] = sum;
                }
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0;
         ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[mode_pq * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    tmp += wsp1[warpsize * k + ilane] *
                           basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    tmp *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += tmp;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = tmp;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM2_MAX = 4;
        TData prod[NM2_MAX - 2]        = {0.0};
        TData topVert                  = 0;
        TData bottomVert               = 0;

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // Store jac * quadrature weight
                    TData tmp = w0[i] * w1[j] * w2[k];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    // top vertex
                    topVert += (basis0[i] * basis1[nq1 + j] +
                                basis0[nq0 + i] * basis1[j] +
                                basis0[nq0 + i] * basis1[nq1 + j]) *
                               basis2[nq2 + k] * in[index] * tmp;

                    // bottom vertex
                    bottomVert += basis0[nq0 + i] * basis1[nq1 + j] *
                                  basis2[k] * in[index] * tmp;

                    // singular edge
                    tmp *= basis1[nq1 + j] * basis0[nq0 + i] * in[index];
                    for (unsigned int r = 1u;
                         r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
                    {
                        prod[r - 1] += basis2[(r + 1) * nq2 + k] * tmp;
                    }
                    for (unsigned int r = NM2_MAX - 1u; r < nm2 - 1u; ++r)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp;
                        }
                    }
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += topVert * scale;
            out[warpsize * nm2 + ilane] += bottomVert * scale;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1] * scale;
            }
        }
        else
        {
            out[warpsize + ilane] += topVert;
            out[warpsize * nm2 + ilane] += bottomVert;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1];
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0;
         ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[mode_pq * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    tmp += wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    tmp *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += tmp;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = tmp;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM2_MAX = 4;
        TData prod[NM2_MAX - 2]        = {0.0};
        TData topVert                  = 0;
        TData bottomVert               = 0;
        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // top vertex
                    topVert += (basis0[i] * basis1[nq1 + j] +
                                basis0[nq0 + i] * basis1[j] +
                                basis0[nq0 + i] * basis1[nq1 + j]) *
                               basis2[nq2 + k] * in[index];

                    // bottom vertex
                    bottomVert += basis0[nq0 + i] * basis1[nq1 + j] *
                                  basis2[k] * in[index];

                    // singular edge
                    TData tmp = basis1[nq1 + j] * basis0[nq0 + i] * in[index];
                    for (unsigned int r = 1u;
                         r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
                    {
                        prod[r - 1] += basis2[(r + 1) * nq2 + k] * tmp;
                    }
                    for (unsigned int r = NM2_MAX - 1; r < nm2 - 1u; ++r)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp;
                        }
                        else
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp;
                        }
                    }
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += topVert * scale;
            out[warpsize * nm2 + ilane] += bottomVert * scale;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1] * scale;
            }
        }
        else
        {
            out[warpsize + ilane] += topVert;
            out[warpsize * nm2 + ilane] += bottomVert;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1];
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
            {
                unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[(mode_pr + r) * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM1_MAX = 4;
        TData prod[NM1_MAX]            = {0.0};

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    TData tmp = basis2[nq2 + k] * basis0[nq0 + i] * w0[i] *
                                w1[j] * w2[k] * in[index];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
                    {
                        prod[q] += tmp * basis1[q * nq1 + j];
                    }
                    for (unsigned int q = NM1_MAX; q < nm1; ++q)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j] * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j];
                        }
                    }
                }
            }
        }

        for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
        {
            if constexpr (SCALE)
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q] * scale;
            }
            else
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q];
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
            {
                unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[(mode_pr + r) * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM1_MAX = 4;
        TData prod[NM1_MAX]            = {0.0};

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    TData tmp = basis2[nq2 + k] * basis0[nq0 + i] * in[index];
                    for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
                    {
                        prod[q] += tmp * basis1[q * nq1 + j];
                    }
                    for (unsigned int q = NM1_MAX; q < nm1; ++q)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j] * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j];
                        }
                    }
                }
            }
        }

        for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
        {
            if constexpr (SCALE)
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q] * scale;
            }
            else
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q];
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < p; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        for (unsigned int q = p; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1; q < nm2; ++q)
        {
            mode2 += nm2 - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        TData prod = 0.0;

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // Store jac * quadrature weight
                    TData tmp = w0[i] * w1[j] * w2[k];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    // top vertex
                    prod += (basis0[i] * basis1[nq1 + j] +
                             basis0[nq0 + i] * basis1[j] +
                             basis0[nq0 + i] * basis1[nq1 + j]) *
                            basis2[nq2 + k] * in[index] * tmp;
                }
            }
        }

        // Add to existing entry.
        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacTOPKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < p; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k +=
                        wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        for (unsigned int q = p; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k +=
                        wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1; q < nm2; ++q)
        {
            mode2 += nm2 - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    // top vertex
                    prod += (basis0[i] * basis1[nq1 + j] +
                             basis0[nq0 + i] * basis1[j] +
                             basis0[nq0 + i] * basis1[nq1 + j]) *
                            basis2[nq2 + k] * in[warpsize * cnt_kji + ilane];
                }
            }
        }

        // Add to existing entry.
        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int p = idx0; p < nm0; p += stride)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            sum += in[i] * basis0[p * nq0 + i];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[p] += sum;
        }
        else
        {
            out[p] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacTOPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    [[maybe_unused]] const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = idx / nm0;
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[q * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacTOPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[idx * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    // Correction for singular vertex in collpased coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            prod += basis0[nq0 + i] * basis1[nq1 + j] * in[idx];
        }

        if constexpr (SCALE)
        {
            prod *= scale;
        }

        blockReduceSum(prod, threadBlock, out + 1);
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += wsp0[cnt_pkj] * basis1[q * nq1 + j];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = (idx / nm0) % nm1;
        const unsigned int r = idx / (nm0 * nm1);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum += wsp1[cnt_pqk] * basis2[r * nq2 + k];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex1,
    const unsigned int *NEK_RESTRICT pindex2,
    const unsigned int *NEK_RESTRICT qindex2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm01 * nq2; idx += stride)
    {
        const unsigned int mode_pq = idx / nq2;
        const unsigned int p       = pindex1[mode_pq];
        const unsigned int k       = idx % nq2;
        unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[mode_pq * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex2[idx];
        const unsigned int q       = qindex2[idx];
        const unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k)
        {
            tmp += wsp1[mode_pq * nq2 + k] * basis2[mode2 * nq2 + k];
        }

        if constexpr (SCALE)
        {
            tmp *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += tmp;
        }
        else
        {
            out[idx] = tmp;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM2_MAX = 8;
        if (nm2 <= NM2_MAX)
        {
            TData prod[NM2_MAX] = {0.0};
            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                prod[nm2 - 1u] +=
                    (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];

                // singular edge
                TData tmp = basis1[nq1 + j] * basis0[nq0 + i] * in[idx];
#pragma unroll
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    prod[r] += basis2[(r + 1u) * nq2 + k] * tmp;
                }

                // bottom vertex
                prod[0] += basis2[k] * tmp;
            }
#if !defined(NEKTAR_ENABLE_SYCL)
#pragma unroll
#endif
            for (unsigned int r = 0u; r < nm2; ++r)
            {
                if constexpr (SCALE)
                {
                    prod[r] *= scale;
                }

                if (r == nm2 - 1u)
                {
                    blockReduceSum(prod[r], threadBlock, out + 1);
                }
                else
                {
                    blockReduceSum(prod[r], threadBlock, out + nm2 + r);
                }
            }
        }
        else
        {
            TData prod0 = 0.0;
            TData prod1 = 0.0;

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                prod0 +=
                    (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];

                // bottom vertex
                prod1 +=
                    basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] * in[idx];
            }

            if constexpr (SCALE)
            {
                prod0 *= scale;
                prod1 *= scale;
            }

            blockReduceSum(prod0, threadBlock, out + 1);
            blockReduceSum(prod1, threadBlock, out + nm2);

            // singular edge
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += basis2[(r + 1u) * nq2 + k] * basis1[nq1 + j] *
                            basis0[nq0 + i] * in[idx];
                }

                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                blockReduceSum(prod, threadBlock, out + nm2 + r);
            }
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex,
    const unsigned int *NEK_RESTRICT qindex,
    const unsigned int *NEK_RESTRICT rindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex[idx];
        const unsigned int q       = qindex[idx];
        const unsigned int r       = rindex[idx];
        const unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u + r;
        unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode_pr * nq2 + k] * wsp1[cnt_pqk];
        }

        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM1_MAX = 8;
        if (nm1 <= NM1_MAX)
        {
            TData prod[NM1_MAX] = {0.0};

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                TData tmp = in[idx] * basis2[nq2 + k] * basis0[nq0 + i];
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    prod[q] += tmp * basis1[q * nq1 + j];
                }
            }

#if !defined(NEKTAR_ENABLE_SYCL)
#pragma unroll
#endif
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                if constexpr (SCALE)
                {
                    prod[q] *= scale;
                }

                blockReduceSum(prod[q], threadBlock, out + nm2 * q + 1);
            }
        }
        else
        {
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += in[idx] * basis2[nq2 + k] * basis1[q * nq1 + j] *
                            basis0[nq0 + i];
                }

                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                blockReduceSum(prod, threadBlock, out + nm2 * q + 1);
            }
        }
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex,
    const unsigned int *NEK_RESTRICT qindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        const unsigned int q = qindex[idx];
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode2 * nq2 + k] * wsp1[cnt_pqk];
        }
        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            // top vertex
            prod += (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];
        }

        if constexpr (SCALE)
        {
            prod *= scale;
        }

        blockReduceSum(prod, threadBlock, out + 1);
    }

    localBarrier(threadBlock);
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase1DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *s_wsp0 = (TData *)shmemptr;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nq0 * e;
        TData *outptr       = out + nm0 * e;

        for (unsigned int i = idx0; i < nq0; i += stride)
        {
            if constexpr (DEFORMED)
            {
                s_wsp0[i] = inptr[i] * jacptr[i] * w0[i];
            }
            else
            {
                s_wsp0[i] = inptr[i] * jacptr[0] * w0[i];
            }
        }

        localBarrier(threadBlock);

        IProductWRTBaseSegSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
            nm0, nq0, basis0, s_wsp0, outptr, scale, threadBlock);
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase2DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const unsigned int *NEK_RESTRICT index0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1;
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

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_basis0  = s_wsp1 + offset;
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
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * e;
        TData *outptr       = out + nmTot * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            if constexpr (DEFORMED)
            {
                s_wsp0[idx] = inptr[idx] * jacptr[idx] * w0[i] * w1[j];
            }
            else
            {
                s_wsp0[idx] = inptr[idx] * jacptr[0] * w0[i] * w1[j];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, s_wsp0,
                outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, s_wsp0, outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, s_wsp0, s_out1ptr, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase3DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const unsigned int *NEK_RESTRICT index0,
    const unsigned int *NEK_RESTRICT index1,
    const unsigned int *NEK_RESTRICT index2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT nodToMod,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = nq0 * nq1 * nq2;
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

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_wsp2    = s_wsp1 + offset0;
    TData *s_basis0  = s_wsp2 + offset1;
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
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * e;
        TData *outptr       = out + nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            if constexpr (DEFORMED)
            {
                s_wsp0[idx] = inptr[idx] * jacptr[idx] * w0[i] * w1[j] * w2[k];
            }
            else
            {
                s_wsp0[idx] = inptr[idx] * jacptr[0] * w0[i] * w1[j] * w2[k];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, s_wsp0, outptr, s_wsp1,
                s_wsp2, scale, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTBase1DKernelLauncher(
        NonTemplated1DSizeParameters sizeParam1D, const size_t nelmt,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase1DSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
        sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt, basis0, w0, jac, in, out,
        scale, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
    unsigned int nm0, unsigned int nq0, typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase1DKernelLauncher(
        [[maybe_unused]] Templated1DSizeParameters<nm0, nq0> sizeParam1D,
        const size_t nelmt, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase1DSumFacTOPKernel<SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, shmemptr,
        threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTBase2DKernelLauncher(
        NonTemplated2DSizeParameters sizeParam2D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, isModified, index0, basis0,
        basis1, w0, w1, nodToMod, jac, in, out, scale, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nmTot, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase2DKernelLauncher(
        [[maybe_unused]] Templated2DSizeParameters<nm0, nm1, nmTot, nq0, nq1>
            sizeParam2D,
        const size_t nelmt, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, nodToMod, jac, in, out, scale, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    IProductWRTBase3DKernelLauncher(
        NonTemplated3DSizeParameters sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, isModified, index0, index1, index2, basis0,
        basis1, basis2, w0, w1, w2, nodToMod, jac, in, out, scale, shmemptr,
        threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase3DKernelLauncher(
        [[maybe_unused]] Templated3DSizeParameters<nm0, nm1, nm2, nmTot, nq0,
                                                   nq1, nq2>
            sizeParam3D,
        const size_t nelmt, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, basis0, basis1, basis2, w0, w1, w2, nodToMod, jac, in, out,
        scale, shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
