///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFacKernels.hpp
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
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline unsigned int IProductWRTBaseSharedMemorySize(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nm0)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline unsigned int IProductWRTBaseSharedMemorySize(
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
inline unsigned int IProductWRTBaseSharedMemorySize(
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2)
{
    return 0;
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1, const TData scale)
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
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
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
NEK_DEVICE_INLINE static void IProductWRTBase1DSumFacKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
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
        const TData *inptr = in + nq0 * warpsize * iwarp;
        TData *outptr      = out + nm0 * warpsize * iwarp;
        IProductWRTBaseSegSumFacKernel<SCALE, APPEND, DEFORMED>(
            ilane, nm0, nq0, basis0, w0, jacptr, inptr, outptr, scale);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase2DSumFacKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
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
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nmTot * warpsize * iwarp;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wspptr = wsp + nq1 * warpsize * iwarp;
            IProductWRTBaseQuadSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr,
                inptr, outptr, wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wspptr = wsp + nq1 * warpsize * iwarp;
            IProductWRTBaseTriSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, inptr, outptr, wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr = wsp + nmTot * warpsize * iwarp;
            TData *wspptr  = wsp + nmTot * nelmt;

            IProductWRTBaseTriSumFacKernel<SCALE, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, inptr, out1ptr, wspptr, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SCALE, bool APPEND,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBase3DSumFacKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
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
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nmTot * warpsize * iwarp;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
            IProductWRTBaseHexSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2, w0,
                w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {

            TData *out1ptr = wsp + nmTot * warpsize * iwarp;
            TData *wsp0    = wsp + nmTot * nelmt + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1    = wsp + nmTot * nelmt + nq1 * nq2 * nelmt +
                          nq2 * warpsize * iwarp;
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr = wsp + nmTot * warpsize * iwarp;
            TData *wsp0    = wsp + nmTot * nelmt + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1    = wsp + nmTot * nelmt + nq1 * nq2 * nelmt +
                          nq2 * warpsize * iwarp;
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nq1 * nq2 * warpsize * iwarp;
            TData *wsp1 = wsp + nq1 * nq2 * nelmt + nq2 * warpsize * iwarp;
            IProductWRTBasePyrSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        e += getGlobalRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    IProductWRTBase1DKernelLauncher(
        NonTemplated1DSizeParameters sizeParam1D, const size_t nelmt,
        const TData *__restrict__ basis0, const TData *__restrict__ w0,
        const TData *__restrict__ jac, const TData *__restrict__ in,
        TData *__restrict__ out, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase1DSumFacKernel<SCALE, APPEND, DEFORMED>(
        sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt, basis0, w0, jac, in, out,
        scale, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
    unsigned int nm0, unsigned int nq0, typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase1DKernelLauncher(
        [[maybe_unused]] Templated1DSizeParameters<nm0, nq0> sizeParam1D,
        const size_t nelmt, const TData *__restrict__ basis0,
        const TData *__restrict__ w0, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase1DSumFacKernel<SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, shmemptr,
        threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    IProductWRTBase2DKernelLauncher(
        NonTemplated2DSizeParameters sizeParam2D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase2DSumFacKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, isModified, basis0, basis1,
        w0, w1, nodToMod, jac, in, out, wsp, scale, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nmTot, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase2DKernelLauncher(
        [[maybe_unused]] Templated2DSizeParameters<nm0, nm1, nmTot, nq0, nq1>
            sizeParam2D,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase2DSumFacKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, basis0, basis1, w0, w1,
        nodToMod, jac, in, out, wsp, scale, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    IProductWRTBase3DKernelLauncher(
        NonTemplated3DSizeParameters sizeParam3D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *__restrict__ index0,
        [[maybe_unused]] const unsigned int *__restrict__ index1,
        [[maybe_unused]] const unsigned int *__restrict__ index2,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ w0,
        const TData *__restrict__ w1, const TData *__restrict__ w2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase3DSumFacKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, isModified, basis0, basis1, basis2, w0, w1,
        w2, nodToMod, jac, in, out, wsp, scale, shmemptr, threadBlock);
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
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) IProductWRTBase3DKernelLauncher(
        [[maybe_unused]] Templated3DSizeParameters<nm0, nm1, nm2, nmTot, nq0,
                                                   nq1, nq2>
            sizeParam3D,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *__restrict__ index0,
        [[maybe_unused]] const unsigned int *__restrict__ index1,
        [[maybe_unused]] const unsigned int *__restrict__ index2,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ basis2, const TData *__restrict__ w0,
        const TData *__restrict__ w1, const TData *__restrict__ w2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ jac,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    IProductWRTBase3DSumFacKernel<SHAPE_TYPE, SCALE, APPEND, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, basis0, basis1,
        basis2, w0, w1, w2, nodToMod, jac, in, out, wsp, scale, shmemptr,
        threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
