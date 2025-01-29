///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFacKernels.hpp
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
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nm0)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        return nm0 + nm0 * nq0;
    }
    else
    {
        return 0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nq1,
                                             const unsigned int nm0,
                                             const unsigned int nm1)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return nq0 * nm0 + nq1 * nm1 + nmTot + nq0 * nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return nm0 * nq0 + nmTot * nq1 + nmTot + nm0 * nq1;
        }
    }
    else
    {
        return 0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int BwdTransSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nq0 * nm0 + nq1 * nm1 +
                   nq2 * nm2 + nmTot + (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot +
                   (nm01 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return nm0 * nq0 + nm1 * nq1 + nm12 * nq2 + nmTot +
                   (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot +
                   (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
    }
    else
    {
        return 0;
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p)
        {
            tmp += in[warpsize * p + ilane] * basis0[p * nq0 + i];
        }
        out[warpsize * i + ilane] = tmp;
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // direction 0
        for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
            {
                tmp += in[warpsize * cnt_qp + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * q + ilane] = tmp;
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += wsp[warpsize * q + ilane] * basis1[q * nq1 + j];
            }
            out[warpsize * (nq0 * j + i) + ilane] = tmp;
        }
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        // direction 1
        for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < (nm1 - p); ++q, ++mode_pq)
            {
                tmp +=
                    in[warpsize * mode_pq + ilane] * basis1[mode_pq * nq1 + j];
            }
            wsp[warpsize * p + ilane] = tmp;
        }

        // direction 0
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            TData tmp = 0.0;

            if (isModified)
            {
                tmp += in[warpsize + ilane] * basis0[nq0 + i] * basis1[nq1 + j];
            }

#pragma unroll
            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += wsp[warpsize * p + ilane] * basis0[p * nq0 + i];
            }

            out[warpsize * cnt_ji + ilane] = tmp;
        }
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // direction 0
        for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u; r < nm2; ++r)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                {
                    tmp += in[warpsize * cnt_rqp + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_rq + ilane] = tmp;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                {
                    tmp +=
                        wsp0[warpsize * cnt_rq + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * r + ilane] = tmp;
            }

            // direction 2
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    tmp += wsp1[warpsize * r + ilane] * basis2[r * nq2 + k];
                }
                out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - p - q;
                     ++r, ++mode2, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[nq2 * mode2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case nm2>nm1
#pragma unroll
            for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
            {
                mode2 += nm2 - p - q;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    tmp += fpq[warpsize * mode_pq + ilane] *
                           basis1[mode_pq * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];

                    // bottom vertex
                    TData tmp1 = basis2[k] * in[warpsize * nm2 + ilane];

                    // singular edge
#pragma unroll
                    for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                    {
                        tmp1 += basis2[(r + 1u) * nq2 + k] *
                                in[warpsize * (nm2 + r) + ilane];
                    }
                    tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransPrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pr = 0u, mode_pq = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[(mode_pr + r) * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }
            mode_pr += nm2 - p;
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
#pragma unroll
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        tmp += basis1[q * nq1 + j] *
                               in[warpsize * (nm2 * q + 1u) + ilane];
                    }
                    tmp *= basis2[nq2 + k] * basis0[nq0 + i];
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void BwdTransPyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ fpq, TData *__restrict__ fp)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - std::max(p, q);
                     ++r, ++mode2, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[mode2 * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case nm2>nm1
#pragma unroll
            for (unsigned int q = nm1; q < nm2; ++q)
            {
                mode2 += nm2 - q;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                out[warpsize * cnt_kji + ilane] = tmp;
            }
        }
    }
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"
