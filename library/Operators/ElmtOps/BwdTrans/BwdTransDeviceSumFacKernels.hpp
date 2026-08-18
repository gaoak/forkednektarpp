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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <Operators/ElmtOps/ElmtHelper.hpp>

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
inline constexpr size_t BwdTransWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    if constexpr (SHAPE_TYPE == LibUtilities::Seg)
    {
        wspsize = 0;
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nm1 = sizeParam2D.nm1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wspsize = nm1 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        wspsize = nm0 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        wspsize = (nm0 + nmTot) * nelmt;
    }

    return wspsize;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();
    const unsigned int nm2 = sizeParam3D.nm2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wspsize = (nm1 * nm2 + nm2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        wspsize = ((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        wspsize = (nm0 * nm1 + nm0) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = ((nm0 * nm1 + nm0) + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wspsize = (nm0 * nm1 + nm0) * nelmt;
    }

    return wspsize;
}

template <typename Implementation, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
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
inline constexpr unsigned int BwdTransSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p)
        {
            tmp += in[warpsize * p + ilane] * basis0[p * nq0 + i];
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += tmp;
        }
        else
        {
            out[warpsize * i + ilane] = tmp;
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

            if constexpr (APPEND)
            {
                out[warpsize * (nq0 * j + i) + ilane] += tmp;
            }
            else
            {
                out[warpsize * (nq0 * j + i) + ilane] = tmp;
            }
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

            if constexpr (APPEND)
            {
                out[warpsize * cnt_ji + ilane] += tmp;
            }
            else
            {
                out[warpsize * cnt_ji + ilane] = tmp;
            }
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                if constexpr (APPEND)
                {
                    out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] +=
                        tmp;
                }
                else
                {
                    out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
                }
            }
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransPrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransPyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

template <typename Implementation, bool APPEND, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    BwdTrans1DKernelLauncher(const TSizeParameter1D sizeParam1D,
                             const size_t nelmt,
                             const TData *NEK_RESTRICT basis0,
                             const TData *NEK_RESTRICT in,
                             TData *NEK_RESTRICT out,
                             [[maybe_unused]] unsigned char *shmemptr,
                             const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nq0 * (nelmt * c + warpsize * iwarp);
        BwdTransSegSumFacKernel<APPEND>(ilane, nm0, nq0, basis0, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    BwdTrans2DKernelLauncher(const TSizeParameter2D sizeParam2D,
                             const size_t nelmt, const bool isModified,
                             const TData *NEK_RESTRICT basis0,
                             const TData *NEK_RESTRICT basis1,
                             const TData *NEK_RESTRICT nodToMod,
                             const TData *NEK_RESTRICT in,
                             TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
                             [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int nqTot = nq0 * nq1;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wspptr = wsp + nm1 * (nelmt * c + warpsize * iwarp);
            BwdTransQuadSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1, basis0,
                                             basis1, inptr, outptr, wspptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wspptr = wsp + nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransTriSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1,
                                            isModified, basis0, basis1, inptr,
                                            outptr, wspptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wspptr = wsp + nmTot * nelmt * ncomp +
                            nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTriSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1,
                                            isModified, basis0, basis1, in1ptr,
                                            outptr, wspptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    BwdTrans3DKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, [[maybe_unused]] const unsigned int *index0,
        [[maybe_unused]] const unsigned int *index1,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int nqTot = nq0 * nq1 * nq2;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nm1 * nm2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm1 * nm2) * nelmt * ncomp +
                          nm2 * (nelmt * c + warpsize * iwarp);
            BwdTransHexSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            basis0, basis1, basis2, inptr,
                                            outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

            TData *wsp0 = wsp + nm01 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm01 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransTetSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            inptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0   = wsp + nmTot * nelmt * ncomp +
                          nm01 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm01 + nmTot) * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTetSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            in1ptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm0 * nm1 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransPrismSumFacKernel<APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0   = wsp + nmTot * nelmt * ncomp +
                          nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm0 * nm1 + nmTot) * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransPrismSumFacKernel<APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, in1ptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm0 * nm1 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransPyrSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            inptr, outptr, wsp0, wsp1);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
