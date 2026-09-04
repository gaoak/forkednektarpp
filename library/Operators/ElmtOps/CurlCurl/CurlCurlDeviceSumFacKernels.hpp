///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceSumFacKernels.hpp
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter2D sizeParam2D)
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
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
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
        return nq0 + 2u * nq1 + nq2;
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

// The curl-curl operator is evaluated in two passes, omega = curl(u) followed
// by out = curl(omega). Both passes run inside a single fused kernel, so the
// only global workspace needed is the one holding the intermediate omega.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t CurlCurlWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t CurlCurlWorkSpaceSize(
    const size_t nelmt, const TPhysSizeParameter2D sizeParam2D)
{
    // In 2D omega only has an out-of-plane component.
    return sizeParam2D.nq0() * sizeParam2D.nq1() * nelmt;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t CurlCurlWorkSpaceSize(
    const size_t nelmt, const TPhysSizeParameter3D sizeParam3D)
{
    return 3u * sizeParam3D.nq0() * sizeParam3D.nq1() * sizeParam3D.nq2() *
           nelmt;
}

// Scalar curl of a two-dimensional vector field, omega = dv/dx - du/dy.
//
// Both components of the tensorial derivative of u and v are accumulated in
// registers so that the chain rule, the collapsed coordinate correction and
// the curl are applied in a single sweep over the quadrature points.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Curl2DScalarSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT omega)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2u * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivatives of both components at once.
            // Direction 0
            TData d0u = 0.0;
            TData d0v = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                const TData Dq             = D0[q * nq0 + i];
                const unsigned int qsource = warpsize * (nq0 * j + q) + ilane;
                d0u += Dq * in[qsource];
                d0v += Dq * in[inoffset + qsource];
            }

            // Direction 1
            TData d1u = 0.0;
            TData d1v = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                const TData Dq             = D1[q * nq1 + j];
                const unsigned int qsource = warpsize * (nq0 * q + i) + ilane;
                d1u += Dq * in[qsource];
                d1v += Dq * in[inoffset + qsource];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0u *= f1[j];
                d1u += d0u * f0[i];
                d0v *= f1[j];
                d1v += d0v * f0[i];
            }

            // Multiply by derivative factors and take the curl.
            const TData dvdx = d0v * df[0u * warpsize + dfindex] +
                               d1v * df[1u * warpsize + dfindex];
            const TData dudy = d0u * df[2u * warpsize + dfindex] +
                               d1u * df[3u * warpsize + dfindex];

            omega[index] = dvdx - dudy;
        }
    }
}

// Vector curl of a two-dimensional scalar field,
// out = {d(omega)/dy, -d(omega)/dx}.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Curl2DVectorSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT omega, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2u * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * omega[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * omega[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors and take the curl.
            out[index] = d0 * df[2u * warpsize + dfindex] +
                         d1 * df[3u * warpsize + dfindex];
            out[outoffset + index] = -(d0 * df[0u * warpsize + dfindex] +
                                       d1 * df[1u * warpsize + dfindex]);
        }
    }
}

// Curl of a three-dimensional vector field. Used for both passes of the
// curl-curl operator, i.e. omega = curl(u) and out = curl(omega).
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Curl3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t inoffset, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                // Compute the tensorial derivatives of the three components
                // at once.
                // Direction 0
                TData d0[3] = {0.0, 0.0, 0.0};
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    const TData Dq = D0[q * nq0 + i];
                    const unsigned int qsource =
                        warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane;
                    d0[0] += Dq * in[qsource];
                    d0[1] += Dq * in[inoffset + qsource];
                    d0[2] += Dq * in[2u * inoffset + qsource];
                }

                // Direction 1
                TData d1[3] = {0.0, 0.0, 0.0};
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    const TData Dq = D1[q * nq1 + j];
                    const unsigned int qsource =
                        warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane;
                    d1[0] += Dq * in[qsource];
                    d1[1] += Dq * in[inoffset + qsource];
                    d1[2] += Dq * in[2u * inoffset + qsource];
                }

                // Direction 2
                TData d2[3] = {0.0, 0.0, 0.0};
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    const TData Dq = D2[q * nq2 + k];
                    const unsigned int qsource =
                        warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane;
                    d2[0] += Dq * in[qsource];
                    d2[1] += Dq * in[inoffset + qsource];
                    d2[2] += Dq * in[2u * inoffset + qsource];
                }

                // Moving from standard to collapsed coordinates.
#pragma unroll
                for (unsigned int c = 0u; c < 3u; ++c)
                {
                    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                                  SHAPE_TYPE == LibUtilities::NodalTet)
                    {
                        TData tmp0 = f1m[j] * f2[k] * d0[c];
                        TData tmp1 = f0[i] * tmp0;
                        TData tmp2 = f2[k] * d1[c];
                        d0[c]      = tmp0;
                        d1[c]      = tmp1 + tmp2;
                        d2[c] += tmp1 + f1[j] * tmp2;
                    }
                    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                       SHAPE_TYPE == LibUtilities::NodalPrism)
                    {
                        d0[c] *= f2[k];
                        d2[c] += f0[i] * d0[c];
                    }
                    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                    {
                        d0[c] *= f2[k];
                        d1[c] *= f2[k];
                        d2[c] += f0[i] * d0[c] + f1[j] * d1[c];
                    }
                }

                // Multiply by the derivative factors and take the curl. Only
                // the six off-diagonal physical derivatives are needed, the
                // diagonal ones cancel out.
                const TData df0 = df[0u * warpsize + dfindex];
                const TData df1 = df[1u * warpsize + dfindex];
                const TData df2 = df[2u * warpsize + dfindex];
                const TData df3 = df[3u * warpsize + dfindex];
                const TData df4 = df[4u * warpsize + dfindex];
                const TData df5 = df[5u * warpsize + dfindex];
                const TData df6 = df[6u * warpsize + dfindex];
                const TData df7 = df[7u * warpsize + dfindex];
                const TData df8 = df[8u * warpsize + dfindex];

                // dw/dy - dv/dz
                out[index] = (d0[2] * df3 + d1[2] * df4 + d2[2] * df5) -
                             (d0[1] * df6 + d1[1] * df7 + d2[1] * df8);
                // du/dz - dw/dx
                out[outoffset + index] =
                    (d0[0] * df6 + d1[0] * df7 + d2[0] * df8) -
                    (d0[2] * df0 + d1[2] * df1 + d2[2] * df2);
                // dv/dx - du/dy
                out[2u * outoffset + index] =
                    (d0[1] * df0 + d1[1] * df1 + d2[1] * df2) -
                    (d0[0] * df3 + d1[0] * df4 + d2[0] * df5);
            }
        }
    }
}

// The curl-curl operator is not defined in one dimension. The launcher is
// only instantiated so that the segment block operator compiles, and simply
// evaluates the derivative in the first direction.
template <typename Implementation, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurl1DKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    [[maybe_unused]] const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * warpsize * iwarp;
        TData *outptr      = out + nq0 * warpsize * iwarp;
        PhysDerivDir1DSumFacKernel<false, DEFORMED, 0>(ilane, ncoord, nq0, D0,
                                                       dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

// Fused two-dimensional curl-curl kernel. Both curls are evaluated in the
// same kernel launch. Each lane owns a complete element, hence the
// intermediate omega it writes to the workspace is read back by the same
// lane and no synchronisation is required between the two passes.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurl2DKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
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

    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
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

    size_t e = getGlobalIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *wspptr      = wsp + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;

        // omega = dv/dx - du/dy
        Curl2DScalarSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, D0, D1, s_f0, s_f1, dfptr, inptr,
            wspptr);

        // out = {d(omega)/dy, -d(omega)/dx}
        Curl2DVectorSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, D0, D1, s_f0, s_f1, dfptr,
            wspptr, outptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

// Fused three-dimensional curl-curl kernel, see the two-dimensional kernel
// above for the rationale.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurl3DKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT out, unsigned char *shmemptr,
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
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
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *wspptr      = wsp + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;

        // omega = curl(u)
        Curl3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, inoffset, D0, D1, D2, s_f0, s_f1,
            s_f1m, s_f2, dfptr, inptr, wspptr);

        // out = curl(omega)
        Curl3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, inoffset, D0, D1, D2, s_f0, s_f1,
            s_f1m, s_f2, dfptr, wspptr, outptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
