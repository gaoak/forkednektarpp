///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceSumFacTOPKernels.hpp
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
//
// The two curls of the curl-curl operator are fused into a single kernel, so
// shared memory holds the input components as well as the intermediate omega.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    const TPhysSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    // Two input components plus the out-of-plane omega.
    return 3 * nq0 * nq1;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Three input components plus the three components of omega.
    return 6 * nq0 * nq1 * nq2;
}

// The intermediate omega is held in shared memory, so no global workspace is
// required by this implementation.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
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
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t CurlCurlWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TPhysSizeParameter2D sizeParam2D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t CurlCurlWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TPhysSizeParameter3D sizeParam3D)
{
    return 0;
}

// Scalar curl of a two-dimensional vector field, omega = dv/dx - du/dy.
//
// The tensorial derivatives of both components are accumulated together so
// that the chain rule, the collapsed coordinate correction and the curl are
// applied in a single sweep over the quadrature points.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Curl2DScalarSumFacTOPKernel(
    [[maybe_unused]] const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT omega,
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

        // Compute tensorial derivatives of both components at once.
        // Direction 0
        TData d0u = 0.0;
        TData d0v = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            const TData Dq             = D0[q * nq0 + i];
            const unsigned int qsource = nq0 * j + q;
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
            const unsigned int qsource = nq0 * q + i;
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
        const TData dvdx =
            d0v * df[0u * dfsize + dfindex] + d1v * df[1u * dfsize + dfindex];
        const TData dudy =
            d0u * df[2u * dfsize + dfindex] + d1u * df[3u * dfsize + dfindex];

        omega[idx] = dvdx - dudy;
    }

    localBarrier(threadBlock);
}

// Vector curl of a two-dimensional scalar field,
// out = {d(omega)/dy, -d(omega)/dx}.
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Curl2DVectorSumFacTOPKernel(
    [[maybe_unused]] const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT omega, TData *NEK_RESTRICT out,
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

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * omega[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * omega[nq0 * q + i];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            d0 *= f1[j];
            d1 += d0 * f0[i];
        }

        // Multiply by derivative factors and take the curl.
        out[idx] =
            d0 * df[2u * dfsize + dfindex] + d1 * df[3u * dfsize + dfindex];
        out[outoffset + idx] =
            -(d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex]);
    }

    localBarrier(threadBlock);
}

// Curl of a three-dimensional vector field. Used for both passes of the
// curl-curl operator, i.e. omega = curl(u) and out = curl(omega).
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Curl3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t inoffset, const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
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

        // Compute the tensorial derivatives of the three components at once.
        // Direction 0
        TData d0[3] = {0.0, 0.0, 0.0};
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            const TData Dq             = D0[q * nq0 + i];
            const unsigned int qsource = nq0 * nq1 * k + nq0 * j + q;
            d0[0] += Dq * in[qsource];
            d0[1] += Dq * in[inoffset + qsource];
            d0[2] += Dq * in[2u * inoffset + qsource];
        }

        // Direction 1
        TData d1[3] = {0.0, 0.0, 0.0};
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            const TData Dq             = D1[q * nq1 + j];
            const unsigned int qsource = nq0 * nq1 * k + nq0 * q + i;
            d1[0] += Dq * in[qsource];
            d1[1] += Dq * in[inoffset + qsource];
            d1[2] += Dq * in[2u * inoffset + qsource];
        }

        // Direction 2
        TData d2[3] = {0.0, 0.0, 0.0};
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            const TData Dq             = D2[q * nq2 + k];
            const unsigned int qsource = nq0 * nq1 * q + nq0 * j + i;
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

        // Multiply by the derivative factors and take the curl. Only the six
        // off-diagonal physical derivatives are needed, the diagonal ones
        // cancel out.
        const TData df0 = df[0u * dfsize + dfindex];
        const TData df1 = df[1u * dfsize + dfindex];
        const TData df2 = df[2u * dfsize + dfindex];
        const TData df3 = df[3u * dfsize + dfindex];
        const TData df4 = df[4u * dfsize + dfindex];
        const TData df5 = df[5u * dfsize + dfindex];
        const TData df6 = df[6u * dfsize + dfindex];
        const TData df7 = df[7u * dfsize + dfindex];
        const TData df8 = df[8u * dfsize + dfindex];

        // dw/dy - dv/dz
        out[idx] = (d0[2] * df3 + d1[2] * df4 + d2[2] * df5) -
                   (d0[1] * df6 + d1[1] * df7 + d2[1] * df8);
        // du/dz - dw/dx
        out[outoffset + idx] = (d0[0] * df6 + d1[0] * df7 + d2[0] * df8) -
                               (d0[2] * df0 + d1[2] * df1 + d2[2] * df2);
        // dv/dx - du/dy
        out[2u * outoffset + idx] = (d0[1] * df0 + d1[1] * df1 + d2[1] * df2) -
                                    (d0[0] * df3 + d1[0] * df4 + d2[0] * df5);
    }

    localBarrier(threadBlock);
}

// The curl-curl operator is not defined in one dimension. The launcher is
// only instantiated so that the segment block operator compiles, and simply
// evaluates the derivative in the first direction.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurlKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    [[maybe_unused]] const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    [[maybe_unused]] TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT out,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    size_t e = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nq0 * e;
        TData *outptr      = out + nq0 * e;
        PhysDerivDir1DSumFacTOPKernel<false, DEFORMED, 0>(
            ncoord, nq0, D0, dfptr, inptr, outptr, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

// Fused two-dimensional curl-curl kernel. Both curls are evaluated in the
// same kernel launch, with the input components and the intermediate omega
// staged through shared memory.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurlKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, [[maybe_unused]] TData *NEK_RESTRICT wsp,
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

    TData *s_in               = (TData *)shmemptr;
    TData *s_omega            = s_in + 2u * nqTot;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy both components to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_in[idx]         = inptr[idx];
            s_in[nqTot + idx] = inptr[inoffset + idx];
        }

        localBarrier(threadBlock);

        // omega = dv/dx - du/dy
        Curl2DScalarSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, s_in, s_omega,
            threadBlock);

        // out = {d(omega)/dy, -d(omega)/dx}
        Curl2DVectorSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, D0, D1, f0, f1, dfptr, s_omega, outptr,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

// One curl's plane part on each plane of a 3DH1 block: three components in,
// three out. See the SumFac kernel of the same name for how the curl splits
// and why the 2D kernels already hold both halves of the plane part.
//
// The three input components fit the shared memory the 2D curl-curl already
// asks for: the first two go where it stages its input and the third where
// it would hold omega, which this pass never forms -- each half of the plane
// part writes straight out to global memory.
//
// Each plane holds one plane curl over the same geometry, so they ride the
// second grid dimension of a single launch and the kernel picks its plane
// from the block index.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void Curl3DH1KernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const size_t inoffset, const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *s_in               = (TData *)shmemptr;
    TData *s_inz              = s_in + 2u * nqTot;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * (nelmt * p + e);
        TData *outptr      = out + nqTot * (nelmt * p + e);

        // Copy all three components to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_in[idx]         = inptr[idx];
            s_in[nqTot + idx] = inptr[inoffset + idx];
            s_inz[idx]        = inptr[2u * inoffset + idx];
        }

        localBarrier(threadBlock);

        // The third component, df_y/dx - df_x/dy.
        Curl2DScalarSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, s_in,
            outptr + 2u * outoffset, threadBlock);

        // The first two components, {df_z/dy, -df_z/dx}.
        Curl2DVectorSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, outoffset, D0, D1, f0, f1, dfptr, s_inz, outptr,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

// Fused three-dimensional curl-curl kernel, see the two-dimensional kernel
// above for the rationale.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurlKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    [[maybe_unused]] TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT out,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *s_in               = (TData *)shmemptr;
    TData *s_omega            = s_in + 3u * nqTot;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy all three components to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_in[idx]              = inptr[idx];
            s_in[nqTot + idx]      = inptr[inoffset + idx];
            s_in[2u * nqTot + idx] = inptr[2u * inoffset + idx];
        }

        localBarrier(threadBlock);

        // omega = curl(u)
        Curl3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, nqTot, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr,
            s_in, s_omega, threadBlock);

        // out = curl(omega)
        Curl3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, nqTot, inoffset, D0, D1, D2, f0, f1, f1m, f2, dfptr,
            s_omega, outptr, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
