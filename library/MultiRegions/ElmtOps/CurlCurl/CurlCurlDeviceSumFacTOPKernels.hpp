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

/**
 * @file CurlCurlDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the curl-curl operator: one element
 * per thread block, threads indexed over the element's quadrature points.
 *
 * @details
 * These kernels compute the same thing as the SumFac ones in
 * CurlCurlDeviceSumFacKernels.hpp -- the tensor-direction derivatives of
 * every component, the collapsed-to-standard coordinate correction, the
 * geometric derivative factors and the curl, twice over for the double
 * curl -- but distribute the work differently. A whole thread block takes
 * one element at a time and its threads stride over the element's
 * quadrature points, each thread computing the output at the points it
 * owns. The element's input components are staged in shared memory first,
 * so that the tensor contractions, which read across the element, hit
 * shared rather than global memory, and the vorticity omega formed by the
 * first curl is written to shared memory too, where the second curl reads
 * it; the derivative factors stay in global memory and are read at the
 * thread's own point index, the collapsed-coordinate tables at the
 * direction indices that point resolves into. Accordingly the data is not
 * interleaved (one element's values are contiguous) and the derivative
 * factors are taken in the transposed layout, with each factor component
 * contiguous over the element's points.
 *
 * The file provides, for each dimensionality: the per-element curl kernels
 * (Curl2DScalarSumFacTOPKernel and Curl2DVectorSumFacTOPKernel in 2D,
 * Curl3DSumFacTOPKernel for both curls in 3D) and the device entry point
 * the block operator launches (the CurlCurlKernelLauncher overload for
 * that dimension), which unpacks the size parameter and walks the block's
 * elements one thread block at a time, staging each element's input
 * components in shared memory before calling the curl kernels.
 * Curl3DH1KernelLauncher, launched with the planes of a multi-plane 3DH1
 * block on the second grid dimension, stages an element's three
 * components the same way and calls each 2D curl kernel once, for the
 * plane part of a single curl. CurlCurlSharedMemorySize reports how much
 * dynamic shared memory such a launch must request; CurlCurlWorkSpaceSize
 * reports that no global workspace is needed. The launchers and helpers
 * are constrained to the SumFacTOP implementation tag and are otherwise
 * identical in name and signature to the SumFac ones, which is how
 * CurlCurlBlockOpImpl::OperatorNDImpl and
 * CurlCurlBlockOpImpl::OperatorCurl3DH1 can launch either strategy from
 * one call site.
 *
 * Everything in the namespace is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY, so it exists only in translation units compiled for
 * a device back-end.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a one-dimensional shape: none, since the 1D
 * launcher reads the element's input straight from global memory.
 *
 * In two and three dimensions the two curls of the operator are fused into
 * a single kernel, so the shared memory holds the element's input
 * components as well as the intermediate omega; see the 2D and 3D
 * overloads.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter1D  1D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Element sizes; unused.
 *
 * @return Zero.
 */
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

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a two-dimensional shape.
 *
 * Three element-sized buffers, whatever the shape: the two input
 * components the launcher stages and the out-of-plane vorticity the first
 * curl kernel writes; Curl3DH1KernelLauncher stages its three input
 * components in the same three buffers. The collapsed-coordinate factor
 * tables are not staged.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return 3 * nq0 * nq1.
 */
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

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a three-dimensional shape.
 *
 * Six element-sized buffers, whatever the shape: the three input
 * components the launcher stages and the three vorticity components the
 * first curl kernel writes.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return 6 * nq0 * nq1 * nq2.
 */
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

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a one-dimensional shape: none.
 *
 * The intermediate omega is held in shared memory, so no overload of this
 * helper requests global workspace for the SumFacTOP strategy.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter1D  1D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block; unused.
 * @param   sizeParam1D  Element sizes; unused.
 *
 * @return Zero.
 */
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

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a two-dimensional shape: none, the vorticity being held in shared
 * memory.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block; unused.
 * @param   sizeParam2D  Element sizes; unused.
 *
 * @return Zero.
 */
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

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a three-dimensional shape: none, the vorticity being held in shared
 * memory.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block; unused.
 * @param   sizeParam3D  Element sizes; unused.
 *
 * @return Zero.
 */
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

/**
 * @brief One element of a 2D block, shared by a thread block: the scalar
 * curl omega = d(u1)/dx0 - d(u0)/dx1 at the quadrature points the calling
 * thread owns, from the element's values of both components.
 *
 * The threads stride over the element's points; for each of them the
 * derivatives of both components along the two tensor directions are
 * formed by contracting @p D0 and @p D1 over the element's values, in
 * registers and at once, so that the collapsed-coordinate correction (for
 * triangles, as in Curl2DScalarSumFacKernel), the chain rule and the curl
 * are applied in a single sweep. Everything is addressed by the plain
 * point index, since the element's data is contiguous; the derivative
 * factors are read in the transposed layout, factor component m of point
 * idx at m * nqTot + idx for deformed geometry and at m for regular
 * geometry. The kernel ends with a block barrier, so that the vorticity is
 * complete before the second curl kernel reads across it.
 *
 * @tparam SHAPE_TYPE   Element shape; selects the collapsed-coordinate
 *                      correction.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   ncoord      Coordinate dimension; unused.
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   inoffset    Stride between the two components of @p in.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df          Derivative factors of the element, transposed
 *                      layout.
 * @param   in          Input values of the element, in shared memory when
 *                      called from the launcher; component 1 starts at
 *                      @p inoffset.
 * @param   omega       Output values of the element.
 * @param   threadBlock Thread-block handle.
 */
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

        // Moving from collapsed to standard coordinates.
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

/**
 * @brief One element of a 2D block, shared by a thread block: the vector
 * curl out = ( d(omega)/dx1, -d(omega)/dx0 ) at the quadrature points the
 * calling thread owns, from the element's values of the vorticity.
 *
 * The tensor-direction derivatives of omega are formed as in
 * Curl2DScalarSumFacTOPKernel, corrected and chain-ruled the same way, and
 * both output components are written by the same thread. The kernel ends
 * with a block barrier, so that the launcher's copy of the next element's
 * input cannot start while a thread is still reading the shared vorticity.
 *
 * @tparam SHAPE_TYPE   Element shape; selects the collapsed-coordinate
 *                      correction.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   ncoord      Coordinate dimension; unused.
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   outoffset   Stride between the two components of @p out.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df          Derivative factors of the element, transposed
 *                      layout.
 * @param   omega       Vorticity values of the element, in shared memory
 *                      when called from the launcher.
 * @param   out         Output values of the element; component 1 starts
 *                      at @p outoffset.
 * @param   threadBlock Thread-block handle.
 */
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

        // Moving from collapsed to standard coordinates.
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

/**
 * @brief One element of a 3D block, shared by a thread block: the curl of
 * a three-component field at the quadrature points the calling thread
 * owns; used for both passes of the curl-curl operator, omega = curl u and
 * out = curl omega.
 *
 * The threads stride over the element's points; for each of them the
 * derivatives of the three components along the three tensor directions
 * are formed by contracting @p D0, @p D1 and @p D2 over the element's
 * values, in registers and at once, then chain-ruled back to the standard
 * coordinates for the simplex-type shapes exactly as in Curl3DSumFacKernel
 * and mapped into Cartesian derivatives by the derivative-factor entry
 * k * 3 + d, read in the transposed layout as in
 * Curl2DScalarSumFacTOPKernel. Only the six off-diagonal Cartesian
 * derivatives are formed, the diagonal ones cancelling out of the curl:
 *     out0 = d(u2)/dx1 - d(u1)/dx2,
 *     out1 = d(u0)/dx2 - d(u2)/dx0,
 *     out2 = d(u1)/dx0 - d(u0)/dx1.
 * The kernel ends with a block barrier, so that its output is complete
 * before the next kernel or the launcher's next copy reads or overwrites
 * shared memory.
 *
 * @tparam SHAPE_TYPE   Element shape; selects the collapsed-coordinate
 *                      correction.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   nq2         Quadrature points in direction 2.
 * @param   inoffset    Stride between the components of @p in.
 * @param   outoffset   Stride between the components of @p out.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   D2          1D derivative matrix of direction 2.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (prisms, pyramids and
 *                      tetrahedra).
 * @param   f1          Collapsed-coordinate factor (1 + eta_1)/2 at the
 *                      direction-1 points (pyramids and tetrahedra).
 * @param   f1m         Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (tetrahedra only).
 * @param   f2          Collapsed-coordinate factor 2/(1 - eta_2) at the
 *                      direction-2 points (prisms, pyramids and
 *                      tetrahedra).
 * @param   df          Derivative factors of the element, transposed
 *                      layout.
 * @param   in          Input values of the element, three components at
 *                      @p inoffset from each other.
 * @param   out         Output values of the element, three components at
 *                      @p outoffset from each other.
 * @param   threadBlock Thread-block handle.
 */
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

        // Moving from collapsed to standard coordinates.
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

/**
 * @brief Device entry point of the 1D SumFacTOP kernel: the curl-curl is
 * undefined on a segment, so each element's single component is
 * differentiated once along its direction.
 *
 * Exists so that the generated segment ShapeBlock() compiles; the block
 * operator's constructor asserts that the block is two- or
 * three-dimensional (see CurlCurlDeviceSumFac.hpp). Launched by
 * CurlCurlBlockOpImpl::OperatorNDImpl through
 * DEVICE_1DGRID_KERNEL_LAUNCHER, which appends the shared-memory pointer
 * (unused here) and the thread-block handle. Unpacks the size parameter --
 * runtime or compile-time, both satisfying IsPhysSizeParameter1D_v as the
 * static_assert enforces. The element index comes from the thread-block
 * index and advances by the number of thread blocks in the grid, so the
 * whole block cooperates on one element at a time and the block's
 * @p nelmt elements (padding included) are covered whatever grid size was
 * launched; the input, output and derivative-factor pointers are offset by
 * whole elements, the data being contiguous rather than warp interleaved,
 * and PhysDerivDir1DSumFacTOPKernel writes the element's derivative with
 * the direction-0 derivative factor. The enable_if pins this overload to
 * the SumFacTOP tag, leaving the SumFac tag to the identically named
 * launcher in CurlCurlDeviceSumFacKernels.hpp.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam1D  Element sizes: coordinate dimension and
 *                       quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   inoffset     Component stride; unused in one dimension.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of the block, transposed
 *                       layout.
 * @param   in           Input values of the block.
 * @param   wsp          Global workspace; unused.
 * @param   out          Output values of the block.
 * @param   shmemptr     Dynamic shared memory of the thread block; unused
 *                       in one dimension.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
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

/**
 * @brief Device entry point of the 2D SumFacTOP kernel: both curls of the
 * operator for every element of the block in one launch.
 *
 * As in 1D, but the shared memory is used: FETCH_SHARED_MEMORY() binds
 * @p shmemptr to the block's dynamic shared memory (a no-op on the
 * back-ends that pass it in), laid out as the element's two input
 * components followed by its vorticity. Each element's input components
 * are copied from global into that shared memory -- the threads share the
 * copy work with the same stride they later compute with -- and a barrier
 * publishes the copy before Curl2DScalarSumFacTOPKernel reads across it to
 * write the vorticity into the third buffer; Curl2DVectorSumFacTOPKernel
 * then reads the vorticity and writes the output straight to global
 * memory. Both kernels end with a barrier of their own, so the vorticity
 * is complete before it is read and the copy for the next element cannot
 * start while a thread is still reading this one.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam2D  Element sizes: coordinate dimension and
 *                       per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   inoffset     Stride between the components of @p in, also used
 *                       between those of @p out.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   f0           Collapsed-coordinate factor of direction 0.
 * @param   f1           Collapsed-coordinate factor of direction 1.
 * @param   df           Derivative factors of the block, transposed
 *                       layout.
 * @param   in           Input values of the block, two components.
 * @param   wsp          Global workspace; unused, the vorticity being
 *                       held in shared memory.
 * @param   out          Output values of the block, two components.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least CurlCurlSharedMemorySize elements of TData:
 *                       three element-sized buffers.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
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

/**
 * @brief Device entry point of the 3D SumFacTOP kernel: both curls of the
 * operator for every element of the block in one launch.
 *
 * As in 2D, with the shared memory laid out as the element's three input
 * components followed by its three vorticity components, and
 * Curl3DSumFacTOPKernel called twice per element: first from the staged
 * input into the vorticity buffers, with the element size nqTot as both
 * component strides, then from the vorticity buffers into the output,
 * reading the vorticity at nqTot and writing the output at @p inoffset.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam3D  Element sizes: per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   inoffset     Stride between the components of @p in, also used
 *                       between those of @p out.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   D2           1D derivative matrix of direction 2.
 * @param   f0           Collapsed-coordinate factor (1 + eta_0)/2 of
 *                       direction 0.
 * @param   f1           Collapsed-coordinate factor (1 + eta_1)/2 of
 *                       direction 1.
 * @param   f1m          Collapsed-coordinate factor 2/(1 - eta_1) of
 *                       direction 1.
 * @param   f2           Collapsed-coordinate factor 2/(1 - eta_2) of
 *                       direction 2.
 * @param   df           Derivative factors of the block, transposed
 *                       layout.
 * @param   in           Input values of the block, three components.
 * @param   wsp          Global workspace; unused, the vorticity being
 *                       held in shared memory.
 * @param   out          Output values of the block, three components.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least CurlCurlSharedMemorySize elements of TData:
 *                       six element-sized buffers.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
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

} // namespace Nektar::MultiRegions::detail
