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

/**
 * @file CurlCurlDeviceSumFacKernels.hpp
 * @brief SumFac device kernels of the curl-curl operator: one element per
 * thread, with the elements of a warp interleaved lane by lane.
 *
 * @details
 * These are the device counterparts of the Serial/AVX sum-factorised
 * kernels in CurlCurlSerialAVXSumFacKernels.hpp, with a warp lane playing
 * the part of a SIMD lane: lane ilane of warp iwarp owns one element, and
 * the block storage is interleaved at the warp size so the lanes of a warp
 * read consecutive addresses. Unlike the host kernels, which receive the
 * reference-space derivatives already formed by the PhysDerivTensor
 * kernels, each curl kernel here contracts the one-dimensional derivative
 * matrices itself over all components of its element and then applies, in
 * the same pass, the collapsed-to-standard coordinate correction for
 * simplex-type shapes, the geometric derivative factors and the curl.
 *
 * The file provides, for each dimensionality: the per-element curl kernels
 * (Curl2DScalarSumFacKernel and Curl2DVectorSumFacKernel in 2D,
 * Curl3DSumFacKernel for both curls in 3D) and the device entry point the
 * block operator launches (the CurlCurlKernelLauncher overload for that
 * dimension), which unpacks the size parameter, stages any
 * collapsed-coordinate tables in shared memory, walks the block's elements
 * with a grid stride and calls the curl kernels twice per element -- omega
 * = curl u into the global workspace, then out = curl omega -- so that the
 * whole double curl is one fused launch. Curl3DH1KernelLauncher, launched
 * with the planes of a multi-plane 3DH1 block on the second grid
 * dimension, instead calls each 2D curl kernel once per element, for the
 * plane part of a single curl, and needs no workspace.
 * CurlCurlSharedMemorySize and CurlCurlWorkSpaceSize report how much
 * dynamic shared memory and global workspace such a launch must request.
 * The launchers and helpers are constrained to the SumFac implementation
 * tag, so the identically named SumFacTOP ones in
 * CurlCurlDeviceSumFacTOPKernels.hpp are the ones that resolve when that
 * tag is used instead.
 *
 * Everything in the namespace is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY, so it exists only in translation units compiled for
 * a device back-end.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Dynamic shared memory a SumFac launch needs per thread block, in
 * elements of TData, for a one-dimensional shape: none, since a segment
 * has no collapsed-coordinate tables to stage.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
 * @tparam TPhysSizeParameter1D  1D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Element sizes; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int CurlCurlSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory a SumFac launch needs per thread block, in
 * elements of TData, for a two-dimensional shape.
 *
 * The only shared data are the collapsed-coordinate factor tables the
 * launcher stages before the element loop, so quadrilaterals need none and
 * triangles need one entry per quadrature point of each direction.
 *
 * @tparam SHAPE_TYPE            Element shape.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
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

/**
 * @brief Dynamic shared memory a SumFac launch needs per thread block, in
 * elements of TData, for a three-dimensional shape.
 *
 * As in two dimensions the shared data are the collapsed-coordinate factor
 * tables: none for hexahedra, one table per direction for pyramids, two
 * (directions 0 and 2) for prisms, and four for tetrahedra, which stage
 * two tables over direction 1.
 *
 * @tparam SHAPE_TYPE            Element shape.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
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

/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a one-dimensional shape: none, since the segment launcher forms no
 * vorticity.
 *
 * Both curls of the operator run inside a single fused kernel, so the only
 * global workspace the 2D and 3D launches need is the one holding the
 * intermediate omega = curl u between them.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
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

/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a two-dimensional shape: one plane of the block's quadrature-point
 * values, holding the out-of-plane vorticity.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, including padding.
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return nq0 * nq1 * nelmt.
 */
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

/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a three-dimensional shape: three planes of the block's quadrature-point
 * values, one per vorticity component.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFac.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, including padding.
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return 3 * nq0 * nq1 * nq2 * nelmt.
 */
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

/**
 * @brief One element of a 2D block, held by one warp lane: the scalar curl
 * omega = d(u1)/dx0 - d(u0)/dx1 at every quadrature point, from the
 * element's values of both components.
 *
 * For each quadrature point the derivatives of both components along the
 * two tensor directions are formed by contracting @p D0 and @p D1 over the
 * element's values, in registers and at once, so that the
 * collapsed-coordinate correction, the chain rule and the curl are applied
 * in a single sweep. For triangles the direction-0 derivatives are
 * multiplied by f1[j] = 2/(1 - eta_1) and the direction-1 derivatives
 * augmented by them times f0[i] = (1 + eta_0)/2, recovering the
 * standard-coordinate derivatives; the derivative-factor entry k * 2 + d
 * then maps the direction-d derivative into the Cartesian-k gradient
 * component and the two off-diagonal Cartesian derivatives are combined
 * into omega.
 *
 * All arrays are addressed lane by lane: entry i of this element sits at
 * warpsize * i + ilane of a warp's slice. The derivative factors follow
 * suit, with the 2 * ncoord factors of a point adjacent when the geometry
 * is deformed and a single set per element otherwise.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane       Lane index within the warp, i.e. which element of
 *                      the warp's slice this thread owns.
 * @param   ncoord      Coordinate dimension; sets the number 2 * ncoord
 *                      of derivative factors per point.
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   inoffset    Stride between the two components of @p in.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df          Derivative factors of this warp's elements.
 * @param   in          Input values of this warp's elements; component 1
 *                      starts at @p inoffset.
 * @param   omega       Output values of this warp's elements.
 */
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
            const TData dvdx = d0v * df[0u * warpsize + dfindex] +
                               d1v * df[1u * warpsize + dfindex];
            const TData dudy = d0u * df[2u * warpsize + dfindex] +
                               d1u * df[3u * warpsize + dfindex];

            omega[index] = dvdx - dudy;
        }
    }
}

/**
 * @brief One element of a 2D block, held by one warp lane: the vector curl
 * out = ( d(omega)/dx1, -d(omega)/dx0 ) at every quadrature point, from
 * the element's values of the vorticity.
 *
 * The tensor-direction derivatives of omega are formed as in
 * Curl2DScalarSumFacKernel, corrected and chain-ruled the same way, and
 * both output components are written by the same sweep.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane       Lane index within the warp, i.e. which element of
 *                      the warp's slice this thread owns.
 * @param   ncoord      Coordinate dimension; sets the number 2 * ncoord
 *                      of derivative factors per point.
 * @param   nq0         Quadrature points in direction 0.
 * @param   nq1         Quadrature points in direction 1.
 * @param   outoffset   Stride between the two components of @p out.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                      direction-0 points (triangles only).
 * @param   f1          Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                      direction-1 points (triangles only).
 * @param   df          Derivative factors of this warp's elements.
 * @param   omega       Vorticity values of this warp's elements.
 * @param   out         Output values of this warp's elements; component 1
 *                      starts at @p outoffset.
 */
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

            // Moving from collapsed to standard coordinates.
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

/**
 * @brief One element of a 3D block, held by one warp lane: the curl of a
 * three-component field at every quadrature point; used for both passes
 * of the curl-curl operator, omega = curl u and out = curl omega.
 *
 * For each quadrature point the derivatives of the three components along
 * the three tensor directions are formed by contracting @p D0, @p D1 and
 * @p D2 over the element's values, in registers and at once. For the
 * simplex-type shapes each component's derivatives are then chain-ruled
 * back to the standard coordinates -- a prism scales direction 0 by f2[k]
 * and adds f0[i] times it to direction 2; a pyramid scales directions 0
 * and 1 by f2[k] and adds f0[i] and f1[j] times them to direction 2; a
 * tetrahedron scales direction 0 by f1m[j] * f2[k], rebuilds direction 1
 * from f0[i] times that and f2[k] times the old direction-1 derivative,
 * and adds both contributions, the second weighted by f1[j], to direction
 * 2; hexahedra need no correction -- before the derivative-factor entry
 * k * 3 + d maps the direction-d derivative into the Cartesian-k gradient
 * component. Only the six off-diagonal Cartesian derivatives are formed,
 * the diagonal ones cancelling out of the curl:
 *     out0 = d(u2)/dx1 - d(u1)/dx2,
 *     out1 = d(u0)/dx2 - d(u2)/dx0,
 *     out2 = d(u1)/dx0 - d(u0)/dx1.
 *
 * All arrays are addressed lane by lane as in Curl2DScalarSumFacKernel,
 * with nine derivative factors per point (deformed) or per element (regular).
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane       Lane index within the warp, i.e. which element of
 *                      the warp's slice this thread owns.
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
 * @param   df          Derivative factors of this warp's elements.
 * @param   in          Input values of this warp's elements, three
 *                      components at @p inoffset from each other.
 * @param   out         Output values of this warp's elements, three
 *                      components at @p outoffset from each other.
 */
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

/**
 * @brief Device entry point of the 1D SumFac kernel: the curl-curl is
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
 * static_assert enforces. Each thread then starts at its global index and
 * strides by the grid's range, so the block's @p nelmt elements (padding
 * included) are covered whatever grid size was launched. The element index
 * splits into the warp it belongs to and the lane within it: the warp's
 * slice of the input, output and derivative-factor arrays is selected by
 * pointer arithmetic and the lane is passed to PhysDerivDir1DSumFacKernel,
 * which writes the element's derivative with the direction-0 derivative
 * factor. The enable_if pins this overload to the SumFac tag, leaving the
 * SumFacTOP tag to the identically named launcher in
 * CurlCurlDeviceSumFacTOPKernels.hpp.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam1D  Element sizes: coordinate dimension and
 *                       quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   inoffset     Component stride; unused in one dimension.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of the block.
 * @param   in           Input values of the block.
 * @param   wsp          Global workspace; unused in one dimension.
 * @param   out          Output values of the block.
 * @param   shmemptr     Dynamic shared memory of the thread block; unused
 *                       in one dimension.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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

/**
 * @brief Device entry point of the 2D SumFac kernel: both curls of the
 * operator for every element of the block in one launch.
 *
 * As in 1D, but the shared memory is used: FETCH_SHARED_MEMORY() binds
 * @p shmemptr to the block's dynamic shared memory (a no-op on the
 * back-ends that pass it in). For triangles the collapsed-coordinate
 * factor tables are then copied from global into the thread block's shared
 * memory once, before the element loop, and a barrier makes them visible
 * to the whole block; the curl kernels then read the shared copies. For
 * quadrilaterals no shared memory is requested and the two table pointers
 * stay null. The grid-stride loop then selects, per element, the warp's
 * slices of the input, workspace, output and derivative-factor arrays and
 * calls Curl2DScalarSumFacKernel to write the element's vorticity into the
 * workspace and Curl2DVectorSumFacKernel to write the output from it. Each
 * lane owns a complete element, so the vorticity it writes is read back by
 * the same lane and no synchronisation is needed between the two curls.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
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
 * @param   df           Derivative factors of the block.
 * @param   in           Input values of the block, two components.
 * @param   wsp          Global workspace of CurlCurlWorkSpaceSize
 *                       elements, receiving the vorticity.
 * @param   out          Output values of the block, two components.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least CurlCurlSharedMemorySize elements of TData.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurlKernelLauncher(
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

// One curl's plane part on each plane of a 3DH1 block: three components in,
// three out.
//
// The curl of a 3DH1 field splits into a part that differentiates in the
// plane, (df_z/dy, -df_z/dx, df_y/dx - df_x/dy), and a part that
// differentiates along z. This writes the first; the operator's z-op adds the
// second, and CurlCurlOpImpl takes the pair twice, the output carrying omega
// in between. Both halves of
// the plane part are already in the 2D kernels: the scalar one is the third
// component, taken from the first two, and the vector one is the first two
// components, taken from the third. Neither writes an intermediate any other
// lane reads, so this pass needs no workspace at all.
//
// Each plane holds one plane curl over the same geometry, so they ride the
// second grid dimension of a single launch and the kernel picks its plane
// from the block index.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * p + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * p + warpsize * iwarp);

        // The third component, df_y/dx - df_x/dy.
        Curl2DScalarSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, D0, D1, s_f0, s_f1, dfptr, inptr,
            outptr + 2u * outoffset);

        // The first two components, {df_z/dy, -df_z/dx}.
        Curl2DVectorSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, outoffset, D0, D1, s_f0, s_f1, dfptr,
            inptr + 2u * inoffset, outptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFac kernel: both curls of the
 * operator for every element of the block in one launch.
 *
 * As in 2D, with the collapsed-coordinate tables staged per shape -- four
 * for tetrahedra, two (directions 0 and 2) for prisms, three for pyramids,
 * none for hexahedra -- and Curl3DSumFacKernel called twice per element:
 * once from the input into the three vorticity planes of the workspace,
 * once from those planes into the output, the component stride
 * @p inoffset serving both.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam3D  Element sizes: per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   inoffset     Stride between the components of @p in, of the
 *                       workspace and of @p out.
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
 * @param   df           Derivative factors of the block.
 * @param   in           Input values of the block, three components.
 * @param   wsp          Global workspace of CurlCurlWorkSpaceSize
 *                       elements, receiving the vorticity.
 * @param   out          Output values of the block, three components.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least CurlCurlSharedMemorySize elements of TData.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void CurlCurlKernelLauncher(
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

} // namespace Nektar::MultiRegions::detail
