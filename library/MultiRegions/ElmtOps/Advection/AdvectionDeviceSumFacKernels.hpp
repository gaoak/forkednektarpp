///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceSumFacKernels.hpp
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
// Description: SumFac device kernels of the advection operator, one element
// per warp lane.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionDeviceSumFacKernels.hpp
 * @brief SumFac device kernels of the advection operator: one element per
 * thread, with the elements of a warp interleaved lane by lane.
 *
 * @details
 * These are the device counterparts of the Serial/AVX sum-factorised
 * kernels in AdvectionSerialAVXSumFacKernels.hpp, with a warp lane playing
 * the part of a SIMD lane: lane ilane of warp iwarp owns one element, and
 * the block storage is interleaved at the warp size so the lanes of a warp
 * read consecutive addresses. Unlike the host kernels, which receive the
 * reference-space derivatives already formed by the PhysDerivTensor
 * kernels, each kernel here contracts the one-dimensional derivative
 * matrices itself and then applies, in the same pass, the
 * collapsed-to-standard coordinate correction for simplex-type shapes, the
 * geometric derivative factors and the dot product with the advection
 * velocity, writing (or accumulating) scale * (V . grad u) per quadrature
 * point.
 *
 * The file provides, for each dimensionality: the per-element kernel
 * (Advection*DSumFacKernel) and the device entry point the block operator
 * launches (the AdvectionKernelLauncher overload for that dimension),
 * which unpacks the size parameter, stages any collapsed-coordinate tables
 * in shared memory, walks the block's elements with a grid stride -- one
 * grid row per component or homogeneous mode -- and calls the per-element
 * kernel. AdvectionSharedMemorySize reports how much dynamic shared memory
 * a launch must request. The launchers are constrained to the SumFac
 * implementation tag, so the identically named SumFacTOP launchers in
 * AdvectionDeviceSumFacTOPKernels.hpp are the ones that resolve when that
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

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

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
inline constexpr unsigned int AdvectionSharedMemorySize(
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
inline constexpr unsigned int AdvectionSharedMemorySize(
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
inline constexpr unsigned int AdvectionSharedMemorySize(
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
 * @brief One element of a 1D block, held by one warp lane:
 * out = scale * du/dxi * sum_d df(d) V_d at every quadrature point.
 *
 * The derivative along the single reference direction is formed by
 * contracting the 1D derivative matrix over the element's values, then
 * multiplied by the derivative factor and velocity component of each of
 * the up to three coordinate directions of a possibly embedded segment --
 * the same arithmetic as Advection1DKernel on the host.
 *
 * All arrays are addressed lane by lane: entry i of this element sits at
 * warpsize * i + ilane of a warp's slice. The derivative factors follow
 * suit, with the ncoord factors of a point adjacent when the geometry is
 * deformed and a single set per warp otherwise.
 *
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane        Lane index within the warp, i.e. which element of
 *                       the warp's slice this thread owns.
 * @param   ncoord       Coordinate dimension, i.e. number of derivative
 *                       factors and velocity components.
 * @param   nq0          Quadrature points per element.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of this warp's elements.
 * @param   advVel       Velocity data of this warp's elements; component
 *                       d starts at d * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this warp's elements.
 * @param   out          Output values of this warp's elements.
 * @param   scale        Scalar factor applied to the result.
 */
template <bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Advection1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[warpsize * q + ilane];
        }

        // Multiply by derivative factors.
        TData tmp = 0.0;
        for (unsigned int d = 0u; d < ncoord; d++)
        {
            tmp += d0 * df[d * warpsize + dfindex] *
                   advVel[d * advVelOffset + index];
        }

        if constexpr (APPEND)
        {
            out[index] += scale * tmp;
        }
        else
        {
            out[index] = scale * tmp;
        }
    }
}

/**
 * @brief One element of a 2D block, held by one warp lane:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d at every quadrature
 * point.
 *
 * The two tensor-direction derivatives are formed by contracting D0 over
 * direction 0 and D1 over direction 1. For triangles the tensor directions
 * are the collapsed coordinates, so d0 is first multiplied by
 * f1[j] = 2/(1 - eta_1) and d1 augmented by d0 * f0[i] with
 * f0[i] = (1 + eta_0)/2, recovering the standard-coordinate derivatives
 * before the 2 * ncoord derivative factors are applied. With
 * @p ncoord == 3 the element is embedded in three-dimensional space and a
 * third factor/velocity pair contributes. This mirrors Advection2DKernel
 * on the host.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane        Lane index within the warp.
 * @param   ncoord       Coordinate dimension, i.e. number of
 *                       gradient/velocity components (2 or 3).
 * @param   nq0          Quadrature points in direction 0.
 * @param   nq1          Quadrature points in direction 1.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   f0           Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                       direction-0 points (triangles only).
 * @param   f1           Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                       direction-1 points (triangles only).
 * @param   df           Derivative factors of this warp's elements,
 *                       2 * ncoord per element or per point.
 * @param   advVel       Velocity data of this warp's elements; component
 *                       k starts at k * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this warp's elements.
 * @param   out          Output values of this warp's elements.
 * @param   scale        Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_DEVICE_INLINE static void Advection2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Get advection velocity.
            TData vx, vy;
            vx = advVel[index];
            vy = advVel[index + 1u * advVelOffset];

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from collapsed to standard coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors and Advection Vel.
            TData tmp;
            tmp = vx * (d0 * df[0u * warpsize + dfindex] +
                        d1 * df[1u * warpsize + dfindex]);
            tmp += vy * (d0 * df[2u * warpsize + dfindex] +
                         d1 * df[3u * warpsize + dfindex]);

            if (ncoord == 3u)
            {
                // Multiply by derivative factors and Advection Vel.
                TData vz = advVel[index + 2u * advVelOffset];
                tmp += vz * (d0 * df[4u * warpsize + dfindex] +
                             d1 * df[5u * warpsize + dfindex]);
            }

            if constexpr (APPEND)
            {
                out[index] += scale * tmp;
            }
            else
            {
                out[index] = scale * tmp;
            }
        }
    }
}

/**
 * @brief One element of a 3D block, held by one warp lane:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d at every quadrature
 * point.
 *
 * The three tensor-direction derivatives are formed by contracting D0, D1
 * and D2 over their directions and then, for the shapes with collapsed
 * coordinates, chain-ruled back to standard coordinates with the
 * tabulated factors: hexahedra need no correction, prisms and pyramids
 * fold the corrected direction-0 (and, for pyramids, direction-1)
 * derivative into direction 2, and tetrahedra combine all four tables. All
 * nine derivative factors are then applied and contracted with the
 * velocity. This mirrors Advection3DKernel on the host, whose
 * documentation spells out the corrections and the factor naming.
 *
 * @tparam SHAPE_TYPE Element shape; selects the collapsed-coordinate
 *                    correction.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Read per-point rather than per-element derivative
 *                    factors.
 *
 * @param   ilane        Lane index within the warp.
 * @param   nq0          Quadrature points in direction 0.
 * @param   nq1          Quadrature points in direction 1.
 * @param   nq2          Quadrature points in direction 2.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   D2           1D derivative matrix of direction 2.
 * @param   f0           (1 + eta_0)/2 at the direction-0 points.
 * @param   f1           (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m          2/(1 - eta_1) at the direction-1 points.
 * @param   f2           2/(1 - eta_2) at the direction-2 points.
 * @param   df           Derivative factors of this warp's elements, nine
 *                       per element or per point.
 * @param   advVel       Velocity data of this warp's elements; component
 *                       k starts at k * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this warp's elements.
 * @param   out          Output values of this warp's elements.
 * @param   scale        Scalar factor applied to the result.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_DEVICE_INLINE static void Advection3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale)
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

                // Get advection velocity.
                TData vx, vy, vz;
                vx = advVel[index];
                vy = advVel[index + 1u * advVelOffset];
                vz = advVel[index + 2u * advVelOffset];

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[q * nq2 + k] *
                          in[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                // Moving from collapsed to standard coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                              SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    d0 *= f2[k];
                    d2 += f0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= f2[k];
                    d1 *= f2[k];
                    d2 += f0[i] * d0 + f1[j] * d1;
                }

                // Multiply by derivative factors and Advection Velocity.
                TData tmp = 0.0;
                tmp       = vx * (d0 * df[0u * warpsize + dfindex] +
                            d1 * df[1u * warpsize + dfindex] +
                            d2 * df[2u * warpsize + dfindex]);
                tmp += vy * (d0 * df[3u * warpsize + dfindex] +
                             d1 * df[4u * warpsize + dfindex] +
                             d2 * df[5u * warpsize + dfindex]);
                tmp += vz * (d0 * df[6u * warpsize + dfindex] +
                             d1 * df[7u * warpsize + dfindex] +
                             d2 * df[8u * warpsize + dfindex]);

                if constexpr (APPEND)
                {
                    out[index] += scale * tmp;
                }
                else
                {
                    out[index] = scale * tmp;
                }
            }
        }
    }
}

/**
 * @brief Device entry point of the 1D SumFac advection kernel.
 *
 * Launched by AdvectionBlockOpImpl::OperatorNDImpl through
 * DEVICE_2DGRID_KERNEL_LAUNCHER, which appends the shared-memory pointer
 * (unused here) and the thread-block handle. Unpacks the size parameter
 * -- runtime or compile-time, both satisfying IsPhysSizeParameter1D_v as
 * the static_assert enforces. Each thread then starts at its global index
 * in the grid's first dimension and strides by that dimension's range, so
 * the block's @p nelmt elements (padding included) are covered whatever
 * grid size was launched. The grid's second dimension gives the index c
 * of the component -- or homogeneous mode -- being advected, read once and
 * fixed for the whole stride loop, since that dimension holds exactly one
 * block row for each. The element index splits into the warp it belongs
 * to and the lane within it: the warp's slice of the input and output
 * arrays is selected by pointer arithmetic that also offsets by c, since
 * @p in and @p out address every component, the velocity slice by the
 * warp and the plane c % @p nhomo, and the derivative-factor slice by the
 * warp alone -- the same velocity and factors serve every component. The
 * lane is then passed to Advection1DSumFacKernel, which computes the
 * element's output. The enable_if pins this overload to the SumFac tag,
 * leaving the SumFacTOP tag to the identically named launcher in
 * AdvectionDeviceSumFacTOPKernels.hpp.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam1D  Element sizes: coordinate dimension and
 *                       quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block; unused
 *                       in one dimension.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const unsigned int nhomo, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale,
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

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nq0 * (nelmt * c + warpsize * iwarp);
        const TData *advVelPtr =
            advVel + nq0 * (nelmt * (c % nhomo) + warpsize * iwarp);
        Advection1DSumFacKernel<APPEND, DEFORMED>(ilane, ncoord, nq0, D0, dfptr,
                                                  advVelPtr, advVelOffset,
                                                  inptr, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFac advection kernel.
 *
 * As in 1D, but the shared memory is used: FETCH_SHARED_MEMORY() binds
 * @p shmemptr to the block's dynamic shared memory (a no-op on the
 * back-ends that pass it in). For triangles the collapsed-coordinate
 * factor tables are then copied from global into the thread block's
 * shared memory once, before the element loop, and a barrier makes them
 * visible to the whole block; the per-element kernel then reads the shared
 * copies. For quadrilaterals no shared memory is requested and the two
 * table pointers stay null. The remaining unpacking and grid-stride loop
 * are as in 1D, calling Advection2DSumFacKernel once per element.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam2D  Element sizes: coordinate dimension and
 *                       per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   f0           Collapsed-coordinate factor of direction 0.
 * @param   f1           Collapsed-coordinate factor of direction 1.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least AdvectionSharedMemorySize elements of TData.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT advVel,
    const size_t advVelOffset, const unsigned int nhomo,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale,
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
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        const TData *advVelPtr =
            advVel + nqTot * (nelmt * (c % nhomo) + warpsize * iwarp);
        Advection2DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED>(
            ilane, ncoord, nq0, nq1, D0, D1, s_f0, s_f1, dfptr, advVelPtr,
            advVelOffset, inptr, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFac advection kernel.
 *
 * Counterpart of the 2D overload for three-dimensional shapes; the size
 * parameter carries no coordinate dimension, since a 3D element always
 * has three. The shape's collapsed-coordinate tables -- four for
 * tetrahedra, two for prisms, three for pyramids, none for hexahedra --
 * are staged in shared memory before the element loop, laid out back to
 * back in the order the shape needs them, and a barrier publishes them to
 * the block before Advection3DSumFacKernel is called once per element.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam3D  Element sizes: per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   D2           1D derivative matrix of direction 2.
 * @param   f0           (1 + eta_0)/2 at the direction-0 points.
 * @param   f1           (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m          2/(1 - eta_1) at the direction-1 points.
 * @param   f2           2/(1 - eta_2) at the direction-2 points.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable; unused in 3D.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least AdvectionSharedMemorySize elements of TData.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    [[maybe_unused]] const unsigned int nhomo, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale, unsigned char *shmemptr,
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
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane     = e % warpsize;
        const size_t iwarp     = e / warpsize;
        const TData *dfptr     = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr     = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr          = out + nqTot * (nelmt * c + warpsize * iwarp);
        const TData *advVelPtr = advVel + nqTot * warpsize * iwarp;
        Advection3DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            advVelPtr, advVelOffset, inptr, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
