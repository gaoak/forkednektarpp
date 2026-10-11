///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFacKernels.hpp
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
// Description: Device SumFac kernels of the physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the physical derivative: one element
 * per warp lane.
 *
 * @details
 * Every kernel here is constrained to the MultiRegions::SumFac strategy
 * and comes in two layers:
 * - the per-element kernels (`PhysDeriv{,Dir}{1,2,3}DSumFacKernel`
 *   taking an @p ilane), which do the whole element's work in one lane:
 *   for each of the element's quadrature points they take the
 *   tensorial derivatives with the 1D differentiation matrices, apply
 *   the shape's collapsed-coordinate correction and contract the
 *   result with the derivative factors. All arrays are warp
 *   interleaved, entry @em n of an element being at
 *   `warpsize * n + ilane`;
 * - the launch entry points, which check the size-parameter type at
 *   compile time, unpack it, stage any collapsed-coordinate tables the
 *   shape needs in shared memory, then grid-stride over the block's
 *   elements: split the global index into lane and warp, offset the
 *   data pointers to that warp and call the per-element kernel. The
 *   three PhysDerivKernelLauncher overloads (one per dimension,
 *   selected by the size-parameter type) are what the block operator
 *   launches through DEVICE_3DGRID_KERNEL_LAUNCHER, whose grid Y and Z
 *   axes carry the homogeneous-mode and component counts so a single
 *   launch covers every component and mode of the block, the one- and
 *   two-dimensional overloads reserving a third output plane per
 *   component once the mode count exceeds one (see the 1D overload).
 *   The per-direction `PhysDerivDir{1,2,3}DKernelLauncher` are not
 *   instantiated anywhere (see their warnings).
 *
 * The derivative factors are untransposed and warp interleaved: the
 * `ndf = coordDim * dimension` factors of one point are adjacent
 * (stride warpsize), and consecutive points follow, so a deformed
 * element addresses factor @em c of point @em n at
 * `warpsize * (ndf * n + c) + ilane` and a regular one simply at
 * `warpsize * c + ilane`. That is what the DEFORMED template flag
 * selects; nothing else about the two instantiations differs.
 *
 * The per-direction ("Dir") kernels produce a single Cartesian
 * component, optionally accumulating; PhysDeriv itself never calls
 * them, the Divergence device SumFac kernels (and, in one dimension,
 * the CurlCurl ones) do. The all-component per-element kernels are in
 * turn shared with the AdvectionDealias, Helmholtz, Laplacian and
 * LinAdvDiffReaction device SumFac kernels.
 *
 * The SumDerivTensor kernels in this file are the transposes of the
 * tensorial derivative and belong to the weak-form families
 * (IProductWRTDerivBase, Laplacian, Helmholtz, LinAdvDiffReaction)
 * rather than to PhysDeriv; they are collected here alongside the
 * forward derivative kernels they mirror.
 *
 * The whole file is compiled only for the device back-ends
 * (NEKTAR_ENABLE_DEVICE together with DEVICE_COMPILE_ONLY).
 *
 * @see PhysDerivDeviceSumFacTOPKernels.hpp for the SumFacTOP
 * counterparts, which spread one element over a whole thread block.
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
 * @brief Dynamic shared memory, in elements of the data type, a 1D
 * SumFac launch needs: none, since a segment has no
 * collapsed-coordinate tables to stage.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Constrained to MultiRegions::SumFac.
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
inline constexpr unsigned int PhysDerivSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory, in elements of the data type, a 2D
 * SumFac launch needs: room for the collapsed-coordinate tables of a
 * triangle, none for a quadrilateral.
 *
 * @tparam SHAPE_TYPE            Quad, Tri or NodalTri.
 * @tparam Implementation        Constrained to MultiRegions::SumFac.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Table entries to allocate per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
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
 * @brief Dynamic shared memory, in elements of the data type, a 3D
 * SumFac launch needs: room for the collapsed-coordinate tables the
 * shape uses, none for a hexahedron.
 *
 * A (nodal) tetrahedron stages four tables (`nq0 + 2 * nq1 + nq2`
 * entries, direction 1 contributing both \f$\tfrac{1}{2}(1+\eta_1)\f$
 * and \f$2/(1-\eta_1)\f$), a (nodal) prism two and a pyramid three.
 *
 * @tparam SHAPE_TYPE            Hex, Tet, NodalTet, Prism, NodalPrism
 *                               or Pyr.
 * @tparam Implementation        Constrained to MultiRegions::SumFac.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Table entries to allocate per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
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
 * @brief One element, one dimension: differentiate a segment in the
 * calling lane and scale by the derivative factors.
 *
 * With one standard direction there is nothing to contract and no
 * collapsed-coordinate correction:
 * `out[k][i] = df[k] * sum_q D0[q * nq0 + i] * in[q]`.
 *
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   ncoord      Cartesian components to produce (coordDim);
 *                      also the number of derivative factors per point.
 * @param   nq0         Quadrature points of the element.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0          1D differentiation matrix, column-major:
 *                      `D0[q * nq0 + i]` differentiates at point i.
 * @param   df          Derivative factors of this lane's warp.
 * @param   in          Input values, warp interleaved.
 * @param   out         Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
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
        for (unsigned int k = 0u; k < ncoord; k++)
        {
            out[k * outoffset + index] = d0 * df[k * warpsize + dfindex];
        }
    }
}

/**
 * @brief One element, one dimension, single component: as
 * PhysDeriv1DSumFacKernel but writing only Cartesian component DIR.
 *
 * Template parameter order is `<APPEND, DEFORMED, DIR, TData>`. This is
 * the entry point the Divergence and CurlCurl device SumFac kernels
 * call into.
 *
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
 * @param   ncoord  Derivative factors per point; sets the per-point
 *                  factor stride when @p DEFORMED.
 * @param   nq0     Quadrature points of the element.
 * @param   D0      1D differentiation matrix, column-major.
 * @param   df      Derivative factors of this lane's warp.
 * @param   in      Input values, warp interleaved.
 * @param   out     Cartesian component DIR, warp interleaved.
 */
template <bool APPEND, bool DEFORMED, unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
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
        if constexpr (APPEND)
        {
            out[index] += d0 * df[DIR * warpsize + dfindex];
        }
        else
        {
            out[index] = d0 * df[DIR * warpsize + dfindex];
        }
    }
}

/**
 * @brief One element, one dimension: apply the transpose of the 1D
 * differentiation matrix, optionally scaled and accumulated.
 *
 * The counterpart of the forward tensorial derivative, indexing
 * `D0[i * nq0 + q]` instead of `D0[q * nq0 + i]`. Not used by
 * PhysDeriv; it serves the weak-form families, which need
 * \f$\sum_d D_d^{\mathsf{T}} v_d\f$.
 *
 * @tparam SCALE      Multiply the result by @p scale.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
 * @param   nq0     Quadrature points of the element.
 * @param   D0      1D differentiation matrix, applied transposed.
 * @param   in0     Input values, warp interleaved.
 * @param   out     Output values, warp interleaved.
 * @param   scale   Factor applied when @p SCALE.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT in0,
    TData *NEK_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[warpsize * q + ilane];
        }

        if constexpr (SCALE)
        {
            d0 *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += d0;
        }
        else
        {
            out[warpsize * i + ilane] = d0;
        }
    }
}

/**
 * @brief One element, two dimensions: differentiate a quadrilateral or
 * triangle in the calling lane and write all @p ncoord Cartesian
 * components.
 *
 * Per point the two collapsed derivatives are formed, the triangle
 * correction `d0 *= f1[j]; d1 += d0 * f0[i]` is applied where the shape
 * requires it, and each output component is
 * `d0 * df[2k] + d1 * df[2k+1]`. The third component is produced only
 * for an element embedded in three dimensions.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   ncoord      Cartesian components to produce (coordDim); the
 *                      derivative factors come `ndf = 2 * ncoord` per
 *                      point.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   f0          \f$\tfrac{1}{2}(1+\eta_0)\f$ at the direction-0
 *                      points (triangles only).
 * @param   f1          \f$2/(1-\eta_1)\f$ at the direction-1 points
 *                      (triangles only).
 * @param   df          Derivative factors of this lane's warp.
 * @param   in          Input values, warp interleaved.
 * @param   out         Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
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

            // Multiply by derivative factors.
            out[index] = d0 * df[0u * warpsize + dfindex] +
                         d1 * df[1u * warpsize + dfindex];
            out[outoffset + index] = d0 * df[2u * warpsize + dfindex] +
                                     d1 * df[3u * warpsize + dfindex];
            if (ncoord == 3u)
            {
                out[2u * outoffset + index] = d0 * df[4u * warpsize + dfindex] +
                                              d1 * df[5u * warpsize + dfindex];
            }
        }
    }
}

/**
 * @brief One element, two dimensions, single component: as
 * PhysDeriv2DSumFacKernel but writing only Cartesian component DIR,
 * `d0 * df[2 * DIR] + d1 * df[2 * DIR + 1]`.
 *
 * Template parameter order is
 * `<SHAPE_TYPE, APPEND, DEFORMED, DIR, TData>`. This is the entry point
 * the Divergence device SumFac kernels call into.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   ncoord      Cartesian components the factor array holds;
 *                      sets the per-point factor stride
 *                      `ndf = 2 * ncoord`.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   f0,f1       Collapsed-coordinate factor tables, as in
 *                      PhysDeriv2DSumFacKernel (triangles only).
 * @param   df          Derivative factors of this lane's warp.
 * @param   in          Input values, warp interleaved.
 * @param   out         Cartesian component DIR, warp interleaved.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
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

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

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

            // Multiply by derivative factors.
            if constexpr (APPEND)
            {
                out[index] += d0 * df[2u * DIR * warpsize + dfindex] +
                              d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
            else
            {
                out[index] = d0 * df[2u * DIR * warpsize + dfindex] +
                             d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
        }
    }
}

/**
 * @brief One element, two dimensions: apply the transposes of both 1D
 * differentiation matrices to their own inputs and sum the two
 * contributions, optionally scaled and accumulated.
 *
 * As SumDerivTensor1DKernel, this serves the weak-form families rather
 * than PhysDeriv.
 *
 * @tparam SCALE      Multiply the sum by @p scale.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   D0,D1       1D differentiation matrices, applied transposed.
 * @param   in0         Input differentiated along direction 0.
 * @param   in1         Input differentiated along direction 1.
 * @param   out         Output values, warp interleaved.
 * @param   scale       Factor applied when @p SCALE.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor2DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    TData *NEK_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[i * nq0 + q] * in0[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[j * nq1 + q] * in1[warpsize * (nq0 * q + i) + ilane];
            }

            if constexpr (APPEND)
            {
                if constexpr (SCALE)
                {
                    out[warpsize * cnt_ji + ilane] += scale * (d0 + d1);
                }
                else
                {
                    out[warpsize * cnt_ji + ilane] += d0 + d1;
                }
            }
            else
            {
                if constexpr (SCALE)
                {
                    out[warpsize * cnt_ji + ilane] = scale * (d0 + d1);
                }
                else
                {
                    out[warpsize * cnt_ji + ilane] = d0 + d1;
                }
            }
        }
    }
}

/**
 * @brief One element, three dimensions: differentiate the element in
 * the calling lane and write all three Cartesian components.
 *
 * Per point the three collapsed derivatives are formed, the shape's
 * collapsed-to-standard correction is applied -- a prism scales
 * direction 0 by @p f2 and adds its @p f0 multiple onto direction 2, a
 * pyramid does the same for directions 0 and 1 with @p f0 and @p f1, a
 * (nodal) tetrahedron scales direction 0 by @p f1m and @p f2 and
 * direction 1 by @p f2 and folds both into directions 1 and 2, a
 * hexahedron needs none -- and each output component is
 * `sum_d d_d * df[3k+d]` over the nine derivative factors.
 *
 * @tparam SHAPE_TYPE Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   outoffset       Stride between the Cartesian planes of
 *                          @p out.
 * @param   D0,D1,D2        1D differentiation matrices, column-major.
 * @param   f0              \f$\tfrac{1}{2}(1+\eta_0)\f$ at the
 *                          direction-0 points.
 * @param   f1              \f$\tfrac{1}{2}(1+\eta_1)\f$ at the
 *                          direction-1 points.
 * @param   f1m             \f$2/(1-\eta_1)\f$ at the direction-1
 *                          points.
 * @param   f2              \f$2/(1-\eta_2)\f$ at the direction-2
 *                          points.
 * @param   df              Derivative factors of this lane's warp,
 *                          nine per point or per element.
 * @param   in              Input values, warp interleaved.
 * @param   out             Cartesian derivatives, plane @em k written
 *                          at `k * outoffset`.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t outoffset,
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

                // Multiply by derivative factors.
                out[index] = d0 * df[0u * warpsize + dfindex] +
                             d1 * df[1u * warpsize + dfindex] +
                             d2 * df[2u * warpsize + dfindex];
                out[outoffset + index] = d0 * df[3u * warpsize + dfindex] +
                                         d1 * df[4u * warpsize + dfindex] +
                                         d2 * df[5u * warpsize + dfindex];
                out[2u * outoffset + index] = d0 * df[6u * warpsize + dfindex] +
                                              d1 * df[7u * warpsize + dfindex] +
                                              d2 * df[8u * warpsize + dfindex];
            }
        }
    }
}

/**
 * @brief One element, three dimensions, single component: as
 * PhysDeriv3DSumFacKernel but writing only Cartesian component DIR,
 * `sum_d d_d * df[3 * DIR + d]`.
 *
 * Template parameter order is
 * `<SHAPE_TYPE, APPEND, DEFORMED, DIR, TData>`. This is the entry point
 * the Divergence device SumFac kernels call into.
 *
 * @tparam SHAPE_TYPE Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam DIR        Cartesian component to produce.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   D0,D1,D2        1D differentiation matrices, column-major.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables, as in
 *                          PhysDeriv3DSumFacKernel.
 * @param   df              Derivative factors of this lane's warp.
 * @param   in              Input values, warp interleaved.
 * @param   out             Cartesian component DIR, warp interleaved.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
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

                // Multiply by derivative factors.
                if constexpr (APPEND)
                {
                    out[index] += d0 * df[(3u * DIR) * warpsize + dfindex] +
                                  d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                  d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
                else
                {
                    out[index] = d0 * df[(3u * DIR) * warpsize + dfindex] +
                                 d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                 d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
            }
        }
    }
}

/**
 * @brief One element, three dimensions: apply the transposes of the
 * three 1D differentiation matrices to their own inputs and sum the
 * contributions, optionally scaled and accumulated.
 *
 * As SumDerivTensor1DKernel, this serves the weak-form families rather
 * than PhysDeriv.
 *
 * @tparam SCALE      Multiply the sum by @p scale.
 * @tparam APPEND     Accumulate onto @p out instead of overwriting.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   D0,D1,D2        1D differentiation matrices, applied
 *                          transposed.
 * @param   in0,in1,in2     Inputs differentiated along directions 0, 1
 *                          and 2.
 * @param   out             Output values, warp interleaved.
 * @param   scale           Factor applied when @p SCALE.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    const TData *NEK_RESTRICT in2, TData *NEK_RESTRICT out,
    const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[i * nq0 + q] *
                          in0[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[j * nq1 + q] *
                          in1[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[k * nq2 + q] *
                          in2[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                if constexpr (APPEND)
                {
                    if constexpr (SCALE)
                    {
                        out[warpsize * cnt_kji + ilane] +=
                            scale * (d0 + d1 + d2);
                    }
                    else
                    {
                        out[warpsize * cnt_kji + ilane] += d0 + d1 + d2;
                    }
                }
                else
                {
                    if constexpr (SCALE)
                    {
                        out[warpsize * cnt_kji + ilane] =
                            scale * (d0 + d1 + d2);
                    }
                    else
                    {
                        out[warpsize * cnt_kji + ilane] = d0 + d1 + d2;
                    }
                }
            }
        }
    }
}

/**
 * @brief SumFac launch entry point for a block of segments: unpacks the
 * size parameter, then grid-strides over the block's segments on the
 * grid's X axis, one element per warp lane, while the Y and Z axes
 * select the homogeneous mode and component of a single fused launch
 * that covers every (component, mode) pair at once.
 *
 * Selected over the SumFacTOP overload of the same name by the
 * Implementation tag, and over the two- and three-dimensional
 * overloads by the size-parameter type, which may be either the
 * runtime or the compile-time form.
 *
 * The global thread index is split into a lane and a warp; the
 * derivative-factor pointer is offset to that warp only -- `ndf *
 * dfsize` values per element, `dfsize` being `nq0` for a deformed
 * element and 1 for a regular one -- since the factors do not depend on
 * component or mode. The data pointers additionally carry the block
 * index `m` (mode, range `nmode`) and `c` (component): the input slab
 * is `nmode * c + m`, matching the component-major, mode-minor order of
 * the block storage, and the output slab `outDim * nmode * c + m`,
 * where `outDim` is `ncoord` unless the launch is homogeneous
 * (`nmode > 1`), in which case it is always three: a homogeneous
 * expansion reserves the output's third (z) plane even for this
 * in-plane kernel, which only ever writes `ncoord` of the `outDim`
 * planes it reserves, leaving the rest for detail::DerivZOpImpl to fill
 * in. No shared memory is used in one dimension; @p shmemptr is
 * accepted so that all three overloads share one launch macro.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter1D 1D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam1D Coordinate dimension and point count of the
 *                      element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0          1D differentiation matrix.
 * @param   df          Derivative factors of the block.
 * @param   in          Input values of the block, warp interleaved and
 *                      spanning every component and homogeneous mode.
 * @param   out         Output values of the block, warp interleaved and
 *                      spanning every component and homogeneous mode.
 * @param   shmemptr    Dynamic shared memory of this thread block;
 *                      unused in one dimension.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
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

    size_t e                  = getGlobalIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nq0 * (nelmt * (outDim * nmode * c + m) + warpsize * iwarp);
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, outoffset, D0,
                                          dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac launch entry point for one Cartesian component of a
 * block of segments: unpacks the size parameter, then grid-strides
 * over the block's segments on the grid's X axis, one element per warp
 * lane, offsetting the data pointers by the block index `c`
 * (component) alone -- there is no homogeneous-mode axis here.
 *
 * @warning Not compilable if ever instantiated: the call below passes
 * eight arguments to the per-element PhysDerivDir1DSumFacKernel, which
 * declares seven; the extra is a literal `0` between `nq0` and @p D0,
 * in the position an `outoffset` parameter would occupy. Harmless
 * only because this launcher currently has no call site; callers
 * needing a single Cartesian component use the per-element
 * PhysDerivDir1DSumFacKernel directly, as the Divergence and CurlCurl
 * kernels do.
 *
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam DIR                  Cartesian component to produce.
 * @tparam TPhysSizeParameter1D 1D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam1D Coordinate dimension and point count of the
 *                      element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   D0          1D differentiation matrix.
 * @param   df          Derivative factors of the block.
 * @param   in          Input values of the block.
 * @param   out         Cartesian component DIR of the block.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <typename Implementation, bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir1DKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
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
        PhysDerivDir1DSumFacKernel<APPEND, DEFORMED, DIR>(
            ilane, ncoord, nq0, 0, D0, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac launch entry point for a block of quadrilaterals or
 * triangles: claims the dynamic shared memory, unpacks the size
 * parameter, stages the collapsed-coordinate tables it needs, then
 * grid-strides over the block's elements on the grid's X axis, one
 * element per warp lane, while the Y and Z axes select the homogeneous
 * mode and component of a single fused launch that covers every
 * (component, mode) pair at once.
 *
 * For a (nodal) triangle the two tables are copied into the dynamic
 * shared memory sized by PhysDerivSharedMemorySize and the block
 * synchronises before the element loop; for a quadrilateral no shared
 * memory is used and null pointers are handed on. Each iteration offsets
 * the derivative-factor pointer to its warp only -- `ndf * dfsize`
 * values per element, `dfsize` being nqTot when deformed and 1
 * otherwise, since the factors do not depend on component or mode --
 * and calls PhysDeriv2DSumFacKernel. The data pointers additionally
 * carry the block index `m` (mode, range `nmode`) and `c` (component),
 * exactly as in the one-dimensional overload above: the input slab is
 * `nmode * c + m` and the output slab `outDim * nmode * c + m`,
 * `outDim` collapsing to `ncoord` unless the launch is homogeneous, in
 * which case it reserves the third plane for the z-derivative backend
 * as described there.
 *
 * @tparam SHAPE_TYPE           Quad, Tri or NodalTri.
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter2D 2D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam2D Coordinate dimension and point counts of the
 *                      element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0,D1       1D differentiation matrices.
 * @param   f0,f1       Collapsed-coordinate factor tables.
 * @param   df          Derivative factors of the block.
 * @param   in          Input values of the block, warp interleaved and
 *                      spanning every component and homogeneous mode.
 * @param   out         Output values of the block, warp interleaved and
 *                      spanning every component and homogeneous mode.
 * @param   shmemptr    Dynamic shared memory of this thread block.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
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

    size_t e                  = getGlobalIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (outDim * nmode * c + m) + warpsize * iwarp);
        PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, outoffset, D0, D1, s_f0, s_f1, dfptr,
            inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac launch entry point for one Cartesian component of a
 * block of quadrilaterals or triangles: claims the dynamic shared
 * memory, unpacks the size parameter, stages the collapsed-coordinate
 * tables it needs, then grid-strides over the block's elements on the
 * grid's X axis, one element per warp lane, offsetting the data
 * pointers by the block index `c` (component) alone -- there is no
 * homogeneous-mode axis here.
 *
 * @warning Not compilable if ever instantiated: the call below omits
 * `ncoord`, which the eleven-parameter per-element
 * PhysDerivDir2DSumFacKernel declares as its second argument; only ten
 * arguments are passed. Nor does the launcher unpack `ncoord` for
 * itself, so its per-element factor stride is fixed at `ndf = 4`, i.e.
 * a coordinate dimension of two. Harmless only because this launcher
 * currently has no call site; callers needing a single Cartesian
 * component use the per-element PhysDerivDir2DSumFacKernel directly, as
 * the Divergence kernels do.
 *
 * @tparam SHAPE_TYPE           Quad, Tri or NodalTri.
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam DIR                  Cartesian component to produce.
 * @tparam TPhysSizeParameter2D 2D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam2D Coordinate dimension and point counts of the
 *                      element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   D0,D1       1D differentiation matrices.
 * @param   f0,f1       Collapsed-coordinate factor tables.
 * @param   df          Derivative factors of the block.
 * @param   in          Input values of the block.
 * @param   out         Cartesian component DIR of the block.
 * @param   shmemptr    Dynamic shared memory of this thread block.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir2DKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    const unsigned int ndf    = 4u;
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

        PhysDerivDir2DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            ilane, nq0, nq1, D0, D1, s_f0, s_f1, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac launch entry point for a block of three-dimensional
 * elements: claims the dynamic shared memory, unpacks the size
 * parameter, stages the shape's collapsed-coordinate tables, then
 * grid-strides over the block's elements on the grid's X axis, one
 * element per warp lane, while the Y and Z axes select the homogeneous
 * mode and component of a single fused launch that covers every
 * (component, mode) pair at once.
 *
 * Which tables are staged depends on the shape -- four for a (nodal)
 * tetrahedron, two for a (nodal) prism, three for a pyramid, none for a
 * hexahedron, matching PhysDerivSharedMemorySize -- after which the
 * block synchronises. Each iteration offsets the derivative-factor
 * pointer to its warp only, `9 * dfsize` values per element, since the
 * factors do not depend on component or mode. The data pointers
 * additionally carry the block index `m` (mode, range `nmode`) and `c`
 * (component), exactly as in the one- and two-dimensional overloads
 * above: the input slab is `nmode * c + m`, and the output slab
 * `3 * nmode * c + m`, the factor of three fixed rather than `outDim`
 * because a three-dimensional element always yields three Cartesian
 * derivatives regardless of homogeneity.
 *
 * @tparam SHAPE_TYPE           Hex, Tet, NodalTet, Prism, NodalPrism or
 *                              Pyr.
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam TPhysSizeParameter3D 3D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam3D     Point counts of the element.
 * @param   nelmt           Elements in the block, including padding.
 * @param   outoffset       Stride between the Cartesian planes of
 *                          @p out.
 * @param   D0,D1,D2        1D differentiation matrices.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables.
 * @param   df              Derivative factors of the block.
 * @param   in              Input values of the block, warp interleaved
 *                          and spanning every component and
 *                          homogeneous mode.
 * @param   out             Output values of the block, warp
 *                          interleaved and spanning every component
 *                          and homogeneous mode.
 * @param   shmemptr        Dynamic shared memory of this thread block.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
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
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, outoffset, D0, D1, D2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac launch entry point for one Cartesian component of a
 * block of three-dimensional elements: claims the dynamic shared
 * memory, unpacks the size parameter, stages the shape's
 * collapsed-coordinate tables, then grid-strides over the block's
 * elements on the grid's X axis, one element per warp lane, calling the
 * per-direction per-element kernel so that only Cartesian component DIR
 * is written.
 *
 * The only per-direction launcher of the three whose argument list
 * matches its per-element kernel; it is nevertheless not instantiated
 * anywhere, the Divergence family launching its own kernels and
 * calling the per-element PhysDeriv kernels from inside them.
 *
 * @tparam SHAPE_TYPE           Hex, Tet, NodalTet, Prism, NodalPrism or
 *                              Pyr.
 * @tparam Implementation       Constrained to MultiRegions::SumFac.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Derivative factors vary point by point.
 * @tparam DIR                  Cartesian component to produce.
 * @tparam TPhysSizeParameter3D 3D physical size-parameter type.
 * @tparam TthreadBlock         Thread-block handle type.
 * @tparam TData                Floating-point type of the field data.
 *
 * @param   sizeParam3D     Point counts of the element.
 * @param   nelmt           Elements in the block, including padding.
 * @param   D0,D1,D2        1D differentiation matrices.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables.
 * @param   df              Derivative factors of the block.
 * @param   in              Input values of the block, warp interleaved.
 * @param   out             Cartesian component DIR of the block, warp
 *                          interleaved.
 * @param   shmemptr        Dynamic shared memory of this thread block.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir3DKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
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
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
