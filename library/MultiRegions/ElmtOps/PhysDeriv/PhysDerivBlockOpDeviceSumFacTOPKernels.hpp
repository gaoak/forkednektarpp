///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernels of the physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the physical derivative: one
 * element per thread block, the block's threads striding over the
 * element's output points.
 *
 * @details
 * The SumFacTOP counterpart of PhysDerivBlockOpDeviceSumFacKernels.hpp. Every
 * kernel here is constrained to the MultiRegions::SumFacTOP strategy and
 * comes in two layers:
 * - the per-element kernels (`PhysDeriv{,Dir}{1,2,3}DSumFacTOPKernel`
 *   without an @p nelmt), in which the calling thread block strides its
 *   local thread index over the element's quadrature points and each
 *   thread computes the tensorial derivatives, collapsed-coordinate
 *   correction and derivative-factor contraction for each output point
 *   it visits. The element's data is not interleaved: entry @em n of the
 *   element is simply at index @em n. Each ends with a block barrier,
 *   so the caller may reuse the shared workspace immediately;
 * - the launch entry points, which check the size-parameter type at
 *   compile time, unpack it, then for each element in turn stride their
 *   thread block over it: in two and three dimensions they first copy
 *   the element's input values into shared memory and synchronise
 *   before calling the per-element kernel on the shared copy; in one
 *   dimension no shared memory is used and the per-element kernel reads
 *   straight from global memory. The three PhysDerivKernelLauncher
 *   overloads (one per dimension, selected by the size-parameter type)
 *   are what the block operator launches through
 *   DEVICE_3DGRID_KERNEL_LAUNCHER, whose grid Y and Z axes carry the
 *   homogeneous-mode and component counts so a single launch covers
 *   every component and mode of the block, the one- and
 *   two-dimensional overloads reserving a third output plane per
 *   component once the mode count exceeds one (see the 1D overload).
 *   The per-direction `PhysDerivDir{1,2,3}DKernelLauncher` are not
 *   instantiated anywhere (see their notes).
 *
 * The derivative factors are fetched transposed for this strategy: per
 * element the `ndf = coordDim * dimension` factors form contiguous
 * planes of `dfsize` values, `dfsize` being the element's point count
 * for a deformed element and 1 for a regular one, so factor @em c at
 * point @em n sits at `c * dfsize + n` and `c` respectively. That is
 * what the DEFORMED template flag selects; nothing else about the two
 * instantiations differs.
 *
 * The per-direction ("Dir") kernels produce a single Cartesian
 * component, optionally accumulating; PhysDeriv itself never calls
 * them, the Divergence device SumFacTOP kernels (and, in one dimension,
 * the CurlCurl ones) do. The all-component per-element kernels are in
 * turn shared with the AdvectionDealias, Helmholtz, Laplacian and
 * LinAdvDiffReaction device SumFacTOP kernels.
 *
 * The SumDerivTensor kernels in this file are the transposes of the
 * tensorial derivative and belong to the weak-form families
 * (IProductWRTDerivBase, Laplacian, Helmholtz, LinAdvDiffReaction)
 * rather than to PhysDeriv.
 *
 * The whole file is compiled only for the device back-ends
 * (NEKTAR_ENABLE_DEVICE together with DEVICE_COMPILE_ONLY).
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
 * SumFacTOP launch needs: none, since the 1D launcher reads the
 * element's input straight from global memory.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Constrained to MultiRegions::SumFacTOP.
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
inline constexpr unsigned int PhysDerivSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory, in elements of the data type, a 2D
 * SumFacTOP launch needs: one element's quadrature-point values.
 *
 * Unlike the SumFac form this does not depend on the shape: the
 * collapsed-coordinate tables are read straight from global memory and
 * the shared memory holds the staged input instead.
 *
 * @tparam SHAPE_TYPE            Shape of the block's elements; unused.
 * @tparam Implementation        Constrained to MultiRegions::SumFacTOP.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Values to allocate per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
    const TPhysSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    return nq0 * nq1;
}

/**
 * @brief Dynamic shared memory, in elements of the data type, a 3D
 * SumFacTOP launch needs: one element's quadrature-point values.
 *
 * @tparam SHAPE_TYPE            Shape of the block's elements; unused.
 * @tparam Implementation        Constrained to MultiRegions::SumFacTOP.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Values to allocate per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    return nq0 * nq1 * nq2;
}

/**
 * @brief One element, one dimension: differentiate a segment with the
 * calling thread block, whose threads stride over the output points, and
 * scale by the derivative factors.
 *
 * With one standard direction there is nothing to contract and no
 * collapsed-coordinate correction:
 * `out[k][i] = df[k] * sum_q D0[q * nq0 + i] * in[q]`. Ends with a
 * block barrier.
 *
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Thread-block handle type of the active device
 *                      back-end.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components to produce (coordDim).
 * @param   nq0         Quadrature points of the element.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0          1D differentiation matrix, column-major.
 * @param   df          Derivative factors of this element, transposed
 *                      layout.
 * @param   in          Input values of this element.
 * @param   out         Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[q];
        }

        // Multiply by derivative factors.
        for (unsigned int k = 0u; k < ncoord; k++)
        {
            out[k * outoffset + i] = d0 * df[k * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, one dimension, single component: as
 * PhysDeriv1DSumFacTOPKernel but writing only Cartesian component DIR.
 *
 * Template parameter order is
 * `<APPEND, DEFORMED, DIR, TthreadBlock, TData>`. This is the entry
 * point the Divergence and CurlCurl device SumFacTOP kernels call into.
 * @p ncoord is unused here: with the transposed factor layout the plane
 * of component DIR is found from @p nq0 alone.
 *
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam DIR          Cartesian component to produce.
 * @tparam TthreadBlock Thread-block handle type of the active device
 *                      back-end.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components the factor array holds;
 *                      unused.
 * @param   nq0         Quadrature points of the element.
 * @param   D0          1D differentiation matrix, column-major.
 * @param   df          Derivative factors of this element, transposed
 *                      layout.
 * @param   in          Input values of this element.
 * @param   out         Cartesian component DIR.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <bool APPEND, bool DEFORMED, unsigned int DIR, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir1DSumFacTOPKernel(
    [[maybe_unused]] const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[q];
        }

        // Multiply by derivative factors.
        if constexpr (APPEND)
        {
            out[i] += d0 * df[DIR * dfsize + dfindex];
        }
        else
        {
            out[i] = d0 * df[DIR * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, one dimension: apply the transpose of the 1D
 * differentiation matrix, optionally scaled and accumulated.
 *
 * The counterpart of the forward tensorial derivative, indexing
 * `D0[i * nq0 + q]` instead of `D0[q * nq0 + i]`. Not used by
 * PhysDeriv; it serves the weak-form families.
 *
 * @tparam SCALE        Multiply the result by @p scale.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nq0         Quadrature points of the element.
 * @param   D0          1D differentiation matrix, applied transposed.
 * @param   in0         Input values of this element.
 * @param   out         Output values of this element.
 * @param   scale       Factor applied when @p SCALE.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor1DSumFacTOPKernel(
    const unsigned int nq0, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT in0, TData *NEK_RESTRICT out, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        // Compute transpose tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[q];
        }

        if constexpr (SCALE)
        {
            d0 *= scale;
        }

        if constexpr (APPEND)
        {
            out[i] += d0;
        }
        else
        {
            out[i] = d0;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, two dimensions: differentiate a quadrilateral or
 * triangle with the calling thread block, whose threads stride over the
 * output points, and write all @p ncoord Cartesian components.
 *
 * Each thread recovers its point's `(i, j)` from the flat index, forms
 * the two collapsed derivatives, applies the triangle correction
 * `d0 *= f1[j]; d1 += d0 * f0[i]` where the shape requires it, and
 * writes `d0 * df[2k] + d1 * df[2k+1]` per component -- the third only
 * for an element embedded in three dimensions. Ends with a block
 * barrier.
 *
 * @tparam SHAPE_TYPE   Quad, Tri or NodalTri.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components to produce (coordDim); the
 *                      derivative factors come `2 * ncoord` planes per
 *                      element.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   f0          \f$\tfrac{1}{2}(1+\eta_0)\f$ at the direction-0
 *                      points (triangles only).
 * @param   f1          \f$2/(1-\eta_1)\f$ at the direction-1 points
 *                      (triangles only).
 * @param   df          Derivative factors of this element, transposed
 *                      layout.
 * @param   in          Input values of this element, typically the
 *                      shared-memory copy staged by the caller.
 * @param   out         Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
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
            d0 += D0[q * nq0 + i] * in[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * q + i];
        }

        // Moving from collapsed to standard coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            d0 *= f1[j];
            d1 += d0 * f0[i];
        }

        // Multiply by derivative factors.
        out[idx] =
            d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex];
        out[outoffset + idx] =
            d0 * df[2u * dfsize + dfindex] + d1 * df[3u * dfsize + dfindex];
        if (ncoord == 3u)
        {
            out[2u * outoffset + idx] =
                d0 * df[4u * dfsize + dfindex] + d1 * df[5u * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, two dimensions, single component: as
 * PhysDeriv2DSumFacTOPKernel but writing only Cartesian component DIR.
 *
 * Template parameter order is
 * `<SHAPE_TYPE, APPEND, DEFORMED, DIR, TthreadBlock, TData>`. This is
 * the entry point the Divergence device SumFacTOP kernels call into.
 * The derivative-factor planes are picked by an if-constexpr chain on
 * DIR rather than by arithmetic, so a DIR above 2 silently writes
 * nothing. @p ncoord is unused: with the transposed factor layout the
 * planes are found from @p nq0 and @p nq1 alone.
 *
 * @tparam SHAPE_TYPE   Quad, Tri or NodalTri.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam DIR          Cartesian component to produce (0, 1 or 2).
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components the factor array holds;
 *                      unused.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   f0,f1       Collapsed-coordinate factor tables, as in
 *                      PhysDeriv2DSumFacTOPKernel (triangles only).
 * @param   df          Derivative factors of this element, transposed
 *                      layout.
 * @param   in          Input values of this element.
 * @param   out         Cartesian component DIR.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir2DSumFacTOPKernel(
    [[maybe_unused]] const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
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
            d0 += D0[q * nq0 + i] * in[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * q + i];
        }

        // Moving from collapsed to standard coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            d0 *= f1[j];
            d1 += d0 * f0[i];
        }

        // Multiply by derivative factors.
        if constexpr (DIR == 0)
        {
            if constexpr (APPEND)
            {
                out[idx] += d0 * df[0u * dfsize + dfindex] +
                            d1 * df[1u * dfsize + dfindex];
            }
            else
            {
                out[idx] = d0 * df[0u * dfsize + dfindex] +
                           d1 * df[1u * dfsize + dfindex];
            }
        }
        else if constexpr (DIR == 1)
        {
            if constexpr (APPEND)
            {
                out[idx] += d0 * df[2u * dfsize + dfindex] +
                            d1 * df[3u * dfsize + dfindex];
            }
            else
            {
                out[idx] = d0 * df[2u * dfsize + dfindex] +
                           d1 * df[3u * dfsize + dfindex];
            }
        }
        else if constexpr (DIR == 2)
        {
            if constexpr (APPEND)
            {
                out[idx] += d0 * df[4u * dfsize + dfindex] +
                            d1 * df[5u * dfsize + dfindex];
            }
            else
            {
                out[idx] = d0 * df[4u * dfsize + dfindex] +
                           d1 * df[5u * dfsize + dfindex];
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, two dimensions: apply the transposes of both 1D
 * differentiation matrices to their own inputs and sum the two
 * contributions, optionally scaled and accumulated.
 *
 * As SumDerivTensor1DSumFacTOPKernel, this serves the weak-form
 * families rather than PhysDeriv.
 *
 * @tparam SCALE        Multiply the sum by @p scale.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   D0,D1       1D differentiation matrices, applied transposed.
 * @param   in0         Input differentiated along direction 0.
 * @param   in1         Input differentiated along direction 1.
 * @param   out         Output values of this element.
 * @param   scale       Factor applied when @p SCALE.
 * @param   threadBlock Thread-block handle supplying the index, range
 *                      and barrier helpers.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor2DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[j * nq1 + q] * in1[nq0 * q + i];
        }

        if constexpr (APPEND)
        {
            if constexpr (SCALE)
            {
                out[idx] += scale * (d0 + d1);
            }
            else
            {
                out[idx] += d0 + d1;
            }
        }
        else
        {
            if constexpr (SCALE)
            {
                out[idx] = scale * (d0 + d1);
            }
            else
            {
                out[idx] = d0 + d1;
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, three dimensions: differentiate the element with
 * the calling thread block, whose threads stride over the output points,
 * and write all three Cartesian components.
 *
 * Each thread recovers its point's `(i, j, k)` from the flat index,
 * forms the three collapsed derivatives, applies the shape's
 * collapsed-to-standard correction -- prism, pyramid and (nodal)
 * tetrahedron as in PhysDeriv3DSumFacKernel, none for a hexahedron --
 * and writes `sum_d d_d * df[3k+d]` per component. Ends with a block
 * barrier.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
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
 * @param   df              Derivative factors of this element,
 *                          transposed layout, nine planes.
 * @param   in              Input values of this element, typically the
 *                          shared-memory copy staged by the caller.
 * @param   out             Cartesian derivatives, plane @em k written
 *                          at `k * outoffset`.
 * @param   threadBlock     Thread-block handle supplying the index,
 *                          range and barrier helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
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

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[q * nq2 + k] * in[nq0 * nq1 * q + nq0 * j + i];
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
        out[idx] = d0 * df[0u * dfsize + dfindex] +
                   d1 * df[1u * dfsize + dfindex] +
                   d2 * df[2u * dfsize + dfindex];
        out[outoffset + idx] = d0 * df[3u * dfsize + dfindex] +
                               d1 * df[4u * dfsize + dfindex] +
                               d2 * df[5u * dfsize + dfindex];
        out[2u * outoffset + idx] = d0 * df[6u * dfsize + dfindex] +
                                    d1 * df[7u * dfsize + dfindex] +
                                    d2 * df[8u * dfsize + dfindex];
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, three dimensions, single component: as
 * PhysDeriv3DSumFacTOPKernel but writing only Cartesian component DIR,
 * `sum_d d_d * df[3 * DIR + d]`.
 *
 * Template parameter order is
 * `<SHAPE_TYPE, APPEND, DEFORMED, DIR, TthreadBlock, TData>`. This is
 * the entry point the Divergence device SumFacTOP kernels call into.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam DIR          Cartesian component to produce.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   D0,D1,D2        1D differentiation matrices, column-major.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables, as in
 *                          PhysDeriv3DSumFacTOPKernel.
 * @param   df              Derivative factors of this element,
 *                          transposed layout.
 * @param   in              Input values of this element.
 * @param   out             Cartesian component DIR.
 * @param   threadBlock     Thread-block handle supplying the index,
 *                          range and barrier helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
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

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[q * nq2 + k] * in[nq0 * nq1 * q + nq0 * j + i];
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
            out[idx] += d0 * df[(3u * DIR) * dfsize + dfindex] +
                        d1 * df[(3u * DIR + 1u) * dfsize + dfindex] +
                        d2 * df[(3u * DIR + 2u) * dfsize + dfindex];
        }
        else
        {
            out[idx] = d0 * df[(3u * DIR) * dfsize + dfindex] +
                       d1 * df[(3u * DIR + 1u) * dfsize + dfindex] +
                       d2 * df[(3u * DIR + 2u) * dfsize + dfindex];
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element, three dimensions: apply the transposes of the
 * three 1D differentiation matrices to their own inputs and sum the
 * contributions, optionally scaled and accumulated.
 *
 * As SumDerivTensor1DSumFacTOPKernel, this serves the weak-form
 * families rather than PhysDeriv.
 *
 * @tparam SCALE        Multiply the sum by @p scale.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   D0,D1,D2        1D differentiation matrices, applied
 *                          transposed.
 * @param   in0,in1,in2     Inputs differentiated along directions 0, 1
 *                          and 2.
 * @param   out             Output values of this element.
 * @param   scale           Factor applied when @p SCALE.
 * @param   threadBlock     Thread-block handle supplying the index,
 *                          range and barrier helpers.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT in0,
    const TData *NEK_RESTRICT in1, const TData *NEK_RESTRICT in2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq0 * nq1);

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[j * nq1 + q] * in1[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[k * nq2 + q] * in2[nq0 * nq1 * q + nq0 * j + i];
        }

        if constexpr (APPEND)
        {
            if constexpr (SCALE)
            {
                out[idx] += scale * (d0 + d1 + d2);
            }
            else
            {
                out[idx] += d0 + d1 + d2;
            }
        }
        else
        {
            if constexpr (SCALE)
            {
                out[idx] = scale * (d0 + d1 + d2);
            }
            else
            {
                out[idx] = d0 + d1 + d2;
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief SumFacTOP launch entry point for a block of segments: unpacks
 * the size parameter, then strides the thread block over the block's
 * segments on the grid's X axis, one element per thread block per pass,
 * while the Y and Z axes select the homogeneous mode and component of a
 * single fused launch that covers every (component, mode) pair at
 * once.
 *
 * Selected over the SumFac overload of the same name by the
 * Implementation tag, and over the two- and three-dimensional
 * overloads by the size-parameter type, which may be either the
 * runtime or the compile-time form.
 *
 * The derivative-factor pointer is offset to the current element only
 * -- `ndf * dfsize` values, `dfsize` being `nq0` for a deformed
 * element and 1 for a regular one -- since the factors do not depend on
 * component or mode; PhysDeriv1DSumFacTOPKernel does the element's
 * work. The data pointers additionally carry the block index `m` (mode,
 * range `nmode`) and `c` (component): the input slab is
 * `nmode * c + m`, and the output slab `outDim * nmode * c + m`, where
 * `outDim` is `ncoord` unless the launch is homogeneous (`nmode > 1`),
 * in which case it is always three: a homogeneous expansion reserves
 * the output's third (z) plane even for this in-plane kernel, which
 * only ever writes `ncoord` of the `outDim` planes it reserves, leaving
 * the rest for detail::DerivZOpImpl to fill in. No shared memory is
 * used in one dimension: the input is read straight from global
 * memory, and @p shmemptr is accepted so that all three overloads
 * share one launch macro.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
 * @param   in          Input values of the block, spanning every
 *                      component and homogeneous mode.
 * @param   out         Output values of the block, spanning every
 *                      component and homogeneous mode.
 * @param   shmemptr    Dynamic shared memory of this thread block;
 *                      unused in one dimension.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    size_t e                  = getBlockIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nq0 * nelmt * (nmode * c + m) + nq0 * e;
        TData *outptr = out + nq0 * nelmt * (outDim * nmode * c + m) + nq0 * e;
        PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, outoffset, D0, dfptr,
                                             inptr, outptr, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP launch entry point for one Cartesian component of a
 * block of segments: unpacks the size parameter, then strides the
 * thread block over the block's segments on the grid's X axis, one
 * element per thread block per pass, offsetting the data pointers by
 * the block index `c` (component) alone -- there is no homogeneous-mode
 * axis here.
 *
 * @warning Not compilable if ever instantiated: the call below passes
 * eight arguments to the per-element PhysDerivDir1DSumFacTOPKernel,
 * which declares seven; the extra is a literal `0` between `nq0` and
 * @p D0, in the position an `outoffset` parameter would occupy.
 * Harmless only because this launcher currently has no call site;
 * callers needing a single Cartesian component use the per-element
 * PhysDerivDir1DSumFacTOPKernel directly, as the Divergence and
 * CurlCurl kernels do.
 *
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nq0 * nelmt * c + nq0 * e;
        TData *outptr      = out + nq0 * nelmt * c + nq0 * e;
        PhysDerivDir1DSumFacTOPKernel<APPEND, DEFORMED, DIR>(
            ncoord, nq0, 0, D0, dfptr, inptr, outptr, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP launch entry point for a block of quadrilaterals or
 * triangles: claims the dynamic shared memory, unpacks the size
 * parameter, then strides the thread block over the block's elements
 * on the grid's X axis, staging each element's input values in shared
 * memory before differentiating it, while the Y and Z axes select the
 * homogeneous mode and component of a single fused launch that covers
 * every (component, mode) pair at once.
 *
 * Per element the threads cooperatively copy the element's nqTot input
 * values into the dynamic shared memory sized by
 * PhysDerivSharedMemorySize, synchronise, and call
 * PhysDeriv2DSumFacTOPKernel on the shared copy -- which every thread
 * reads from repeatedly, once per term of its two sums. The
 * collapsed-coordinate tables are read from global memory. The
 * per-element kernel's closing barrier is what makes it safe to
 * overwrite the shared copy on the next pass. As in the one-dimensional
 * overload, the derivative-factor pointer is offset to the current
 * element alone, while the data pointers additionally carry the block
 * index `m` (mode, range `nmode`) and `c` (component): the input slab
 * is `nmode * c + m`, and the output slab `outDim * nmode * c + m`,
 * `outDim` collapsing to `ncoord` unless the launch is homogeneous, in
 * which case it reserves the third plane for the z-derivative backend
 * as described there.
 *
 * @tparam SHAPE_TYPE           Quad, Tri or NodalTri.
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
 * @param   in          Input values of the block, spanning every
 *                      component and homogeneous mode.
 * @param   out         Output values of the block, spanning every
 *                      component and homogeneous mode.
 * @param   shmemptr    Dynamic shared memory of this thread block.
 * @param   threadBlock Thread-block handle supplied by the launch
 *                      macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e                  = getBlockIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * (nmode * c + m) + nqTot * e;
        TData *outptr =
            out + nqTot * nelmt * (outDim * nmode * c + m) + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, outoffset, D0, D1, f0, f1, dfptr, s_wsp0, outptr,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP launch entry point for one Cartesian component of a
 * block of quadrilaterals or triangles: claims the dynamic shared
 * memory, unpacks the size parameter, then strides the thread block
 * over the block's elements on the grid's X axis, staging each
 * element's input values in shared memory before differentiating it,
 * offsetting the data pointers by the block index `c` (component)
 * alone -- there is no homogeneous-mode axis here.
 *
 * @warning Not compilable if ever instantiated: the call below omits
 * `ncoord`, which the eleven-parameter per-element
 * PhysDerivDir2DSumFacTOPKernel declares as its first argument; only
 * ten arguments are passed. Nor does the launcher unpack `ncoord` for
 * itself, so its per-element factor stride is fixed at `ndf = 4`, i.e.
 * a coordinate dimension of two. Harmless only because this launcher
 * currently has no call site; callers needing a single Cartesian
 * component use the per-element PhysDerivDir2DSumFacTOPKernel directly,
 * as the Divergence kernels do.
 *
 * @tparam SHAPE_TYPE           Quad, Tri or NodalTri.
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr      = out + nqTot * nelmt * c + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir2DSumFacTOPKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            nq0, nq1, D0, D1, f0, f1, dfptr, s_wsp0, outptr, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP launch entry point for a block of three-dimensional
 * elements: claims the dynamic shared memory, unpacks the size
 * parameter, then strides the thread block over the block's elements,
 * staging each element's input values in shared memory before
 * differentiating it.
 *
 * As the two-dimensional overload, with three differentiation matrices
 * and four collapsed-coordinate tables read from global memory, and the
 * Y and Z grid axes again selecting the homogeneous mode `m` and
 * component `c` of a fused launch: the input slab is `nmode * c + m`
 * and the output slab `3 * nmode * c + m`, the factor of three fixed
 * rather than `outDim` because a three-dimensional element always
 * yields three Cartesian derivatives regardless of homogeneity.
 *
 * @tparam SHAPE_TYPE           Hex, Tet, NodalTet, Prism, NodalPrism or
 *                              Pyr.
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
 * @param   in              Input values of the block, spanning every
 *                          component and homogeneous mode.
 * @param   out             Output values of the block, spanning every
 *                          component and homogeneous mode.
 * @param   shmemptr        Dynamic shared memory of this thread block.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * (nmode * c + m) + nqTot * e;
        TData *outptr = out + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, outoffset, D0, D1, D2, f0, f1, f1m, f2, dfptr,
            s_wsp0, outptr, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP launch entry point for one Cartesian component of a
 * block of three-dimensional elements: claims the dynamic shared
 * memory, unpacks the size parameter, then strides the thread block
 * over the block's elements, staging each element's input values in
 * shared memory before calling the per-direction per-element kernel so
 * that only Cartesian component DIR is written.
 *
 * The only per-direction launcher of the three whose argument list
 * matches its per-element kernel; it is nevertheless not instantiated
 * anywhere, the Divergence family launching its own kernels and
 * calling the per-element PhysDeriv kernels from inside them.
 *
 * @tparam SHAPE_TYPE           Hex, Tet, NodalTet, Prism, NodalPrism or
 *                              Pyr.
 * @tparam Implementation       Constrained to MultiRegions::SumFacTOP.
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
 * @param   in              Input values of the block.
 * @param   out             Cartesian component DIR of the block.
 * @param   shmemptr        Dynamic shared memory of this thread block.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr      = out + nqTot * nelmt * c + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir3DSumFacTOPKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, s_wsp0, outptr,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
