///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceBlockOpDeviceSumFacKernels.hpp
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
 * @file DivergenceBlockOpDeviceSumFacKernels.hpp
 * @brief SumFac device kernel launchers of the block divergence -- one
 * element per thread -- together with their shared-memory sizing
 * helpers.
 *
 * @details
 * The DivergenceKernelLauncher overloads of this file, one per
 * dimension, write \f$\sum_c \partial u_c/\partial x_c\f$ to the
 * output. In 1D that is PhysDerivDir1DSumFacKernel
 * (PhysDeriv/PhysDerivBlockOpDeviceSumFacKernels.hpp, included here).
 * In 2D and 3D, Divergence{2,3}DSumFacKernel differentiate every
 * component in every reference direction at a point, apply the
 * collapsed-coordinate correction where the shape requires it, contract
 * each component with its own row of the derivative factors and store
 * the sum once, so the output is written once per point.
 * DivergenceSharedMemorySize reports how much dynamic shared memory a
 * launch must request.
 *
 * Each thread owns one element: element e is handled by lane
 * `e % warpSize` of warp `e / warpSize`, matching the warp-interleaved
 * block storage the host side reshapes to, and the kernels grid-stride
 * over the elements left after the first pass (padding elements
 * included -- their results are ignored). The homogeneous planes ride
 * the second grid dimension: each thread block offsets the input and
 * output to the plane given by its second block index, all planes
 * sharing the derivative factors. The launchers are
 * constrained to the SumFac implementation tag, so the identically
 * named SumFacTOP launchers in DivergenceBlockOpDeviceSumFacTOPKernels.hpp
 * are the ones that resolve when that tag is used instead. The whole
 * file is compiled only for the device back-ends (NEKTAR_ENABLE_DEVICE
 * together with DEVICE_COMPILE_ONLY).
 *
 * @see DivergenceBlockOpDeviceSumFac.hpp for the host side configuring and
 * launching these kernels.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 1D SumFac divergence kernel requires: none, since
 * a segment has no collapsed-coordinate tables to stage.
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
inline constexpr unsigned int DivergenceSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 2D SumFac divergence kernel requires.
 *
 * Quadrilaterals need none. The shapes carrying a collapsed
 * coordinate stage their two factor tables in shared memory before
 * the element loop, one entry per quadrature point of the direction
 * each table belongs to.
 *
 * @tparam SHAPE_TYPE            Element shape.
 * @tparam Implementation        Constrained to MultiRegions::SumFac.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Entry count; the caller multiplies by sizeof(TData).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int DivergenceSharedMemorySize(
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
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 3D SumFac divergence kernel requires.
 *
 * Hexahedra need none. The remaining shapes stage the factor tables
 * their collapsed coordinates need: tetrahedra four tables (two of
 * them on direction 1), prisms two and pyramids three, each holding
 * one entry per quadrature point of its direction.
 *
 * @tparam SHAPE_TYPE            Element shape.
 * @tparam Implementation        Constrained to MultiRegions::SumFac.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Entry count; the caller multiplies by sizeof(TData).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int DivergenceSharedMemorySize(
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
 * @brief One element, two dimensions: the divergence
 * \f$\partial u/\partial x + \partial v/\partial y\f$ of the two
 * components at @p in and @p in + @p inoffset, written once per point.
 *
 * Per point each component is differentiated in both reference directions,
 * the triangle correction is applied where the shape requires it, and its
 * derivatives are contracted with its own row of the derivative factors; the
 * sum is kept in a register and stored once.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   ncoord      Cartesian components the factor array holds;
 *                      sets the per-point factor stride
 *                      `ndf = 2 * ncoord`.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   inoffset    Stride from one input component to the next.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   f0,f1       Collapsed-coordinate factor tables (triangles only).
 * @param   df          Derivative factors of this lane's warp.
 * @param   in          First input component, warp interleaved.
 * @param   out         Divergence, warp interleaved.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Divergence2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT D0,
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

            TData div = 0.0;
#pragma unroll
            for (unsigned int c = 0u; c < 2u; ++c)
            {
                const TData *inc = in + c * inoffset;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 +=
                        D0[q * nq0 + i] * inc[warpsize * (nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 +=
                        D1[q * nq1 + j] * inc[warpsize * (nq0 * q + i) + ilane];
                }

                // Moving from collapsed to standard coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                              SHAPE_TYPE == LibUtilities::NodalTri)
                {
                    d0 *= f1[j];
                    d1 += d0 * f0[i];
                }

                // Multiply by derivative factors.
                div += d0 * df[2u * c * warpsize + dfindex] +
                       d1 * df[(2u * c + 1u) * warpsize + dfindex];
            }

            out[index] = div;
        }
    }
}

/**
 * @brief One element, three dimensions: the divergence
 * \f$\partial u/\partial x + \partial v/\partial y
 * + \partial w/\partial z\f$ of the three components at @p in,
 * @p in + @p inoffset and @p in + 2 @p inoffset, written once per point.
 *
 * As Divergence2DSumFacKernel, with the collapsed-coordinate corrections of
 * the three-dimensional shapes and nine derivative factors per point.
 *
 * @tparam SHAPE_TYPE Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam DEFORMED   Derivative factors vary point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   inoffset        Stride from one input component to the next.
 * @param   D0,D1,D2        1D differentiation matrices, column-major.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables, as in
 *                          PhysDeriv3DSumFacKernel.
 * @param   df              Derivative factors of this lane's warp.
 * @param   in              First input component, warp interleaved.
 * @param   out             Divergence, warp interleaved.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void Divergence3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t inoffset, const TData *NEK_RESTRICT D0,
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

                TData div = 0.0;
#pragma unroll
                for (unsigned int c = 0u; c < 3u; ++c)
                {
                    const TData *inc = in + c * inoffset;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
#pragma unroll
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += D0[q * nq0 + i] *
                              inc[warpsize * (nq0 * nq1 * k + nq0 * j + q) +
                                  ilane];
                    }

                    // Direction 1
                    TData d1 = 0.0;
#pragma unroll
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += D1[q * nq1 + j] *
                              inc[warpsize * (nq0 * nq1 * k + nq0 * q + i) +
                                  ilane];
                    }

                    // Direction 2
                    TData d2 = 0.0;
#pragma unroll
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += D2[q * nq2 + k] *
                              inc[warpsize * (nq0 * nq1 * q + nq0 * j + i) +
                                  ilane];
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
                    div += d0 * df[(3u * c) * warpsize + dfindex] +
                           d1 * df[(3u * c + 1u) * warpsize + dfindex] +
                           d2 * df[(3u * c + 2u) * warpsize + dfindex];
                }

                out[index] = div;
            }
        }
    }
}

/**
 * @brief Kernel entry point of the 1D SumFac divergence: unpacks the
 * size parameter, then differentiates one component of the block and
 * scales it by the element's derivative factor for direction 0,
 * overwriting the output.
 *
 * Resolves only when the Implementation tag is SumFac, so that the
 * launch site in DivergenceBlockOpImpl::OperatorNDImpl picks the right
 * kernel family without a further switch; a static_assert requires
 * the size parameter to be one of the 1D physical-space forms --
 * runtime or compile-time, see ElmtHelper.hpp. Launched through
 * DEVICE_2DGRID_KERNEL_LAUNCHER, which appends the shared-memory
 * pointer (unused here) and the back-end's thread-block handle.
 *
 * The thread takes element `e` = global thread index and grid-strides
 * until the block's elements are exhausted; the warp's interleaved
 * data is addressed through `iwarp` and lane `ilane`, and
 * PhysDerivDir1DSumFacKernel differentiates the element's data.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam DEFORMED             Derivative factors vary per quadrature
 *                              point rather than per element.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam1D Element sizes (coordinate dimension and point
 *                      count).
 * @param   nelmt       Elements of one plane of the block, padding
 *                      included.
 * @param   inoffset    Component stride; unused in one dimension.
 * @param   D0          1D derivative matrix.
 * @param   df          Derivative factors of the block,
 *                      warp-interleaved.
 * @param   in          Component data of the block.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block;
 *                      unused in one dimension.
 * @param   threadBlock Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void DivergenceKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    [[maybe_unused]] const size_t inoffset, const TData *NEK_RESTRICT D0,
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

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * (nelmt * p + warpsize * iwarp);
        TData *outptr      = out + nq0 * (nelmt * p + warpsize * iwarp);
        PhysDerivDir1DSumFacKernel<false, DEFORMED, 0>(ilane, ncoord, nq0, D0,
                                                       dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Kernel entry point of the 2D SumFac divergence: binds the
 * dynamic shared memory and accumulates
 * \f$\partial u/\partial x + \partial v/\partial y\f$, one element
 * per thread.
 *
 * Constrained as the 1D launcher, on the SumFac tag alone; a
 * static_assert requires one of the 2D physical-space size
 * parameters. FETCH_SHARED_MEMORY points @p shmemptr at the thread
 * block's dynamic shared memory on the back-ends that provide it
 * that way; the launch site sizes that allocation with
 * DivergenceSharedMemorySize.
 *
 * For the shapes with a collapsed coordinate the two factor tables
 * are first copied into that shared memory, the quadrilateral
 * instantiation leaving the pointers null since its kernels ignore
 * them. Each thread then takes element `e` = global thread index and
 * grid-strides over the block: with the derivative factors, input and
 * output advanced to the thread's warp, Divergence2DSumFacKernel writes
 * \f$\partial u/\partial x + \partial v/\partial y\f$, the second
 * component at @p inoffset, to the output once per point.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam DEFORMED             Derivative factors vary per quadrature
 *                              point rather than per element.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam2D Element sizes (coordinate dimension and point
 *                      counts).
 * @param   nelmt       Elements of one plane of the block, padding
 *                      included.
 * @param   inoffset    Stride from one input component to the next,
 *                      a component spanning all the planes.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   f0          Collapsed-coordinate factor table of
 *                      direction 0; unused for quadrilaterals.
 * @param   f1          Collapsed-coordinate factor table of
 *                      direction 1; unused for quadrilaterals.
 * @param   df          Derivative factors of the block,
 *                      warp-interleaved.
 * @param   in          First component of the block's vector field.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block,
 *                      sized by DivergenceSharedMemorySize.
 * @param   threadBlock Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void DivergenceKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
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

        Divergence2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, D0, D1, s_f0, s_f1, dfptr, inptr,
            outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Kernel entry point of the 3D SumFac divergence: binds the
 * dynamic shared memory and accumulates
 * \f$\partial u/\partial x + \partial v/\partial y
 * + \partial w/\partial z\f$, one element per thread.
 *
 * Constrained as the other launchers, on the SumFac tag alone; a
 * static_assert requires one of the 3D physical-space size
 * parameters.
 *
 * The factor tables the shape's collapsed coordinates need are staged
 * in that shared memory (none for hexahedra), then each thread
 * grid-strides over the elements and calls Divergence3DSumFacKernel,
 * which reads the components at @p in, @p inoffset and twice
 * @p inoffset and writes the output once per point. The element is assumed to
 * have coordinate dimension three, so it carries nine derivative factors.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFac here.
 * @tparam DEFORMED             Derivative factors vary per quadrature
 *                              point rather than per element.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam3D Element sizes (point counts per direction).
 * @param   nelmt       Elements of one plane of the block, padding
 *                      included.
 * @param   inoffset    Stride from one input component to the next,
 *                      a component spanning all the planes.
 * @param   D0          1D derivative matrix of direction 0.
 * @param   D1          1D derivative matrix of direction 1.
 * @param   D2          1D derivative matrix of direction 2.
 * @param   f0          \f$\tfrac{1}{2}(1+\eta_0)\f$ at the direction-0
 *                      points.
 * @param   f1          \f$\tfrac{1}{2}(1+\eta_1)\f$ at the direction-1
 *                      points.
 * @param   f1m         \f$2/(1-\eta_1)\f$ at the direction-1 points.
 * @param   f2          \f$2/(1-\eta_2)\f$ at the direction-2 points.
 *                      All four are unused for hexahedra.
 * @param   df          Derivative factors of the block,
 *                      warp-interleaved.
 * @param   in          First component of the block's vector field.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block,
 *                      sized by DivergenceSharedMemorySize.
 * @param   threadBlock Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void DivergenceKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const size_t inoffset, const TData *NEK_RESTRICT D0,
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
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * p + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * p + warpsize * iwarp);
        Divergence3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2,
            dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
