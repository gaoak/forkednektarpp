///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseBlockOpDeviceSumFacKernels.hpp
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
// Description: SumFac device kernels of the inner product with the
// gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseBlockOpDeviceSumFacKernels.hpp
 * @brief SumFac device kernels of the inner product with the gradient of
 * the basis: one element per thread, with the elements of a warp
 * interleaved lane by lane and the intermediates held in a global-memory
 * workspace.
 *
 * @details
 * These are the kernels the Device/SumFac block implementation launches
 * (see IProductWRTDerivBaseBlockOpDeviceSumFac.hpp); the SumFacTOP
 * counterparts, where a whole thread block cooperates on one element, are in
 * IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp.
 *
 * One thread handles one element. The block storage is reshaped to
 * interleave width warpSize before the launch, so that the elements of a
 * warp are interleaved: element e is handled by lane ilane = e % warpSize
 * of warp iwarp = e / warpSize, and value n of that element sits at offset
 * warpSize * n + ilane within its warp's group. The per-element kernels
 * stride over the elements of the block by the whole grid, which is a
 * whole number of warps -- the launch uses one warp per thread block, see
 * GetDeviceBlockSize -- so a thread keeps its lane index for every element
 * it visits. The workspace and the geometric factors are interleaved the
 * same way, and the grid's y and z dimensions select the homogeneous mode
 * and output component a thread block works on. On a multi-plane 3DH1
 * expansion the input holds three directions per output component, of
 * which only the first coordDim are read here, so the launchers step between
 * output components by three.
 *
 * Each element passes through the stages described in
 * IProductWRTDerivBaseBlockOpSerialAVXSumFac.hpp:
 * -# a StdAlignDerivBaseNDSumFacKernel from this header contracts the
 *    input's coordinate components against the derivative factors,
 *    applies the collapsed-coordinate corrections and multiplies by the
 *    Jacobian and the quadrature weights, leaving one field per reference
 *    direction in the workspace;
 * -# SumDerivTensorNDKernel (PhysDerivBlockOpDeviceSumFacKernels.hpp) applies
 *    each direction's transposed one-dimensional derivative matrix and
 *    sums the directions;
 * -# on the Coeff path an IProductWRTBase...SumFacKernel
 *    (IProductWRTBaseBlockOpDeviceSumFacKernels.hpp) contracts the result
 *    against the one-dimensional basis tables and, for the nodal shapes,
 *    MatVecKernel applies the transposed nodal-to-modal matrix. On the
 *    Phys path stage two writes the output directly.
 * Stage three carries the operator's scale factor and the append flag on
 * the Coeff path; on the Phys path stage two does.
 *
 * The derivative factors are read untransposed, the ncoord * dimension
 * factors of one point adjacent (each warp-interleaved), indexed
 * k * dimension + d; on a regular (non-deformed) block there is one set
 * and one Jacobian per element. IProductWRTDerivBaseWorkSpaceSize and
 * IProductWRTDerivBaseSharedMemorySize report how much global workspace
 * and dynamic shared memory a launch must request. The launchers and
 * helpers are constrained to the SumFac implementation tag, so the
 * identically named SumFacTOP ones in
 * IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp are the ones that
 * resolve when that tag is used instead.
 *
 * Everything in the namespace is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY, so it exists only in translation units compiled for
 * a device back-end.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a one-dimensional shape.
 *
 * One flat global-memory region serves the whole launch: a stage needing
 * s values per element occupies s * nelmt consecutive values,
 * warp-interleaved exactly like the field data, the stages follow one
 * another, and the block operator scales the total by the number of
 * (mode, component) pairs of the grid. In one dimension the Phys output
 * holds the single direction field, nq0 values per element, and the Coeff
 * output that field plus its transposed derivative, 2 * nq0.
 *
 * The output field state is encoded in the size-parameter type: a
 * physical-space size parameter selects the Phys variant, a modal one the
 * Coeff variant.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam1D  Sizes of the expansion.
 *
 * @return Workspace requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (IsPhysSizeParameter1D_v<TSizeParameter1D>)
    {
        wspsize = nq0 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Seg)
        {
            wspsize = 2 * nq0 * nelmt;
        }
    }

    return wspsize;
}

/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a two-dimensional shape.
 *
 * Laid out as in the 1D overload, the output field state again encoded in
 * the size-parameter type. Per element that is
 * - Phys: the two direction fields, 2 * nq0 * nq1 values, since the summed
 *   transposed derivative goes straight to the output;
 * - Coeff: those two plus the summed field, 3 * nq0 * nq1, then the nq1
 *   values the IProductWRTBase stage of the quadrilateral and triangle
 *   uses; the nodal triangle adds the nmTot values of the coefficients it
 *   converts from.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam2D  Sizes per direction.
 *
 * @return Workspace requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        wspsize = 2 * nq0 * nq1 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            wspsize = (3 * nq0 * nq1 + nq1) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            wspsize = (3 * nq0 * nq1 + nq1 + nmTot) * nelmt;
        }
    }

    return wspsize;
}

/**
 * @brief Global workspace a SumFac launch needs, in elements of TData, for
 * a three-dimensional shape.
 *
 * Laid out as in the 1D overload, the output field state again encoded in
 * the size-parameter type. Per element that is
 * - Phys: the three direction fields, 3 * nq0 * nq1 * nq2 values;
 * - Coeff: the three direction fields plus the summed field, then the
 *   nq1 * nq2 and nq2 values of the two IProductWRTBase stages; the nodal
 *   shapes add their total mode count nmTot.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam3D  Sizes per direction.
 *
 * @return Workspace requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        wspsize = 3 * nq0 * nq1 * nq2 * nelmt;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 + nmTot) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 + nmTot) * nelmt;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
        }
    }

    return wspsize;
}

/**
 * @brief Dynamic shared memory a SumFac launch needs per thread block, in
 * elements of TData, for a one-dimensional shape: none.
 *
 * The SumFac kernels hold their intermediates in the global-memory
 * workspace and read the tables straight from global memory; only the
 * collapsed-coordinate factor tables of the simplex shapes are staged, and
 * a segment has none. The SumFacTOP overloads of the same name
 * (IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp) return the size of
 * the staging area that strategy does use.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TData             Floating-point type of the field data; unused.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Sizes of the expansion; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory a SumFac launch needs per thread block, in
 * elements of TData, for a two-dimensional shape.
 *
 * Zero for the quadrilateral; for the triangles the two
 * collapsed-coordinate factor tables, nq0 + nq1 values, which the
 * launchers stage once per thread block because every element reads them.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TData             Floating-point type of the field data; unused.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
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
 * Zero for the hexahedron; otherwise the collapsed-coordinate factor
 * tables the shape uses, staged once per thread block: nq0 + nq2 for the
 * prisms, nq0 + nq1 + nq2 for the pyramid and nq0 + 2 * nq1 + nq2 for the
 * tetrahedra, which need both direction-1 tables.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFac.
 * @tparam TData             Floating-point type of the field data; unused.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
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
        return nq0 + 2 * nq1 + nq2;
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
 * @brief Segment geometric stage: contract the input vector with the
 * derivative factors and apply the quadrature metric, for one element.
 *
 * Writes, for each point i of the calling lane's element,
 * \f$ g_0 = w_0[i]\, J \sum_k (\partial\xi_0/\partial x_k) f_k \f$. All
 * pointers address the lane's warp group, so the point index is scaled by
 * the warp size and offset by @p ilane.
 *
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam TData       Floating-point type of the field data.
 *
 * @param   ilane       Lane index of the element within its warp.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   nq0         Quadrature points of the element.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   w0          Quadrature weights of the element.
 * @param   df          Derivative factors of the warp group.
 * @param   jac         Jacobians of the warp group.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   out         Receives \f$g_0\f$ in the workspace.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            sum += df[k * warpsize + dfindex] * in[k * inoffset + index];
        }

        if constexpr (DEFORMED)
        {
            out[index] = sum * jac[index] * w0[i];
        }
        else
        {
            out[index] = sum * jac[0] * w0[i];
        }
    }
}

/**
 * @brief Quadrilateral or triangle geometric stage, for one element.
 *
 * Writes the two weighted direction fields in point order (direction 0
 * fastest); for the triangles the collapsed correction is applied to the
 * first before the weighting, using the shared-memory factor tables @p f0
 * and @p f1, which are unused for quadrilaterals.
 *
 * @tparam SHAPE_TYPE  Shape of the block's elements.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam TData       Floating-point type of the field data.
 *
 * @param   ilane       Lane index of the element within its warp.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read (2 or 3).
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0          Table of (1 + eta_0)/2 (triangles).
 * @param   f1          Table of 2/(1 - eta_1) (triangles).
 * @param   df          Derivative factors of the warp group, 2 * ncoord
 *                      per point.
 * @param   jac         Jacobians of the warp group.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   out0,out1   Receive the two direction fields.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1)
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

            TData tmp  = in[index];
            TData sum1 = df[0u * warpsize + dfindex] * tmp;
            TData sum2 = df[1u * warpsize + dfindex] * tmp;
            tmp        = in[inoffset + index];
            sum1 += df[2u * warpsize + dfindex] * tmp;
            sum2 += df[3u * warpsize + dfindex] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + index];
                sum1 += df[4u * warpsize + dfindex] * tmp;
                sum2 += df[5u * warpsize + dfindex] * tmp;
            }

            TData tmpQ = w0[i] * w1[j];
            if constexpr (DEFORMED)
            {
                tmpQ *= jac[index];
            }
            else
            {
                tmpQ *= jac[0];
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                out0[index] = sum1 * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                               SHAPE_TYPE == LibUtilities::NodalTri)
            {
                out0[index] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
        }
    }
}

/**
 * @brief Three-dimensional geometric stage, for one element.
 *
 * Writes the three weighted direction fields in point order. A 3D
 * element's coordinate dimension is always three, so three input
 * components are read and the derivative factors have nine entries per
 * point. The number of additive collapsed correction terms differs per
 * shape -- none for the hexahedron, one for the prisms, two for the
 * pyramid and the tetrahedra -- and they are applied before the
 * weighting. That count is not the number of factor tables a shape reads,
 * which is at least as large and which IProductWRTDerivBaseSharedMemorySize
 * sizes; the tables a shape does not read are ignored.
 *
 * @tparam SHAPE_TYPE  Shape of the block's elements.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam TData       Floating-point type of the field data.
 *
 * @param   ilane           Lane index of the element within its warp.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   w0,w1,w2        Quadrature weights per direction.
 * @param   f0              Table of (1 + eta_0)/2.
 * @param   f1              Table of (1 + eta_1)/2.
 * @param   f1m             Table of 2/(1 - eta_1).
 * @param   f2              Table of 2/(1 - eta_2).
 * @param   df              Derivative factors of the warp group, nine per
 *                          point.
 * @param   jac             Jacobians of the warp group.
 * @param   in              Input vector \f${\bf f}\f$.
 * @param   out0,out1,out2  Receive the three direction fields.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1,
    TData *NEK_RESTRICT out2)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                TData tmp  = in[index];
                TData sum1 = df[0u * warpsize + dfindex] * tmp;
                TData sum2 = df[1u * warpsize + dfindex] * tmp;
                TData sum3 = df[2u * warpsize + dfindex] * tmp;
                tmp        = in[inoffset + index];
                sum1 += df[3u * warpsize + dfindex] * tmp;
                sum2 += df[4u * warpsize + dfindex] * tmp;
                sum3 += df[5u * warpsize + dfindex] * tmp;
                tmp = in[2u * inoffset + index];
                sum1 += df[6u * warpsize + dfindex] * tmp;
                sum2 += df[7u * warpsize + dfindex] * tmp;
                sum3 += df[8u * warpsize + dfindex] * tmp;

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[index];
                }
                else
                {
                    tmpQ *= jac[0];
                }

                if constexpr (SHAPE_TYPE == LibUtilities::Hex)
                {
                    out0[index] = sum1 * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                                   SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    out0[index] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + sum3 * f0[i]) * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
            }
        }
    }
}

/**
 * @brief SumFac segment launcher, Coeff output: unpacks the element sizes
 * and computes, for every element of the block, the inner product of the
 * input vector with the derivative of the basis.
 *
 * Each thread grid-strides over the elements, slicing the geometric
 * factors, the field data and the workspace for its warp and its (mode,
 * component) pair, and runs the three stages: the geometric stage into
 * the first workspace plane, SumDerivTensor1DKernel into the second, and
 * IProductWRTBaseSegSumFacKernel from there onto the output, where the
 * scale and the append flag take effect.
 *
 * The enable_if constrains this definition to the SumFac tag, so the
 * identically named SumFacTOP launcher in
 * IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp resolves for that tag
 * instead; the two share one signature, which is why some arguments go
 * unused on one side or the other.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements (Seg).
 * @tparam Implementation      MultiRegions::SumFac.
 * @tparam APPEND              Accumulate onto the output instead of
 *                             overwriting it.
 * @tparam DEFORMED            True when the geometric factors vary within
 *                             an element.
 * @tparam TSizeParameter1D    Runtime or compile-time element sizes.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam1D Sizes of the expansion.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      One-dimensional basis table of the expansion.
 * @param   D0          One-dimensional derivative matrix.
 * @param   w0          Quadrature weights of the expansion.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector \f${\bf f}\f$.
 * @param   out         Coefficient-space output.
 * @param   wsp         Global-memory workspace, sized by
 *                      IProductWRTDerivBaseWorkSpaceSize.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory; unused here.
 * @param   threadBlock Thread-block handle supplying the index and range
 *                      helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nm0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + nq0 * nelmt * nmode * ncomp +
                       nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase1DSumFacKernel<DEFORMED>(
            ilane, ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv0);
        SumDerivTensor1DKernel<false, false>(ilane, nq0, D0, deriv0, deriv,
                                             scale);
        IProductWRTBaseSegSumFacKernel<true, APPEND>(ilane, nm0, nq0, basis0,
                                                     deriv, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac segment launcher, Phys output: unpacks the element sizes
 * and leaves the same weak derivative at the quadrature points.
 *
 * As the Coeff launcher, but the geometric stage is followed by
 * SumDerivTensor1DKernel writing straight to the output -- applying the
 * scale and the append flag -- instead of being contracted against the
 * basis, so only the direction field is held in the workspace. There is
 * no coordinate-dimension argument: the physical-space size parameter
 * @p sizeParam1D supplies it.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements (Seg).
 * @tparam Implementation          MultiRegions::SumFac.
 * @tparam APPEND                  Accumulate onto the output instead of
 *                                 overwriting it.
 * @tparam DEFORMED                True when the geometric factors vary
 *                                 within an element.
 * @tparam TPhysSizeParameter1D    Runtime or compile-time element sizes,
 *                                 physical-space output.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 *
 * @param   sizeParam1D Sizes of the expansion.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   D0          One-dimensional derivative matrix.
 * @param   w0          Quadrature weights of the expansion.
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector \f${\bf f}\f$.
 * @param   out         Physical-space output.
 * @param   wsp         Global-memory workspace.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory; unused here.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter1D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase1DSumFacKernel<DEFORMED>(
            ilane, ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv);
        SumDerivTensor1DKernel<true, APPEND>(ilane, nq0, D0, deriv, outptr,
                                             scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac quadrilateral or triangle launcher, Coeff output: unpacks
 * the element sizes and computes the inner product.
 *
 * The block's threads first stage the collapsed-coordinate factor tables
 * of the triangles in shared memory, then grid-stride over the elements
 * running the three stages: StdAlignDerivBase2DSumFacKernel into two
 * workspace planes, SumDerivTensor2DKernel into a third, and the
 * IProductWRTBase kernel of the shape onto the output. The nodal triangle
 * writes its coefficients to a further workspace slice and finishes with
 * MatVecKernel applying the transposed nodal-to-modal matrix, which is
 * where its append flag takes effect.
 *
 * The mode-index table @p index0 is unused here: it is part of the
 * signature the SumFacTOP launcher shares, whose kernels use it to locate
 * a thread's mode in the collapsed ordering.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements.
 * @tparam Implementation      MultiRegions::SumFac.
 * @tparam APPEND              Accumulate onto the output instead of
 *                             overwriting it.
 * @tparam DEFORMED            True when the geometric factors vary within
 *                             an element.
 * @tparam TSizeParameter2D    Runtime or compile-time element sizes.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam2D     Sizes per direction.
 * @param   ncoord          Coordinate dimension (2 or 3).
 * @param   nelmt           Elements in the block, padding included.
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   isModified      First-direction basis is eModified_A.
 * @param   index0          Mode-index table; unused here.
 * @param   basis0,basis1   One-dimensional basis tables per direction.
 * @param   D0,D1           One-dimensional derivative matrices per
 *                          direction.
 * @param   w0,w1           Quadrature weights per direction.
 * @param   f0,f1           Collapsed-coordinate factor tables (triangles).
 * @param   nodToMod        Nodal-to-modal matrix (nodal triangle).
 * @param   df              Derivative factors of the block.
 * @param   jac             Jacobians of the block.
 * @param   in              Physical-space input vector.
 * @param   out             Coefficient-space output.
 * @param   wsp             Global-memory workspace.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory for the factor tables.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Pre-compute factor.
    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        const unsigned int idx0   = getLocalIdx<0>(threadBlock);
        const unsigned int stride = getLocalRange<0>(threadBlock);

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

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                       nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, w0, w1, s_f0, s_f1, dfptr,
            jacptr, inptr, deriv0, deriv1);
        SumDerivTensor2DKernel<false, false>(ilane, nq0, nq1, D0, D1, deriv0,
                                             deriv1, deriv, scale);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, deriv, outptr, wsp0,
                scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, deriv,
                outptr, wsp0, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr =
                wsp + 3 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 3 * nqTot) * nelmt * nmode * ncomp +
                          nq1 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<true, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, deriv,
                out1ptr, wsp0, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac quadrilateral or triangle launcher, Phys output: unpacks
 * the element sizes and leaves the derivative at the quadrature points.
 *
 * As the Coeff launcher without the basis contraction:
 * SumDerivTensor2DKernel writes the summed transposed derivative straight
 * to the output, applying the scale and the append flag, so only the two
 * direction fields are held in the workspace. The basis tables, the
 * mode-index table and the nodal-to-modal matrix are absent from this
 * signature: the Phys path stops before the basis contraction.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements.
 * @tparam Implementation          MultiRegions::SumFac.
 * @tparam APPEND                  Accumulate onto the output instead of
 *                                 overwriting it.
 * @tparam DEFORMED                True when the geometric factors vary
 *                                 within an element.
 * @tparam TPhysSizeParameter2D    Runtime or compile-time element sizes,
 *                                 physical-space output.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 *
 * @param   sizeParam2D Sizes per direction.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   D0,D1       One-dimensional derivative matrices per direction.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0,f1       Collapsed-coordinate factor tables (triangles).
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector.
 * @param   out         Physical-space output.
 * @param   wsp         Global-memory workspace.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory for the factor tables.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter2D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Pre-compute factor.
    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        const unsigned int idx0   = getLocalIdx<0>(threadBlock);
        const unsigned int stride = getLocalRange<0>(threadBlock);

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

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * (nelmt * (inDim * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, inoffset, w0, w1, s_f0, s_f1, dfptr,
            jacptr, inptr, deriv0, deriv1);
        SumDerivTensor2DKernel<true, APPEND>(ilane, nq0, nq1, D0, D1, deriv0,
                                             deriv1, outptr, scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac three-dimensional launcher, Coeff output: unpacks the
 * element sizes and computes the inner product.
 *
 * The 3D form of the quadrilateral/triangle launcher: the shape's
 * collapsed-coordinate factor tables are staged in shared memory, then
 * each thread grid-strides over the elements running the geometric stage,
 * SumDerivTensor3DKernel and the IProductWRTBase kernel of the shape, the
 * nodal shapes finishing with MatVecKernel applying the transposed
 * nodal-to-modal matrix. There is no coordinate-dimension argument -- a 3D
 * element's is always three -- and the mode-index tables @p index0 ...
 * @p index2 are unused here, being part of the signature shared with the
 * SumFacTOP launcher.
 *
 * @note The workspace opens with four planes of
 * nqTot * nelmt * nmode * ncomp values (nmode and ncomp being the grid's y
 * and z ranges) holding deriv0, deriv1, deriv2 and deriv; the shape's
 * wsp0, and its wsp1 or out1ptr where it has one, follow at their smaller
 * per-shape widths, every base scaled by the same nelmt * nmode * ncomp
 * factor.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements.
 * @tparam Implementation      MultiRegions::SumFac.
 * @tparam APPEND              Accumulate onto the output instead of
 *                             overwriting it.
 * @tparam DEFORMED            True when the geometric factors vary within
 *                             an element.
 * @tparam TSizeParameter3D    Runtime or compile-time element sizes.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam3D             Sizes per direction.
 * @param   ncoord                  Coordinate dimension; unused.
 * @param   nelmt                   Elements in the block, padding
 *                                  included.
 * @param   inoffset                Stride between the input's coordinate
 *                                  components.
 * @param   isModified              First-direction basis is eModified_A.
 * @param   index0,index1,index2    Mode-index tables; unused here.
 * @param   basis0,basis1,basis2    One-dimensional basis tables per
 *                                  direction.
 * @param   D0,D1,D2                One-dimensional derivative matrices
 *                                  per direction.
 * @param   w0,w1,w2                Quadrature weights per direction.
 * @param   f0,f1,f1m,f2            Collapsed-coordinate factor tables.
 * @param   nodToMod                Nodal-to-modal matrix (nodal shapes).
 * @param   df                      Derivative factors of the block.
 * @param   jac                     Jacobians of the block.
 * @param   in                      Physical-space input vector.
 * @param   out                     Coefficient-space output.
 * @param   wsp                     Global-memory workspace.
 * @param   scale                   Factor applied to the result.
 * @param   shmemptr                Dynamic shared memory for the factor
 *                                  tables.
 * @param   threadBlock             Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const size_t inoffset, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, TData scale, unsigned char *shmemptr,
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

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Pre-compute factor.
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
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr =
            in + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv2 = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv = wsp + 3 * nqTot * nelmt * nmode * ncomp +
                       nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, w0, w1, w2, s_f0, s_f1, s_f1m, s_f2,
            dfptr, jacptr, inptr, deriv0, deriv1, deriv2);
        SumDerivTensor3DKernel<false, false>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                             deriv0, deriv1, deriv2, deriv,
                                             scale);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *out1ptr =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 =
                wsp + (4 * nqTot + nmTot) * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 =
                wsp + (4 * nqTot + nq1 * nq2 + nmTot) * nelmt * nmode * ncomp +
                nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<true, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nmTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp0 =
                wsp + (4 * nqTot + nmTot) * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 =
                wsp + (4 * nqTot + nq1 * nq2 + nmTot) * nelmt * nmode * ncomp +
                nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<true, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * nmode * ncomp +
                nq1 * nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (4 * nqTot + nq1 * nq2) * nelmt * nmode * ncomp +
                          nq2 * (nelmt * (nmode * c + m) + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<true, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, deriv, outptr, wsp0, wsp1, scale);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief SumFac three-dimensional launcher, Phys output: unpacks the
 * element sizes and leaves the derivative at the quadrature points.
 *
 * As the Coeff launcher without the basis contraction:
 * SumDerivTensor3DKernel writes the summed transposed derivative straight
 * to the output, applying the scale and the append flag, so only the three
 * direction fields are held in the workspace. The basis tables, the
 * mode-index tables, the nodal-to-modal matrix and the modified-basis flag
 * are absent from this signature, all of which serve the basis
 * contraction the Phys path does not reach.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements.
 * @tparam Implementation          MultiRegions::SumFac.
 * @tparam APPEND                  Accumulate onto the output instead of
 *                                 overwriting it.
 * @tparam DEFORMED                True when the geometric factors vary
 *                                 within an element.
 * @tparam TPhysSizeParameter3D    Runtime or compile-time element sizes,
 *                                 physical-space output.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 *
 * @param   sizeParam3D     Sizes per direction.
 * @param   nelmt           Elements in the block, padding included.
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   D0,D1,D2        One-dimensional derivative matrices per
 *                          direction.
 * @param   w0,w1,w2        Quadrature weights per direction.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables.
 * @param   df              Derivative factors of the block.
 * @param   jac             Jacobians of the block.
 * @param   in              Physical-space input vector.
 * @param   out             Physical-space output.
 * @param   wsp             Global-memory workspace.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory for the factor tables.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter3D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
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
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Pre-compute factor.
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
    const unsigned int ncomp = getBlockRange<2>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr =
            in + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv0 =
            wsp + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *deriv2 = wsp + 2 * nqTot * nelmt * nmode * ncomp +
                        nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);

        StdAlignDerivBase3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, inoffset, w0, w1, w2, s_f0, s_f1, s_f1m, s_f2,
            dfptr, jacptr, inptr, deriv0, deriv1, deriv2);
        SumDerivTensor3DKernel<true, APPEND>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                             deriv0, deriv1, deriv2, outptr,
                                             scale);

        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
