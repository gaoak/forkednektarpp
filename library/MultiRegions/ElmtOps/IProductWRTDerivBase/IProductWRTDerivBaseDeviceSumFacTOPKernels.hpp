///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceSumFacTOPKernels.hpp
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
// Description: SumFacTOP device kernels of the inner product with the
// gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the inner product with the gradient
 * of the basis: one element per thread block, with the intermediates
 * staged in shared memory.
 *
 * @details
 * These kernels compute the same thing as the SumFac ones in
 * IProductWRTDerivBaseDeviceSumFacKernels.hpp but distribute the work
 * differently. A whole thread block takes one element at a time and
 * strides over the block's elements by the grid, the grid's y and z
 * dimensions selecting the homogeneous mode and output component (on a
 * multi-plane 3DH1 expansion the input holds three directions per output
 * component, of which only the first coordDim are read here); within
 * an element the block's threads stride over the work items of each stage
 * -- the element's quadrature points in the geometric and derivative
 * stages, its modes in the basis contraction. The field data is therefore
 * not interleaved (interleave width 1): an element's values are
 * contiguous, and the kernels index them directly. Each stage ends with a
 * local barrier, so the intermediates one stage writes to shared memory
 * are visible to the next.
 *
 * Because the intermediates live in shared memory, these kernels need no
 * global-memory workspace at all -- the IProductWRTDerivBaseWorkSpaceSize
 * overloads below return zero -- and the launch instead requests the
 * dynamic shared memory reported by IProductWRTDerivBaseSharedMemorySize,
 * which also holds the one-dimensional basis or derivative tables the
 * launchers copy in once per thread block.
 *
 * The stages are those of the SumFac kernels: the geometric stage of this
 * header, the transposed derivative (SumDerivTensorNDSumFacTOPKernel,
 * PhysDerivDeviceSumFacTOPKernels.hpp) and, on the Coeff path, the basis
 * contraction (IProductWRTBaseDeviceSumFacTOPKernels.hpp) with the nodal
 * shapes finishing on MatVecSumFacTOPKernel and the transposed
 * nodal-to-modal matrix. The derivative factors are read from the
 * transposed DerivFactorKey layout, in which the values of one factor
 * over an element's points are contiguous; on a regular (non-deformed)
 * block there is one set of factors and one Jacobian per element. The
 * launchers and helpers are constrained to the SumFacTOP implementation
 * tag and are otherwise identical in name and signature to the SumFac
 * ones, which is how IProductWRTDerivBaseBlockOpImpl::OperatorNDImpl can
 * launch either strategy from one call site.
 *
 * Everything in the namespace is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY, so it exists only in translation units compiled for
 * a device back-end.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a one-dimensional shape: none, the intermediates being held in
 * shared memory.
 *
 * No overload of this helper requests global workspace for the SumFacTOP
 * strategy; the SumFac overloads in
 * IProductWRTDerivBaseDeviceSumFacKernels.hpp do.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block; unused.
 * @param   sizeParam1D  Sizes of the expansion; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Global workspace a SumFacTOP launch needs for a two-dimensional
/// shape: none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Global workspace a SumFacTOP launch needs for a
/// three-dimensional shape: none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a one-dimensional shape.
 *
 * The Phys output holds the single direction field, nq0 values; the Coeff
 * output that field plus its transposed derivative, 2 * nq0. The tables
 * the 1D kernels contract against are read straight from global memory.
 * The output field state is encoded in the size-parameter type: a
 * physical-space size parameter selects the Phys variant, a modal one the
 * Coeff variant.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Sizes of the expansion.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (IsPhysSizeParameter1D_v<TSizeParameter1D>)
    {
        return nq0;
    }
    else
    {
        return 2 * nq0;
    }
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a two-dimensional shape.
 *
 * The staging area a thread block claims, in the order the launchers
 * carve it up:
 * - Phys: the two direction fields, then the two one-dimensional
 *   derivative matrices, 2 * nq0 * nq1 + nq0 * nq0 + nq1 * nq1.
 * - Coeff: the two direction fields and their summed transposed derivative
 *   (3 * nq0 * nq1), the nm0 * nq1 intermediate of the basis contraction,
 *   and the two one-dimensional basis tables -- the second of which is
 *   indexed by the shape's total mode count for the triangles, whose
 *   second direction carries the collapsed mode ordering. The nodal
 *   triangle adds its coefficient vector, nmTot.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   sizeParam2D  Sizes per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        return 2 * nq0 * nq1 + nq0 * nq0 + nq1 * nq1;
    }
    else
    {
        const unsigned int nm0 = sizeParam2D.nm0();
        const unsigned int nm1 = sizeParam2D.nm1();

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return nm0 * nq0 + nm1 * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            return nm0 * nq0 + nmTot * nq1 + 3 * nq0 * nq1 + nm0 * nq1 + nmTot;
        }
    }
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a three-dimensional shape.
 *
 * Laid out as in the 2D overload:
 * - Phys: the three direction fields and the three one-dimensional
 *   derivative matrices.
 * - Coeff: four fields of nqTot -- the three direction fields and their
 *   summed transposed derivative -- the nm0 * nq1 * nq2 and second
 *   intermediate of the basis contraction, and the three one-dimensional
 *   basis tables. The mode counts indexing the second and third tables
 *   follow the shape's collapsed mode ordering, which is why they are
 *   computed rather than simply nm1 and nm2; the nodal shapes add their
 *   coefficient vector of nmTot values.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   sizeParam3D  Sizes per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
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

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        return 3 * nq0 * nq1 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2;
    }
    else
    {
        const unsigned int nm0 = sizeParam3D.nm0();
        const unsigned int nm1 = sizeParam3D.nm1();
        const unsigned int nm2 = sizeParam3D.nm2();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm01 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm01 * nq2 + nmTot;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nmTot;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
                   nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
    }
}

/**
 * @brief Segment geometric stage: contract the input vector with the
 * derivative factors and apply the quadrature metric, for one element.
 *
 * The calling block's threads stride over the element's points, writing
 * \f$ g_0 = w_0[i]\, J \sum_k (\partial\xi_0/\partial x_k) f_k \f$ into
 * shared memory, and synchronise before returning.
 *
 * @tparam DEFORMED        True when the geometric factors vary within an
 *                         element.
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   nq0         Quadrature points of the element.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   w0          Quadrature weights of the element.
 * @param   df          Derivative factors of the element.
 * @param   jac         Jacobian(s) of the element.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   out         Receives \f$g_0\f$ in shared memory.
 * @param   threadBlock Thread-block handle supplying the index, range and
 *                      barrier helpers.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t inoffset,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            sum += df[k * dfsize + dfindex] * in[k * inoffset + i];
        }

        if constexpr (DEFORMED)
        {
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
            out[i] = sum * jac[0] * w0[i];
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Quadrilateral or triangle geometric stage, for one element.
 *
 * The block's threads stride over the element's points, each recovering
 * its (i, j) index pair, and write the two weighted direction fields into
 * shared memory; for the triangles the collapsed correction is applied to
 * the first before the weighting, using @p f0 and @p f1, which are unused
 * for quadrilaterals. Ends with a local barrier.
 *
 * @tparam SHAPE_TYPE      Shape of the block's elements.
 * @tparam DEFORMED        True when the geometric factors vary within an
 *                         element.
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   ncoord      Coordinate dimension (2 or 3).
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0          Table of (1 + eta_0)/2 (triangles).
 * @param   f1          Table of 2/(1 - eta_1) (triangles).
 * @param   df          Derivative factors of the element, 2 * ncoord
 *                      fields.
 * @param   jac         Jacobian(s) of the element.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   out0,out1   Receive the two direction fields.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1,
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

        TData tmp  = in[idx];
        TData sum1 = df[0u * dfsize + dfindex] * tmp;
        TData sum2 = df[1u * dfsize + dfindex] * tmp;
        tmp        = in[inoffset + idx];
        sum1 += df[2u * dfsize + dfindex] * tmp;
        sum2 += df[3u * dfsize + dfindex] * tmp;
        if (ncoord == 3u)
        {
            tmp = in[2u * inoffset + idx];
            sum1 += df[4u * dfsize + dfindex] * tmp;
            sum2 += df[5u * dfsize + dfindex] * tmp;
        }

        TData tmpQ = w0[i] * w1[j];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            out0[idx] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Three-dimensional geometric stage, for one element.
 *
 * The block's threads stride over the element's points, each recovering
 * its (i, j, k) index triple, and write the three weighted direction
 * fields into shared memory. A 3D element's coordinate dimension is always
 * three, so three input components are read and nine derivative factors
 * are used. The number of additive collapsed correction terms differs per
 * shape -- none for the hexahedron, one for the prisms, two for the
 * pyramid and the tetrahedra -- and they are applied before the
 * weighting; that count is not the number of factor tables a shape reads,
 * which is larger. Ends with a local barrier.
 *
 * @tparam SHAPE_TYPE      Shape of the block's elements.
 * @tparam DEFORMED        True when the geometric factors vary within an
 *                         element.
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   w0,w1,w2        Quadrature weights per direction.
 * @param   f0              Table of (1 + eta_0)/2.
 * @param   f1              Table of (1 + eta_1)/2.
 * @param   f1m             Table of 2/(1 - eta_1).
 * @param   f2              Table of 2/(1 - eta_2).
 * @param   df              Derivative factors of the element, nine
 *                          fields.
 * @param   jac             Jacobian(s) of the element.
 * @param   in              Input vector \f${\bf f}\f$.
 * @param   out0,out1,out2  Receive the three direction fields.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void StdAlignDerivBase3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT out2,
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

        TData tmp  = in[idx];
        TData sum1 = df[0u * dfsize + dfindex] * tmp;
        TData sum2 = df[1u * dfsize + dfindex] * tmp;
        TData sum3 = df[2u * dfsize + dfindex] * tmp;
        tmp        = in[inoffset + idx];
        sum1 += df[3u * dfsize + dfindex] * tmp;
        sum2 += df[4u * dfsize + dfindex] * tmp;
        sum3 += df[5u * dfsize + dfindex] * tmp;
        tmp = in[2u * inoffset + idx];
        sum1 += df[6u * dfsize + dfindex] * tmp;
        sum2 += df[7u * dfsize + dfindex] * tmp;
        sum3 += df[8u * dfsize + dfindex] * tmp;

        TData tmpQ = w0[i] * w1[j] * w2[k];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            out0[idx] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + sum3 * f0[i]) * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief SumFacTOP segment launcher, Coeff output: unpacks the element
 * sizes and computes, for every element of the block, the inner product
 * of the input vector with the derivative of the basis.
 *
 * The thread block strides over the elements of the block, slicing the
 * geometric factors and the field data for the element it is on and for
 * its (mode, component) pair, and runs the three stages: the geometric
 * stage into the first half of the shared-memory allocation,
 * SumDerivTensor1DSumFacTOPKernel into the second, and
 * IProductWRTBaseSegSumFacTOPKernel from there onto the output, where the
 * scale and the append flag take effect.
 *
 * The enable_if on the Implementation tag constrains this definition to
 * SumFacTOP, so the identically named SumFac launcher in
 * IProductWRTDerivBaseDeviceSumFacKernels.hpp resolves for that tag
 * instead. The two share one signature, which is why the global-memory
 * workspace @p wsp is accepted and ignored here -- this strategy keeps
 * its intermediates in shared memory.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements (Seg).
 * @tparam Implementation      MultiRegions::SumFacTOP.
 * @tparam APPEND              Accumulate onto the output instead of
 *                             overwriting it.
 * @tparam DEFORMED            True when the geometric factors vary within
 *                             an element.
 * @tparam TSizeParameter1D    Runtime or compile-time element sizes.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam1D Sizes of the expansion.
 * @param   ncoord      Coordinate dimension.
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
 * @param   wsp         Global-memory workspace; unused here.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory holding the direction field
 *                      and its transposed derivative.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
        [[maybe_unused]] TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv  = deriv0 + nq0;

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
        TData *outptr = out + nm0 * nelmt * (nmode * c + m) + nm0 * e;

        StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, inoffset, w0,
                                                     dfptr, jacptr, inptr,
                                                     deriv0, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<false, false>(nq0, D0, deriv0, deriv,
                                                      scale, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<true, APPEND>(
            nm0, nq0, basis0, deriv, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP segment launcher, Phys output: unpacks the element
 * sizes and leaves the same weak derivative at the quadrature points.
 *
 * As the Coeff launcher, but the geometric stage is followed by
 * SumDerivTensor1DSumFacTOPKernel writing straight to the output,
 * applying the scale and the append flag. There is no
 * coordinate-dimension argument: the physical-space size parameter
 * @p sizeParam1D supplies it. See the Coeff launcher above for the
 * SumFac/SumFacTOP dispatch scheme.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements (Seg).
 * @tparam Implementation          MultiRegions::SumFacTOP.
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
 * @param   in          Physical-space input vector.
 * @param   out         Physical-space output.
 * @param   wsp         Global-memory workspace; unused here.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory holding the direction field.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter1D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
        const size_t inoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *deriv = (TData *)shmemptr;

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
        TData *outptr = out + nq0 * nelmt * (nmode * c + m) + nq0 * e;

        StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, inoffset, w0,
                                                     dfptr, jacptr, inptr,
                                                     deriv, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<true, APPEND>(nq0, D0, deriv, outptr,
                                                      scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP quadrilateral or triangle launcher, Coeff output:
 * unpacks the element sizes and computes the inner product.
 *
 * The dynamic shared-memory allocation is carved into the two direction
 * fields, their summed transposed derivative, the intermediate of the
 * basis contraction and the two one-dimensional basis tables (plus, for
 * the nodal triangle, its coefficient vector), in that order and with the
 * sizes IProductWRTDerivBaseSharedMemorySize accounts for; the locals
 * nmode0 and nmode1 hold the row counts of the two basis tables, which
 * for the triangles follow the collapsed mode ordering. The block's
 * threads copy the basis tables in, then stride over the elements running
 * the three stages; the nodal triangle writes its coefficients to shared
 * memory and finishes with MatVecSumFacTOPKernel applying the transposed
 * nodal-to-modal matrix, which is where its append flag takes effect.
 *
 * Unlike the SumFac launcher of the same name, this one does use the
 * mode-index table @p index0 -- for the triangles, whose basis
 * contraction needs it to map a thread's flat mode index back to the
 * collapsed ordering. For the quadrilateral it is null and no kernel reads
 * it.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements.
 * @tparam Implementation      MultiRegions::SumFacTOP.
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
 * @param   index0          Mode-index table of the collapsed ordering
 *                          (triangles).
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
 * @param   wsp             Global-memory workspace; unused here.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory, carved up as above.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const size_t inoffset, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *deriv0    = (TData *)shmemptr;
    TData *deriv1    = deriv0 + nqTot;
    TData *deriv     = deriv1 + nqTot;
    TData *s_wsp0    = deriv + nqTot;
    TData *s_basis0  = s_wsp0 + offset;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_out1ptr = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * nelmt * (inDim * nmode * c + m) + nqTot * e;
        TData *outptr = out + nmTot * nelmt * (nmode * c + m) + nmTot * e;

        StdAlignDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr, inptr,
            deriv0, deriv1, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<false, false>(
            nq0, nq1, D0, D1, deriv0, deriv1, deriv, scale, threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, deriv,
                outptr, s_wsp0, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, outptr, s_wsp0, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<true, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, deriv, s_out1ptr, s_wsp0, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP quadrilateral or triangle launcher, Phys output:
 * unpacks the element sizes and leaves the derivative at the quadrature
 * points.
 *
 * As the Coeff launcher without the basis contraction: the shared memory
 * holds the two direction fields and the two one-dimensional derivative
 * matrices, which the block's threads stage there because every element
 * reads them, and SumDerivTensor2DSumFacTOPKernel writes the summed
 * transposed derivative straight to the output, applying the scale and
 * the append flag. The basis tables, the mode-index table and the
 * nodal-to-modal matrix are absent from this signature: the Phys path
 * stops before the basis contraction.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements.
 * @tparam Implementation          MultiRegions::SumFacTOP.
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
 * @param   wsp         Global-memory workspace; unused here.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory, carved up as above.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv1 = deriv0 + nqTot;
    TData *s_D0   = deriv1 + nqTot;
    TData *s_D1   = s_D0 + nq0 * nq0;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
    {
        s_D0[idx] = D0[idx];
    }

    for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
    {
        s_D1[idx] = D1[idx];
    }

    size_t e                 = getBlockIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        // In 3DH1 (nmode > 1) the input holds three directions per
        // variable, the z one read by DerivZOp, so the stride between
        // variables is three even though only ncoord are consumed here.
        const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
        const TData *inptr =
            in + nqTot * nelmt * (inDim * nmode * c + m) + nqTot * e;
        TData *outptr = out + nqTot * nelmt * (nmode * c + m) + nqTot * e;

        StdAlignDerivBase2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr, inptr,
            deriv0, deriv1, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<true, APPEND>(
            nq0, nq1, s_D0, s_D1, deriv0, deriv1, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP three-dimensional launcher, Coeff output: unpacks the
 * element sizes and computes the inner product.
 *
 * The 3D form of the quadrilateral/triangle launcher: the shared memory
 * holds the three direction fields, their summed transposed derivative,
 * the two intermediates of the basis contraction and the three
 * one-dimensional basis tables (plus, for the nodal shapes, their
 * coefficient vector); the locals nmode0 ... nmode2 and offset0 / offset1
 * hold the table row counts and intermediate sizes of the shape's
 * collapsed mode ordering, matching IProductWRTDerivBaseSharedMemorySize.
 * The block's threads copy the basis tables in, then stride over the
 * elements running the geometric stage, SumDerivTensor3DSumFacTOPKernel
 * and the IProductWRTBase kernel of the shape, the nodal shapes finishing
 * on MatVecSumFacTOPKernel and the transposed nodal-to-modal matrix.
 * There is no coordinate-dimension argument -- a 3D element's is always
 * three -- and the mode-index tables @p index0 ... @p index2 are used here
 * by the basis contraction.
 *
 * As in the 2D launcher, the summed transposed derivative always
 * overwrites its shared-memory intermediate, which is reused across the
 * element loop; the append flag takes effect in the closing basis
 * contraction onto the output.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements.
 * @tparam Implementation      MultiRegions::SumFacTOP.
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
 * @param   index0,index1,index2    Mode-index tables of the collapsed
 *                                  ordering.
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
 * @param   wsp                     Global-memory workspace; unused here.
 * @param   scale                   Factor applied to the result.
 * @param   shmemptr                Dynamic shared memory, carved up as
 *                                  above.
 * @param   threadBlock             Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const size_t inoffset, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *deriv0    = (TData *)shmemptr;
    TData *deriv1    = deriv0 + nqTot;
    TData *deriv2    = deriv1 + nqTot;
    TData *deriv     = deriv2 + nqTot;
    TData *s_wsp0    = deriv + nqTot;
    TData *s_wsp1    = s_wsp0 + offset0;
    TData *s_basis0  = s_wsp1 + offset1;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_basis2  = s_basis1 + nmode1 * nq1;
    TData *s_out1ptr = s_basis2 + nmode2 * nq2;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr =
            in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
        TData *outptr = out + nmTot * nelmt * (nmode * c + m) + nmTot * e;

        StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr, jacptr,
            inptr, deriv0, deriv1, deriv2, threadBlock);
        // Always overwrite the shared-memory deriv scratch (it is reused
        // across the element loop); the op-level append is handled by the
        // IProductWRTBase kernels writing the global output below.
        SumDerivTensor3DSumFacTOPKernel<false, false>(
            nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, deriv, scale,
            threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, deriv, outptr, s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<true, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, outptr,
                s_wsp0, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<true, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, deriv, s_out1ptr,
                s_wsp0, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<true, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, deriv, outptr, s_wsp0,
                s_wsp1, scale, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief SumFacTOP three-dimensional launcher, Phys output: unpacks the
 * element sizes and leaves the derivative at the quadrature points.
 *
 * As the Coeff launcher without the basis contraction: the shared memory
 * holds the three direction fields and the three one-dimensional
 * derivative matrices, and SumDerivTensor3DSumFacTOPKernel writes the
 * summed transposed derivative straight to the output, applying the scale
 * and the append flag. The basis tables, the mode-index tables and the
 * nodal-to-modal matrix are absent from this signature, all of which
 * serve the basis contraction the Phys path does not reach.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements.
 * @tparam Implementation          MultiRegions::SumFacTOP.
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
 * @param   wsp             Global-memory workspace; unused here.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory, carved up as above.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        TData scale, unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *deriv0 = (TData *)shmemptr;
    TData *deriv1 = deriv0 + nqTot;
    TData *deriv2 = deriv1 + nqTot;
    TData *s_D0   = deriv2 + nqTot;
    TData *s_D1   = s_D0 + nq0 * nq0;
    TData *s_D2   = s_D1 + nq1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
    {
        s_D0[idx] = D0[idx];
    }

    for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
    {
        s_D1[idx] = D1[idx];
    }

    for (unsigned int idx = idx0; idx < nq2 * nq2; idx += stride)
    {
        s_D2[idx] = D2[idx];
    }

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr =
            in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
        TData *outptr = out + nqTot * nelmt * (nmode * c + m) + nqTot * e;

        StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr, jacptr,
            inptr, deriv0, deriv1, deriv2, threadBlock);
        SumDerivTensor3DSumFacTOPKernel<true, APPEND>(
            nq0, nq1, nq2, s_D0, s_D1, s_D2, deriv0, deriv1, deriv2, outptr,
            scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
