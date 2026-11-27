///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp
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
 * @file IProductWRTDerivBaseBlockOpDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the inner product with the gradient
 * of the basis: one element, or a group of them, per thread block at a
 * time, with the intermediates staged in shared memory.
 *
 * @details
 * These kernels compute the same thing as the SumFac ones in
 * IProductWRTDerivBaseBlockOpDeviceSumFacKernels.hpp but distribute the work
 * differently. A whole thread block takes one element, or a group of them
 * (GetDeviceElmtBlock()), at a time and strides over the block's elements
 * by the grid, the grid's y and z
 * dimensions selecting the homogeneous mode and output component (on a
 * multi-plane 3DH1 expansion the input holds three directions per output
 * component, of which only the first coordDim are read here); within
 * an element the block's threads stride over the work items of each stage
 * -- the element's quadrature points in the geometric and derivative
 * stages, its modes in the basis contraction. The field data is therefore
 * not interleaved (interleave width 1): an element's values are
 * contiguous, and the kernels index them directly. Each stage ends with a
 * local barrier, so the intermediates one stage writes are visible to the
 * next.
 *
 * The launchers copy the one-dimensional basis and derivative tables into
 * shared memory once per thread block. The per-element intermediates form
 * the workspace: with useSharedMemoryWsp in shared memory after the basis data,
 * aligned and row-padded (IProductWRTDerivBaseSharedWorkSpaceSize), else
 * packed and unpadded in the element's share of the thread block's slice of
 * the global-memory
 * workspace (IProductWRTDerivBaseWorkSpaceSize). The launch requests the
 * dynamic shared memory reported by IProductWRTDerivBaseSharedMemorySize.
 *
 * The stages are those of the SumFac kernels: the geometric stage of this
 * header, the transposed derivative (SumDerivTensorNDSumFacTOPKernel,
 * PhysDerivBlockOpDeviceSumFacTOPKernels.hpp) and, on the Coeff path, the basis
 * contraction (IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp) with the nodal
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

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Unpadded sizes, in TData values, of the regions the Coeff launchers
 * use besides the direction fields and their summed derivative: the basis
 * contraction's intermediates (nm0 * nq1 in 2D; nm0 * nq1 * nq2 and then
 * nm01 * nq2 for the tetrahedra, nm0 * nm1 * nq2 otherwise, in 3D) and the
 * nodal shapes' coefficient vector.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTDerivBaseStage0Size(
    const TSizeParameter sizeParam)
{
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        return sizeParam.nm0() * sizeParam.nq1();
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        return sizeParam.nm0() * sizeParam.nq1() * sizeParam.nq2();
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTDerivBaseStage1Size(
    const TSizeParameter sizeParam)
{
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        return 0u;
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nm0 = sizeParam.nm0();
        const unsigned int nm1 = sizeParam.nm1();

        return ((SHAPE_TYPE == LibUtilities::Tet ||
                 SHAPE_TYPE == LibUtilities::NodalTet)
                    ? (2u * nm1 - nm0 + 1u) * nm0 / 2u
                    : nm0 * nm1) *
               sizeParam.nq2();
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTDerivBaseModalSize(
    const TSizeParameter sizeParam)
{
    constexpr bool isNodal = SHAPE_TYPE == LibUtilities::NodalTri ||
                             SHAPE_TYPE == LibUtilities::NodalTet ||
                             SHAPE_TYPE == LibUtilities::NodalPrism ||
                             SHAPE_TYPE == LibUtilities::NodalPyr;

    return isNodal ? sizeParam.nmTot() : 0u;
}

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a one-dimensional shape.
 *
 * Per thread block, the direction field and, for the Coeff output, its
 * transposed derivative, nq0 values each.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   nelmt        Elements whose workspace is held at once: those of
 *                       one group per thread block (see
 *                       WorkSpaceNumElmts).
 * @param   sizeParam1D  Sizes of the expansion.
 *
 * @return Workspace in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter1D sizeParam1D)
{
    if constexpr (IsPhysSizeParameter1D_v<TSizeParameter1D>)
    {
        return sizeParam1D.nq0() * nelmt;
    }
    else if constexpr (IsSizeParameter1D_v<TSizeParameter1D>)
    {
        return 2 * sizeParam1D.nq0() * nelmt;
    }
}

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a two-dimensional shape.
 *
 * Per thread block, the regions the 2D launchers lay out in global memory,
 * packed and unpadded:
 * - Phys: the two direction fields, 2 * nq0 * nq1.
 * - Coeff: the two direction fields and their summed transposed
 *   derivative, then the nm0 * nq1 intermediate of the basis contraction;
 *   the nodal triangle adds its coefficient vector, nmTot.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   nelmt        Elements whose workspace is held at once: those of
 *                       one group per thread block (see
 *                       WorkSpaceNumElmts).
 * @param   sizeParam2D  Sizes per direction.
 *
 * @return Workspace in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    const unsigned int nqTot = sizeParam2D.nqTot();

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        return 2u * nqTot * nelmt;
    }
    else if constexpr (IsSizeParameter2D_v<TSizeParameter2D>)
    {
        return (3u * nqTot +
                IProductWRTDerivBaseStage0Size<SHAPE_TYPE>(sizeParam2D) +
                IProductWRTDerivBaseModalSize<SHAPE_TYPE>(sizeParam2D)) *
               nelmt;
    }
}

/**
 * @brief Global workspace a SumFacTOP launch needs, in elements of TData,
 * for a three-dimensional shape.
 *
 * As the 2D overload: the three direction fields for the Phys output; for
 * the Coeff output those and their summed transposed derivative, the two
 * intermediates of the basis contraction and, for the nodal shapes, their
 * coefficient vector.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   nelmt        Elements whose workspace is held at once: those of
 *                       one group per thread block (see
 *                       WorkSpaceNumElmts).
 * @param   sizeParam3D  Sizes per direction.
 *
 * @return Workspace in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t IProductWRTDerivBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    const unsigned int nqTot = sizeParam3D.nqTot();

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        return 3u * nqTot * nelmt;
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter3D>)
    {
        return (4u * nqTot +
                IProductWRTDerivBaseStage0Size<SHAPE_TYPE>(sizeParam3D) +
                IProductWRTDerivBaseStage1Size<SHAPE_TYPE>(sizeParam3D) +
                IProductWRTDerivBaseModalSize<SHAPE_TYPE>(sizeParam3D)) *
               nelmt;
    }
}

/**
 * @brief One element's workspace when staged in shared memory, in TData
 * values. As IProductWRTDerivBaseWorkSpaceSize, except that for the 2D and
 * 3D Coeff output the summed derivative and each intermediate of the basis
 * contraction are rounded to 16 bytes and the intermediates carry the inner
 * product's row padding.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTDerivBaseSharedWorkSpaceSize(
    const TSizeParameter sizeParam)
{
    if constexpr (IsSizeParameter1D_v<TSizeParameter> ||
                  IsPhysSizeParameter1D_v<TSizeParameter> ||
                  IsPhysSizeParameter2D_v<TSizeParameter> ||
                  IsPhysSizeParameter3D_v<TSizeParameter>)
    {
        return static_cast<unsigned int>(
            IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE, SumFacTOP>(
                1u, sizeParam));
    }
    else if constexpr (IsSizeParameter2D_v<TSizeParameter> ||
                       IsSizeParameter3D_v<TSizeParameter>)
    {
        constexpr bool is3D = IsSizeParameter3D_v<TSizeParameter>;

        unsigned int nqTot = sizeParam.nq0() * sizeParam.nq1();
        if constexpr (is3D)
        {
            nqTot *= sizeParam.nq2();
        }

        const unsigned int wspPadSize =
            IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam);

        return (is3D ? 3u : 2u) * nqTot +
               SharedMemoryAlignedSize<TData>(nqTot) +
               SharedMemoryAlignedSize<TData>(
                   IProductWRTDerivBaseStage0Size<SHAPE_TYPE>(sizeParam) +
                   wspPadSize) +
               (is3D ? SharedMemoryAlignedSize<TData>(
                           IProductWRTDerivBaseStage1Size<SHAPE_TYPE>(
                               sizeParam) +
                           wspPadSize)
                     : 0u) +
               IProductWRTDerivBaseModalSize<SHAPE_TYPE>(sizeParam);
    }
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a one-dimensional shape.
 *
 * The transposed differentiation matrix, nq0 * nq0, rounded to 16 bytes;
 * with @p useSharedMemoryWsp the workspace
 * IProductWRTDerivBaseSharedWorkSpaceSize gives, per element, for
 * @p elmtsPerBlock elements follows. The basis table is
 * read straight from global memory. The output field state is encoded in the
 * size-parameter type: a physical-space size parameter selects the Phys
 * variant, a modal one the Coeff variant.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TData             Floating-point type of the field data; sets the
 *                           alignment.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Sizes of the expansion.
 * @param   elmtsPerBlock  Elements per thread block, from
 *                       GetElmtsPerBlock().
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter1D_v<TSizeParameter1D> ||
                                IsPhysSizeParameter1D_v<TSizeParameter1D>),
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter1D sizeParam1D, const unsigned int elmtsPerBlock,
    const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? elmtsPerBlock *
                  IProductWRTDerivBaseSharedWorkSpaceSize<
                      SHAPE_TYPE, Implementation, TData>(sizeParam1D)
            : 0u;

    return SharedMemoryAlignedSize<TData>(nq0 * nq0) + wspSize;
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a two-dimensional shape.
 *
 * The basis data a thread block stages, rounded to 16 bytes, in the order the
 * launchers carve them up:
 * - Phys: the two transposed one-dimensional derivative matrices,
 *   nq0 * nq0 + nq1 * nq1.
 * - Coeff: the two one-dimensional basis tables -- the second of which is
 *   indexed by the shape's total mode count for the triangles, whose
 *   second direction carries the collapsed mode ordering -- then the two
 *   transposed derivative matrices.
 *
 * With @p useSharedMemoryWsp the workspace
 * IProductWRTDerivBaseSharedWorkSpaceSize gives, per element, for
 * @p elmtsPerBlock elements follows.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TData             Floating-point type of the field data; sets the
 *                           padding.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   sizeParam2D  Sizes per direction.
 * @param   elmtsPerBlock  Elements per thread block, from
 *                       GetElmtsPerBlock().
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter2D_v<TSizeParameter2D> ||
                                IsPhysSizeParameter2D_v<TSizeParameter2D>),
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter2D sizeParam2D, const unsigned int elmtsPerBlock,
    const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1;

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? elmtsPerBlock *
                  IProductWRTDerivBaseSharedWorkSpaceSize<
                      SHAPE_TYPE, Implementation, TData>(sizeParam2D)
            : 0u;

    if constexpr (IsPhysSizeParameter2D_v<TSizeParameter2D>)
    {
        return SharedMemoryAlignedSize<TData>(nDT) + wspSize;
    }
    else if constexpr (IsSizeParameter2D_v<TSizeParameter2D>)
    {
        return SharedMemoryAlignedSize<TData>(
                   nDT + BasisDataSize2D<SHAPE_TYPE>(sizeParam2D).total()) +
               wspSize;
    }
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a three-dimensional shape.
 *
 * Laid out as in the 2D overload:
 * - Phys: the three transposed one-dimensional derivative matrices.
 * - Coeff: the three one-dimensional basis tables, then the three
 *   transposed derivative matrices. The mode counts indexing the second
 *   and third tables follow the shape's collapsed mode ordering, which is
 *   why they are computed rather than simply nm1 and nm2.
 *
 * With @p useSharedMemoryWsp the workspace
 * IProductWRTDerivBaseSharedWorkSpaceSize gives, per element, for
 * @p elmtsPerBlock elements follows.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Implementation tag; this overload is
 *                           constrained to SumFacTOP.
 * @tparam TData             Floating-point type of the field data; sets the
 *                           padding.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space
 *                           (see ElmtHelper.hpp); its type selects the
 *                           Phys or Coeff variant.
 *
 * @param   sizeParam3D  Sizes per direction.
 * @param   elmtsPerBlock  Elements per thread block, from
 *                       GetElmtsPerBlock().
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               (IsSizeParameter3D_v<TSizeParameter3D> ||
                                IsPhysSizeParameter3D_v<TSizeParameter3D>),
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTDerivBaseSharedMemorySize(
    const TSizeParameter3D sizeParam3D, const unsigned int elmtsPerBlock,
    const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1 + nq2 * nq2;

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? elmtsPerBlock *
                  IProductWRTDerivBaseSharedWorkSpaceSize<
                      SHAPE_TYPE, Implementation, TData>(sizeParam3D)
            : 0u;

    if constexpr (IsPhysSizeParameter3D_v<TSizeParameter3D>)
    {
        return SharedMemoryAlignedSize<TData>(nDT) + wspSize;
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter3D>)
    {
        return SharedMemoryAlignedSize<TData>(
                   nDT + BasisDataSize3D<SHAPE_TYPE>(sizeParam3D).total()) +
               wspSize;
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
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
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
 * The thread block strides over the block's element groups
 * (GetDeviceElmtBlock()), slicing the
 * geometric factors and the field data for the element it is on and for
 * its (mode, component) pair, and runs the three stages: the geometric
 * stage into the first half of the shared-memory allocation,
 * SumDerivTensor1DSumFacTOPKernel into the second, and
 * IProductWRTBaseSegSumFacTOPKernel from there onto the output, where the
 * scale and the append flag take effect.
 *
 * The enable_if on the Implementation tag constrains this definition to
 * SumFacTOP, so the identically named SumFac launcher in
 * IProductWRTDerivBaseBlockOpDeviceSumFacKernels.hpp resolves for that tag
 * instead. The two share one signature. The direction field and its
 * transposed derivative form the workspace, after the derivative matrix in
 * shared memory with @p useSharedMemoryWsp, else in the element's share of
 * the thread block's slice of @p wsp.
 *
 * @tparam SHAPE_TYPE          Shape of the block's elements (Seg).
 * @tparam Implementation      MultiRegions::SumFacTOP.
 * @tparam APPEND              Accumulate onto the output instead of
 *                             overwriting it.
 * @tparam DEFORMED            True when the geometric factors vary within
 *                             an element.
 * @tparam TSizeParameter1D    Runtime or compile-time element sizes.
 * @tparam TElmtsPerBlock      SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 * @tparam TUseSharedWsp       Type of @p useSharedMemoryWsp: `bool`, or a
 *                             `std::bool_constant` SelectSharedWorkSpace()
 *                             fixed.
 *
 * @param   sizeParam1D Sizes of the expansion.
 * @param   ncoord      Coordinate dimension.
 * @param   nelmt       Elements in the block, padding included.
 * @param   elmtsPerBlock Elements per thread block, from
 *                      GetDeviceKernelParameters().
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
 * @param   basis0      One-dimensional basis table of the expansion.
 * @param   D0          One-dimensional derivative matrix.
 * @param   w0          Quadrature weights of the expansion.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector \f${\bf f}\f$.
 * @param   out         Coefficient-space output.
 * @param   wsp         Global-memory workspace, sized by
 *                      IProductWRTDerivBaseWorkSpaceSize times the
 *                      (mode, component) pairs; unused if
 *                      @p useSharedMemoryWsp.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory holding the transposed
 *                      derivative matrix and, with @p useSharedMemoryWsp,
 *                      the workspace.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const TElmtsPerBlock elmtsPerBlock,
        const size_t inoffset, [[maybe_unused]] const bool isModified,
        const TUseSharedWsp useSharedMemoryWsp,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, TData scale, unsigned char *shmemptr,
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

    const unsigned int basisDataSize = nq0 * nq0;
    TData *s_D0T                     = (TData *)shmemptr;

    // Copy to shared memory.
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: the element's share after the derivative matrix, or of the
        // block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv0 =
            isShared ? s_D0T + SharedMemoryAlignedSize<TData>(basisDataSize) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam1D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam1D) *
                                 globalWspIdx;
        TData *deriv = deriv0 + nq0;

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            // In 3DH1 (nmode > 1) the input holds three directions per
            // variable, the z one read by DerivZOp, so the stride between
            // variables is three even though only ncoord are consumed here.
            const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
            const TData *inptr =
                in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
            TData *outptr = out + nm0 * nelmt * (nmode * c + m) + nm0 * e;

            StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(
                ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv0,
                elmtBlock);
            SumDerivTensor1DSumFacTOPKernel<false, false>(
                nq0, s_D0T, deriv0, deriv, scale, elmtBlock);
            IProductWRTBaseSegSumFacTOPKernel<true, APPEND>(
                nm0, nq0, basis0, deriv, outptr, scale, elmtBlock);
        }
    });
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
 * @tparam TElmtsPerBlock          SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 * @tparam TUseSharedWsp           Type of @p useSharedMemoryWsp: `bool`, or a
 *                                 `std::bool_constant` SelectSharedWorkSpace()
 *                                 fixed.
 *
 * @param   sizeParam1D Sizes of the expansion.
 * @param   nelmt       Elements in the block, padding included.
 * @param   elmtsPerBlock Elements per thread block, from
 *                      GetDeviceKernelParameters().
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
 * @param   D0          One-dimensional derivative matrix.
 * @param   w0          Quadrature weights of the expansion.
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector.
 * @param   out         Physical-space output.
 * @param   wsp         Global-memory workspace, sized by
 *                      IProductWRTDerivBaseWorkSpaceSize times the
 *                      (mode, component) pairs; unused if
 *                      @p useSharedMemoryWsp.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory holding the transposed
 *                      derivative matrix and, with @p useSharedMemoryWsp,
 *                      the direction field.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter1D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
        const TElmtsPerBlock elmtsPerBlock, const size_t inoffset,
        const TUseSharedWsp useSharedMemoryWsp, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    const unsigned int basisDataSize = nq0 * nq0;
    TData *s_D0T                     = (TData *)shmemptr;

    // Copy to shared memory.
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: the element's share after the derivative matrix, or of the
        // block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv =
            isShared ? s_D0T + SharedMemoryAlignedSize<TData>(basisDataSize) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam1D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam1D) *
                                 globalWspIdx;

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            // In 3DH1 (nmode > 1) the input holds three directions per
            // variable, the z one read by DerivZOp, so the stride between
            // variables is three even though only ncoord are consumed here.
            const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
            const TData *inptr =
                in + nq0 * nelmt * (inDim * nmode * c + m) + nq0 * e;
            TData *outptr = out + nq0 * nelmt * (nmode * c + m) + nq0 * e;

            StdAlignDerivBase1DSumFacTOPKernel<DEFORMED>(
                ncoord, nq0, inoffset, w0, dfptr, jacptr, inptr, deriv,
                elmtBlock);
            SumDerivTensor1DSumFacTOPKernel<true, APPEND>(
                nq0, s_D0T, deriv, outptr, scale, elmtBlock);
        }
    });
}

/**
 * @brief SumFacTOP quadrilateral or triangle launcher, Coeff output:
 * unpacks the element sizes and computes the inner product.
 *
 * The dynamic shared-memory allocation starts with the two
 * one-dimensional basis tables and the two transposed derivative matrices.
 * The workspace -- the two direction fields, their summed transposed
 * derivative, the intermediate of the basis contraction and, for the nodal
 * triangle, its coefficient vector, in that order -- follows them with
 * @p useSharedMemoryWsp, else is the element's share of the thread block's
 * slice of @p wsp; the
 * sizes are those IProductWRTDerivBaseSharedMemorySize and
 * IProductWRTDerivBaseWorkSpaceSize account for; BasisDataSize2D gives the
 * two basis tables, which for the triangles follow the collapsed mode
 * ordering. The block's
 * threads copy the basis tables and derivative matrices in, then stride
 * over the elements running
 * the three stages; the nodal triangle writes its coefficients to the
 * workspace and finishes with MatVecSumFacTOPKernel applying the transposed
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
 * @tparam TElmtsPerBlock      SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 * @tparam TUseSharedWsp       Type of @p useSharedMemoryWsp: `bool`, or a
 *                             `std::bool_constant` SelectSharedWorkSpace()
 *                             fixed.
 *
 * @param   sizeParam2D     Sizes per direction.
 * @param   ncoord          Coordinate dimension (2 or 3).
 * @param   nelmt           Elements in the block, padding included.
 * @param   elmtsPerBlock   Elements per thread block, from
 *                          GetDeviceKernelParameters().
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   isModified      First-direction basis is eModified_A.
 * @param   useSharedMemoryWsp       Stage the workspace in @p shmemptr
 *                          rather than in @p wsp.
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
 * @param   wsp             Global-memory workspace, sized by
 *                          IProductWRTDerivBaseWorkSpaceSize times the
 *                          (mode, component) pairs; unused if
 *                          @p useSharedMemoryWsp.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory, carved up as above.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const TElmtsPerBlock elmtsPerBlock,
        const size_t inoffset, const bool isModified,
        const TUseSharedWsp useSharedMemoryWsp,
        const unsigned int *NEK_RESTRICT index0,
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

    constexpr bool isNodal = SHAPE_TYPE == LibUtilities::NodalTri;

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int wsp0Size = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wsp0Size = nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        wsp0Size = nm0 * nq1;
    }

    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_D0T    = s_basis1 + basisDataSize.basis1();
    TData *s_D1T    = s_D0T + nq0 * nq0;

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // In shared memory the inner product reads rows of length
        // SharedMemoryRowStride -- when nq0 is shorter its input is loaded
        // into the direction slots, consumed by then -- and its intermediate
        // carries the row padding, each region rounded to 16 bytes; in
        // global memory the regions are packed and unpadded.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: the element's share after the basis data, or of the thread
        // block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv0 =
            isShared ? s_basis0 +
                           SharedMemoryAlignedSize<TData>(
                               basisDataSize.total() + nq0 * nq0 + nq1 * nq1) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam2D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam2D) *
                                 globalWspIdx;
        const unsigned int ldi =
            padInput ? SharedMemoryRowStride<TData>(nq0) : nq0;
        const bool isPadded = (ldi != nq0);

        TData *deriv1 = deriv0 + nqTot;
        TData *deriv  = deriv1 + nqTot;
        TData *ipb    = (isPadded) ? deriv0 : deriv;
        TData *wsp0 =
            deriv + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *modal =
            wsp0 +
            (isShared ? SharedMemoryAlignedSize<TData>(
                            wsp0Size +
                            IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                                sizeParam2D))
                      : wsp0Size);

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
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
                ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr,
                inptr, deriv0, deriv1, elmtBlock);
            SumDerivTensor2DSumFacTOPKernel<false, false>(
                nq0, nq1, s_D0T, s_D1T, deriv0, deriv1, deriv, scale,
                elmtBlock);
            if (isPadded)
            {
                LoadWithPaddingToSharedMemory(nq0, nq1, ldi, deriv, ipb,
                                              elmtBlock);
                localBarrier(elmtBlock);
            }

            IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, true,
                                             APPEND && !isNodal, padInput>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, isNodal ? modal : outptr, wsp0, scale,
                elmtBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, modal,
                                                    outptr, elmtBlock);
            }
        }
    });
}

/**
 * @brief SumFacTOP quadrilateral or triangle launcher, Phys output:
 * unpacks the element sizes and leaves the derivative at the quadrature
 * points.
 *
 * As the Coeff launcher without the basis contraction: the shared memory
 * holds the transposes of the two one-dimensional derivative matrices,
 * which the block's threads stage there because every element reads them,
 * and the workspace the two direction fields, and
 * SumDerivTensor2DSumFacTOPKernel writes the summed transposed derivative
 * straight to the output, applying the scale and the append flag. The basis
 * tables, the mode-index table and the nodal-to-modal matrix are absent from
 * this signature: the Phys path stops before the basis contraction.
 *
 * @tparam SHAPE_TYPE              Shape of the block's elements.
 * @tparam Implementation          MultiRegions::SumFacTOP.
 * @tparam APPEND                  Accumulate onto the output instead of
 *                                 overwriting it.
 * @tparam DEFORMED                True when the geometric factors vary
 *                                 within an element.
 * @tparam TPhysSizeParameter2D    Runtime or compile-time element sizes,
 *                                 physical-space output.
 * @tparam TElmtsPerBlock          SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 * @tparam TUseSharedWsp           Type of @p useSharedMemoryWsp: `bool`, or a
 *                                 `std::bool_constant` SelectSharedWorkSpace()
 *                                 fixed.
 *
 * @param   sizeParam2D Sizes per direction.
 * @param   nelmt       Elements in the block, padding included.
 * @param   elmtsPerBlock Elements per thread block, from
 *                      GetDeviceKernelParameters().
 * @param   inoffset    Stride between the input's coordinate components.
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
 * @param   D0,D1       One-dimensional derivative matrices per direction.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0,f1       Collapsed-coordinate factor tables (triangles).
 * @param   df          Derivative factors of the block.
 * @param   jac         Jacobians of the block.
 * @param   in          Physical-space input vector.
 * @param   out         Physical-space output.
 * @param   wsp         Global-memory workspace, sized by
 *                      IProductWRTDerivBaseWorkSpaceSize times the
 *                      (mode, component) pairs; unused if
 *                      @p useSharedMemoryWsp.
 * @param   scale       Factor applied to the result.
 * @param   shmemptr    Dynamic shared memory, carved up as above.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter2D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
        const TElmtsPerBlock elmtsPerBlock, const size_t inoffset,
        const TUseSharedWsp useSharedMemoryWsp, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
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
    const unsigned int nqTot  = sizeParam2D.nqTot();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    const unsigned int basisDataSize = nq0 * nq0 + nq1 * nq1;
    TData *s_D0T                     = (TData *)shmemptr;
    TData *s_D1T                     = s_D0T + nq0 * nq0;

    // Copy to shared memory.
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: the element's share after the derivative matrices, or of
        // the block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv0 =
            isShared ? s_D0T + SharedMemoryAlignedSize<TData>(basisDataSize) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam2D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam2D) *
                                 globalWspIdx;
        TData *deriv1 = deriv0 + nqTot;

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
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
                ncoord, nq0, nq1, inoffset, w0, w1, f0, f1, dfptr, jacptr,
                inptr, deriv0, deriv1, elmtBlock);
            SumDerivTensor2DSumFacTOPKernel<true, APPEND>(
                nq0, nq1, s_D0T, s_D1T, deriv0, deriv1, outptr, scale,
                elmtBlock);
        }
    });
}

/**
 * @brief SumFacTOP three-dimensional launcher, Coeff output: unpacks the
 * element sizes and computes the inner product.
 *
 * The 3D form of the quadrilateral/triangle launcher: the shared memory
 * holds the three one-dimensional basis tables and the three transposed
 * derivative matrices, and the workspace the three direction fields, their
 * summed transposed derivative and the two intermediates of the basis
 * contraction (plus, for the nodal shapes, their coefficient vector);
 * BasisDataSize3D and the locals wsp0Size / wsp1Size hold the table and
 * intermediate sizes of the shape's collapsed mode ordering, matching
 * IProductWRTDerivBaseSharedMemorySize and
 * IProductWRTDerivBaseWorkSpaceSize.
 * The block's threads copy the basis tables in, then stride over the
 * elements running the geometric stage, SumDerivTensor3DSumFacTOPKernel
 * and the IProductWRTBase kernel of the shape, the nodal shapes finishing
 * on MatVecSumFacTOPKernel and the transposed nodal-to-modal matrix.
 * There is no coordinate-dimension argument -- a 3D element's is always
 * three -- and the mode-index tables @p index0 ... @p index2 are used here
 * by the basis contraction.
 *
 * As in the 2D launcher, the summed transposed derivative always
 * overwrites its workspace intermediate, which is reused across the
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
 * @tparam TElmtsPerBlock      SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 * @tparam TUseSharedWsp       Type of @p useSharedMemoryWsp: `bool`, or a
 *                             `std::bool_constant` SelectSharedWorkSpace()
 *                             fixed.
 *
 * @param   sizeParam3D             Sizes per direction.
 * @param   ncoord                  Coordinate dimension; unused.
 * @param   nelmt                   Elements in the block, padding
 *                                  included.
 * @param   elmtsPerBlock           Elements per thread block, from
 *                                  GetDeviceKernelParameters().
 * @param   inoffset                Stride between the input's coordinate
 *                                  components.
 * @param   isModified              First-direction basis is eModified_A.
 * @param   useSharedMemoryWsp      Stage the workspace in @p shmemptr
 *                                  rather than in @p wsp.
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
 * @param   wsp                     Global-memory workspace, sized by
 *                                  IProductWRTDerivBaseWorkSpaceSize times
 *                                  the (mode, component) pairs; unused if
 *                                  @p useSharedMemoryWsp.
 * @param   scale                   Factor applied to the result.
 * @param   shmemptr                Dynamic shared memory, carved up as
 *                                  above.
 * @param   threadBlock             Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTDerivBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        const TElmtsPerBlock elmtsPerBlock, const size_t inoffset,
        const bool isModified, const TUseSharedWsp useSharedMemoryWsp,
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
        TData *NEK_RESTRICT wsp, TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    constexpr bool isNodal = SHAPE_TYPE == LibUtilities::NodalTet ||
                             SHAPE_TYPE == LibUtilities::NodalPrism ||
                             SHAPE_TYPE == LibUtilities::NodalPyr;

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = sizeParam3D.nqTot();

    constexpr unsigned int ndf = 9u;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int wsp0Size = 0, wsp1Size = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wsp0Size = nm0 * nq1 * nq2;
        wsp1Size = nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        wsp0Size = nm0 * nq1 * nq2;
        wsp1Size = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        wsp0Size = nm0 * nq1 * nq2;
        wsp1Size = nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
    {
        wsp0Size = nm0 * nq1 * nq2;
        wsp1Size = nm0 * nm1 * nq2;
    }

    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();
    TData *s_D0T    = s_basis2 + basisDataSize.basis2();
    TData *s_D1T    = s_D0T + nq0 * nq0;
    TData *s_D2T    = s_D1T + nq1 * nq1;

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);
    LoadTransposeToSharedMemory(nq2, nq2, D2, s_D2T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Laid out as in 2D, with a second intermediate.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: the element's share after the basis data, or of the thread
        // block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv0 =
            isShared ? s_basis0 +
                           SharedMemoryAlignedSize<TData>(
                               basisDataSize.total() + nq0 * nq0 + nq1 * nq1 +
                               nq2 * nq2) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam3D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam3D) *
                                 globalWspIdx;
        const unsigned int ldi =
            padInput ? SharedMemoryRowStride<TData>(nq0) : nq0;
        const bool isPadded = (ldi != nq0);
        const unsigned int wspPadSize =
            padInput ? IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                           sizeParam3D)
                     : 0u;

        TData *deriv1 = deriv0 + nqTot;
        TData *deriv2 = deriv1 + nqTot;
        TData *deriv  = deriv2 + nqTot;
        TData *ipb    = (isPadded) ? deriv0 : deriv;
        TData *wsp0 =
            deriv + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *wsp1 =
            wsp0 + (isShared
                        ? SharedMemoryAlignedSize<TData>(wsp0Size + wspPadSize)
                        : wsp0Size);
        TData *wsp2 =
            wsp1 + (isShared
                        ? SharedMemoryAlignedSize<TData>(wsp1Size + wspPadSize)
                        : wsp1Size);

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr =
                in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
            TData *outptr = out + nmTot * nelmt * (nmode * c + m) + nmTot * e;

            StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, inptr, deriv0, deriv1, deriv2, elmtBlock);
            // Always overwrite the deriv scratch (it is reused
            // across the element loop); the op-level append is handled by the
            // IProductWRTBase kernels writing the global output below.
            SumDerivTensor3DSumFacTOPKernel<false, false>(
                nq0, nq1, nq2, s_D0T, s_D1T, s_D2T, deriv0, deriv1, deriv2,
                deriv, scale, elmtBlock);
            if (isPadded)
            {
                LoadWithPaddingToSharedMemory(nq0, nq1 * nq2, ldi, deriv, ipb,
                                              elmtBlock);
                localBarrier(elmtBlock);
            }

            IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, true,
                                             APPEND && !isNodal, padInput>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb,
                isNodal ? wsp2 : outptr, wsp0, wsp1, scale, elmtBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, wsp2,
                                                    outptr, elmtBlock);
            }
        }
    });
}

/**
 * @brief SumFacTOP three-dimensional launcher, Phys output: unpacks the
 * element sizes and leaves the derivative at the quadrature points.
 *
 * As the Coeff launcher without the basis contraction: the shared memory
 * holds the transposes of the three one-dimensional derivative matrices,
 * the workspace the three direction fields, and SumDerivTensor3DSumFacTOPKernel
 * writes the summed transposed derivative straight to the output, applying the
 * scale and the append flag. The basis tables, the mode-index tables and the
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
 * @tparam TElmtsPerBlock          SingleElmtPerBlock or MultipleElmtPerBlock.
 * @tparam TthreadBlock            Back-end thread-block handle type.
 * @tparam TData                   Floating-point type of the field data.
 * @tparam TUseSharedWsp           Type of @p useSharedMemoryWsp: `bool`, or a
 *                                 `std::bool_constant` SelectSharedWorkSpace()
 *                                 fixed.
 *
 * @param   sizeParam3D     Sizes per direction.
 * @param   nelmt           Elements in the block, padding included.
 * @param   elmtsPerBlock   Elements per thread block, from
 *                          GetDeviceKernelParameters().
 * @param   inoffset        Stride between the input's coordinate
 *                          components.
 * @param   useSharedMemoryWsp       Stage the workspace in @p shmemptr
 *                          rather than in @p wsp.
 * @param   D0,D1,D2        One-dimensional derivative matrices per
 *                          direction.
 * @param   w0,w1,w2        Quadrature weights per direction.
 * @param   f0,f1,f1m,f2    Collapsed-coordinate factor tables.
 * @param   df              Derivative factors of the block.
 * @param   jac             Jacobians of the block.
 * @param   in              Physical-space input vector.
 * @param   out             Physical-space output.
 * @param   wsp             Global-memory workspace, sized by
 *                          IProductWRTDerivBaseWorkSpaceSize times the
 *                          (mode, component) pairs; unused if
 *                          @p useSharedMemoryWsp.
 * @param   scale           Factor applied to the result.
 * @param   shmemptr        Dynamic shared memory, carved up as above.
 * @param   threadBlock     Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TPhysSizeParameter3D>()))
    IProductWRTDerivBasePhysKernelLauncher(
        const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
        const TElmtsPerBlock elmtsPerBlock, const size_t inoffset,
        const TUseSharedWsp useSharedMemoryWsp, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = sizeParam3D.nqTot();

    constexpr unsigned int ndf = 9u;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    const unsigned int basisDataSize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2;
    TData *s_D0T                     = (TData *)shmemptr;
    TData *s_D1T                     = s_D0T + nq0 * nq0;
    TData *s_D2T                     = s_D1T + nq1 * nq1;

    // Copy to shared memory.
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);
    LoadTransposeToSharedMemory(nq2, nq2, D2, s_D2T, threadBlock);

    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: the element's share after the derivative matrices, or of
        // the block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) +
             getBlockRange<0>(threadBlock) * (m + nmode * c)) *
                elmtsPerBlock +
            wspIdx;
        TData *deriv0 =
            isShared ? s_D0T + SharedMemoryAlignedSize<TData>(basisDataSize) +
                           IProductWRTDerivBaseSharedWorkSpaceSize<
                               SHAPE_TYPE, Implementation, TData>(sizeParam3D) *
                               wspIdx
                     : wsp + IProductWRTDerivBaseWorkSpaceSize<SHAPE_TYPE,
                                                               Implementation>(
                                 1u, sizeParam3D) *
                                 globalWspIdx;
        TData *deriv1 = deriv0 + nqTot;
        TData *deriv2 = deriv1 + nqTot;

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e      = elmtBlock.elmtIdx;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr =
                in + nqTot * nelmt * (3 * nmode * c + m) + nqTot * e;
            TData *outptr = out + nqTot * nelmt * (nmode * c + m) + nqTot * e;

            StdAlignDerivBase3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, inoffset, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, inptr, deriv0, deriv1, deriv2, elmtBlock);
            SumDerivTensor3DSumFacTOPKernel<true, APPEND>(
                nq0, nq1, nq2, s_D0T, s_D1T, s_D2T, deriv0, deriv1, deriv2,
                outptr, scale, elmtBlock);
        }
    });
}

#endif

} // namespace Nektar::MultiRegions::detail
