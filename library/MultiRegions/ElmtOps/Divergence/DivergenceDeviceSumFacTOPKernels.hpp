///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceDeviceSumFacTOPKernels.hpp
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
 * @file DivergenceDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernel launchers of the block divergence --
 * one element per thread block -- together with their shared-memory
 * sizing helpers.
 *
 * @details
 * The kernels compute the same quantity as their SumFac counterparts
 * (see DivergenceDeviceSumFacKernels.hpp) and are assembled from the
 * same PhysDeriv building blocks, here the SumFacTOP forms
 * PhysDerivDir1DSumFacTOPKernel<APPEND, DEFORMED, DIR> and
 * PhysDerivDir{2,3}DSumFacTOPKernel<SHAPE_TYPE, APPEND, DEFORMED,
 * DIR> from PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp: per
 * component, all reference-direction derivatives are formed,
 * corrected for a collapsed coordinate where the shape requires it,
 * and contracted with row DIR of the derivative factors, overwriting
 * the running sum for the first component and accumulating onto it
 * for the others.
 *
 * The parallelisation differs: one element is assigned to a whole
 * thread block, whose threads stride over the element's quadrature
 * points, and the blocks stride over the elements of the block of
 * elements (padding included). The homogeneous planes ride the second
 * grid dimension, each thread block offsetting the input and output to
 * the plane given by its second block index, all planes sharing the
 * derivative factors. In two and three dimensions the
 * component being differentiated and the running sum are staged in
 * two shared-memory workspaces, the sum being written out once all
 * components have been added; the PhysDerivDir SumFacTOP kernels
 * close with a barrier, so the next component may be copied in
 * immediately afterwards. Element data is expected contiguous
 * (interleave width one) and the derivative factors in the
 * factor-major layout, which is what
 * DivergenceBlockOpImpl::m_implInterleaveWidth and its DerivFactorKey
 * request for this strategy. The DivergenceKernelLauncher overloads
 * are constrained to the SumFacTOP implementation tag and are
 * otherwise identical in name and signature to the SumFac ones, which
 * is how DivergenceBlockOpImpl::OperatorNDImpl can launch either
 * strategy from one call site; DivergenceSharedMemorySize reports how
 * much dynamic shared memory such a launch must request.
 *
 * The whole file is compiled only for the device back-ends
 * (NEKTAR_ENABLE_DEVICE together with DEVICE_COMPILE_ONLY).
 *
 * @see DivergenceDeviceSumFac.hpp for the host side configuring and
 * launching these kernels.
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
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 1D SumFacTOP divergence kernel requires: none,
 * since the 1D launcher reads the element's input straight from global
 * memory.
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
inline constexpr unsigned int DivergenceSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 2D SumFacTOP divergence kernel requires: two
 * per-element workspaces of nq0 * nq1 entries.
 *
 * One workspace holds the component currently being differentiated,
 * the other the running sum. Unlike the SumFac overload this does not
 * depend on the shape: the collapsed-coordinate factor tables are
 * read straight from global memory.
 *
 * @tparam SHAPE_TYPE            Element shape; not used.
 * @tparam Implementation        Constrained to MultiRegions::SumFacTOP.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Entry count; the caller multiplies by sizeof(TData).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int DivergenceSharedMemorySize(
    const TPhysSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    return 2 * nq0 * nq1;
}

/**
 * @brief Dynamic shared memory, counted in TData entries, that one
 * thread block of the 3D SumFacTOP divergence kernel requires: two
 * per-element workspaces of nq0 * nq1 * nq2 entries.
 *
 * As in two dimensions, one workspace holds the component currently
 * being differentiated and the other the running sum, independently
 * of the shape.
 *
 * @tparam SHAPE_TYPE            Element shape; not used.
 * @tparam Implementation        Constrained to MultiRegions::SumFacTOP.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Entry count; the caller multiplies by sizeof(TData).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int DivergenceSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    return 2 * nq0 * nq1 * nq2;
}

/**
 * @brief Kernel entry point of the 1D SumFacTOP divergence: unpacks
 * the size parameter, then differentiates one component of the block
 * and scales the result by the element's derivative factor for
 * direction 0, overwriting the output.
 *
 * Resolves only when the Implementation tag is SumFacTOP, so that it
 * and the SumFac launcher of the same name can coexist, the launch
 * site in DivergenceBlockOpImpl::OperatorNDImpl selecting between them
 * through the tag alone. The size parameter must be one of the 1D
 * physical-space forms -- runtime or compile-time, see
 * ElmtHelper.hpp -- enforced by a static_assert rather than by the
 * overload constraint. Launched through DEVICE_2DGRID_KERNEL_LAUNCHER,
 * which appends the shared-memory pointer (unused here) and the
 * back-end's thread-block handle.
 *
 * The thread block takes element `e` = block index and strides over
 * the block's elements; per element PhysDerivDir1DSumFacTOPKernel
 * differentiates the contiguous element data -- its threads sharing
 * out the quadrature points. No shared memory is needed, both
 * operands living in global memory.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFacTOP here.
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
 * @param   df          Derivative factors of the block.
 * @param   in          Component data of the block.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block;
 *                      unused in one dimension.
 * @param   threadBlock Back-end thread-block handle supplying the
 *                      block and thread indices.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nq0 * nelmt * p + nq0 * e;
        TData *outptr      = out + nq0 * nelmt * p + nq0 * e;
        PhysDerivDir1DSumFacTOPKernel<false, DEFORMED, 0>(
            ncoord, nq0, D0, dfptr, inptr, outptr, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Kernel entry point of the 2D SumFacTOP divergence: binds the
 * dynamic shared memory and accumulates
 * \f$\partial u/\partial x + \partial v/\partial y\f$, one element
 * per thread block.
 *
 * Constrained as the 1D launcher, on the SumFacTOP tag alone, a
 * static_assert requiring one of the 2D physical-space size
 * parameters. FETCH_SHARED_MEMORY points @p shmemptr at the thread
 * block's dynamic shared memory on the back-ends that provide it
 * that way; the launch site sizes that allocation with
 * DivergenceSharedMemorySize.
 *
 * The dynamic shared memory is split into two workspaces of nqTot
 * entries, `s_wsp0` staging the component being differentiated and
 * `s_wsp1` holding the running sum. Per element the threads copy the
 * u component into `s_wsp0`, barrier, and call
 * PhysDerivDir2DSumFacTOPKernel with DIR = 0, which writes
 * \f$\partial u/\partial x\f$ to `s_wsp1`; the v component from
 * @p inoffset is then staged the same way and added with DIR = 1 and
 * APPEND, and `s_wsp1` is copied out to global memory before the
 * block moves on to its next element.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
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
 * @param   df          Derivative factors of the block, factor-major
 *                      per element.
 * @param   in          First component of the block's vector field.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block,
 *                      sized by DivergenceSharedMemorySize.
 * @param   threadBlock Back-end thread-block handle supplying the
 *                      block and thread indices.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    TData *s_wsp0             = (TData *)shmemptr;
    TData *s_wsp1             = (TData *)shmemptr + nqTot;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * p + nqTot * e;
        TData *outptr      = out + nqTot * nelmt * p + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir2DSumFacTOPKernel<SHAPE_TYPE, false, DEFORMED, 0>(
            ncoord, nq0, nq1, D0, D1, f0, f1, dfptr, s_wsp0, s_wsp1,
            threadBlock);

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[inoffset + idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir2DSumFacTOPKernel<SHAPE_TYPE, true, DEFORMED, 1>(
            ncoord, nq0, nq1, D0, D1, f0, f1, dfptr, s_wsp0, s_wsp1,
            threadBlock);

        // Copy to global memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            outptr[idx] = s_wsp1[idx];
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Kernel entry point of the 3D SumFacTOP divergence: binds the
 * dynamic shared memory and accumulates
 * \f$\partial u/\partial x + \partial v/\partial y
 * + \partial w/\partial z\f$, one element per thread block.
 *
 * Constrained as the other launchers, on the SumFacTOP tag alone, a
 * static_assert requiring one of the 3D physical-space size
 * parameters.
 *
 * The structure is that of the 2D launcher with a third component:
 * the u, v and w components are staged in turn in the shared
 * workspace `s_wsp0` and PhysDerivDir3DSumFacTOPKernel is called with
 * DIR = 0, 1 and 2, the first overwriting the running sum in
 * `s_wsp1` and the others accumulating onto it, after which `s_wsp1`
 * is copied out to global memory. The element is assumed to have
 * coordinate dimension three, so it carries nine derivative factors.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
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
 * @param   df          Derivative factors of the block, factor-major
 *                      per element.
 * @param   in          First component of the block's vector field.
 * @param   out         Output data of the block.
 * @param   shmemptr    Dynamic shared memory of the thread block,
 *                      sized by DivergenceSharedMemorySize.
 * @param   threadBlock Back-end thread-block handle supplying the
 *                      block and thread indices.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    FETCH_SHARED_MEMORY(shmemptr);

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    TData *s_wsp0             = (TData *)shmemptr;
    TData *s_wsp1             = (TData *)shmemptr + nqTot;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * p + nqTot * e;
        TData *outptr      = out + nqTot * nelmt * p + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir3DSumFacTOPKernel<SHAPE_TYPE, false, DEFORMED, 0>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, s_wsp0, s_wsp1,
            threadBlock);

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[inoffset + idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir3DSumFacTOPKernel<SHAPE_TYPE, true, DEFORMED, 1>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, s_wsp0, s_wsp1,
            threadBlock);

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[2 * inoffset + idx];
        }

        localBarrier(threadBlock);

        PhysDerivDir3DSumFacTOPKernel<SHAPE_TYPE, true, DEFORMED, 2>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, s_wsp0, s_wsp1,
            threadBlock);

        // Copy to global memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            outptr[idx] = s_wsp1[idx];
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
