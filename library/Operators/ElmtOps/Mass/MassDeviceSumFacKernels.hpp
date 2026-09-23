///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFacKernels.hpp
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
// Description: Device SumFac kernels of the mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the mass operator: one element per
 * thread, the two composed families' shape kernels run back to back.
 *
 * @details
 * These are the kernels the Device/SumFac block implementation launches
 * (see MassDeviceSumFac.hpp); the SumFacTOP counterparts, where a whole
 * thread block cooperates on one element, are in
 * MassDeviceSumFacTOPKernels.hpp. Everything below is compiled only into
 * device translation units of a device-enabled build (NEKTAR_ENABLE_DEVICE
 * and DEVICE_COMPILE_ONLY).
 *
 * The MassKernelLauncher overloads own no mathematics of their own. Each
 * grid-strides over the block's elements, and for its element calls the
 * BwdTrans shape kernel of BwdTransDeviceSumFacKernels.hpp followed by the
 * IProductWRTBase shape kernel of IProductWRTBaseDeviceSumFacKernels.hpp,
 * the transform instantiated with APPEND false and the inner product with
 * SCALE and APPEND false and given a scale of 1; the transform kernels have
 * no SCALE parameter and take no scale argument. The quadrature weights and
 * the Jacobian therefore enter exactly where they do in a standalone inner
 * product: inside the IProductWRTBase shape kernel, which receives the
 * per-direction weight tables and the warp's Jacobian slice and whose
 * DEFORMED parameter decides whether that slice holds one value per element
 * or one per quadrature point. For the nodal shapes the element's
 * coefficients are mapped to the modal basis by MatVecKernel before the
 * transform and mapped back by the same kernel with TRANSPOSE set
 * afterwards.
 *
 * Element data is interleaved at the warp size: lane e % warpsize of warp
 * e / warpsize owns element e, and every workspace slice is addressed by
 * warp, so a shape kernel writes value k of its element at offset
 * warpsize * k + ilane. One launch covers every component of the block:
 * the grid's second dimension is the component count, a thread's own
 * component c comes from getBlockIdx<1>, and in, out and every workspace
 * region span all components back to back, a component's share starting at
 * element index nelmt * c, ahead of the warp interleaving above.
 *
 * This file also provides the SumFac overloads of the two launch-size
 * queries the block implementation calls, MassWorkSpaceSize and
 * MassSharedMemorySize; the SumFacTOP overloads of the same names live in
 * MassDeviceSumFacTOPKernels.hpp.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Global-memory workspace a 1D SumFac mass operator needs, in TData
 * values.
 *
 * Per element the nq0 physical values handed from the transform to the
 * inner product; the segment kernels need no further scratch.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    Operators::SumFac; the SumFacTOP overload lives
 *                           in MassDeviceSumFacTOPKernels.hpp.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return Workspace size in TData values per component, 0 for unhandled
 * shapes.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t MassWorkSpaceSize(const size_t nelmt,
                                          const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    const unsigned int nq0 = sizeParam1D.nq0();

    if constexpr (SHAPE_TYPE == LibUtilities::Seg)
    {
        wspsize = nq0 * nelmt;
    }

    return wspsize;
}

/**
 * @brief Global-memory workspace a 2D SumFac mass operator needs, in TData
 * values.
 *
 * Per element the nq0 * nq1 physical values passed between the two stages,
 * then one scratch area the two stages share, sized for whichever needs
 * more: nq1 values for the quadrilateral (nq1 for the inner product, nm1
 * for the transform, and nq1 >= nm1 for a standard quadrature),
 * max(nq1, nm0) for the triangles. The nodal triangle adds its modal
 * coefficient count nmTot for the converted coefficients. Each region is
 * laid out warp-interleaved like the field data and scaled by the
 * component count by the caller (see the file notes).
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Workspace size in TData values per component, 0 for unhandled
 * shapes.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t MassWorkSpaceSize(const size_t nelmt,
                                          const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wspsize = (nq0 * nq1 + nq1) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        wspsize = (nq0 * nq1 + std::max(nq1, nm0)) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        wspsize = (nq0 * nq1 + std::max(nq1, nm0) + nmTot) * nelmt;
    }

    return wspsize;
}

/**
 * @brief Global-memory workspace a 3D SumFac mass operator needs, in TData
 * values.
 *
 * Per element the nq0 * nq1 * nq2 physical values passed between the two
 * stages, then the two scratch areas the two stages share. For the
 * collapsed shapes each is sized to the larger of the two stages' demands
 * -- the inner product's nq1 * nq2 and nq2, the transform's
 * (2 nm1 - nm0 + 1) nm0 / 2 mode pairs and nm0 for the tetrahedra,
 * nm0 * nm1 and nm0 for the prisms and the pyramid -- while the
 * hexahedron's entry takes the inner product's nq1 * nq2 and nq2 directly,
 * which covers the transform's nm1 * nm2 and nm2 whenever each direction
 * has at least as many quadrature points as modes. The nodal shapes add
 * their modal coefficient count for the converted coefficients.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Workspace size in TData values per component, 0 for unhandled
 * shapes.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t MassWorkSpaceSize(const size_t nelmt,
                                          const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wspsize = (nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

        wspsize =
            (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm01) + std::max(nq2, nm0)) *
            nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();
        const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

        wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm01) +
                   std::max(nq2, nm0) + nmTot) *
                  nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0)) *
                  nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0) + nmTot) *
                  nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                   std::max(nq2, nm0)) *
                  nelmt;
    }

    return wspsize;
}

/**
 * @brief Dynamic shared memory a 1D SumFac launch needs: none.
 *
 * The SumFac kernels keep the physical intermediate and the stage scratch
 * in the global-memory workspace and read the basis and weight tables
 * straight from global memory, so no dynamic shared memory is requested at
 * launch in any dimension. The overloads mirror the SumFacTOP ones in
 * MassDeviceSumFacTOPKernels.hpp, which do use shared memory, so that the
 * block implementation can query the size the same way for either
 * strategy.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Sizes of the expansion; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Dynamic shared memory a 2D SumFac launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Dynamic shared memory a 3D SumFac launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Device entry point of the 1D SumFac mass operator: unpacks the
 * size parameter and, grid-striding over the block's elements, transforms
 * each thread's segment into its warp's slice of @p wsp and integrates it
 * straight back out, the weights and the Jacobian being applied inside the
 * inner-product shape kernel.
 *
 * Enabled only for the SumFac tag, so the SumFacTOP overload of the same
 * name (MassDeviceSumFacTOPKernels.hpp) never competes for the call; a size
 * parameter that is not 1D is rejected by the static_assert rather than by
 * overload resolution. Both the runtime and the compile-time
 * size-parameter forms are accepted; for this tag the launch bounds are
 * the warp size either way (see GetMaxThreadPerBlock).
 *
 * @tparam SHAPE_TYPE       Seg.
 * @tparam Implementation   Operators::SumFac.
 * @tparam DEFORMED         @p jac holds one interleaved value per
 *                          quadrature point rather than a single
 *                          per-element value.
 * @tparam TSizeParameter1D 1D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam1D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      Basis table (nm0 rows of nq0).
 * @param   w0          Quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   jac         Block's Jacobians, warp-interleaved.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace, MassWorkSpaceSize values per
 *                      component, back to back.
 * @param   shmemptr    Dynamic shared-memory base; unused here.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    MassKernelLauncher(const TSizeParameter1D sizeParam1D, const size_t nelmt,
                       [[maybe_unused]] const bool isModified,
                       const TData *NEK_RESTRICT basis0,
                       const TData *NEK_RESTRICT w0,
                       [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
                       const TData *NEK_RESTRICT jac,
                       const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
                       TData *NEK_RESTRICT wsp,
                       [[maybe_unused]] unsigned char *shmemptr,
                       const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nm0 * (nelmt * c + warpsize * iwarp);
        TData *wspptr      = wsp + nq0 * (nelmt * c + warpsize * iwarp);
        BwdTransSegSumFacKernel<false>(ilane, nm0, nq0, basis0, inptr, wspptr);
        IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
            ilane, nm0, nq0, basis0, w0, jacptr, wspptr, outptr, (TData)1.0);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFac mass operator: unpacks the
 * size parameter and, grid-striding over the block's elements, transforms
 * and integrates one quadrilateral or triangle per thread.
 *
 * The workspace is sliced as MassWorkSpaceSize describes it, each region
 * scaled by the component count (see the file notes): bwd takes the
 * leading nqTot * nelmt * ncomp values and holds the physical values
 * between the stages, and the scratch area behind it is passed to both
 * shape kernels in turn. For a nodal triangle the coefficients are first
 * mapped to the modal basis into a further slice, and the inner product
 * writes back into that slice so that the transposed mapping can produce
 * the nodal output.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   Operators::SumFac.
 * @tparam DEFORMED         @p jac holds one interleaved value per
 *                          quadrature point rather than a single
 *                          per-element value.
 * @tparam TSizeParameter2D 2D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam2D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Mode-index table; unused by this strategy and
 *                      passed null, the parameter being present so that
 *                      both strategies' launchers take the same argument
 *                      list.
 * @param   basis0      Direction-0 basis table.
 * @param   basis1      Direction-1 basis table; for the triangles indexed
 *                      by the combined (p,q) mode.
 * @param   w0,w1       Per-direction quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix (nodal shapes only).
 * @param   jac         Block's Jacobians, warp-interleaved.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace: MassWorkSpaceSize's regions, each
 *                      scaled by the component count (see above).
 * @param   shmemptr    Dynamic shared-memory base; unused here.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    MassKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + nqTot * nelmt * ncomp +
                          nq1 * (nelmt * c + warpsize * iwarp);
            BwdTransQuadSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1, basis0,
                                            basis1, inptr, bwd, wsp0);
            IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr, bwd,
                outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + nqTot * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, inptr,
                                           bwd, wsp0);
            IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, bwd, outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *modes = wsp + nqTot * nelmt * ncomp +
                           nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nqTot + nmTot) * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, modes,
                                           bwd, wsp0);
            IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, bwd, modes, wsp0, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFac mass operator: unpacks the
 * size parameter and, grid-striding over the block's elements, transforms
 * and integrates one element per thread.
 *
 * As in 2D, with two shared scratch areas instead of one, each region
 * scaled by the component count (see the file notes).
 *
 * @note For the two tetrahedral shapes the layout computed here does not
 * match MassWorkSpaceSize in two places: the first scratch area is given
 * the inner product's stride nq1 * nq2 where the reservation is
 * max(nq1 * nq2, nm01), nm01 = (2 nm1 - nm0 + 1) nm0 / 2 being the number
 * of mode pairs the transform stages there, and the second area's base
 * offset is placed behind the first at that same nq1 * nq2 per element
 * (times nelmt * ncomp) instead of at the reserved maximum. Layout and
 * reservation therefore agree only while nq1 * nq2 >= nm01. The
 * hexahedron's reservation takes no such maximum to begin with, so its
 * matching plain nq1 * nq2 stride never disagrees with it; the second
 * area's stride, and every offset of the other shapes, are the ones
 * MassWorkSpaceSize reserves.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation   Operators::SumFac.
 * @tparam DEFORMED         @p jac holds one interleaved value per
 *                          quadrature point rather than a single
 *                          per-element value.
 * @tparam TSizeParameter3D 3D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam3D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2,index3
 *                      Mode-index tables; unused by this strategy and
 *                      passed null (see the 2D launcher).
 * @param   basis0      Direction-0 basis table.
 * @param   basis1      Direction-1 basis table; combined-mode indexed for
 *                      the tetrahedra.
 * @param   basis2      Direction-2 basis table; combined-mode indexed for
 *                      the collapsed shapes.
 * @param   w0,w1,w2    Per-direction quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix (nodal shapes only).
 * @param   jac         Block's Jacobians, warp-interleaved.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace: MassWorkSpaceSize's regions, each
 *                      scaled by the component count (see above).
 * @param   shmemptr    Dynamic shared-memory base; unused here.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    MassKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nqTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            BwdTransHexSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           basis0, basis1, basis2, inptr, bwd,
                                           wsp0, wsp1);
            IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2, w0,
                w1, w2, jacptr, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
            IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *modes = wsp + nqTot * nelmt * ncomp +
                           nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nqTot + nmTot) * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nqTot + nq1 * nq2 + nmTot) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           modes, bwd, wsp0, wsp1);
            IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, modes, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp + (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, inptr, bwd, wsp0, wsp1);
            IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *modes = wsp + nqTot * nelmt * ncomp +
                           nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 =
                wsp + (nqTot + nmTot) * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nqTot + nmTot + std::max(nq1 * nq2, nm0 * nm1)) *
                              nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, modes);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, modes, bwd, wsp0, wsp1);
            IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, modes, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, modes, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp + (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPyrSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
            IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                (TData)1.0);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
