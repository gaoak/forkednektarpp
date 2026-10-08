///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFacTOPKernels.hpp
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
// Description: SumFacTOP device kernels of the mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the mass operator: the threads of one
 * device block cooperate on one element at a time, with the intermediates
 * staged in shared memory.
 *
 * @details
 * These are the kernels the Device/SumFacTOP block implementation launches
 * (see MassDeviceSumFac.hpp); the SumFac counterparts, with one element
 * per thread, are in MassDeviceSumFacKernels.hpp. Everything below is
 * compiled only into device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * The 2D and 3D MassKernelLauncher overloads first stage the per-direction
 * basis tables in shared memory once for the whole thread block, then
 * stride their blocks over the elements. For each element they stage the
 * coefficients (or, for a nodal shape, their modal image), call the
 * BwdTrans shape kernel of BwdTransDeviceSumFacTOPKernels.hpp, and call the
 * IProductWRTBase shape kernel of IProductWRTBaseDeviceSumFacTOPKernels.hpp,
 * the transform with APPEND false and the inner product with SCALE and
 * APPEND false. The 1D overload stages only the physical values passed
 * between the two stages: it hands the basis table, the coefficients and
 * the output to the segment kernels straight from global memory.
 *
 * Unlike the SumFac kernels, which hand the weights and the Jacobian to the
 * inner-product shape kernel, the kernels here apply the whole quadrature
 * metric themselves, in a thread-strided loop over the element's
 * quadrature points between the two stages: each staged physical value is
 * multiplied by the tensor product of the per-direction weights and by the
 * element Jacobian (one value per element, or one per point when
 * DEFORMED), and a barrier follows before the inner product reads them.
 * The IProductWRTBase shape kernels are therefore called in their
 * metric-free form, the one that expects its input to carry the metric
 * already.
 *
 * The MassSharedMemorySize overloads here return what that staging costs;
 * the MassWorkSpaceSize overloads carry the SumFacTOP tag but return the
 * same non-zero global-memory sizes as their SumFac counterparts, even
 * though the launchers below mark their workspace argument unused and no
 * kernel here touches it.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Global-memory workspace a 1D SumFacTOP mass launch requests, in
 * TData values.
 *
 * Repeats the size of the SumFac overload in MassDeviceSumFacKernels.hpp
 * (nq0 values per element), so the block implementation requests a
 * workspace of that size before every SumFacTOP launch -- although the
 * kernels of this file stage their intermediates in shared memory and
 * never read the workspace pointer (see the file notes). The 2D and 3D
 * overloads below do the same for their dimensions.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP.
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
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
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

/// @brief Global-memory workspace a 2D SumFacTOP mass launch requests:
/// the SumFac size, unread by the kernels here (see the 1D overload).
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
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

/// @brief Global-memory workspace a 3D SumFacTOP mass launch requests:
/// the SumFac size, unread by the kernels here (see the 1D overload).
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
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
 * @brief Dynamic shared memory a 1D SumFacTOP mass launch requests, in
 * TData values.
 *
 * The 1D MassKernelLauncher stages one region only, the element's nq0
 * physical values; this returns twice that, so nq0 values of the request
 * go unused. They still enter the occupancy heuristic of
 * GetDeviceGridSize.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    return 2 * nq0;
}

/**
 * @brief Dynamic shared memory a 2D SumFacTOP mass launch requests, in
 * TData values.
 *
 * The sum of the regions the 2D MassKernelLauncher lays out: the element's
 * nmTot coefficients, its nq0 * nq1 physical values, the scratch the two
 * stages share -- max(nq0 * nm1, nm0 * nq1) for the quadrilateral,
 * nm0 * nq1 for the triangles -- and copies of both basis tables, the
 * direction-1 table having nmTot rows for the triangles because its rows
 * are indexed by the combined mode.
 *
 * @tparam SHAPE_TYPE        Quad, Tri or NodalTri.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + nmTot + nq0 * nq1 +
               std::max(nq0 * nm1, nm0 * nq1);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq0 + nmTot * nq1 + nmTot + nq0 * nq1 + nm0 * nq1;
    }
}

/**
 * @brief Dynamic shared memory a 3D SumFacTOP mass launch requests, in
 * TData values.
 *
 * As in 2D, with three basis tables and two shared scratch areas. For the
 * hexahedron each area is the maximum of what the two stages ask of it.
 * For the collapsed shapes the two areas are sized to the transform's two
 * demands exactly and the inner product reuses them in the opposite order
 * -- its first stage fills the area the transform's second stage filled --
 * which is why the 3D MassKernelLauncher passes them to the inner-product
 * shape kernel with the two arguments swapped. The direction-2 table is
 * combined-mode indexed for every collapsed shape, and the direction-1
 * table for the tetrahedra as well, hence the nm01, nm02 and nmode2 row
 * counts.
 *
 * @tparam SHAPE_TYPE        Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) +
               std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
}

/**
 * @brief Device entry point of the 1D SumFacTOP mass operator: unpacks the
 * size parameter and, striding the device block over the block's
 * elements, transforms and integrates one segment at a time.
 *
 * The staged physical values are multiplied in place by the weights and
 * the Jacobian between the two shape kernels, so the inner product runs in
 * its metric-free form. Neither shape kernel stages the basis table: the
 * segment kernels read @p basis0, @p in and @p out straight from global
 * memory.
 *
 * Enabled only for the SumFacTOP tag, so the SumFac overload of the same
 * name (MassDeviceSumFacKernels.hpp) never competes for the call; a
 * static_assert then rejects anything but a 1D size parameter. The @p wsp
 * argument is accepted, so that both strategies' launchers take the same
 * argument list, and ignored.
 *
 * @tparam SHAPE_TYPE       Seg.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value.
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
 * @param   jac         Block's Jacobians.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread block's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread block's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace; unused.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
                       [[maybe_unused]] TData *NEK_RESTRICT wsp,
                       unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *bwd = (TData *)shmemptr;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * nelmt * c + nm0 * e;
        TData *outptr       = out + nm0 * nelmt * c + nm0 * e;
        BwdTransSegSumFacTOPKernel<false>(nm0, nq0, basis0, inptr, bwd,
                                          threadBlock);

        for (unsigned int i = idx0; i < nq0; i += stride)
        {
            if constexpr (DEFORMED)
            {
                bwd[i] *= jacptr[i] * w0[i];
            }
            else
            {
                bwd[i] *= jacptr[0] * w0[i];
            }
        }

        localBarrier(threadBlock);

        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFacTOP mass operator: unpacks the
 * size parameter and, striding the device block over the block's
 * elements, transforms and integrates one quadrilateral or triangle at a
 * time.
 *
 * The shared-memory regions are carved out in the order tmp (the element's
 * coefficients), bwd (its physical values), the shared stage scratch, then
 * the two basis tables, which the thread block copies in once before the
 * element loop. Per element the coefficients are staged -- through the
 * nodal-to-modal matrix for a nodal triangle -- transformed, multiplied by
 * the weights of both directions and the Jacobian, and integrated back;
 * for a nodal triangle the inner product writes into tmp and the
 * transposed nodal mapping produces the output.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value.
 * @tparam TSizeParameter2D 2D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam2D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Table giving p for each flat (p,q) mode of the
 *                      triangles, read by the inner-product shape kernel;
 *                      unused for the quadrilateral.
 * @param   basis0,basis1   Per-direction basis tables in global memory,
 *                      copied to shared memory here.
 * @param   w0,w1       Per-direction quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix (nodal shapes only).
 * @param   jac         Block's Jacobians.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread block's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread block's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace; unused.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    MassKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
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

    unsigned int offset{0}, nmode0{0}, nmode1{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = std::max(nm0 * nq1, nm1 * nq0);
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

    TData *tmp      = (TData *)shmemptr;
    TData *bwd      = tmp + nmTot;
    TData *s_wsp0   = bwd + nqTot;
    TData *s_basis0 = s_wsp0 + offset;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

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

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel<false>(nm0, nm1, nq0, nq1, nqTot,
                                               s_basis0, s_basis1, tmp, bwd,
                                               s_wsp0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel<false>(nm0, nm1, nq0, nq1, nqTot,
                                              isModified, s_basis0, s_basis1,
                                              tmp, bwd, s_wsp0, threadBlock);
        }

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            if constexpr (DEFORMED)
            {
                bwd[idx] *= jacptr[idx] * w0[i] * w1[j];
            }
            else
            {
                bwd[idx] *= jacptr[0] * w0[i] * w1[j];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, tmp, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFacTOP mass operator: unpacks the
 * size parameter and, striding the device block over the block's
 * elements, transforms and integrates one element at a time.
 *
 * As in 2D, with three basis tables and two shared scratch areas. The two
 * areas are handed to the inner-product shape kernel in the reverse order
 * for the collapsed shapes, because there each area is sized for exactly
 * one of the transform's two intermediates and the inner product's two
 * intermediates take those sizes the other way round; the hexahedron,
 * whose areas are maxima over both stages, passes them in the same order.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value.
 * @tparam TSizeParameter3D 3D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam3D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2,index3   Mode-index tables of the
 *                      collapsed shapes; which of them a shape's two
 *                      kernels read follows from the calls below, and
 *                      the unused ones are null.
 * @param   basis0,basis1,basis2    Per-direction basis tables in global
 *                      memory, copied to shared memory here.
 * @param   w0,w1,w2    Per-direction quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix (nodal shapes only).
 * @param   jac         Block's Jacobians.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread block's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread block's own component is c,
 *                      overwritten.
 * @param   wsp         Global workspace; unused.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    MassKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
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

    unsigned int offset0{0}, offset1{0}, nmode0{0}, nmode1{0}, nmode2{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
        offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *tmp      = (TData *)shmemptr;
    TData *bwd      = tmp + nmTot;
    TData *s_wsp0   = bwd + nqTot;
    TData *s_wsp1   = s_wsp0 + offset0;
    TData *s_basis0 = s_wsp1 + offset1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

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
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTet ||
                      SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel<false>(nm0, nm1, nm2, nq0, nq1, nq2,
                                              nqTot, isModified, index0, index3,
                                              s_basis0, s_basis1, s_basis2, tmp,
                                              bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            if constexpr (DEFORMED)
            {
                bwd[idx] *= jacptr[idx] * w0[i] * w1[j] * w2[k];
            }
            else
            {
                bwd[idx] *= jacptr[0] * w0[i] * w1[j] * w2[k];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, bwd, outptr, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
