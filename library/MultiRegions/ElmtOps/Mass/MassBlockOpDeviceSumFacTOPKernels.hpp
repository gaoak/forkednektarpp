///////////////////////////////////////////////////////////////////////////////
//
// File: MassBlockOpDeviceSumFacTOPKernels.hpp
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
 * @file MassBlockOpDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the mass operator: the threads of one
 * device block cooperate on one element at a time, with the intermediates
 * staged in shared memory.
 *
 * @details
 * These are the kernels the Device/SumFacTOP block implementation launches
 * (see MassBlockOpDeviceSumFac.hpp); the SumFac counterparts, with one element
 * per thread, are in MassBlockOpDeviceSumFacKernels.hpp. Everything below is
 * compiled only into device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * The 2D and 3D MassKernelLauncher overloads first stage the per-direction
 * basis tables in shared memory once for the whole thread block, then
 * stride their blocks over the elements. For each element they stage the
 * coefficients (or, for a nodal shape, their modal image), call the
 * BwdTrans shape kernel of BwdTransBlockOpDeviceSumFacTOPKernels.hpp, and call
 * the IProductWRTBase shape kernel of
 * IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp, the transform with APPEND
 * false and the inner product with the launcher's SCALE and APPEND; for a
 * nodal shape the inner product overwrites the modal slice and APPEND
 * governs the mapping back. The 1D overload
 * stages only the physical values passed between the two stages: it hands the
 * basis table, the coefficients and the output to the segment kernels straight
 * from global memory.
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
 * The basis tables always go in shared memory. The per-element regions --
 * the coefficients, the physical values and the stage scratch -- form the
 * workspace: with useSharedMemoryWsp in shared memory after the basis data,
 * aligned, row-padded and with a padded inner-product input of its own
 * (MassSharedWorkSpaceSize), else packed and unpadded in the thread block's
 * slice of the global-memory workspace (MassWorkSpaceSize).
 * MassSharedMemorySize returns what the shared staging costs.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Unpadded size of the first stage scratch of a 2D or 3D SumFacTOP
 * mass launch, in TData values: what the transform's first intermediate
 * needs, or for the quadrilateral and hexahedron the larger of what the two
 * stages ask of it.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int MassStage0Size(
    const TSizeParameter sizeParam)
{
    const unsigned int nm0 = sizeParam.nm0();
    const unsigned int nm1 = sizeParam.nm1();
    const unsigned int nq0 = sizeParam.nq0();
    const unsigned int nq1 = sizeParam.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return (nm0 * nq1 > nm1 * nq0) ? nm0 * nq1 : nm1 * nq0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        const unsigned int nm2 = sizeParam.nm2();
        const unsigned int nq2 = sizeParam.nq2();
        return (nq0 * nm1 * nm2 > nm0 * nq1 * nq2) ? nq0 * nm1 * nm2
                                                   : nm0 * nq1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return (2u * nm1 - nm0 + 1u) * nm0 / 2u * sizeParam.nq2();
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism ||
                       SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
    {
        return nm0 * nm1 * sizeParam.nq2();
    }
}

/**
 * @brief Unpadded size of the second stage scratch of a 3D SumFacTOP mass
 * launch, in TData values, as MassStage0Size; 2D launches have none.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int MassStage1Size(
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
        const unsigned int nm2 = sizeParam.nm2();
        const unsigned int nq0 = sizeParam.nq0();
        const unsigned int nq1 = sizeParam.nq1();
        const unsigned int nq2 = sizeParam.nq2();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return (nq0 * nq1 * nm2 > nm0 * nm1 * nq2) ? nq0 * nq1 * nm2
                                                       : nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet ||
                           SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism ||
                           SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
        {
            return nm0 * nq1 * nq2;
        }
    }
}

/**
 * @brief Global workspace a 1D SumFacTOP mass launch needs, in TData values:
 * the element's nq0 physical values per thread block.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP; the SumFac overload
 *                           lives in MassBlockOpDeviceSumFacKernels.hpp.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements whose workspace is held at once: one per
 *                       thread block (see WorkSpaceNumElmts).
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return Number of TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t MassWorkSpaceSize(
    const size_t nelmt, const TSizeParameter1D sizeParam1D)
{
    return sizeParam1D.nq0() * nelmt;
}

/**
 * @brief Global workspace a 2D SumFacTOP mass launch needs, in TData values.
 *
 * Per thread block, the regions the 2D MassKernelLauncher lays out in global
 * memory, packed and unpadded: the element's nmTot coefficients, its
 * nq0 * nq1 physical values, which are also the inner product's input, and
 * the stage scratch of MassStage0Size.
 *
 * @tparam SHAPE_TYPE        Quad, Tri or NodalTri.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements whose workspace is held at once: one per
 *                       thread block (see WorkSpaceNumElmts).
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t MassWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    return (sizeParam2D.nmTot() + sizeParam2D.nqTot() +
            MassStage0Size<SHAPE_TYPE>(sizeParam2D)) *
           nelmt;
}

/**
 * @brief Global workspace a 3D SumFacTOP mass launch needs, in TData values.
 *
 * As in 2D, with the two stage scratch areas of MassStage0Size and
 * MassStage1Size.
 *
 * @tparam SHAPE_TYPE        Hex, Tet, NodalTet, Prism, NodalPrism, Pyr or
 *                           NodalPyr.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements whose workspace is held at once: one per
 *                       thread block (see WorkSpaceNumElmts).
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t MassWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    return (sizeParam3D.nmTot() + sizeParam3D.nqTot() +
            MassStage0Size<SHAPE_TYPE>(sizeParam3D) +
            MassStage1Size<SHAPE_TYPE>(sizeParam3D)) *
           nelmt;
}

/**
 * @brief One element's 2D or 3D workspace when staged in shared memory, in
 * TData values: the regions of MassWorkSpaceSize, each 16-byte aligned, with
 * the inner product's padded input as a region of its own and its row
 * padding added to each stage scratch.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int MassSharedWorkSpaceSize(
    const TSizeParameter sizeParam)
{
    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam);
    const unsigned int wsp1Size = MassStage1Size<SHAPE_TYPE>(sizeParam);

    return SharedMemoryAlignedSize<TData>(sizeParam.nmTot()) +
           SharedMemoryAlignedSize<TData>(sizeParam.nqTot()) +
           SharedMemoryAlignedSize<TData>(
               IProductWRTBasePaddedInputSize<TData>(sizeParam)) +
           SharedMemoryAlignedSize<TData>(
               MassStage0Size<SHAPE_TYPE>(sizeParam) + wspPadSize) +
           (IsSizeParameter3D_v<TSizeParameter>
                ? SharedMemoryAlignedSize<TData>(wsp1Size + wspPadSize)
                : 0u);
}

/**
 * @brief Dynamic shared memory a 1D SumFacTOP mass launch requests, in
 * TData values.
 *
 * The 1D MassKernelLauncher stages one region only, the element's nq0
 * physical values, which form its workspace; with @p useSharedMemoryWsp
 * that is the request, else nothing.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TData             Floating-point type of the field data; unused.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter1D sizeParam1D, const bool useSharedMemoryWsp)
{
    return useSharedMemoryWsp ? sizeParam1D.nq0() : 0u;
}

/**
 * @brief Dynamic shared memory a 2D SumFacTOP mass launch requests, in
 * TData values.
 *
 * The copies of both basis tables, the direction-1 table having nmTot rows
 * for the triangles because its rows are indexed by the combined mode,
 * rounded to 16 bytes; with @p useSharedMemoryWsp one element's workspace as
 * MassSharedWorkSpaceSize gives it follows.
 *
 * @tparam SHAPE_TYPE        Quad, Tri or NodalTri.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TData             Floating-point type of the field data; sets the
 *                           padding.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter2D sizeParam2D, const bool useSharedMemoryWsp)
{
    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);
    const unsigned int wspSize =
        useSharedMemoryWsp
            ? MassSharedWorkSpaceSize<SHAPE_TYPE, Implementation, TData>(
                  sizeParam2D)
            : 0u;

    return SharedMemoryAlignedSize<TData>(basisDataSize.total()) + wspSize;
}

/**
 * @brief Dynamic shared memory a 3D SumFacTOP mass launch requests, in
 * TData values.
 *
 * As in 2D, with three basis tables and two scratch areas in the
 * workspace. For the hexahedron each area is the maximum of what the two
 * stages ask of it. For the collapsed shapes the two areas are sized to the
 * transform's two demands exactly and the inner product reuses them in the
 * opposite order
 * -- its first stage fills the area the transform's second stage filled --
 * which is why the 3D MassKernelLauncher passes them to the inner-product
 * shape kernel with the two arguments swapped. The direction-2 table is
 * combined-mode indexed for every collapsed shape, and the direction-1
 * table for the tetrahedra as well (BasisDataSize3D).
 *
 * @tparam SHAPE_TYPE        Hex, Tet, NodalTet, Prism, NodalPrism, Pyr or
 * NodalPyr.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TData             Floating-point type of the field data; sets the
 *                           padding.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 * @param   useSharedMemoryWsp    Stage the workspace in shared memory.
 *
 * @return Shared memory per thread block, in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int MassSharedMemorySize(
    const TSizeParameter3D sizeParam3D, const bool useSharedMemoryWsp)
{
    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);
    const unsigned int wspSize =
        useSharedMemoryWsp
            ? MassSharedWorkSpaceSize<SHAPE_TYPE, Implementation, TData>(
                  sizeParam3D)
            : 0u;

    return SharedMemoryAlignedSize<TData>(basisDataSize.total()) + wspSize;
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
 * name (MassBlockOpDeviceSumFacKernels.hpp) never competes for the call; a
 * static_assert then rejects anything but a 1D size parameter. The staged
 * physical values go in @p shmemptr with @p useSharedMemoryWsp, else in the
 * thread block's slice of @p wsp.
 *
 * @tparam SHAPE_TYPE       Seg.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
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
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
 * @param   basis0      Basis table (nm0 rows of nq0).
 * @param   w0          Quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's coefficients, every component back to
 *                      back; this thread block's own component is c.
 * @param   out         Block's coefficients, every component back to
 *                      back; this thread block's own component is c,
 *                      overwritten, or accumulated into when APPEND.
 * @param   wsp         Global workspace, sized by MassWorkSpaceSize times
 *                      ncomp; unused if @p useSharedMemoryWsp.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    MassKernelLauncher(const TSizeParameter1D sizeParam1D, const size_t nelmt,
                       [[maybe_unused]] const bool isModified,
                       const bool useSharedMemoryWsp,
                       const TData *NEK_RESTRICT basis0,
                       const TData *NEK_RESTRICT w0,
                       [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
                       const TData *NEK_RESTRICT jac,
                       const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
                       TData *NEK_RESTRICT wsp, const TData scale,
                       unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);

    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: shared memory, or the thread block's slice.
        const size_t iblock =
            getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c;
        TData *bwd = isShared ? (TData *)shmemptr : wsp + nq0 * iblock;

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

            IProductWRTBaseSegSumFacTOPKernel<SCALE, APPEND>(
                nm0, nq0, basis0, bwd, outptr, scale, threadBlock);
            e += getBlockRange<0>(threadBlock);
        }
    });
}

/**
 * @brief Device entry point of the 2D SumFacTOP mass operator: unpacks the
 * size parameter and, striding the device block over the block's
 * elements, transforms and integrates one quadrilateral or triangle at a
 * time.
 *
 * The two basis tables go first in shared memory, copied in once before the
 * element loop. The workspace is carved out in the order modal (the
 * element's coefficients), bwd (its physical values), the inner product's
 * padded input, then the stage scratch: after the basis data, 16-byte aligned,
 * with @p useSharedMemoryWsp, else packed and unpadded, without the padded
 * input, in the thread block's slice of @p wsp. Per element the coefficients
 * are staged -- through the nodal-to-modal matrix for a nodal triangle --
 * transformed, multiplied by the weights of both directions and the Jacobian,
 * and integrated back; for a nodal triangle the inner product writes into
 * modal and the transposed nodal mapping produces the output.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
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
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
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
 *                      overwritten, or accumulated into when APPEND.
 * @param   wsp         Global workspace, sized by MassWorkSpaceSize times
 *                      ncomp; unused if @p useSharedMemoryWsp.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    MassKernelLauncher(const TSizeParameter2D sizeParam2D, const size_t nelmt,
                       const bool isModified, const bool useSharedMemoryWsp,
                       const unsigned int *NEK_RESTRICT index0,
                       const TData *NEK_RESTRICT basis0,
                       const TData *NEK_RESTRICT basis1,
                       const TData *NEK_RESTRICT w0,
                       const TData *NEK_RESTRICT w1,
                       const TData *NEK_RESTRICT nodToMod,
                       const TData *NEK_RESTRICT jac,
                       const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
                       TData *NEK_RESTRICT wsp, const TData scale,
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

    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);

    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // In shared memory the inner product's input and stage rows are
        // padded to SharedMemoryRowStride and IProductWRTBaseStageStride,
        // the padded input has its own region, and every region starts
        // 16-byte aligned, as MassSharedWorkSpaceSize sizes them; in global
        // memory the regions are packed and unpadded, as MassWorkSpaceSize
        // sizes them.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: after the basis data, or the thread block's slice.
        const size_t iblock =
            getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c;
        const unsigned int ldi =
            padInput ? SharedMemoryRowStride<TData>(nq0) : nq0;

        TData *modal =
            isShared ? s_basis0 +
                           SharedMemoryAlignedSize<TData>(basisDataSize.total())
                     : wsp + MassWorkSpaceSize<SHAPE_TYPE, Implementation>(
                                 1u, sizeParam2D) *
                                 iblock;
        TData *bwd =
            modal + (isShared ? SharedMemoryAlignedSize<TData>(nmTot) : nmTot);
        TData *padded =
            bwd + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *wsp0 =
            isShared
                ? padded +
                      SharedMemoryAlignedSize<TData>(
                          IProductWRTBasePaddedInputSize<TData>(sizeParam2D))
                : padded;
        TData *ipb = ldi == nq0 ? bwd : padded;

        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
            TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

            if constexpr (isNodal)
            {
                // Nodal to Modal in the workspace.
                MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, modal,
                                      threadBlock);
            }
            else
            {
                // Copy to the workspace.
                LoadToSharedMemory(nmTot, inptr, modal, threadBlock);
                localBarrier(threadBlock);
            }

            BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, false>(
                nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1,
                modal, bwd, wsp0, threadBlock);

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = idx / nq0;
                if constexpr (DEFORMED)
                {
                    ipb[ldi * j + i] = bwd[idx] * (jacptr[idx] * w0[i] * w1[j]);
                }
                else
                {
                    ipb[ldi * j + i] = bwd[idx] * (jacptr[0] * w0[i] * w1[j]);
                }
            }

            localBarrier(threadBlock);

            IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, SCALE,
                                             APPEND && !isNodal, padInput>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, isNodal ? modal : outptr, wsp0, scale,
                threadBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, modal,
                                                    outptr, threadBlock);
            }

            e += getBlockRange<0>(threadBlock);
        }
    });
}

/**
 * @brief Device entry point of the 3D SumFacTOP mass operator: unpacks the
 * size parameter and, striding the device block over the block's
 * elements, transforms and integrates one element at a time.
 *
 * As in 2D, with three basis tables and two scratch areas. The two
 * areas are handed to the inner-product shape kernel in the reverse order
 * for the collapsed shapes, because there each area is sized for exactly
 * one of the transform's two intermediates and the inner product's two
 * intermediates take those sizes the other way round; the hexahedron,
 * whose areas are maxima over both stages, passes them in the same order.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism, Pyr or
 * NodalPyr.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam SCALE            Multiply the result by @p scale.
 * @tparam APPEND           Accumulate onto @p out instead of overwriting.
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
 * @param   useSharedMemoryWsp   Stage the workspace in @p shmemptr rather
 *                      than in @p wsp.
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
 *                      overwritten, or accumulated into when APPEND.
 * @param   wsp         Global workspace, sized by MassWorkSpaceSize times
 *                      ncomp; unused if @p useSharedMemoryWsp.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      MassSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    MassKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, const bool useSharedMemoryWsp,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData scale, unsigned char *shmemptr,
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

    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    const unsigned int wsp0Size = MassStage0Size<SHAPE_TYPE>(sizeParam3D);

    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);

    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Laid out as in 2D, with the stage scratch's padding in shared
        // memory only.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: after the basis data, or the thread block's slice.
        const size_t iblock =
            getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c;
        const unsigned int ldi =
            padInput ? SharedMemoryRowStride<TData>(nq0) : nq0;

        TData *modal =
            isShared ? s_basis0 +
                           SharedMemoryAlignedSize<TData>(basisDataSize.total())
                     : wsp + MassWorkSpaceSize<SHAPE_TYPE, Implementation>(
                                 1u, sizeParam3D) *
                                 iblock;
        TData *bwd =
            modal + (isShared ? SharedMemoryAlignedSize<TData>(nmTot) : nmTot);
        TData *padded =
            bwd + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *wsp0 =
            isShared
                ? padded +
                      SharedMemoryAlignedSize<TData>(
                          IProductWRTBasePaddedInputSize<TData>(sizeParam3D))
                : padded;
        TData *wsp1 =
            isShared
                ? wsp0 + SharedMemoryAlignedSize<TData>(
                             wsp0Size +
                             IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                                 sizeParam3D))
                : wsp0 + wsp0Size;
        TData *ipb = ldi == nq0 ? bwd : padded;

        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
            TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

            if constexpr (isNodal)
            {
                // Nodal to Modal in the workspace.
                MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, modal,
                                      threadBlock);
            }
            else
            {
                // Copy to the workspace.
                LoadToSharedMemory(nmTot, inptr, modal, threadBlock);
                localBarrier(threadBlock);
            }

            BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
                s_basis0, s_basis1, s_basis2, modal, bwd, wsp0, wsp1,
                threadBlock);

            // Weight the physical values into the inner-product input.
            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);
                if constexpr (DEFORMED)
                {
                    ipb[ldi * (nq1 * k + j) + i] =
                        bwd[idx] * (jacptr[idx] * w0[i] * w1[j] * w2[k]);
                }
                else
                {
                    ipb[ldi * (nq1 * k + j) + i] =
                        bwd[idx] * (jacptr[0] * w0[i] * w1[j] * w2[k]);
                }
            }

            localBarrier(threadBlock);

            // The collapsed shapes take the two scratches the other way round.
            constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
            IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, SCALE,
                                             APPEND && !isNodal, padInput>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb,
                isNodal ? modal : outptr, hex ? wsp0 : wsp1, hex ? wsp1 : wsp0,
                scale, threadBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, modal,
                                                    outptr, threadBlock);
            }

            e += getBlockRange<0>(threadBlock);
        }
    });
}

#endif

} // namespace Nektar::MultiRegions::detail
