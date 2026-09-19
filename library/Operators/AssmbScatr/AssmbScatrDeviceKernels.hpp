///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrDeviceKernels.hpp
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
// Description: Device kernels of the assemble-scatter operator
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AssmbScatrDeviceKernels.hpp
 * @brief Device-side kernels, and their host-side launch wrappers, for
 * the AssmbScatr assemble--scatter operator (CUDA, HIP and SYCL
 * back-ends).
 *
 * @details
 * ### What this file provides
 * The AssmbScatr operator makes a continuous-Galerkin coefficient field
 * globally consistent: for every global degree of freedom (DOF) shared by
 * several elements it sums ("assembles") all local contributions and
 * writes ("scatters") the assembled value back over each local copy, in
 * place. This file holds the Device execution-space implementation; the
 * Serial and AVX counterparts live in AssmbScatrSerialAVXKernels.hpp.
 * Every back-end exposes wrapper functions with identical signatures,
 * selected by SFINAE on the ExecSpace template argument, so
 * AssmbScatrOpImpl can call AssembleScatrKernel<ExecSpace>(...) without
 * knowing which back-end it is driving.
 *
 * The whole file is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY (see LibUtilities/Backends/Backends.hpp), so its
 * contents are compiled only when a device back-end is enabled. Under
 * CUDA and HIP the latter macro is set only in translation units
 * compiled by the CUDA/HIP compiler; under SYCL it is defined
 * build-wide, so every translation unit sees the kernels.
 * Host-only builds see an empty namespace.
 *
 * ### The three kernels and the multi-rank flow
 * On a single device AssembleScatrKernel does the whole job. Across
 * several ranks AssmbScatrOpImpl::v_Apply overlaps communication with
 * computation (see AssmbScatrOpImpl.hpp):
 * -# AssembleScatrBndKernel forms the rank-local partial sum of every
 *    partition-boundary DOF and packs it into the send buffer;
 * -# while AssemblyComm exchanges buffers (BeginComm()/EndComm()), the
 *    interior AssembleScatrKernel processes the DOFs whose
 *    contributions reside entirely on this device (those the map
 *    builder selected as needing work -- see the kernel's own notes);
 * -# AssembleFromBndKernel folds the received partial sums into the
 *    local one, in a globally agreed rank order, and scatters the
 *    result.
 *
 * ### Map layout
 * All kernels are driven by integer maps built by
 * MultiRegions::LocalToGlobalDataCreator (see
 * MultiRegions/DataWarehouse/LocalToGlobalDataWarehouseDef.hpp) and
 * cached in the data warehouse. Two layouts are in use:
 * - The interior index/sign maps are interleaved with stride
 *   NektarSpaces::Device::warpSize so that consecutive threads of a warp
 *   read consecutive memory locations (coalesced access). The kernel
 *   bakes the same constant in, so kernel and map builder agree by
 *   construction.
 * - The boundary maps are contiguous: each DOF owns a run of nassemble
 *   local-coefficient indices followed by nbndvals communication-buffer
 *   slot indices.
 *
 * ### Sign entries
 * Sign entries take the values -1, 0 or +1. The +/-1 factors come from
 * the assembly map's local-to-global sign data (AssemblyMapCG) and
 * reconcile the differing sign conventions of coincident local DOFs on
 * neighbouring elements; they are applied on gather and again on
 * scatter, so each local coefficient's own convention is restored. A 0
 * masks an entry completely -- it contributes nothing when gathered and
 * zeroes the local value when scattered -- which is how the
 * zero-Dirichlet operator variant annihilates Dirichlet DOFs and how
 * padding entries of the interleaved layout are kept harmless.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Assemble and scatter, in place, the global DOFs whose local
 * contributions all reside on this device and which need work.
 *
 * One grid-stride iteration handles one assembly target -- one global
 * DOF of one field component. Not every device-interior DOF appears:
 * the map builder omits targets with a single local contribution
 * (assemble--scatter is the identity on them) and keeps targets with
 * more than one, plus -- when the maps are built for the zero-Dirichlet
 * variant -- global Dirichlet targets of any valence, included so that
 * their local copies are zeroed. The first inner loop gathers the
 * signed sum of the target's nassemble[idx] local coefficients; the
 * second overwrites each of those coefficients with the sign-weighted
 * assembled value, i.e. the sum expressed in that coefficient's own
 * sign convention. Distinct targets own disjoint sets of local
 * coefficients, so threads never write to the same location.
 *
 * The index and sign arrays are interleaved: target idx's j-th entry
 * sits at offset[idx] + j * warpSize, where warpSize is the compile-time
 * constant NektarSpaces::Device::warpSize. Consecutive targets map to
 * consecutive threads, so a warp reads index/sign contiguously
 * (coalesced). The maps must have been built with the same interleave
 * width; AssmbScatrOpImpl guarantees this by keying the data warehouse
 * with NektarSpaces::vector_width<ExecSpace, TData>, which equals
 * warpSize for the Device space.
 *
 * @tparam TthreadBlock Back-end thread-block handle type (appended by
 *                      the launcher macro, e.g. hipcudaBlock<1>).
 * @tparam TData        Coefficient data type (float or double).
 *
 * @param   nvals       Number of assembly targets, i.e. device-interior
 *                      (global DOF, component) pairs selected by the
 *                      map builder as needing work (see above).
 * @param   nassemble   Per-target number of local coefficients mapping
 *                      to the target (its valence).
 * @param   index       Interleaved positions of those coefficients in
 *                      @p inoutptr.
 * @param   offset      Per-target start position of its entries in
 *                      @p index and @p sign.
 * @param   sign        Interleaved factors in {-1, 0, +1}; see the
 *                      file-level notes on sign entries.
 * @param   inoutptr    Coefficient array in non-interleaved (width-1)
 *                      layout; read and overwritten in place.
 * @param   threadBlock Thread-block handle used by getGlobalIdx() and
 *                      getGlobalRange() to form the grid-stride loop.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const unsigned *offset, const int *sign, TData *inoutptr,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpSize = NektarSpaces::Device::warpSize;

    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nassemb = nassemble[idx];
        for (unsigned j = 0; j < nassemb; ++j)
        {
            const unsigned ind = ioffset + j * warpSize;
            ass += inoutptr[index[ind]] * sign[ind];
        }

        for (unsigned j = 0; j < nassemb; ++j)
        {
            const unsigned ind   = ioffset + j * warpSize;
            inoutptr[index[ind]] = ass * sign[ind];
        }
    }
}

/**
 * @brief Form the rank-local partial sum of each partition-boundary DOF
 * and pack it into the communication send buffer.
 *
 * This is stage one of the multi-rank assembly (see the file-level
 * overview). For each boundary target the kernel gathers the signed sum
 * of its nassemble[idx] local coefficients, then:
 * - parks the partial sum in the first local coefficient only, stored
 *   in that coefficient's own sign convention: sign[offset[idx]] is
 *   applied on park and re-applied on retrieval, which restores the sum
 *   for the +/-1 factors, while a 0 factor (zero-Dirichlet variant)
 *   masks a partial sum that is itself already 0, so the round trip
 *   stays consistent. AssembleFromBndKernel picks the parked value up
 *   from there after communication;
 * - writes the plain (globally sign-corrected) partial sum into each of
 *   the target's nbndvals[idx] send-buffer slots, one per exchange entry
 *   involving this DOF.
 * The remaining local coefficients keep their raw values and are only
 * made consistent by AssembleFromBndKernel, so the field is in a mixed
 * state between the two kernels. That is safe because the interior
 * AssembleScatrKernel, which runs in between, touches a disjoint set of
 * DOFs.
 *
 * Unlike the interior kernel, the boundary maps are contiguous rather
 * than warp-interleaved: target idx owns entries [offset[idx],
 * offset[idx] + nassemble[idx] + nbndvals[idx]) of @p index, with the
 * local-coefficient entries first and the buffer-slot entries after.
 *
 * @tparam TthreadBlock Back-end thread-block handle type (appended by
 *                      the launcher macro, e.g. hipcudaBlock<1>).
 * @tparam TData        Coefficient data type (float or double).
 *
 * @param   nvals       Number of partition-boundary assembly targets,
 *                      i.e. (global DOF, component) pairs shared with
 *                      other ranks.
 * @param   nassemble   Per-target number of local coefficient
 *                      contributions; the map builder always provides at
 *                      least one (the parking slot).
 * @param   nbndvals    Per-target number of send-buffer slots to fill.
 * @param   index       Contiguous map: per target, nassemble positions
 *                      in @p inoutptr followed by nbndvals positions in
 *                      @p bndptr.
 * @param   offset      Per-target start position within @p index and
 *                      @p sign.
 * @param   sign        Factors in {-1, 0, +1} for the local-coefficient
 *                      entries; the trailing buffer-slot entries are
 *                      zero-filled and never read.
 * @param   inoutptr    Coefficient array (width-1 layout); only the
 *                      first local coefficient per target is updated.
 * @param   bndptr      Send buffer of the assembly communicator; write
 *                      only.
 * @param   threadBlock Thread-block handle used by getGlobalIdx() and
 *                      getGlobalRange() to form the grid-stride loop.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    TData *inoutptr, TData *bndptr, const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nidx    = nassemble[idx];
        const unsigned nbnd    = nbndvals[idx];

        // assemble values
        for (unsigned j = 0; j < nidx; ++j)
        {
            const unsigned ind = ioffset + j;
            ass += inoutptr[index[ind]] * sign[ind];
        }

        // copy one assembled values back to local values
        inoutptr[index[ioffset]] = ass * sign[ioffset];

        // put assembled values into boudnary array
        for (unsigned j = 0; j < nbnd; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            bndptr[index[ind]] = ass;
        }
    }
}

/**
 * @brief Complete the assembly of partition-boundary DOFs from received
 * partial sums and scatter the result to all local coefficients.
 *
 * Runs after AssemblyComm::EndComm() has filled @p bndptr with the
 * partial sums computed by the other sharing ranks. For each target the
 * summation follows a globally agreed rank order: the norder[idx]
 * received values originating from ranks numbered below this one are
 * added first, then this rank's own partial sum (parked in the first
 * local coefficient by AssembleScatrBndKernel and recovered by
 * re-applying its sign), then the remaining received values. The map
 * builder stores the buffer slots in ascending source-rank order
 * precisely so that every rank folds the same values in the same
 * sequence and therefore computes an identical result for a shared DOF,
 * not merely one that agrees to round-off. Finally the assembled value
 * is scattered, with signs, over all nassemble[idx] local coefficients.
 *
 * The map layout is the contiguous boundary layout described in
 * AssembleScatrBndKernel; the same nassemble/nbndvals/index/offset/sign
 * arrays are passed to both kernels.
 *
 * @tparam TthreadBlock Back-end thread-block handle type (appended by
 *                      the launcher macro, e.g. hipcudaBlock<1>).
 * @tparam TData        Coefficient data type (float or double).
 *
 * @param   nvals       Number of partition-boundary assembly targets.
 * @param   nassemble   Per-target number of local coefficients to
 *                      scatter to.
 * @param   nbndvals    Per-target number of receive-buffer slots to
 *                      fold in.
 * @param   index       Contiguous map: per target, nassemble positions
 *                      in @p inoutptr followed by nbndvals positions in
 *                      @p bndptr.
 * @param   offset      Per-target start position within @p index and
 *                      @p sign.
 * @param   sign        Factors in {-1, 0, +1} for the local-coefficient
 *                      entries.
 * @param   norder      Per-target position of this rank's own partial
 *                      sum in the rank-ordered summation, i.e. the
 *                      number of sharing ranks with a lower rank id.
 * @param   bndptr      Receive buffer holding the other ranks' partial
 *                      sums; read only.
 * @param   inoutptr    Coefficient array (width-1 layout); the parked
 *                      partial sum is read from it and the final values
 *                      are written back.
 * @param   threadBlock Thread-block handle used by getGlobalIdx() and
 *                      getGlobalRange() to form the grid-stride loop.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    const unsigned *norder, const TData *bndptr, TData *inoutptr,
    const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nidx    = nassemble[idx];
        const unsigned nbnd    = nbndvals[idx];
        const unsigned nord    = norder[idx];

        // assemble bndptr components wtih local ids
        for (unsigned j = 0; j < nord; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            ass += bndptr[index[ind]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[index[ioffset]] * sign[ioffset];

        // assemble rest of points from where we left off
        for (unsigned j = nord; j < nbnd; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            ass += bndptr[index[ind]];
        }

        // copy rank assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            const unsigned ind   = ioffset + j;
            inoutptr[index[ind]] = ass * sign[ind];
        }
    }
}

/**
 * @brief Launch the interior assemble--scatter kernel on the device.
 *
 * Device execution-space overload, selected by SFINAE against the Serial
 * and AVX overloads in AssmbScatrSerialAVXKernels.hpp. Launches a 1-D
 * grid of Device::defaultBlockSize-thread blocks, sized to give one
 * thread per assembly target, on the back-end's stream 0. Arguments are
 * forwarded unchanged to the device kernel above, which documents their
 * meanings and the required interleaved map layout.
 */
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const unsigned *offset, const int *sign, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(AssembleScatrKernel, gridSize,
                                          blockSize, 0, nvals, nassemble, index,
                                          offset, sign, inoutptr);
}

/**
 * @brief Launch the boundary send-buffer kernel on the device.
 *
 * Device execution-space overload, selected by SFINAE against the
 * Serial/AVX overload in AssmbScatrSerialAVXKernels.hpp. Launches a 1-D
 * grid of Device::defaultBlockSize-thread blocks, sized to give one
 * thread per boundary target, on the back-end's stream 0. Arguments are
 * forwarded unchanged to the device kernel above, which documents their
 * meanings and the contiguous boundary map layout. The caller is
 * responsible for making @p bndptr visible to the communication layer
 * afterwards (see AssmbScatrOpImpl::v_Apply).
 */
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    TData *inoutptr, TData *bndptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        AssembleScatrBndKernel<>, gridSize, blockSize, 0, nvals, nassemble,
        nbndvals, index, offset, sign, inoutptr, bndptr);
}

/**
 * @brief Launch the boundary receive-side assembly kernel on the device.
 *
 * Device execution-space overload, selected by SFINAE against the
 * Serial/AVX overload in AssmbScatrSerialAVXKernels.hpp. Launches a 1-D
 * grid of Device::defaultBlockSize-thread blocks, sized to give one
 * thread per boundary target, on the back-end's stream 0. Arguments are
 * forwarded unchanged to the device kernel above. Must only run once
 * communication has completed and @p bndptr holds the received partial
 * sums in the memory space the kernel executes in (see
 * AssmbScatrOpImpl::v_Apply for the synchronisation logic).
 */
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    const unsigned *norder, const TData *bndptr, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        AssembleFromBndKernel<>, gridSize, blockSize, 0, nvals, nassemble,
        nbndvals, index, offset, sign, norder, bndptr, inoutptr);
}
#endif

} // namespace Nektar::Operators::detail
