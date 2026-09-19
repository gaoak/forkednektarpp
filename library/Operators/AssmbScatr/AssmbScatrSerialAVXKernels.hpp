///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrSerialAVXKernels.hpp
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
// Description: Serial/AVX kernels of the assemble-scatter operator
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AssmbScatrSerialAVXKernels.hpp
 * @brief Host-side (Serial and AVX execution space) kernels of the
 * AssmbScatr operator: gather, sum and scatter of shared coefficients in a
 * continuous (C0) expansion.
 *
 * @details
 * ### What these kernels do
 * A continuous expansion stores every element's coefficients separately, so
 * a global degree of freedom (DOF) shared by several elements exists as
 * several local copies. Assemble--scatter makes those copies consistent: for
 * each global DOF, gather every local copy and sum the contributions
 * (assemble), then write the summed value back to every copy (scatter). The
 * kernels below are the host counterparts of the device kernels in
 * AssmbScatrDeviceKernels.hpp; AssmbScatrOpImpl picks the right overload via
 * `std::enable_if` on the ExecSpace template parameter.
 *
 * ### Common parameter conventions
 * All map arrays are built by MultiRegions::LocalToGlobalDataCreator
 * from the expansion's AssemblyMapCG:
 * - `nvals`     -- number of global DOFs to process.
 * - `nassemble` -- per global DOF, the number of local copies on this
 *                  device/rank.
 * - `index`     -- positions of those copies in `inoutptr` (and, for the
 *                  boundary kernels, of slots in the communication
 *                  buffers).
 * - `offset`    -- per global DOF, the start of its entries in `index` and
 *                  `sign`.
 * - `sign`      -- per `index` entry, a factor from the assembly map's
 *                  local-to-global sign convention: -1 or +1 for genuine
 *                  entries, 0 for AVX interleave padding and, in the
 *                  zero-Dirichlet operator variant, for Dirichlet DOFs,
 *                  and 0 in the boundary maps' trailing buffer-slot
 *                  positions, which are never read.
 *
 * Multiplying a local copy by its sign converts it to the global sign
 * convention before summation; multiplying the assembled value by the same
 * sign on write-back converts it back (the factors square to one, and a
 * zero sign masks the entry out entirely, which is how the zero-Dirichlet
 * variant annihilates Dirichlet values).
 *
 * ### Interior versus boundary kernels
 * AssembleScatrKernel handles global DOFs whose copies all live on this
 * device/rank and performs the whole operation in one pass. Global DOFs
 * shared with other ranks are treated in three stages so communication can
 * overlap with the interior work (see AssmbScatrOpImpl::v_Apply):
 * AssembleScatrBndKernel forms this rank's partial sums and packs them into
 * a send buffer, the exchange runs while the interior kernel executes, and
 * AssembleFromBndKernel combines the received partial sums and performs the
 * final scatter.
 *
 * @see AssmbScatrDeviceKernels.hpp for the device (CUDA/HIP/SYCL)
 *      counterparts.
 * @see MultiRegions/DataWarehouse/LocalToGlobalDataWarehouseDef.hpp for
 *      the construction of the map arrays and the layout guarantees the
 *      kernels rely on.
 */

#pragma once

namespace Nektar::Operators::detail
{
/**
 * @brief Assemble and scatter the device-interior global DOFs; AVX variant
 * reading SIMD-interleaved map data.
 *
 * Global DOFs are processed in blocks of `vector_width` (the SIMD lane
 * count of TData in the AVX space). Within a block the map data is
 * lane-interleaved: contribution `j` of the DOF handled by lane `i`
 * sits at `offset[idx + i] + j * vector_width`, so consecutive lanes read
 * consecutive `index`/`sign` entries. Each block spans `vector_width`
 * times its largest valence, and all arrays are padded to a multiple of
 * `vector_width`, so @p nvals itself need not be. The loops here are
 * still scalar -- the layout mirrors the warp-interleaved device format
 * rather than exploiting intrinsics.
 *
 * The first contribution of every lane is gathered before @p nassemble is
 * consulted. This is safe on padding lanes (`nassemble == 0`) only
 * because LocalToGlobalDataCreator guarantees that padded `index` slots
 * duplicate a valid location (padded `sign` entries are additionally
 * zeroed); the gathered value is discarded, since the write-back loop is
 * bounded by @p nassemble. Genuine DOFs must have `nassemble[idx] >= 1`.
 *
 * @tparam  ExecSpace   Must be NektarSpaces::AVX (overload selected by
 *                      enable_if).
 * @tparam  TData       Scalar type of the coefficient data.
 * @param   nvals       Number of global DOFs to process.
 * @param   nassemble   Local copies per global DOF; zero on padding lanes.
 * @param   index       Interleaved positions of the copies in @p inoutptr.
 * @param   offset      Start of each DOF's entries in @p index / @p sign.
 * @param   sign        Interleaved sign factors (see file documentation).
 * @param   inoutptr    Coefficient array, updated in place.
 */
template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
NEK_FORCE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const unsigned *offset, const int *sign, TData *inoutptr)
{
    constexpr unsigned int vector_width =
        NektarSpaces::vector_width<NektarSpaces::AVX, TData>::value;
    for (unsigned idx = 0; idx < nvals; idx += vector_width)
    {
        for (unsigned i = 0; i < vector_width; ++i)
        {
            TData ass =
                inoutptr[index[offset[idx + i]]] * sign[offset[idx + i]];

            // might be able to write this in a simd loop
            // but not sure it will provide much asdditional speed.
            for (unsigned j = 1; j < nassemble[idx + i]; ++j)
            {
                ass += inoutptr[index[offset[idx + i] + j * vector_width]] *
                       sign[offset[idx + i] + j * vector_width];
            }

            for (unsigned j = 0; j < nassemble[idx + i]; ++j)
            {
                inoutptr[index[offset[idx + i] + j * vector_width]] =
                    ass * sign[offset[idx + i] + j * vector_width];
            }
        }
    }
}

/**
 * @brief Assemble and scatter the device-interior global DOFs; serial
 * variant reading contiguously packed map data.
 *
 * With an interleave width of one, each global DOF's entries in @p ind and
 * @p sign are stored back to back in DOF order, so the running counter
 * `cnt` -- the prefix sum of @p nassemble -- equals `offset[idx]` at every
 * step. The @p offset array is therefore accepted (to keep the signature
 * shared with the AVX overload) but never read; this kernel silently
 * relies on the packed layout staying equivalent to the offsets produced
 * by LocalToGlobalDataCreator.
 *
 * @tparam  ExecSpace   Must be NektarSpaces::Serial (overload selected by
 *                      enable_if).
 * @tparam  TData       Scalar type of the coefficient data.
 * @param   nvals       Number of global DOFs to process.
 * @param   nassemble   Local copies per global DOF; must be >= 1 since the
 *                      first copy is read before the count is checked.
 * @param   ind         Positions of the copies in @p inoutptr, packed
 *                      contiguously in DOF order.
 * @param   offset      Unused (see above).
 * @param   sign        Sign factors (see file documentation).
 * @param   inoutptr    Coefficient array, updated in place.
 */
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *ind,
    [[maybe_unused]] const unsigned *offset, const int *sign, TData *inoutptr)

{
    unsigned cnt = 0;

    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        TData ass = inoutptr[ind[cnt]] * sign[cnt];

        for (unsigned j = 1; j < nassemble[idx]; ++j)
        {
            ass += inoutptr[ind[cnt + j]] * sign[cnt + j];
        }

        for (unsigned j = 0; j < nassemble[idx]; ++j)
        {
            inoutptr[ind[cnt + j]] = ass * sign[cnt + j];
        }
        cnt += nassemble[idx];
    }
}

/**
 * @brief Partially assemble the parallel-boundary global DOFs and pack the
 * partial sums into the communication send buffer (Serial and AVX).
 *
 * First stage of the three-stage parallel flow (see the file
 * documentation). For each global DOF shared with other ranks this kernel:
 * -# gathers this rank's `nassemble[idx]` local copies into a partial sum
 *    held in the global sign convention;
 * -# parks that sum in the *first* local slot only (multiplied by the
 *    slot's sign), where AssembleFromBndKernel later re-reads it -- the
 *    remaining copies are deliberately left untouched until the final
 *    scatter;
 * -# writes the raw partial sum, with no sign factor, into the
 *    `nbndvals[idx]` send-buffer slots, one per exchange entry with the
 *    neighbouring ranks sharing the DOF (exchanged values always travel
 *    in the global convention).
 *
 * Per DOF, the first `nassemble[idx]` entries of @p index (from
 * `offset[idx]`) address local coefficients in @p inoutptr and the next
 * `nbndvals[idx]` entries address slots in @p bndptr. Unlike the interior
 * AVX kernel, the boundary maps are never interleaved, so one scalar loop
 * serves both host execution spaces.
 *
 * @tparam  ExecSpace   NektarSpaces::Serial or NektarSpaces::AVX (overload
 *                      selected by enable_if).
 * @tparam  TData       Scalar type of the coefficient data.
 * @param   nvals       Number of parallel-boundary global DOFs.
 * @param   nassemble   Local copies per boundary DOF; must be >= 1.
 * @param   nbndvals    Send-buffer slots per boundary DOF.
 * @param   index       Local-coefficient positions followed by buffer
 *                      positions, per DOF.
 * @param   offset      Start of each DOF's entries in @p index / @p sign.
 * @param   sign        Sign factors (see file documentation).
 * @param   inoutptr    Coefficient array; only the first local slot of
 *                      each boundary DOF is written.
 * @param   bndptr      Send buffer receiving the partial sums.
 */
template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
NEK_FORCE_INLINE static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    TData *inoutptr, TData *bndptr)
{
    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        const unsigned ind  = offset[idx];
        const unsigned nidx = nassemble[idx];
        const unsigned nbnd = nbndvals[idx];

        TData ass = inoutptr[index[ind]] * sign[ind];

        for (unsigned j = 1; j < nidx; ++j)
        {
            ass += inoutptr[index[ind + j]] * sign[ind + j];
        }

        // keep one local copy for full assembly
        inoutptr[index[ind]] = ass * sign[ind];

        // put assembled values into boudnary array
        for (unsigned j = 0; j < nbnd; ++j)
        {
            bndptr[index[ind + nidx + j]] = ass;
        }
    }
}

/**
 * @brief Complete the assembly of the parallel-boundary global DOFs from
 * the received partial sums and scatter the result (Serial and AVX).
 *
 * Final stage of the three-stage parallel flow, run once communication has
 * finished. For each boundary DOF the received partial sums in @p bndptr
 * are combined with the sum this rank parked in the DOF's first local slot
 * during AssembleScatrBndKernel -- re-multiplying that slot by its sign
 * undoes the sign applied when it was stored -- and the total is then
 * scattered to all `nassemble[idx]` local copies.
 *
 * The accumulation order is deliberate. The buffer slots of each DOF are
 * sorted by ascending contributing rank, and `norder[idx]` is this rank's
 * own position in that ordering, so the sum takes the lower-ranked
 * contributions first, then this rank's partial sum, then the remainder.
 * Every rank sharing the DOF therefore adds the same values in the same
 * order and arrives at an identical assembled value, keeping the ranks
 * consistent despite floating-point non-associativity.
 *
 * The map layout is identical to AssembleScatrBndKernel: per DOF, local
 * coefficient positions followed by buffer positions, never interleaved.
 *
 * @tparam  ExecSpace   NektarSpaces::Serial or NektarSpaces::AVX (overload
 *                      selected by enable_if).
 * @tparam  TData       Scalar type of the coefficient data.
 * @param   nvals       Number of parallel-boundary global DOFs.
 * @param   nassemble   Local copies per boundary DOF.
 * @param   nbndvals    Receive-buffer slots per boundary DOF.
 * @param   index       Local-coefficient positions followed by buffer
 *                      positions, per DOF.
 * @param   offset      Start of each DOF's entries in @p index / @p sign.
 * @param   sign        Sign factors (see file documentation).
 * @param   norder      This rank's position among the ranks sharing each
 *                      DOF, i.e. how many buffer slots to accumulate
 *                      before adding the local partial sum.
 * @param   bndptr      Receive buffer holding the other ranks' partial
 *                      sums.
 * @param   inoutptr    Coefficient array; all local copies of each
 *                      boundary DOF are overwritten with the assembled
 *                      value.
 */
template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
NEK_FORCE_INLINE static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    const unsigned *norder, const TData *bndptr, TData *inoutptr)
{
    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        const unsigned ind  = offset[idx];
        const unsigned nidx = nassemble[idx];
        const unsigned nbnd = nbndvals[idx];
        const unsigned nord = norder[idx];

        TData ass = 0;

        // assemble bndptr components wtih local ids
        for (unsigned j = 0; j < nord; ++j)
        {
            ass += bndptr[index[ind + nidx + j]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[index[ind]] * sign[ind];

        // assemble rest of points from where we left off
        for (unsigned j = nord; j < nbnd; ++j)
        {
            ass += bndptr[index[ind + nidx + j]];
        }

        // copy rank ordered assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            inoutptr[index[ind + j]] = ass * sign[ind + j];
        }
    }
}

} // namespace Nektar::Operators::detail
