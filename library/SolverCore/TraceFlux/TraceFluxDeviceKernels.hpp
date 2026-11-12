///////////////////////////////////////////////////////////////////////////////
//
// File: TraceFluxDeviceKernels.hpp
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
// Description: The in-place interior trace flux kernel for the Device
// execution space, for any point flux.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file TraceFluxDeviceKernels.hpp
 * @brief The in-place interior trace flux kernel for the Device execution
 * space: the flux of a collocated interior block read from and written to
 * the element-local trace fields through the trace point maps, for any
 * point flux (see TraceFluxKernels.hpp).
 *
 * @note TraceFluxOpImpl.hpp includes this header and
 * TraceFluxSerialAVXKernels.hpp. Each defines
 * InteriorBlockInPlaceKernelLauncher() for its own execution spaces only, so
 * the two never coincide.
 */

#pragma once

#include <SolverCore/TraceFlux/TraceFluxKernels.hpp>

#include <type_traits>

namespace Nektar::SolverCore::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief The interior flux of one block read and written in place, one
 * thread per trace point of the block.
 *
 * For a block whose two sides are both collocated with the global trace:
 * each thread reads both sides' state and second input straight from the
 * element-local trace fields through the per-point maps
 * (TraceBlockView::ptMapFwd), and its trace normal and penalty factor from
 * the warehouse's global-trace arrays, where consecutive threads read
 * consecutive points; evaluates the point flux through the normal in
 * registers; and writes the flux to both element-local sides, negated on the
 * backward one. Every element-local trace point belongs to one trace point
 * of one block, so the writes do not collide.
 *
 * The fields are read at whatever interleave width each trace's element
 * block carries (TraceBlockView::stride), so a warp's threads, which take
 * consecutive points of one trace, read consecutive words at width 1 and
 * words a width apart otherwise. Partition blocks keep the packed path on
 * Device
 * (TraceFluxOpImpl::SetUpInPlaceBlocks()). The kernel does not stage through
 * shared memory: on a GH200 a staged form was 8-16 % slower under the
 * average-of-normal-fluxes form and at most 7 % faster under the
 * interior-penalty form at P3-P4, nothing at P6.
 *
 * @tparam NDIM           Spatial dimension: components of the normal.
 * @tparam APPEND         Add to the flux rather than overwrite it.
 * @tparam PARALLEL       A partition block; never on Device.
 * @tparam PointFlux      The point flux (see TraceFluxKernels.hpp).
 * @param  views          The block's two sides and global-trace offsets.
 * @param  pointFlux      The point flux.
 * @param  inPtr,in1      The state field, and side 1's: the same field.
 * @param  secondPtr,second1  The second input, likewise; unread by a point
 *                        flux without one.
 * @param  fluxPtr        The flux field.
 * @param  normPtr,penPtr The global-trace normals and penalty factors, as
 *                        the data warehouse holds them; the penalty factors
 *                        unread by a point flux without a penalty.
 * @param  streamID       The stream the launch runs on; reset to the null
 *                        stream afterwards.
 */
template <
    unsigned int NDIM, bool APPEND, bool PARALLEL, typename ExecSpace,
    typename PointFlux, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
void InteriorBlockInPlaceKernelLauncher(
    const InteriorBlockViews &views, const PointFlux &pointFlux,
    const TData *inPtr, const TData *in1, const TData *secondPtr,
    const TData *second1, TData *fluxPtr, const TData *normPtr,
    const TData *penPtr, const unsigned int streamID)
{
    static_assert(!PARALLEL, "Partition blocks keep the packed path on Device");
    constexpr unsigned nVar    = PointFlux::numVar;
    constexpr unsigned nSecond = nVar * PointFlux::secondPerVar;
    static_assert(IsPointFlux<PointFlux, NDIM>(),
                  "PointFlux must present the point-flux interface of "
                  "TraceFluxKernels.hpp");

    // Copies: the device lambda captures them by value.
    const auto T0                 = views.side0;
    const auto T1                 = views.side1;
    const auto glo                = views.glo;
    const unsigned npts           = T0.ptsPerTrace;
    const size_t numTrace         = T0.numTrace;
    const PointFlux pointFluxCopy = pointFlux;

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * npts, NEKTAR_LAMBDA(const size_t i) {
            // nvcc cannot capture a variable whose first use in an extended
            // lambda is inside an `if constexpr`: name these first.
            constexpr unsigned numSecond      = nSecond;
            const TData *const secondOwn      = secondPtr;
            const TData *const secondNbr      = second1;
            const TData *const penaltyFactors = penPtr;
            TData *const fluxOut              = fluxPtr;
            const size_t trace                = i / npts;
            const size_t ptInTrace            = i - trace * npts;
            const size_t cs0 = T0.compSize[trace], cs1 = T1.compSize[trace];
            const size_t s0 = T0.offset_st[trace], s1 = T1.offset_st[trace];
            const size_t b0 = T0.offset[trace], b1 = T1.offset[trace];
            const TData *nrmTrace = normPtr + glo.gloTOffset[trace];
            const size_t nrmCS    = glo.compSize[trace];
            const size_t q0 = T0.MapFwd(trace, ptInTrace) * T0.stride[trace];
            const size_t q1 = T1.MapFwd(trace, ptInTrace) * T1.stride[trace];
            const size_t o0 = s0 * nVar + b0 + q0;
            const size_t o1 = s1 * nVar + b1 + q1;

            TData fwd[nVar], bwd[nVar], flux[nVar];
            // One element, unread, for a point flux without a second input.
            [[maybe_unused]] TData secondFwd[nSecond > 0 ? nSecond : 1],
                secondBwd[nSecond > 0 ? nSecond : 1];
            for (unsigned n = 0; n < nVar; ++n)
            {
                fwd[n] = inPtr[o0 + n * cs0];
                bwd[n] = in1[o1 + n * cs1];
            }
            if constexpr (numSecond > 0)
            {
                const size_t d0 = s0 * numSecond + b0 + q0;
                const size_t d1 = s1 * numSecond + b1 + q1;
                for (unsigned c = 0; c < numSecond; ++c)
                {
                    secondFwd[c] = secondOwn[d0 + c * cs0];
                    secondBwd[c] = secondNbr[d1 + c * cs1];
                }
            }

            TData normal[NDIM];
            for (unsigned d = 0; d < NDIM; ++d)
            {
                normal[d] = nrmTrace[d * nrmCS + ptInTrace];
            }
            TData penalty = TData(0.0);
            if constexpr (PointFlux::usesPenalty)
            {
                penalty = penaltyFactors[glo.scalarOffset[trace] + ptInTrace];
            }

            pointFluxCopy(fwd, bwd, secondFwd, secondBwd, normal, penalty,
                          flux);

            for (unsigned n = 0; n < nVar; ++n)
            {
                if constexpr (APPEND)
                {
                    fluxOut[o0 + n * cs0] += flux[n];
                    fluxOut[o1 + n * cs1] -= flux[n];
                }
                else
                {
                    fluxOut[o0 + n * cs0] = flux[n];
                    fluxOut[o1 + n * cs1] = -flux[n];
                }
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

#endif

} // namespace Nektar::SolverCore::detail
