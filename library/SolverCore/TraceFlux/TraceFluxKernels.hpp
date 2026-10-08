///////////////////////////////////////////////////////////////////////////////
//
// File: TraceFluxKernels.hpp
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

#pragma once

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractKernels.hpp>

#include "LocalRegions/ReOrientFaceKernel.hpp"
#include "StdRegions/StdRegions.hpp"

namespace Nektar::SolverCore::detail
{
using LocalRegions::ReOrientFaceKernel;
// Host and device: called from the host loops of the per-trace gather/scatter
// helpers and from inside the block launchers' parallel_for kernels.
// Their checks are host-device asserts: live on the host paths, nothing in
// device code, where an assert message cannot be built.
template <bool APPEND, bool NEGATE_INPUT, typename TData>
NEK_HOSTDEVICE_INLINE static void ReOrientEdgeKernel(
    const StdRegions::Orientation orient, const unsigned nq0, const TData *in,
    TData *out)
{
    NEK_HOSTDEVICE_ASSERTL1(
        in != out, "This routine cannot use the same input and output");

    // Input sign change if required.
    TData sign = (NEGATE_INPUT) ? -1.0 : 1.0;

    switch (orient)
    {
        case StdRegions::eForwards:
            // straight copy and add
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < nq0; ++i)
                {
                    out[i] += sign * in[i];
                }
            }
            else
            {
                for (unsigned i = 0; i < nq0; ++i)
                {
                    out[i] = sign * in[i];
                }
            }
            break;
        case StdRegions::eBackwards:
        {
            TData store;
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < (nq0 + 1u) / 2; ++i)
                {
                    store = sign * in[i];
                    out[i] += sign * in[nq0 - 1u - i];
                    out[nq0 - 1u - i] += store;
                }
            }
            else
            {
                for (unsigned i = 0; i < (nq0 + 1u) / 2; ++i)
                {
                    store             = sign * in[i];
                    out[i]            = sign * in[nq0 - 1u - i];
                    out[nq0 - 1u - i] = store;
                }
            }
        }
        break;
        default:
            NEK_HOSTDEVICE_ASSERTL1(false, "Unknown orientation");
            break;
    }
}

// Version with input and output offsets for interleaving, the edge
// counterpart of the offset ReOrientFaceKernel() below.
template <bool APPEND, bool NEGATE_INPUT, typename TData>
NEK_HOSTDEVICE_INLINE static void ReOrientEdgeKernel(
    const StdRegions::Orientation orient, const unsigned nq0, const TData *in,
    const unsigned inOffset, TData *out, const unsigned outOffset)
{
    NEK_HOSTDEVICE_ASSERTL1(
        in != out, "This routine cannot use the same input and output");

    // Input sign change if required.
    TData sign = (NEGATE_INPUT) ? -1.0 : 1.0;

    switch (orient)
    {
        case StdRegions::eForwards:
            // straight copy and add
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < nq0; ++i)
                {
                    out[i * outOffset] += sign * in[i * inOffset];
                }
            }
            else
            {
                for (unsigned i = 0; i < nq0; ++i)
                {
                    out[i * outOffset] = sign * in[i * inOffset];
                }
            }
            break;
        case StdRegions::eBackwards:
        {
            TData store;
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < (nq0 + 1u) / 2; ++i)
                {
                    store = sign * in[i * inOffset];
                    out[i * outOffset] += sign * in[(nq0 - 1u - i) * inOffset];
                    out[(nq0 - 1u - i) * outOffset] += store;
                }
            }
            else
            {
                for (unsigned i = 0; i < (nq0 + 1u) / 2; ++i)
                {
                    store              = sign * in[i * inOffset];
                    out[i * outOffset] = sign * in[(nq0 - 1u - i) * inOffset];
                    out[(nq0 - 1u - i) * outOffset] = store;
                }
            }
        }
        break;
        default:
            NEK_HOSTDEVICE_ASSERTL1(false, "Unknown orientation");
            break;
    }
}

/// The frame-changing face kernel lives in LocalRegions (see
/// LocalRegions/ReOrientFaceKernel.hpp for what the frames and @p Forwards
/// mean); the overload below adds the per-face offsets the gathers use.

// version with input and output offset for interleaving. Host-device for the
// same reason as the scalar ReOrientEdgeKernel above: the block launchers call
// it from inside their parallel_for kernels.
template <bool APPEND, bool NEGATE_INPUT, typename TData>
NEK_HOSTDEVICE_INLINE static void ReOrientFaceKernel(
    const StdRegions::Orientation orient, const unsigned nq0,
    const unsigned nq1, const TData *in, const unsigned inOffset, TData *out,
    const unsigned outOffset, bool Forwards)
{
    NEK_HOSTDEVICE_ASSERTL1(
        in != out, "This routine cannot use the same input and output");

    // Input sign change if required.
    TData sign = (NEGATE_INPUT) ? -1.0 : 1.0;
    switch (orient)
    {
        case StdRegions::eDir1FwdDir1_Dir2FwdDir2: // used for Tris & Quads
        {
            // straight copy
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < nq0 * nq1; ++i)
                {
                    out[i * outOffset] += sign * in[i * inOffset];
                }
            }
            else
            {
                for (unsigned i = 0; i < nq0 * nq1; ++i)
                {
                    out[i * outOffset] = sign * in[i * inOffset];
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir1_Dir2FwdDir2: // used for Tris & Quads
        {
            // Direction A negative and B positive
            TData store;
            unsigned nq0half = (nq0 + 1u) / 2;
            if constexpr (APPEND)
            {
                for (unsigned j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = jnq0 + nq0 - 1u;
                    for (unsigned i = 0; i < nq0half; ++i)
                    {
                        store = sign * in[(jnq0 + i) * inOffset];
                        out[(jnq0 + i) * outOffset] +=
                            sign * in[(jfac - i) * inOffset];
                        out[(jfac - i) * outOffset] += store;
                    }
                }
            }
            else
            {
                for (unsigned j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = jnq0 + nq0 - 1u;
                    for (unsigned i = 0; i < nq0half; ++i)
                    {
                        store = sign * in[(jnq0 + i) * inOffset];
                        out[(jnq0 + i) * outOffset] =
                            sign * in[(jfac - i) * inOffset];
                        out[(jfac - i) * outOffset] = store;
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir1_Dir2BwdDir2:
        {
            // Direction A positive and B negative
            if constexpr (APPEND)
            {
                for (int j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = nq0 * (nq1 - 1u - j);
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[(jnq0 + i) * outOffset] +=
                            sign * in[(jfac + i) * inOffset];
                    }
                }
            }
            else
            {
                for (int j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = nq0 * (nq1 - 1u - j);
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[(jnq0 + i) * outOffset] =
                            sign * in[(jfac + i) * inOffset];
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir1_Dir2BwdDir2:
        {
            // Direction A positive and B negative
            if constexpr (APPEND)
            {
                for (int j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = nq0 * nq1 - 1u - jnq0;
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[(jnq0 + i) * outOffset] +=
                            sign * in[(jfac - i) * inOffset];
                    }
                }
            }
            else
            {
                for (int j = 0; j < nq1; j++)
                {
                    auto jnq0 = j * nq0;
                    auto jfac = nq0 * nq1 - 1u - jnq0;
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[(jnq0 + i) * outOffset] =
                            sign * in[(jfac - i) * inOffset];
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir2_Dir2FwdDir1:
        {
            // Transposed, Direction A and B positive
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i * nq1 + j) * outOffset] +=
                                sign * in[(i + jnq0) * inOffset];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(jnq0 + i) * outOffset] +=
                                sign * in[(i * nq1 + j) * inOffset];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i * nq1 + j) * outOffset] =
                                sign * in[(i + jnq0) * inOffset];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(jnq0 + i) * outOffset] =
                                sign * in[(i * nq1 + j) * inOffset];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir2_Dir2BwdDir1:
        {
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    // forward case (element to trace) or (loc trace to trace)
                    // Transposed, Direction A positive and B negative
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = i + nq0 * (nq1 - 1u);
                        for (int j = 0; j < nq1; ++j)
                        {

                            out[(inq1 + j) * outOffset] +=
                                sign * in[(ifac - j * nq0) * inOffset];
                        }
                    }
                }
                else
                {
                    // inverse case (trace to element)
                    // Transposed, Direction A positive and B negative
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq1 - 1u - j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(jnq0 + i) * outOffset] +=
                                sign * in[(jfac + i * nq1) * inOffset];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    // forward case (element to trace) or (loc trace to trace)
                    // Transposed, Direction A positive and B negative
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = i + nq0 * (nq1 - 1u);
                        for (int j = 0; j < nq1; ++j)
                        {

                            out[(inq1 + j) * outOffset] =
                                sign * in[(ifac - j * nq0) * inOffset];
                        }
                    }
                }
                else
                {
                    // inverse case (trace to element)
                    // Transposed, Direction A positive and B negative
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq1 - 1u - j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(jnq0 + i) * outOffset] =
                                sign * in[(jfac + i * nq1) * inOffset];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir2_Dir2FwdDir1:
        {
            // Transposed, Direction A negative and B positive
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = nq0 - 1u - i;
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[(inq1 + j) * outOffset] +=
                                sign * in[(ifac + j * nq0) * inOffset];
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq1 * (nq0 - 1u) + j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i + jnq0) * outOffset] +=
                                sign * in[(jfac - i * nq1) * inOffset];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = nq0 - 1u - i;
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[(inq1 + j) * outOffset] =
                                sign * in[(ifac + j * nq0) * inOffset];
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq1 * (nq0 - 1u) + j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i + jnq0) * outOffset] =
                                sign * in[(jfac - i * nq1) * inOffset];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir2_Dir2BwdDir1:
        {
            // Transposed, Direction A and B negative
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = nq0 * nq1 - 1u - i;
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[(inq1 + j) * outOffset] +=
                                sign * in[(ifac - j * nq0) * inOffset];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq0 * nq1 - 1u - j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i + jnq0) * outOffset] +=
                                sign * in[(jfac - i * nq1) * inOffset];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        auto inq1 = i * nq1;
                        auto ifac = nq0 * nq1 - 1u - i;
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[(inq1 + j) * outOffset] =
                                sign * in[(ifac - j * nq0) * inOffset];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        auto jnq0 = j * nq0;
                        auto jfac = nq0 * nq1 - 1u - j;
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[(i + jnq0) * outOffset] =
                                sign * in[(jfac - i * nq1) * inOffset];
                        }
                    }
                }
            }
        }
        break;
        default:
            NEK_HOSTDEVICE_ASSERTL1(false, "Unknown orientation");
            break;
    }
}

/**
 * @brief Per-trace metadata for one side of a block, in the execution space's
 * memory.
 *
 * Trivially copyable and small, so it can be captured by value into a kernel.
 * The arrays it points at are owned by the operator and hold one entry per
 * trace in the block, in the order the traces were appended; @p orient is null
 * when the traces are points rather than edges or faces. Orientations are
 * stored as `int` rather than StdRegions::Orientation so that the backing
 * store is a plain arithmetic MemoryRegion.
 *
 * @see TraceFluxOpImpl::TraceDeviceData, which owns the storage.
 */
struct TraceBlockView
{
    const size_t *offset_st = nullptr;
    const size_t *offset    = nullptr;
    const size_t *compSize  = nullptr;
    const int *orient       = nullptr;
    size_t numTrace         = 0;
    /// Distance between consecutive points of one local edge or face: the
    /// element-local field's interleave width, whose layout @p offset then
    /// addresses. A point trace has one point, so its gather needs no stride.
    unsigned stride = 1;
    /// Per-point reorientation, built for the Device only: for trace @p t and
    /// global-trace point @p p, `ptMapFwd[t * ptsPerTrace + p]` is the local
    /// point that lands there; `ptMapBwd` is the same the other way, local
    /// point to the global point it takes. Null where no map was built.
    const unsigned *ptMapFwd = nullptr;
    const unsigned *ptMapBwd = nullptr;
    unsigned ptsPerTrace     = 0;
};

/**
 * @brief Where each trace of a block sits in the whole-mesh global trace, in
 * the execution space's memory.
 *
 * TraceBlockView addresses the *element-local* side of a block; this addresses
 * the other end, the fields carried on the global trace itself - the trace
 * normals, the advection velocity, the interior-penalty factor. Those are
 * mesh-wide arrays that the operators gather from, one trace at a time, into
 * the block-local packed layout the flux kernels read.
 *
 * Trivially copyable and small, so it can be captured by value into a kernel.
 * The arrays hold one entry per trace in the block, in the order the traces
 * were appended.
 *
 * @see TraceFluxOpImpl::GloTraceOffsetDeviceData, which owns the storage.
 */
struct GloTraceOffsetView
{
    /// Offset of the trace within a multi-component global-trace field.
    const size_t *gloTOffset = nullptr;
    /// Offset of the trace within a single-component one, which is packed
    /// differently and so cannot share the offset above.
    const size_t *scalarOffset = nullptr;
    /// Stride between components of a multi-component field, per trace.
    const size_t *compSize = nullptr;
};

/**
 * @brief Extents of a face trace after reorientation onto the global trace.
 *
 * TraceDetails::npts holds the *local* extents, in local direction order, and
 * that is what arrives here as @p nptsIn0 / @p nptsIn1. ReOrientFaceKernel()
 * consumes them directly: it reads the local trace, so its `nq0`/`nq1` are the
 * extents of its input.
 *
 * Its output is on the global trace, and a transposed orientation swaps the
 * two directions on the way. Anything working downstream of the reorientation
 * — PhysInterpFaceKernel(), whose interpolation matrices are indexed by global
 * direction — therefore needs the swapped pair, which is what this returns.
 * A no-op for every non-transposed orientation, and for a transposed one on an
 * isotropic face.
 *
 * @param orient - Orientation of the local trace relative to the global one.
 * @param n0     - Local extent in local direction 0.
 * @param n1     - Local extent in local direction 1.
 * @param nRe0   - Set to the reoriented extent in global direction 0.
 * @param nRe1   - Set to the reoriented extent in global direction 1.
 */
NEK_HOSTDEVICE_INLINE static void ReorientedFaceExtents(
    const StdRegions::Orientation orient, const unsigned n0, const unsigned n1,
    unsigned &nRe0, unsigned &nRe1)
{
    if (orient >= StdRegions::eDir1FwdDir2_Dir2FwdDir1) // transposed
    {
        nRe0 = n1;
        nRe1 = n0;
    }
    else
    {
        nRe0 = n0;
        nRe1 = n1;
    }
}

/**
 * @brief Gather every local trace of a block into the global trace, one
 * thread per point.
 *
 * The Device form of LocEdgeToGloEdgeTraces() and LocFaceToGloFaceTraces()
 * for a collocated side. Consecutive threads write consecutive points of the
 * global trace, so a warp's stores coalesce, and each reads its source through
 * the block's point map, which holds the reorientation the per-trace loop
 * would perform; the trace's dimension is already folded into the map, so
 * edges and faces share this kernel.
 *
 * @param numComp   - Components per trace point.
 * @param T         - Per-trace metadata for the local side of the block,
 *                    with its point map built.
 * @param npts      - Points per trace, local and global alike.
 * @param inPtr     - Element-local input, at the field base.
 * @param outPtr    - Packed global-trace output, at the base of the block.
 * @param outOffset - Component stride within @p outPtr.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocTraceToGloTracePoints(
    const unsigned numComp, const TraceBlockView &T, const unsigned npts,
    const TData *inPtr, TData *outPtr, const unsigned outOffset)
{
    ASSERTL1(T.ptMapFwd != nullptr && T.ptsPerTrace == npts,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * npts, NEKTAR_LAMBDA(const size_t idx) {
            const size_t p = idx % npts;
            const size_t r = idx / npts;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const TData *in = inPtr + T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t];
            outPtr[npts * t + n * outOffset + p] =
                in[T.ptMapFwd[t * npts + p] * T.stride];
        });
}

/**
 * @brief Scatter the global trace of a block back onto every local trace, one
 * thread per point.
 *
 * The inverse of LocTraceToGloTracePoints(): one thread per local point,
 * reading the global point the map assigns it and writing the element-local
 * field contiguously, with the sign and the append folded in.
 *
 * @tparam APPEND       Accumulate into @p outPtr rather than overwriting it.
 * @tparam NEGATE_INPUT Negate on the way out; see GloEdgeToLocEdgeBlock().
 *
 * @param numComp  - Components per trace point.
 * @param T        - Per-trace metadata for the local side of the block, with
 *                   its point map built.
 * @param npts     - Points per trace, local and global alike.
 * @param inPtr    - Packed global-trace input, at the base of the block.
 * @param inOffset - Component stride within @p inPtr.
 * @param outPtr   - Element-local output, at the field base.
 */
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloTraceToLocTracePoints(
    const unsigned numComp, const TraceBlockView &T, const unsigned npts,
    const TData *inPtr, const unsigned inOffset, TData *outPtr)
{
    ASSERTL1(T.ptMapBwd != nullptr && T.ptsPerTrace == npts,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;
    const TData sign      = NEGATE_INPUT ? TData(-1.0) : TData(1.0);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * npts, NEKTAR_LAMBDA(const size_t idx) {
            const size_t q = idx % npts;
            const size_t r = idx / npts;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const TData v =
                sign *
                inPtr[npts * t + n * inOffset + T.ptMapBwd[t * npts + q]];
            TData &o = outPtr[T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t] + q * T.stride];
            if (APPEND)
            {
                o += v;
            }
            else
            {
                o = v;
            }
        });
}

/**
 * @brief Gather every local edge of a block into the global trace, one
 * thread per global point, interpolating on the way.
 *
 * The Device form of LocEdgeToGloEdgeTraces() for a side that does not share
 * the global trace's point distribution. Each thread forms one global point
 * as the interpolation of the local points, read through the block's point
 * map, which holds the reorientation; the matrix is the one the per-trace
 * loop hands to PhysInterpEdgeKernel(), in its layout.
 *
 * @param numComp   - Components per trace point.
 * @param T         - Per-trace metadata for the local side, with its point
 *                    map built.
 * @param interp    - Local-to-global interpolation, `nptsIn` by `nptsOut`,
 *                    entry `[i * nptsOut + p]`.
 * @param nptsIn    - Points on the local edge.
 * @param inPtr     - Element-local input, at the field base.
 * @param nptsOut   - Points on the global edge.
 * @param outPtr    - Packed global-trace output, at the base of the block.
 * @param outOffset - Component stride within @p outPtr.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocEdgeToGloEdgePoints(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, const unsigned nptsOut,
    TData *outPtr, const unsigned outOffset)
{
    ASSERTL1(T.ptMapFwd != nullptr && T.ptsPerTrace == nptsIn,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * nptsOut, NEKTAR_LAMBDA(const size_t idx) {
            const size_t p = idx % nptsOut;
            const size_t r = idx / nptsOut;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const TData *in = inPtr + T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t];
            const unsigned *map = T.ptMapFwd + t * nptsIn;

            TData sum = 0.0;
            for (unsigned i = 0; i < nptsIn; ++i)
            {
                sum += in[map[i] * T.stride] * interp[i * nptsOut + p];
            }
            outPtr[nptsOut * t + n * outOffset + p] = sum;
        });
}

/**
 * @brief Scatter the global trace of a block back onto every local edge, one
 * thread per local point, interpolating on the way.
 *
 * The inverse of LocEdgeToGloEdgePoints(): each thread forms one local point
 * from the global points through the global-to-local matrix, at the index
 * the backward map assigns it, with the sign and the append folded in.
 *
 * @tparam APPEND       Accumulate into @p outPtr rather than overwriting it.
 * @tparam NEGATE_INPUT Negate on the way out; see GloEdgeToLocEdgeBlock().
 *
 * @param numComp  - Components per trace point.
 * @param T        - Per-trace metadata for the local side, with its point
 *                   map built.
 * @param interp   - Global-to-local interpolation, `nptsIn` by `nptsOut`,
 *                   entry `[p * nptsOut + k]`.
 * @param nptsIn   - Points on the global edge.
 * @param inPtr    - Packed global-trace input, at the base of the block.
 * @param inOffset - Component stride within @p inPtr.
 * @param nptsOut  - Points on the local edge.
 * @param outPtr   - Element-local output, at the field base.
 */
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloEdgeToLocEdgePoints(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, const unsigned inOffset,
    const unsigned nptsOut, TData *outPtr)
{
    ASSERTL1(T.ptMapBwd != nullptr && T.ptsPerTrace == nptsOut,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;
    const TData sign      = NEGATE_INPUT ? TData(-1.0) : TData(1.0);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * nptsOut, NEKTAR_LAMBDA(const size_t idx) {
            const size_t q = idx % nptsOut;
            const size_t r = idx / nptsOut;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const TData *in  = inPtr + nptsIn * t + n * inOffset;
            const unsigned k = T.ptMapBwd[t * nptsOut + q];

            TData sum = 0.0;
            for (unsigned p = 0; p < nptsIn; ++p)
            {
                sum += in[p] * interp[p * nptsOut + k];
            }
            TData &o = outPtr[T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t] + q * T.stride];
            if (APPEND)
            {
                o += sign * sum;
            }
            else
            {
                o = sign * sum;
            }
        });
}

/**
 * @brief Gather every local face of a block into the global trace, one
 * thread per global point, interpolating on the way.
 *
 * The face form of LocEdgeToGloEdgePoints(). The local points are read in
 * the global frame through the point map, whose extents are the local ones
 * reordered by the trace's orientation (ReorientedFaceExtents()), and the
 * two-direction interpolation is summed directly; a collocated direction
 * contributes the identity, as in PhysInterpFaceKernel().
 *
 * @param numComp     - Components per trace point.
 * @param T           - Per-trace metadata for the local side, with its point
 *                      map built.
 * @param interp0     - Local-to-global interpolation in trace direction 0,
 *                      null if that direction is collocated.
 * @param interp1     - As @p interp0, for direction 1.
 * @param nptsIn0     - Points on the local face in direction 0.
 * @param nptsIn1     - Points on the local face in direction 1.
 * @param inPtr       - Element-local input, at the field base.
 * @param nptsOut0    - Points on the global face in direction 0.
 * @param nptsOut1    - Points on the global face in direction 1.
 * @param outPtr      - Packed global-trace output, at the base of the block.
 * @param outOffset   - Component stride within @p outPtr.
 * @param Collocated0 - Direction 0 needs no interpolation.
 * @param Collocated1 - Direction 1 needs no interpolation.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocFaceToGloFacePoints(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, const unsigned nptsOut0, const unsigned nptsOut1,
    TData *outPtr, const unsigned outOffset, const bool Collocated0,
    const bool Collocated1)
{
    const auto nptsIn  = nptsIn0 * nptsIn1;
    const auto nptsOut = nptsOut0 * nptsOut1;

    ASSERTL1(T.ptMapFwd != nullptr && T.ptsPerTrace == nptsIn,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * nptsOut, NEKTAR_LAMBDA(const size_t idx) {
            const size_t p = idx % nptsOut;
            const size_t r = idx / nptsOut;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const unsigned p0 = p % nptsOut0;
            const unsigned p1 = p / nptsOut0;

            const TData *in = inPtr + T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t];
            const unsigned *map = T.ptMapFwd + t * nptsIn;

            // Extents of the local points once reoriented into the global
            // frame, which is the frame the map and the matrices work in.
            unsigned nRe0, nRe1;
            ReorientedFaceExtents(StdRegions::Orientation(T.orient[t]), nptsIn0,
                                  nptsIn1, nRe0, nRe1);

            // A collocated direction takes the one matching point.
            const unsigned i0 = Collocated0 ? p0 : 0u;
            const unsigned i1 = Collocated0 ? p0 + 1u : nRe0;
            const unsigned j0 = Collocated1 ? p1 : 0u;
            const unsigned j1 = Collocated1 ? p1 + 1u : nRe1;

            TData sum = 0.0;
            for (unsigned j = j0; j < j1; ++j)
            {
                const TData w1 =
                    Collocated1 ? TData(1.0) : interp1[j * nptsOut1 + p1];
                for (unsigned i = i0; i < i1; ++i)
                {
                    const TData w0 =
                        Collocated0 ? TData(1.0) : interp0[i * nptsOut0 + p0];
                    sum += in[map[i + j * nRe0] * T.stride] * w0 * w1;
                }
            }
            outPtr[nptsOut * t + n * outOffset + p] = sum;
        });
}

/**
 * @brief Scatter the global trace of a block back onto every local face, one
 * thread per local point, interpolating on the way.
 *
 * The inverse of LocFaceToGloFacePoints(): the backward map gives each local
 * point its index in the reoriented frame, whose extents are the local ones
 * reordered by the orientation, and the two-direction interpolation from the
 * global points is summed there, with the sign and the append folded in.
 *
 * @param numComp     - Components per trace point.
 * @param T           - Per-trace metadata for the local side, with its point
 *                      map built.
 * @param interp0     - Global-to-local interpolation in trace direction 0,
 *                      null if that direction is collocated.
 * @param interp1     - As @p interp0, for direction 1.
 * @param nptsIn0     - Points on the global face in direction 0.
 * @param nptsIn1     - Points on the global face in direction 1.
 * @param inPtr       - Packed global-trace input, at the base of the block.
 * @param inOffset    - Component stride within @p inPtr.
 * @param nptsOut0    - Points on the local face in direction 0.
 * @param nptsOut1    - Points on the local face in direction 1.
 * @param outPtr      - Element-local output, at the field base.
 * @param Collocated0 - Direction 0 needs no interpolation.
 * @param Collocated1 - Direction 1 needs no interpolation.
 */
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloFaceToLocFacePoints(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, const unsigned inOffset, const unsigned nptsOut0,
    const unsigned nptsOut1, TData *outPtr, const bool Collocated0,
    const bool Collocated1)
{
    const auto nptsIn  = nptsIn0 * nptsIn1;
    const auto nptsOut = nptsOut0 * nptsOut1;

    ASSERTL1(T.ptMapBwd != nullptr && T.ptsPerTrace == nptsOut,
             "No point map for this block at this size");

    const size_t numTrace = T.numTrace;
    const TData sign      = NEGATE_INPUT ? TData(-1.0) : TData(1.0);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp * nptsOut, NEKTAR_LAMBDA(const size_t idx) {
            const size_t q = idx % nptsOut;
            const size_t r = idx / nptsOut;
            const size_t t = r % numTrace;
            const size_t n = r / numTrace;

            const TData *in = inPtr + nptsIn * t + n * inOffset;

            unsigned nRe0, nRe1;
            ReorientedFaceExtents(StdRegions::Orientation(T.orient[t]),
                                  nptsOut0, nptsOut1, nRe0, nRe1);

            const unsigned k  = T.ptMapBwd[t * nptsOut + q];
            const unsigned k0 = k % nRe0;
            const unsigned k1 = k / nRe0;

            const unsigned p00 = Collocated0 ? k0 : 0u;
            const unsigned p01 = Collocated0 ? k0 + 1u : nptsIn0;
            const unsigned p10 = Collocated1 ? k1 : 0u;
            const unsigned p11 = Collocated1 ? k1 + 1u : nptsIn1;

            TData sum = 0.0;
            for (unsigned p1 = p10; p1 < p11; ++p1)
            {
                const TData w1 =
                    Collocated1 ? TData(1.0) : interp1[p1 * nRe1 + k1];
                for (unsigned p0 = p00; p0 < p01; ++p0)
                {
                    const TData w0 =
                        Collocated0 ? TData(1.0) : interp0[p0 * nRe0 + k0];
                    sum += in[p0 + p1 * nptsIn0] * w0 * w1;
                }
            }
            TData &o = outPtr[T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t] + q * T.stride];
            if (APPEND)
            {
                o += sign * sum;
            }
            else
            {
                o = sign * sum;
            }
        });
}

// One thread per trace: the loop LocEdgeToGloEdgeBlock() runs on Serial and
// AVX.
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocEdgeToGloEdgeTraces(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, TData *wsp,
    const unsigned nptsOut, TData *outPtr, const unsigned outOffset,
    const bool Collocated)
{
    ASSERTL1(!Collocated || (nptsIn == nptsOut), "Input is not collocated");

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            const TData *in = inPtr + T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t];
            TData *out = outPtr + nptsOut * t + n * outOffset;

            const auto orient = StdRegions::Orientation(T.orient[t]);

            if (Collocated)
            {
                ReOrientEdgeKernel<false, false>(orient, nptsIn, in, T.stride,
                                                 out, 1u);
            }
            else
            {
                // Private slice: every thread reorients before interpolating.
                TData *w = wsp + idx * nptsIn;

                ReOrientEdgeKernel<false, false>(orient, nptsIn, in, T.stride,
                                                 w, 1u);
                MultiRegions::detail::PhysInterpEdgeKernel<TData>(
                    1u, nptsIn, nptsOut, interp, w, out, false);
            }
        });
}

/**
 * @brief Gather every edge trace of a block onto the global trace.
 *
 * One launch covers all `numTrace * numComp` edges. Each thread owns one
 * (trace, component) pair, reads its input through @p T rather than from
 * host-side offsets, and writes `nptsOut` contiguous points, so no two threads
 * touch the same output.
 *
 * Addressing the block through @p T is what makes this usable on a device at
 * all: the alternative is a host loop over the traces, reading the offsets from
 * the host-side TraceDetails and forming pointers into the trace field, which
 * is in the execution space's memory and so cannot be dereferenced from the
 * host.
 *
 * @param numComp   - Components per trace point.
 * @param T         - Per-trace metadata for this side of the block.
 * @param interp    - Local-to-global interpolation matrix, null if collocated.
 * @param nptsIn    - Points on the local trace.
 * @param inPtr     - Physical-space input, block layout, at the field base.
 * @param wsp       - Scratch, at least `numTrace * numComp * nptsIn`. Unused
 *                    when @p Collocated.
 * @param nptsOut   - Points on the global trace.
 * @param outPtr    - Packed global-trace output, at the base of the block.
 * @param outOffset - Component stride within @p outPtr.
 * @param Collocated - Local and global traces share a point distribution, so
 *                     reorientation alone suffices.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocEdgeToGloEdgeBlock(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, TData *wsp,
    const unsigned nptsOut, TData *outPtr, const unsigned outOffset,
    const bool Collocated, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // The device runs one thread per point through the block's point map,
    // interpolating in the thread where the side needs it; Serial and AVX
    // run one thread per trace.
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        if (Collocated)
        {
            LocTraceToGloTracePoints<ExecSpace>(numComp, T, nptsIn, inPtr,
                                                outPtr, outOffset);
        }
        else
        {
            LocEdgeToGloEdgePoints<ExecSpace>(numComp, T, interp, nptsIn, inPtr,
                                              nptsOut, outPtr, outOffset);
        }
    }
    else
    {
        LocEdgeToGloEdgeTraces<ExecSpace>(numComp, T, interp, nptsIn, inPtr,
                                          wsp, nptsOut, outPtr, outOffset,
                                          Collocated);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

// One thread per trace: the loop GloEdgeToLocEdgeBlock() runs on Serial and
// AVX.
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloEdgeToLocEdgeTraces(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, const unsigned inOffset,
    TData *wsp, const unsigned nptsOut, TData *outPtr, const bool Collocated)
{
    ASSERTL1(!Collocated || (nptsIn == nptsOut), "Input is not collocated");

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            const TData *in = inPtr + nptsIn * t + n * inOffset;
            TData *out      = outPtr + T.offset_st[t] * numComp + T.offset[t] +
                         n * T.compSize[t];

            const auto orient = StdRegions::Orientation(T.orient[t]);

            if (Collocated)
            {
                ReOrientEdgeKernel<APPEND, NEGATE_INPUT>(orient, nptsIn, in, 1u,
                                                         out, T.stride);
            }
            else
            {
                // Private slice: every thread interpolates before reorienting.
                TData *w = wsp + idx * nptsOut;

                MultiRegions::detail::PhysInterpEdgeKernel<TData>(
                    1u, nptsIn, nptsOut, interp, in, w, false);
                ReOrientEdgeKernel<APPEND, NEGATE_INPUT>(orient, nptsOut, w, 1u,
                                                         out, T.stride);
            }
        });
}

/**
 * @brief Scatter the global trace of a block back onto every local edge.
 *
 * The inverse of LocEdgeToGloEdgeBlock(), and its mirror image in structure:
 * one launch, one thread per (trace, component) pair, the local side addressed
 * through @p T. Interpolation now precedes reorientation, since the data
 * travels from the global trace to the local one.
 *
 * Each thread writes the `nptsOut` points of one component of one local edge.
 * Distinct traces occupy distinct element-local storage, so with @p APPEND set
 * the accumulation is still race-free - the same assumption
 * LocEdgeToGloEdgeBlock() makes about its outputs.
 *
 * @tparam APPEND       Accumulate into @p outPtr rather than overwriting it.
 * @tparam ExecSpace    Execution space the loop runs in.
 * @tparam NEGATE_INPUT Negate on the way out. Used for the backward side of an
 *                      interior trace, which sees the opposite normal, so that
 *                      the sign costs no extra pass over the flux.
 * @tparam TData        Floating-point representation.
 *
 * @param numComp    - Components per trace point.
 * @param T          - Per-trace metadata for the local side of the block.
 * @param interp     - Global-to-local interpolation matrix, null if collocated.
 * @param nptsIn     - Points on the global trace.
 * @param inPtr      - Packed global-trace input, at the base of the block.
 * @param inOffset   - Component stride within @p inPtr.
 * @param wsp        - Scratch, at least `numTrace * numComp * nptsOut`. Unused
 *                     when @p Collocated.
 * @param nptsOut    - Points on the local trace.
 * @param outPtr     - Element-local output, block layout, at the field base.
 * @param Collocated - Local and global traces share a point distribution, so
 *                     reorientation alone suffices.
 */
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloEdgeToLocEdgeBlock(
    const unsigned numComp, const TraceBlockView &T, const TData *interp,
    const unsigned nptsIn, const TData *inPtr, const unsigned inOffset,
    TData *wsp, const unsigned nptsOut, TData *outPtr, const bool Collocated,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // The device runs one thread per point through the block's point map,
    // interpolating in the thread where the side needs it; Serial and AVX
    // run one thread per trace.
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        if (Collocated)
        {
            GloTraceToLocTracePoints<APPEND, ExecSpace, NEGATE_INPUT>(
                numComp, T, nptsOut, inPtr, inOffset, outPtr);
        }
        else
        {
            GloEdgeToLocEdgePoints<APPEND, ExecSpace, NEGATE_INPUT>(
                numComp, T, interp, nptsIn, inPtr, inOffset, nptsOut, outPtr);
        }
    }
    else
    {
        GloEdgeToLocEdgeTraces<APPEND, ExecSpace, NEGATE_INPUT>(
            numComp, T, interp, nptsIn, inPtr, inOffset, wsp, nptsOut, outPtr,
            Collocated);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

// One thread per trace: the loop LocFaceToGloFaceBlock() runs on Serial and
// AVX.
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocFaceToGloFaceTraces(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, TData *loc_wsp, TData *wsp, const unsigned nptsOut0,
    const unsigned nptsOut1, TData *outPtr, const unsigned outOffset,
    const bool Collocated0, const bool Collocated1)
{
    // Orientations vary within a block, so the per-direction correspondence
    // between the local and global extents is only known per trace; compare
    // the totals, which hold either way. Same reasoning in
    // GloFaceToLocFaceBlock().
    ASSERTL1(!(Collocated0 && Collocated1) ||
                 (nptsIn0 * nptsIn1 == nptsOut0 * nptsOut1),
             "Input is not collocated");

    const size_t numTrace = T.numTrace;
    const auto nptsIn     = nptsIn0 * nptsIn1;
    const auto nptsOut    = nptsOut0 * nptsOut1;

    // Scratch one PhysInterpFaceKernel() call consumes: its nqto0 by its
    // nqfrom1. Here nqfrom1 is the reoriented extent, so it is nptsIn1 or
    // nptsIn0 depending on the trace's orientation - take the larger.
    const auto wspStride = nptsOut0 * std::max(nptsIn0, nptsIn1);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            const TData *in = inPtr + T.offset_st[t] * numComp + T.offset[t] +
                              n * T.compSize[t];
            TData *out = outPtr + nptsOut * t + n * outOffset;

            const auto orient = StdRegions::Orientation(T.orient[t]);

            // nptsIn0/nptsIn1 are the local extents, which is what the
            // reorientation wants.
            if (Collocated0 && Collocated1)
            {
                ReOrientFaceKernel<false, false>(orient, nptsIn0, nptsIn1, in,
                                                 T.stride, out, 1u, true);
            }
            else
            {
                // Private slices: every thread reorients then interpolates.
                TData *lw = loc_wsp + idx * nptsIn;

                ReOrientFaceKernel<false, false>(orient, nptsIn0, nptsIn1, in,
                                                 T.stride, lw, 1u, true);

                // The interpolation runs on the reoriented data, so it needs
                // the extents reordered into the global frame.
                unsigned nRe0, nRe1;
                ReorientedFaceExtents(orient, nptsIn0, nptsIn1, nRe0, nRe1);

                MultiRegions::detail::PhysInterpFaceKernel<TData>(
                    1u, nRe0, nRe1, nptsOut0, nptsOut1, interp0, interp1,
                    wsp + idx * wspStride, lw, out, Collocated0, Collocated1);
            }
        });
}

/**
 * @brief Gather every face trace of a block onto the global trace.
 *
 * The face counterpart of LocEdgeToGloEdgeBlock(): one launch, one thread per
 * (trace, component) pair, inputs addressed through @p T, and each thread
 * writing its own `nptsOut0 * nptsOut1` points.
 *
 * @param numComp    - Components per trace point.
 * @param T          - Per-trace metadata for this side of the block.
 * @param interp0    - Local-to-global interpolation in trace direction 0,
 *                     null if that direction is collocated.
 * @param interp1    - As @p interp0, for direction 1.
 * @param nptsIn0    - Points on the local trace in direction 0.
 * @param nptsIn1    - Points on the local trace in direction 1.
 * @param inPtr      - Physical-space input, block layout, at the field base.
 * @param loc_wsp    - Scratch holding the reoriented local trace, at least
 *                     `numTrace * numComp * nptsIn0 * nptsIn1`.
 * @param wsp        - Scratch for the interpolation itself, at least
 *                     `numTrace * numComp * nptsOut0 * nptsIn1`. Both are
 *                     unused when the trace is collocated in both directions.
 * @param nptsOut0   - Points on the global trace in direction 0.
 * @param nptsOut1   - Points on the global trace in direction 1.
 * @param outPtr     - Packed global-trace output, at the base of the block.
 * @param outOffset  - Component stride within @p outPtr.
 * @param Collocated0 - Direction 0 needs no interpolation.
 * @param Collocated1 - Direction 1 needs no interpolation.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocFaceToGloFaceBlock(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, TData *loc_wsp, TData *wsp, const unsigned nptsOut0,
    const unsigned nptsOut1, TData *outPtr, const unsigned outOffset,
    const bool Collocated0, const bool Collocated1,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // The device runs one thread per point through the block's point map,
    // interpolating in the thread where the side needs it; Serial and AVX
    // run one thread per trace.
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        if (Collocated0 && Collocated1)
        {
            LocTraceToGloTracePoints<ExecSpace>(numComp, T, nptsIn0 * nptsIn1,
                                                inPtr, outPtr, outOffset);
        }
        else
        {
            LocFaceToGloFacePoints<ExecSpace>(
                numComp, T, interp0, interp1, nptsIn0, nptsIn1, inPtr, nptsOut0,
                nptsOut1, outPtr, outOffset, Collocated0, Collocated1);
        }
    }
    else
    {
        LocFaceToGloFaceTraces<ExecSpace>(
            numComp, T, interp0, interp1, nptsIn0, nptsIn1, inPtr, loc_wsp, wsp,
            nptsOut0, nptsOut1, outPtr, outOffset, Collocated0, Collocated1);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

// One thread per trace: the loop GloFaceToLocFaceBlock() runs on Serial and
// AVX.
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloFaceToLocFaceTraces(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, const unsigned inOffset, TData *glo_wsp, TData *wsp,
    const unsigned nptsOut0, const unsigned nptsOut1, TData *outPtr,
    const bool Collocated0, const bool Collocated1)
{
    ASSERTL1(!(Collocated0 && Collocated1) ||
                 (nptsIn0 * nptsIn1 == nptsOut0 * nptsOut1),
             "Input is not collocated");

    const size_t numTrace = T.numTrace;
    const auto nptsIn     = nptsIn0 * nptsIn1;
    const auto nptsOut    = nptsOut0 * nptsOut1;

    // Scratch one PhysInterpFaceKernel() call consumes: its nqto0 by its
    // nqfrom1. Here nqto0 is the reoriented local extent, so nptsOut0 or
    // nptsOut1 depending on the trace's orientation - take the larger, as
    // LocFaceToGloFaceBlock() does on its own side.
    const auto wspStride = std::max(nptsOut0, nptsOut1) * nptsIn1;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            const TData *in = inPtr + nptsIn * t + n * inOffset;
            TData *out      = outPtr + T.offset_st[t] * numComp + T.offset[t] +
                         n * T.compSize[t];

            const auto orient = StdRegions::Orientation(T.orient[t]);

            // nptsOut0/nptsOut1 are the local extents, which is what the
            // reorientation wants.
            if (Collocated0 && Collocated1)
            {
                ReOrientFaceKernel<APPEND, NEGATE_INPUT>(
                    orient, nptsOut0, nptsOut1, in, 1u, out, T.stride, false);
            }
            else
            {
                // Private slices: every thread interpolates then reorients.
                TData *gw = glo_wsp + idx * nptsOut;

                // The interpolation feeds the reorientation, so it targets the
                // local extents reordered into the global frame.
                unsigned nRe0, nRe1;
                ReorientedFaceExtents(orient, nptsOut0, nptsOut1, nRe0, nRe1);

                MultiRegions::detail::PhysInterpFaceKernel<TData>(
                    1u, nptsIn0, nptsIn1, nRe0, nRe1, interp0, interp1,
                    wsp + idx * wspStride, in, gw, Collocated0, Collocated1);

                ReOrientFaceKernel<APPEND, NEGATE_INPUT>(
                    orient, nptsOut0, nptsOut1, gw, 1u, out, T.stride, false);
            }
        });
}

/**
 * @brief Scatter the global trace of a block back onto every local face.
 *
 * The inverse of LocFaceToGloFaceBlock(), and the face counterpart of
 * GloEdgeToLocEdgeBlock(). Going this way @p nptsIn0 / @p nptsIn1 are the
 * global trace and @p nptsOut0 / @p nptsOut1 the local one; the reorientation
 * runs last, mapping global back to local, and takes the local extents
 * directly.
 *
 * @tparam APPEND       Accumulate into @p outPtr rather than overwriting it.
 * @tparam ExecSpace    Execution space the loop runs in.
 * @tparam NEGATE_INPUT Negate on the way out; see GloEdgeToLocEdgeBlock().
 * @tparam TData        Floating-point representation.
 *
 * @param numComp     - Components per trace point.
 * @param T           - Per-trace metadata for the local side of the block.
 * @param interp0     - Global-to-local interpolation in trace direction 0,
 *                      null if that direction is collocated.
 * @param interp1     - As @p interp0, for direction 1.
 * @param nptsIn0     - Points on the global trace in direction 0.
 * @param nptsIn1     - Points on the global trace in direction 1.
 * @param inPtr       - Packed global-trace input, at the base of the block.
 * @param inOffset    - Component stride within @p inPtr.
 * @param glo_wsp     - Scratch holding the interpolated trace before it is
 *                      reoriented, at least
 *                      `numTrace * numComp * nptsOut0 * nptsOut1`.
 * @param wsp         - Scratch for the interpolation itself, at least
 *                      `numTrace * numComp * max(nptsOut0, nptsOut1) *
 *                      max(nptsIn0, nptsIn1)`. Both are unused when the trace
 *                      is collocated in both directions.
 * @param nptsOut0    - Points on the local trace in direction 0.
 * @param nptsOut1    - Points on the local trace in direction 1.
 * @param outPtr      - Element-local output, block layout, at the field base.
 * @param Collocated0 - Direction 0 needs no interpolation.
 * @param Collocated1 - Direction 1 needs no interpolation.
 */
template <bool APPEND, typename ExecSpace, bool NEGATE_INPUT, typename TData>
NEK_FORCE_INLINE static void GloFaceToLocFaceBlock(
    const unsigned numComp, const TraceBlockView &T, const TData *interp0,
    const TData *interp1, const unsigned nptsIn0, const unsigned nptsIn1,
    const TData *inPtr, const unsigned inOffset, TData *glo_wsp, TData *wsp,
    const unsigned nptsOut0, const unsigned nptsOut1, TData *outPtr,
    const bool Collocated0, const bool Collocated1,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // The device runs one thread per point through the block's point map,
    // interpolating in the thread where the side needs it; Serial and AVX
    // run one thread per trace.
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        if (Collocated0 && Collocated1)
        {
            GloTraceToLocTracePoints<APPEND, ExecSpace, NEGATE_INPUT>(
                numComp, T, nptsOut0 * nptsOut1, inPtr, inOffset, outPtr);
        }
        else
        {
            GloFaceToLocFacePoints<APPEND, ExecSpace, NEGATE_INPUT>(
                numComp, T, interp0, interp1, nptsIn0, nptsIn1, inPtr, inOffset,
                nptsOut0, nptsOut1, outPtr, Collocated0, Collocated1);
        }
    }
    else
    {
        GloFaceToLocFaceTraces<APPEND, ExecSpace, NEGATE_INPUT>(
            numComp, T, interp0, interp1, nptsIn0, nptsIn1, inPtr, inOffset,
            glo_wsp, wsp, nptsOut0, nptsOut1, outPtr, Collocated0, Collocated1);
    }

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Scatter the global trace of a block back onto every local point.
 *
 * The one-dimensional member of the family, alongside GloEdgeToLocEdgeBlock()
 * and GloFaceToLocFaceBlock(). A point trace carries a single value, so there
 * is nothing to reorient and nothing to interpolate: the whole operation is
 * one indexed write per (trace, component) pair. It is the exact inverse of
 * the `TRACEDIM == 0` branch of TraceFluxOpImpl::GetInteriorTraces().
 *
 * @tparam APPEND    Accumulate into @p outPtr rather than overwriting it.
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam TData     Floating-point representation.
 *
 * @param numComp  - Components per trace point.
 * @param T        - Per-trace metadata for the local side of the block.
 * @param npTBlock - Component stride within @p inPtr.
 * @param sign     - Scaling applied on the way out. The backward side of an
 *                   interior trace passes -1, seeing the opposite normal;
 *                   everything else passes 1.
 * @param inPtr    - Packed global-trace input, at the base of the block.
 * @param outPtr   - Element-local output, block layout, at the field base.
 */
/**
 * @brief Impose the Neumann boundary gradient on the exterior side of a block.
 *
 * The gradient counterpart of FillDirBCTraceBlock(). A Neumann condition
 * prescribes the normal derivative, so the exterior gradient is the interior
 * one with twice the shortfall in its normal component added back, leaving
 * the average normal derivative equal to the prescribed value. Components
 * that are not Neumann keep the exterior gradient equal to the interior one.
 *
 * The boundary trace defines the global trace here, so the reorientation
 * FillDirBCTraceBlock() applies is the identity and the prescribed values are
 * read straight through.
 *
 * @param numBlock,numComp - Traces in the block, and components per point.
 * @param ndim       - Gradient components per field component.
 * @param singleBlock,singleBlockId,singleBlockBase,compStride - The fast path
 *                   for a block drawing on one storage block at one stride.
 * @param blk,off,cOff,bases - Per-trace addressing of the boundary storage
 *                   otherwise.
 * @param bcType,numBCComp,neuCode - Condition type per storage block and
 *                   component, and the code that means Neumann.
 * @param npTot,npTBlock - Points per trace and the component stride.
 * @param norms      - Trace normals, packed as the gradient is.
 * @param gloDerivT0 - Interior gradient, read.
 * @param gloDerivT1 - Exterior gradient, written.
 */
template <typename ExecSpace, typename TData, typename TBlk, typename TOff,
          typename TCode>
NEK_FORCE_INLINE static void FillNeuBCDerivBlock(
    const size_t numBlock, const unsigned numComp, const unsigned ndim,
    const bool singleBlock, const unsigned singleBlockId,
    const TData *singleBlockBase, const size_t compStride, const TBlk *blk,
    const TOff *off, const TOff *cOff, const TData *const *bases,
    const TCode *bcType, const unsigned numBCComp, const unsigned neuCode,
    const size_t npTot, const size_t npTBlock, const TData *norms,
    const TData *gloDerivT0, TData *gloDerivT1, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numBlock * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numBlock;
            const size_t v = idx / numBlock;

            const size_t tOff = npTot * t;

            const unsigned sBlk = singleBlock ? singleBlockId : blk[t];

            if (bcType[sBlk * numBCComp + v] != neuCode)
            {
                for (unsigned d = 0; d < ndim; ++d)
                {
                    const size_t o = (v * ndim + d) * npTBlock + tOff;
                    for (unsigned i = 0; i < npTot; ++i)
                    {
                        gloDerivT1[o + i] = gloDerivT0[o + i];
                    }
                }
                return;
            }

            const TData *f = singleBlock
                                 ? singleBlockBase + off[t] + v * compStride
                                 : bases[blk[t]] + off[t] + v * cOff[t];

            for (unsigned i = 0; i < npTot; ++i)
            {
                TData nDotG = TData(0);
                for (unsigned d = 0; d < ndim; ++d)
                {
                    nDotG += norms[d * npTBlock + tOff + i] *
                             gloDerivT0[(v * ndim + d) * npTBlock + tOff + i];
                }

                const TData corr = TData(2) * (f[i] - nDotG);

                for (unsigned d = 0; d < ndim; ++d)
                {
                    const size_t o = (v * ndim + d) * npTBlock + tOff + i;
                    gloDerivT1[o] =
                        gloDerivT0[o] + corr * norms[d * npTBlock + tOff + i];
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Impose the Dirichlet boundary state on the exterior side of a block.
 *
 * Each trace and component either takes its value from the boundary storage,
 * reoriented onto the global trace, or - where the condition is not Dirichlet
 * - copies the interior side across so the trace carries no jump. For a
 * Neumann component that copy is also what makes the penalty term vanish,
 * which is correct: the state is not prescribed there and the condition is
 * carried by the gradient instead.
 *
 * @tparam TRACEDIM Trace dimension: a point, an edge or a face.
 *
 * @param numBlock,numComp - Traces in the block, and components per point.
 * @param singleBlock,singleBlockId,singleBlockBase,compStride - The fast path
 *                   for a block drawing on one storage block at one stride.
 * @param blk,off,cOff,bases - Per-trace addressing of the boundary storage
 *                   otherwise.
 * @param bcType,numBCComp,dirCode - Condition type per storage block and
 *                   component, and the code that means Dirichlet.
 * @param npTot,npTBlock,np0,np1 - Points per trace, component stride, and the
 *                   in-trace point counts the reorientation needs.
 * @param gloT0      - Interior side, read where the condition is not Dirichlet.
 * @param gloT1      - Exterior side, written.
 */
template <typename ExecSpace, unsigned TRACEDIM, typename TData, typename TBlk,
          typename TOff, typename TCode>
NEK_FORCE_INLINE static void FillDirBCTraceBlock(
    const size_t numBlock, const unsigned numComp, const bool singleBlock,
    const unsigned singleBlockId, const TData *singleBlockBase,
    const size_t compStride, const TBlk *blk, const TOff *off, const TOff *cOff,
    const TData *const *bases, const TCode *bcType, const unsigned numBCComp,
    const unsigned dirCode, const size_t npTot, const size_t npTBlock,
    const unsigned np0, const unsigned np1, const TData *gloT0, TData *gloT1,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numBlock * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numBlock;
            const size_t n = idx / numBlock;

            // singleBlock is a scalar argument, so this branch is the same
            // for every thread in the launch
            const unsigned sBlk = singleBlock ? singleBlockId : blk[t];

            if (bcType[sBlk * numBCComp + n] != dirCode)
            {
                TData *out       = gloT1 + npTot * t + n * npTBlock;
                const TData *src = gloT0 + npTot * t + n * npTBlock;
                for (unsigned i = 0; i < npTot; ++i)
                {
                    out[i] = src[i];
                }
                return;
            }

            const TData *in = singleBlock
                                  ? singleBlockBase + off[t] + n * compStride
                                  : bases[blk[t]] + off[t] + n * cOff[t];

            // Plain if, not if constexpr: np0 and np1 are first captured
            // inside these branches, and nvcc's extended lambda does not
            // record a capture whose first use sits in a constexpr-if branch;
            // TRACEDIM is a template constant, so the dead branches fold
            // either way.
            if (TRACEDIM == 0)
            {
                gloT1[n * npTBlock + t] = in[0];
            }
            else if (TRACEDIM == 1)
            {
                ReOrientEdgeKernel<false, false>(StdRegions::eForwards, np0, in,
                                                 gloT1 + npTot * t +
                                                     n * npTBlock);
            }
            else if (TRACEDIM == 2)
            {
                ReOrientFaceKernel<false, false>(
                    StdRegions::eDir1FwdDir1_Dir2FwdDir2, np0, np1, in,
                    gloT1 + npTot * t + n * npTBlock, true);
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Scatter the interior state of a boundary block into its storage.
 *
 * Writes what the gather put on the global trace back into the boundary
 * condition operator's own blocks, so a caller sees the interior state in
 * the orientation an evaluated condition would have had.
 *
 * @param numComp  - Components per trace point.
 * @param numBlock - Traces in the block.
 * @param npTot    - Points per trace.
 * @param npTBlock - Component stride within @p gloT0.
 * @param blk,off,cOff - Per-trace addressing of the boundary storage.
 * @param owned    - Whether the caller will transform each storage block;
 *                   blocks it will not are left as the session filled them.
 * @param bases    - Base pointer of each storage block.
 * @param gloT0    - Packed global-trace input, at the base of the block.
 */
template <typename ExecSpace, typename TData, typename TBlk, typename TOff,
          typename TFlag>
NEK_FORCE_INLINE static void ScatterGloTraceToBndStore(
    const unsigned numComp, const size_t numBlock, const size_t npTot,
    const size_t npTBlock, const TBlk *blk, const TOff *off, const TOff *cOff,
    const TFlag *owned, TData *const *bases, const TData *gloT0,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numBlock * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numBlock;
            const size_t n = idx / numBlock;

            // Leave alone any block the caller will not transform: its
            // values came from the session and nothing would put them back.
            if (!owned[blk[t]])
            {
                return;
            }

            TData *out      = bases[blk[t]] + off[t] + n * cOff[t];
            const TData *in = gloT0 + npTot * t + n * npTBlock;

            for (unsigned i = 0; i < npTot; ++i)
            {
                out[i] = in[i];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Pack this rank's side of every parallel trace into a send buffer.
 *
 * Element-local order throughout, unreoriented and uninterpolated: the
 * receiving rank applies both itself, knowing this trace's orientation and
 * point distribution from the connectivity exchange.
 *
 * @param numEntry  - Traces to send on this channel.
 * @param numComp   - Components per trace point.
 * @param offset_st,offset,compSize - Per-entry addressing of the local side.
 * @param npts      - Points per entry.
 * @param bufOffset - Where each entry starts in the send buffer.
 * @param inPtr     - Element-local input, at the field base.
 * @param sendPtr   - Send buffer, at its base.
 * @param stride    - Point stride of @p inPtr, its interleave width.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void PackParallelSendBlock(
    const size_t numEntry, const unsigned numComp, const size_t *offset_st,
    const size_t *offset, const size_t *compSize, const size_t *npts,
    const size_t *bufOffset, const TData *inPtr, TData *sendPtr,
    const unsigned stride, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numEntry * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t e = idx % numEntry;
            const size_t n = idx / numEntry;

            const TData *src =
                inPtr + offset_st[e] * numComp + offset[e] + n * compSize[e];
            TData *dst = sendPtr + bufOffset[e] * numComp + n * npts[e];

            for (size_t i = 0; i < npts[e]; ++i)
            {
                dst[i] = src[i * stride];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Copy the forward side of a block's global trace onto the backward.
 *
 * Used where a quantity has no exterior value to impose, so the trace is
 * given no jump: every component of every trace in the block is copied.
 *
 * @param numComp  - Components per trace point.
 * @param numBlock - Traces in the block.
 * @param npTot    - Points per trace.
 * @param npTBlock - Component stride within both sides.
 * @param gloT0    - Forward side, read.
 * @param gloT1    - Backward side, written.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyGloTraceFwdToBwd(
    const unsigned numComp, const size_t numBlock, const size_t npTot,
    const size_t npTBlock, const TData *gloT0, TData *gloT1,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numBlock * numComp * npTot, NEKTAR_LAMBDA(const size_t idx) {
            const size_t i = idx % npTot;
            const size_t r = idx / npTot;
            const size_t t = r % numBlock;
            const size_t n = r / numBlock;

            const auto k = n * npTBlock + npTot * t + i;
            gloT1[k]     = gloT0[k];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Gather a block's point traces into the global-trace layout.
 *
 * The mirror of GloPointToLocPointBlock(): one value per trace and
 * component, read from the element-local block layout and written to the
 * packed global-trace one. A point has no in-trace direction, so there is
 * neither an interpolation nor an orientation to apply.
 *
 * @param numComp  - Components per trace point.
 * @param T        - Per-trace metadata for the local side of the block.
 * @param inPtr    - Element-local input, block layout, at the field base.
 * @param outPtr   - Packed global-trace output, at the base of the block.
 * @param npTBlock - Component stride within @p outPtr.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LocPointToGloPointBlock(
    const unsigned numComp, const TraceBlockView &T, const TData *inPtr,
    TData *outPtr, const size_t npTBlock, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            outPtr[n * npTBlock + t] = inPtr[T.offset_st[t] * numComp +
                                             T.offset[t] + n * T.compSize[t]];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <bool APPEND, typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void GloPointToLocPointBlock(
    const unsigned numComp, const TraceBlockView &T, const size_t npTBlock,
    const TData sign, const TData *inPtr, TData *outPtr,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const size_t numTrace = T.numTrace;

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * numComp, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx % numTrace;
            const size_t n = idx / numTrace;

            TData *out = outPtr + T.offset_st[t] * numComp + T.offset[t] +
                         n * T.compSize[t];

            const TData v = sign * inPtr[n * npTBlock + t];

            // if constexpr is legal here: the branches use only lambda-locals,
            // so no variable is first captured inside one.
            if constexpr (APPEND)
            {
                *out += v;
            }
            else
            {
                *out = v;
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Gather a multi-component global-trace field into a block's packed
 * layout.
 *
 * The trace normals, and the advection velocity that follows them, live on the
 * global trace in a component-major layout whose component stride varies from
 * trace to trace. The flux kernels want them block-local instead: component
 * major with @p npTBlock between components, and the traces of the block laid
 * end to end at @p npTot apart. This does that repacking.
 *
 * A free function taking raw pointers rather than a member of the operator,
 * because NEKTAR_LAMBDA expands to `[=] __device__` under CUDA and HIP and a
 * lambda written inside a member function captures `this` - a host pointer the
 * device cannot follow. The per-trace metadata arrives through @p G for the
 * same reason: it lives on the host as `std::vector`, which a kernel cannot
 * read.
 *
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam TData     Floating-point representation.
 *
 * @param numTrace - Traces in the block.
 * @param npTot    - Points on one trace.
 * @param npTBlock - Component stride of the block, points per trace times the
 *                   trace count, rounded up to the vector width.
 * @param numComp  - Components to gather.
 * @param srcBase  - Global-trace field, at its base.
 * @param G        - Per-trace offsets into @p srcBase.
 * @param dst      - Block-local output, written.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void GatherGloTraceComponentsKernel(
    const size_t numTrace, const size_t npTot, const size_t npTBlock,
    const unsigned numComp, const TData *srcBase, const GloTraceOffsetView &G,
    TData *dst, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * npTot, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx / npTot;
            const size_t i = idx - t * npTot;

            const TData *src    = srcBase + G.gloTOffset[t];
            const size_t stride = G.compSize[t];
            const size_t o      = npTot * t + i;

            for (unsigned n = 0; n < numComp; ++n)
            {
                dst[o + n * npTBlock] = src[i + n * stride];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Scatter a block's packed data out to a multi-component global-trace
 * field.
 *
 * The inverse of GatherGloTraceComponentsKernel(): reads the block-local
 * packed layout and writes each trace back to its own offset and component
 * stride in the mesh-wide field. Used to *store* a quantity on the global
 * trace - the trace advection velocity, which is assembled once at setup and
 * then gathered on every apply.
 *
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam AVERAGE   Write the mean of the two sides rather than the forward
 *                   side alone. Interior and parallel traces average;
 *                   boundary traces carry no exterior value worth averaging
 *                   and pass null for @p srcT1.
 * @tparam TData     Floating-point representation.
 *
 * @param numTrace - Traces in the block.
 * @param npTot    - Points on one trace.
 * @param npTBlock - Component stride of the block.
 * @param numComp  - Components to scatter.
 * @param srcT0    - Block-local forward-side data.
 * @param srcT1    - Block-local backward-side data, or null.
 * @param G        - Per-trace offsets into @p dstBase.
 * @param dstBase  - Global-trace field, at its base, written.
 */
template <typename ExecSpace, bool AVERAGE, typename TData>
NEK_FORCE_INLINE static void ScatterGloTraceComponentsKernel(
    const size_t numTrace, const size_t npTot, const size_t npTBlock,
    const unsigned numComp, const TData *srcT0, const TData *srcT1,
    const GloTraceOffsetView &G, TData *dstBase,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * npTot, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx / npTot;
            const size_t i = idx - t * npTot;

            TData *dst          = dstBase + G.gloTOffset[t];
            const size_t stride = G.compSize[t];
            const size_t o      = npTot * t + i;

            for (unsigned n = 0; n < numComp; ++n)
            {
                // Plain if, not if constexpr: srcT1's first capture must not
                // sit inside an if-constexpr branch (nvcc), and AVERAGE is a
                // template constant, so the dead branch folds either way.
                if (AVERAGE)
                {
                    dst[i + n * stride] =
                        TData(0.5) *
                        (srcT0[o + n * npTBlock] + srcT1[o + n * npTBlock]);
                }
                else
                {
                    dst[i + n * stride] = srcT0[o + n * npTBlock];
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Turn a block's exterior boundary state into a Dirichlet ghost.
 *
 * Overwrites gloT1 with `2*gloT1 - gloT0` across every point and component of
 * the block: a flux kernel that averages half-and-half then sees an average of
 * the imposed value g and a jump of 2(g - u+). Components whose condition is
 * not Dirichlet arrive with gloT1 = gloT0, for which the transformation is the
 * identity, so applying it uniformly is safe.
 *
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam TData     Floating-point representation.
 *
 * @param numTrace - Traces in the block.
 * @param npTot    - Points on one trace.
 * @param npTBlock - Component stride of the block.
 * @param numflux  - Flux components.
 * @param gloT0    - Interior state.
 * @param gloT1    - Imposed exterior state, overwritten in place.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void MakeDirichletGhostStateKernel(
    const size_t numTrace, const size_t npTot, const size_t npTBlock,
    const unsigned numflux, const TData *gloT0, TData *gloT1,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const size_t npTrace = numTrace * npTot;

    Nektar::parallel_for<ExecSpace>(
        0u, npTrace * numflux, NEKTAR_LAMBDA(const size_t idx) {
            const size_t f = idx / npTrace;
            const size_t r = idx - f * npTrace;

            const size_t o = f * npTBlock + r;

            gloT1[o] = TData(2.0) * gloT1[o] - gloT0[o];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Gather a single-component global-trace field into a block's packed
 * layout.
 *
 * GatherGloTraceComponentsKernel() for a field carrying one value per trace
 * point - the interior-penalty factor. Such a field is packed to its own
 * layout rather than to the normals', so it is indexed by
 * GloTraceOffsetView::scalarOffset and needs no component stride.
 *
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam TData     Floating-point representation.
 *
 * @param numTrace - Traces in the block.
 * @param npTot    - Points on one trace.
 * @param srcBase  - Global-trace field, at its base.
 * @param G        - Per-trace offsets into @p srcBase.
 * @param dst      - Block-local output, written.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void GatherGloTraceScalarKernel(
    const size_t numTrace, const size_t npTot, const TData *srcBase,
    const GloTraceOffsetView &G, TData *dst, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * npTot, NEKTAR_LAMBDA(const size_t idx) {
            const size_t t = idx / npTot;
            const size_t i = idx - t * npTot;

            dst[npTot * t + i] = srcBase[G.scalarOffset[t] + i];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Broadcast one value per trace over the points of that trace.
 *
 * For quantities that are a property of the whole trace rather than of a
 * point - the boundary energy weight, which follows the boundary condition -
 * but which the flux kernels read per point.
 *
 * @tparam ExecSpace Execution space the loop runs in.
 * @tparam TData     Floating-point representation.
 *
 * @param numTrace     - Traces in the block.
 * @param npTot        - Points on one trace.
 * @param valuePerTrace - One value per trace.
 * @param dst          - Block-local output, written.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void BroadcastGloTraceValueKernel(
    const size_t numTrace, const size_t npTot, const TData *valuePerTrace,
    TData *dst, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, numTrace * npTot, NEKTAR_LAMBDA(const size_t idx) {
            dst[idx] = valuePerTrace[idx / npTot];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
