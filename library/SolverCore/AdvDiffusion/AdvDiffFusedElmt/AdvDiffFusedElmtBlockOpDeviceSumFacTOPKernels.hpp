///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernel of the fused element-local advection-
// diffusion operator: one element per thread block, one thread per point.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernel of the fused element-local
 * advection-diffusion operator: one element per thread block, one thread per
 * point.
 *
 * @details
 * AdvDiffFusedElmtKernel() runs the stages of the advection kernel
 * (AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp), through the same
 * functions, with the viscous stages between them:
 * -# the state, variable by variable, staged in shared memory: the point's
 *    gradient (PhysDeriv{2,3}DPointSumFacTOP()), formed while the variable
 *    is staged, and the variable on the traces
 *    (AdvectionFusedElmtStateTrace());
 * -# on the elements whose gradient trace is asked for, each gradient
 *    component staged and extracted;
 * -# the viscous flux (`ViscFlux`), each directional component staged and
 *    extracted, and contracted with the trace normal on the trace points;
 * -# the advective flux (`AdvFlux`), formed last so that it is not held in
 *    registers across the viscous stages, the flux vector
 *    `viscFlux - advFlux` and its derivative inner product, negated
 *    (AdvectionFusedElmtFluxDerivative()).
 *
 * The gradient trace is written at its own width. The whole file is
 * compiled only for the device back-ends.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp>

#include <cstddef>

namespace Nektar::SolverCore::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief The fields of one block the advection-diffusion kernel reads and
 * writes: the advection kernel's, with the viscous ones, in device memory at
 * interleave width one but for the gradient trace.
 *
 * @tparam TData  Floating-point type of the field data.
 */
template <typename TData>
struct AdvDiffFusedElmtDeviceFields : AdvectionFusedElmtDeviceFields<TData>
{
    /// The outward normal viscous flux on the traces, laid out as the
    /// state's trace.
    TData *normVisc = nullptr;
    /// Each element's outward normals on its traces, laid out as the trace
    /// (component size traceCompSize).
    const TData *normals = nullptr;
    /// The gradient trace, its component size and interleave width, and
    /// each element's offset in it, or #noGradTrace for an element whose
    /// gradient trace is not written; null when none is.
    TData *gradTrace                    = nullptr;
    size_t gradTraceCompSize            = 0;
    unsigned int gradTraceWidth         = 1;
    const size_t *gradTraceOffset       = nullptr;
    static constexpr size_t noGradTrace = ~size_t(0);
};

/**
 * @brief The dynamic shared memory of one thread block of the
 * advection-diffusion kernel, in elements of the data type: the advection
 * kernel's (AdvectionFusedElmtSharedMemorySize()), followed by one element's
 * values on its traces, `nqTotTrace`.
 *
 * @param   sizeParam   The volume and trace point counts.
 * @param   nqTotTrace  Points of one element's traces.
 */
template <unsigned int NDIM, typename TTraceSize>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvDiffFusedElmtSharedMemorySize(
    const TTraceSize sizeParam, const unsigned int nqTotTrace)
{
    return AdvectionFusedElmtSharedMemorySize<NDIM>(sizeParam) + nqTotTrace;
}

/**
 * @brief The Cartesian gradient of one staged field at the calling thread's
 * point.
 *
 * Under SumFacTOP each thread of the block owns one quadrature point of the
 * element, (@p i, @p j, @p k) with flat index @p idx, and forms that point's
 * derivatives from the element's values in shared memory.
 *
 * @param   staged  The value at every point of the element, in shared
 *                  memory.
 * @param   deriv   Out: its derivatives at the point.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          typename TTraceSize, typename TData>
NEK_DEVICE_INLINE static void AdvDiffFusedElmtDerivPoint(
    const TTraceSize sizeParam, const unsigned int i, const unsigned int j,
    const unsigned int k, const unsigned int idx,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT staged,
    TData (&deriv)[3])
{
    static_assert(NDIM >= 1 && NDIM <= 3,
                  "The fused element-local kernels are for one, two or three "
                  "dimensions.");
    if constexpr (NDIM == 1)
    {
        const unsigned int nm0 = sizeParam.nm0();
        const TData *D0        = basisData.derivMatrix[0];
        TData d0               = 0.0;
        for (unsigned int q = 0u; q < nm0; ++q)
        {
            d0 += D0[q * nm0 + i] * staged[q];
        }
        deriv[0] = d0 * df[DEFORMED ? idx : 0u];
    }
    else if constexpr (NDIM == 2)
    {
        MultiRegions::detail::PhysDeriv2DPointSumFacTOP<SHAPE, DEFORMED>(
            2u, sizeParam.nm0(), sizeParam.nm1(), i, j, idx,
            basisData.derivMatrix[0], basisData.derivMatrix[1],
            basisData.factors[0], basisData.factors[1], df, staged, deriv);
    }
    else if constexpr (NDIM == 3)
    {
        MultiRegions::detail::PhysDeriv3DPointSumFacTOP<SHAPE, DEFORMED>(
            sizeParam.nm0(), sizeParam.nm1(), sizeParam.nm2(), i, j, k, idx,
            basisData.derivMatrix[0], basisData.derivMatrix[1],
            basisData.derivMatrix[2], basisData.factors[0],
            basisData.factors[1], basisData.factors[2], basisData.factors[3],
            df, staged, deriv);
    }
}

/**
 * @brief The state of element @p e on its traces and its gradient at the
 * calling thread's point, a variable at a time: each variable staged in
 * shared memory (AdvectionFusedElmtStagePoint()), its gradient formed at the
 * point (AdvDiffFusedElmtDerivPoint()) and the variable extracted
 * (AdvectionFusedElmtExtract()), as AdvectionFusedElmtStateTrace() does
 * without the gradient.
 *
 * @param   df    The element's derivative factors.
 * @param   grad  Out: the gradient at the point, `grad[d][n]` the derivative
 *                of variable n along direction d; untouched at a thread
 *                without a point.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          unsigned int NVAR, typename TTraceSize, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void AdvDiffFusedElmtStateTrace(
    const TTraceSize sizeParam,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const AdvDiffFusedElmtDeviceFields<TData> &fields, const size_t e,
    const unsigned int nqTotTrace,
    const AdvectionFusedElmtSharedMemoryData<TData> &sharedMemoryData,
    const AdvectionFusedElmtThread &thread, const TData *NEK_RESTRICT df,
    const TData (&state)[NVAR], TData (&grad)[NDIM][NVAR],
    const TthreadBlock &threadBlock)
{
    for (unsigned int n = 0; n < NVAR; ++n)
    {
        AdvectionFusedElmtStagePoint(thread.active, thread.idx0, state[n],
                                     sharedMemoryData.staged, threadBlock);
        if (thread.active)
        {
            TData deriv[3];
            AdvDiffFusedElmtDerivPoint<SHAPE, DEFORMED, NDIM>(
                sizeParam, thread.i, thread.j, thread.k, thread.idx0, basisData,
                df, sharedMemoryData.staged, deriv);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                grad[d][n] = deriv[d];
            }
        }
        AdvectionFusedElmtExtract<SHAPE, NDIM>(
            sizeParam, basisData, sharedMemoryData.staged, sharedMemoryData.wsp,
            fields.trace + n * fields.traceCompSize + e * nqTotTrace,
            threadBlock);
    }
}

/**
 * @brief Zero the outputs of the padding elements, `fields.nelmt` to
 * `fields.nelmtPad`: those of the advection kernel
 * (AdvectionFusedElmtZeroPadding()) and the normal viscous flux on the
 * traces. The gradient trace holds listed elements only, and so no padding
 * element.
 */
template <unsigned int NVAR, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvDiffFusedElmtZeroPadding(
    const AdvDiffFusedElmtDeviceFields<TData> &fields, const unsigned int nqTot,
    const unsigned int nqTotTrace, const AdvectionFusedElmtThread &thread,
    const TthreadBlock &threadBlock)
{
    AdvectionFusedElmtZeroPadding<NVAR>(fields, nqTot, nqTotTrace, thread,
                                        threadBlock);
    for (size_t e = fields.nelmt + getBlockIdx<0>(threadBlock);
         e < fields.nelmtPad; e += getBlockRange<0>(threadBlock))
    {
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            for (unsigned int p = thread.idx0; p < nqTotTrace;
                 p += thread.stride)
            {
                fields.normVisc[n * fields.traceCompSize + e * nqTotTrace + p] =
                    0.0;
            }
        }
    }
}

/**
 * @brief The fused advection-diffusion pass over a block's elements: the
 * state on the traces, the outward normal viscous flux on the traces and,
 * where asked for, the gradient on the traces, and the negated derivative
 * inner products of the flux vector.
 *
 * The thread blocks stride over the elements; each takes one element at a
 * time with one thread per point, `threadBlock` holding at least as many
 * threads as the element has points (MultiRegions::GetDeviceKernelParameters()
 * with the maximum block size). The stages are those of the file description.
 * The padding elements, beyond `fields.nelmt`, then have their outputs
 * zeroed (AdvDiffFusedElmtZeroPadding()).
 *
 * @tparam SHAPE       Shape of the block's elements.
 * @tparam DEFORMED    Geometric factors vary point by point.
 * @tparam Physics     The flux policy (AdvDiffFusedElmtKernels.hpp).
 * @tparam TTraceSize  Trace size parameter, compile-time or not.
 *
 * @param   sizeParam    The volume and trace point counts.
 * @param   physics      The flux policy.
 * @param   basisData    The block's basis data.
 * @param   fields       The block's fields.
 * @param   shmemptr     Dynamic shared memory,
 *                       AdvDiffFusedElmtSharedMemorySize() entries.
 * @param   threadBlock  Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, typename Physics,
          typename TTraceSize, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (AdvectionFusedElmtMaxThreads<TTraceSize>()))
    AdvDiffFusedElmtKernel(
        const TTraceSize sizeParam, const Physics physics,
        const AdvectionFusedElmtDeviceBasisData<TData> basisData,
        const AdvDiffFusedElmtDeviceFields<TData> fields,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int NDIM = Physics::NDIM;
    constexpr unsigned int NVAR = Physics::NVAR;
    constexpr unsigned int NAUX = Physics::NAUX;
    static_assert(NDIM >= 1 && NDIM <= 3,
                  "The fused element-local kernels are for one, two or three "
                  "dimensions.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nqTot      = AdvectionFusedElmtPoints(sizeParam);
    const unsigned int nqTotTrace = sizeParam.template nqTotTrace<SHAPE>();
    const unsigned int dfsize     = DEFORMED ? nqTot : 1u;
    const auto sharedMemoryData =
        AdvectionFusedElmtSharedMemoryDataLayout<NDIM>(sizeParam,
                                                       (TData *)shmemptr);
    // One element's values on its traces, after the advection layout.
    TData *traceBuf = sharedMemoryData.end;
    const auto thread =
        AdvectionFusedElmtThreadOf<NDIM>(sizeParam, threadBlock);

    AdvectionFusedElmtLoadMatrices<NDIM>(sizeParam, basisData, sharedMemoryData,
                                         threadBlock);

    for (size_t e = getBlockIdx<0>(threadBlock); e < fields.nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const size_t traceOffset = e * nqTotTrace;

        const TData *df  = fields.df + NDIM * NDIM * dfsize * e;
        const TData *jac = fields.jac + dfsize * e;

        TData state[NVAR], aux[NAUX > 0 ? NAUX : 1];
        AdvectionFusedElmtLoadPoint<NVAR, NAUX>(fields, e, nqTot, thread, state,
                                                aux);
        const TData *auxPt = NAUX > 0 ? aux : nullptr;

        // 1. The state on the traces and its gradient, a variable at a time:
        // the gradient while the variable is staged.
        TData grad[NDIM][NVAR] = {};
        AdvDiffFusedElmtStateTrace<SHAPE, DEFORMED, NDIM, NVAR>(
            sizeParam, basisData, fields, e, nqTotTrace, sharedMemoryData,
            thread, df, state, grad, threadBlock);

        // 2. The gradient on the traces, where asked for, at the gradient
        // trace's width.
        if (fields.gradTraceOffset &&
            fields.gradTraceOffset[e] !=
                AdvDiffFusedElmtDeviceFields<TData>::noGradTrace)
        {
            const size_t gradOffset = fields.gradTraceOffset[e];
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    AdvectionFusedElmtStagePoint(
                        thread.active, thread.idx0, grad[d][n],
                        sharedMemoryData.staged, threadBlock);
                    AdvectionFusedElmtExtract<SHAPE, NDIM>(
                        sizeParam, basisData, sharedMemoryData.staged,
                        sharedMemoryData.wsp, traceBuf, threadBlock);
                    TData *dest = fields.gradTrace +
                                  (n * NDIM + d) * fields.gradTraceCompSize +
                                  gradOffset;
                    for (unsigned int p = thread.idx0; p < nqTotTrace;
                         p += thread.stride)
                    {
                        dest[p * fields.gradTraceWidth] = traceBuf[p];
                    }
                    localBarrier(threadBlock);
                }
            }
        }

        // 3. The viscous flux, and its outward normal component on the
        // traces: each directional component extracted, then contracted
        // with the normal point by point, each trace point owned by one
        // thread throughout.
        TData viscFlux[NDIM][NVAR] = {};
        if (thread.active)
        {
            physics.ViscFlux(state, grad, auxPt, viscFlux);
        }
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            TData *normVisc =
                fields.normVisc + n * fields.traceCompSize + traceOffset;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                AdvectionFusedElmtStagePoint(
                    thread.active, thread.idx0, viscFlux[d][n],
                    sharedMemoryData.staged, threadBlock);
                AdvectionFusedElmtExtract<SHAPE, NDIM>(
                    sizeParam, basisData, sharedMemoryData.staged,
                    sharedMemoryData.wsp, traceBuf, threadBlock);
                const TData *normal =
                    fields.normals + d * fields.traceCompSize + traceOffset;
                for (unsigned int p = thread.idx0; p < nqTotTrace;
                     p += thread.stride)
                {
                    const TData value = normal[p] * traceBuf[p];
                    normVisc[p]       = (d == 0) ? value : normVisc[p] + value;
                }
                localBarrier(threadBlock);
            }
        }

        // 4. The flux vector viscFlux - advFlux and its derivative inner
        // product, negated. The advective flux is formed only now, so that
        // it is not held across the viscous stages.
        TData flux[NDIM][NVAR] = {};
        if (thread.active)
        {
            physics.AdvFlux(state, auxPt, flux);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                for (unsigned int n = 0; n < NVAR; ++n)
                {
                    flux[d][n] = viscFlux[d][n] - flux[d][n];
                }
            }
        }
        AdvectionFusedElmtFluxDerivative<SHAPE, DEFORMED, NDIM, NVAR>(
            sizeParam, basisData, fields, e, sharedMemoryData, thread, df, jac,
            flux, threadBlock);
    }

    AdvDiffFusedElmtZeroPadding<NVAR>(fields, nqTot, nqTotTrace, thread,
                                      threadBlock);
}

#endif

} // namespace Nektar::SolverCore::detail
