///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernel of the fused element-local advection
// operator, and the device stages it shares with the advection-diffusion
// kernel.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernel of the fused element-local advection
 * operator, and the device stages the advection-diffusion kernel shares:
 * one element per thread block, one thread per point.
 *
 * @details
 * AdvectionFusedElmtKernel() takes a block's elements one per thread block,
 * with one thread per quadrature point, so that each thread keeps its
 * point's state and flux in registers from one stage to the next and only
 * the values a sum over points needs pass through shared memory, one
 * variable at a time:
 * -# the state, variable by variable, staged in shared memory and extracted
 *    onto the traces (PhysTraceExtract{2,3}DSumFacTOPKernelCore(), the two
 *    end points in one dimension), AdvectionFusedElmtStateTrace();
 * -# the flux vector `-advFlux` (`AdvFlux`), each variable's flux vector
 *    aligned with the metric at the point (AdvectionFusedElmtAlignPoint())
 *    into shared memory and contracted with the transposed derivative
 *    matrices (SumDerivTensor{1,2,3}DSumFacTOPKernel(), scale -1),
 *    AdvectionFusedElmtFluxDerivative().
 *
 * The advection-diffusion kernel
 * (AdvDiffFusedElmtBlockOpDeviceSumFacTOPKernels.hpp) runs the same stages,
 * through the same functions but for the first, which also forms the gradient
 * there (AdvDiffFusedElmtStateTrace()), with its viscous stages between them.
 *
 * It composes the device functions the separate operators' SumFacTOP
 * kernels share, so it covers the same shapes and rules; the arithmetic is
 * that of the host kernels
 * (AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp) up to rounding. Only
 * the metric alignment is written out here, at one point, in place of
 * StdAlignDerivBase{1,2,3}DSumFacTOPKernel(), whose flux input would need all
 * the directions staged at once.
 *
 * The data is at interleave width one, an element's values contiguous,
 * with the derivative factors transposed (a plane per factor) as the
 * SumFacTOP kernels of PhysDeriv read them.
 *
 * The whole file is compiled only for the device back-ends.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOpDeviceSumFacTOPKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtKernels.hpp>

#include <cstddef>
#include <type_traits>

namespace Nektar::SolverCore::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief The basis data of one block the kernels read, in device memory, as
 * the separate operators fetch them from the data warehouse.
 *
 * @tparam TData  Floating-point type of the data.
 */
template <typename TData> struct AdvectionFusedElmtDeviceBasisData
{
    /// Differentiation matrix and quadrature weights per direction.
    const TData *derivMatrix[3] = {};
    const TData *weights[3]     = {};
    /// Collapsed-coordinate factors: none in one dimension, `(f0, f1)` in
    /// two and `(f0, f1, f1m, f2)` in three, as PhysDeriv names them.
    const TData *factors[4] = {};
    /// End-point interpolants per direction, then the in-face interpolants
    /// per face pair (MultiRegions::detail::FetchTraceTables()).
    const TData *traceBasis[9] = {};
    /// Whether each direction's rule contains its end points, and whether
    /// each face pair's trace rule is the volume rule.
    bool endPts[3] = {false, false, false};
    bool colloc[6] = {false, false, false, false, false, false};
};

/**
 * @brief The fields of one block the advection kernel reads and writes, in
 * device memory at interleave width one. The advection-diffusion kernel's
 * fields extend them.
 *
 * @tparam TData  Floating-point type of the field data.
 */
template <typename TData> struct AdvectionFusedElmtDeviceFields
{
    /// Elements of the block, and with padding.
    size_t nelmt    = 0;
    size_t nelmtPad = 0;
    /// Component size of the volume fields (the state, the auxiliary fields
    /// and the derivative inner products) and of the element-local trace
    /// fields: the block's padded element count times the points of one
    /// element and of its traces.
    size_t volCompSize   = 0;
    size_t traceCompSize = 0;
    /// The state and the auxiliary fields.
    const TData *state = nullptr;
    const TData *aux   = nullptr;
    /// The negated derivative inner products.
    TData *out = nullptr;
    /// The state on the traces.
    TData *trace = nullptr;
    /// Derivative factors (transposed) and Jacobians of the block.
    const TData *df  = nullptr;
    const TData *jac = nullptr;
};

/**
 * @brief Points of one element of the trace size parameter @p sizeParam.
 */
template <typename TTraceSize>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionFusedElmtPoints(
    const TTraceSize sizeParam)
{
    return sizeParam.nmTot();
}

/**
 * @brief The dynamic shared memory of one thread block of the advection
 * kernel, in elements of the data type, laid out in this order
 * (AdvectionFusedElmtSharedMemoryDataLayout()):
 * - the transposed differentiation matrices, `nm_d^2` per direction;
 * - the aligned flux vector of one variable, `NDIM * nqTot`, whose first
 *   `nqTot` entries also stage one value per point for the trace extraction;
 * - the trace extraction's workspace, as
 *   MultiRegions::detail::PhysTraceExtractSharedMemorySize() sizes it less
 *   the volume copy (none in one dimension).
 *
 * A kernel that needs more appends it after these.
 *
 * @tparam NDIM  Dimension.
 *
 * @param   sizeParam   The volume and trace point counts.
 */
template <unsigned int NDIM, typename TTraceSize>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionFusedElmtSharedMemorySize(
    const TTraceSize sizeParam)
{
    unsigned int size = sizeParam.nm0() * sizeParam.nm0();
    unsigned int wsp  = 0;
    if constexpr (NDIM >= 2)
    {
        size += sizeParam.nm1() * sizeParam.nm1();
    }
    if constexpr (NDIM == 2)
    {
        wsp = MultiRegions::detail::PhysTraceExtractBlockSize(sizeParam);
    }
    if constexpr (NDIM == 3)
    {
        size += sizeParam.nm2() * sizeParam.nm2();
        wsp = 3u * MultiRegions::detail::PhysTraceExtractBlockSize(sizeParam);
    }
    return size + NDIM * AdvectionFusedElmtPoints(sizeParam) + wsp;
}

/**
 * @brief The bound on the threads of a launch for the `__LAUNCH_BOUNDS__`
 * annotation, at most fusedElmtMaxBlockSize: the kernels keep one thread
 * per point and cannot stride.
 *
 * MultiRegions::detail::GetMaxThreadPerBlock() bounds a compile-time size
 * parameter by its points, but does not know the trace size parameters, to
 * which it gives the cap. Since only the total enters the bound, a
 * compile-time trace size parameter is passed as one flat count of its
 * points (AdvectionFusedElmtPoints()); any other takes the cap.
 */
template <typename TTraceSize>
static constexpr unsigned int AdvectionFusedElmtMaxThreads()
{
    constexpr unsigned int maxBlockSize = fusedElmtMaxBlockSize;
    if constexpr (MultiRegions::detail::IsTemplatedTraceSizeParameter1D<
                      TTraceSize>::value ||
                  MultiRegions::detail::IsTemplatedTraceSizeParameter2D<
                      TTraceSize>::value ||
                  MultiRegions::detail::IsTemplatedTraceSizeParameter3D<
                      TTraceSize>::value)
    {
        return MultiRegions::detail::GetMaxThreadPerBlock<
            MultiRegions::SumFacTOP,
            MultiRegions::detail::TemplatedPhysSizeParameter1D<
                1u, TTraceSize::nmTot()>>(maxBlockSize);
    }
    else
    {
        return MultiRegions::detail::GetMaxThreadPerBlock<
            MultiRegions::SumFacTOP, TTraceSize>(maxBlockSize);
    }
}

/**
 * @brief The shared memory of one thread block, as
 * AdvectionFusedElmtSharedMemorySize() lays it out.
 */
template <typename TData> struct AdvectionFusedElmtSharedMemoryData
{
    /// The transposed differentiation matrices.
    TData *DT[3];
    /// The aligned flux vector of one variable, `NDIM * nqTot`.
    TData *aligned;
    /// One value per point staged for a sum over points: the first `nqTot`
    /// entries of #aligned.
    TData *staged;
    /// The trace extraction's workspace.
    TData *wsp;
    /// The first entry past the layout, where a kernel needing more shared
    /// memory starts its own.
    TData *end;
};

/**
 * @brief The shared memory of the calling thread block, from @p shmemptr.
 */
template <unsigned int NDIM, typename TTraceSize, typename TData>
NEK_DEVICE_INLINE static AdvectionFusedElmtSharedMemoryData<TData>
AdvectionFusedElmtSharedMemoryDataLayout(const TTraceSize sizeParam,
                                         TData *shmemptr)
{
    const unsigned int nm0 = sizeParam.nm0();
    unsigned int nm1 = 1u, nm2 = 1u, nws = 0u;
    if constexpr (NDIM >= 2)
    {
        nm1 = sizeParam.nm1();
    }
    if constexpr (NDIM == 2)
    {
        nws = MultiRegions::detail::PhysTraceExtractBlockSize(sizeParam);
    }
    if constexpr (NDIM == 3)
    {
        nm2 = sizeParam.nm2();
        nws = 3u * MultiRegions::detail::PhysTraceExtractBlockSize(sizeParam);
    }
    AdvectionFusedElmtSharedMemoryData<TData> sharedMemoryData;
    sharedMemoryData.DT[0] = shmemptr;
    sharedMemoryData.DT[1] = sharedMemoryData.DT[0] + nm0 * nm0;
    sharedMemoryData.DT[2] =
        sharedMemoryData.DT[1] + (NDIM >= 2 ? nm1 * nm1 : 0u);
    sharedMemoryData.aligned =
        sharedMemoryData.DT[2] + (NDIM == 3 ? nm2 * nm2 : 0u);
    sharedMemoryData.staged = sharedMemoryData.aligned;
    sharedMemoryData.wsp =
        sharedMemoryData.aligned + NDIM * AdvectionFusedElmtPoints(sizeParam);
    sharedMemoryData.end = sharedMemoryData.wsp + nws;
    return sharedMemoryData;
}

/**
 * @brief The calling thread's point in an element under SumFacTOP: one
 * thread per quadrature point, the threads past the element's points
 * inactive.
 */
struct AdvectionFusedElmtThread
{
    /// The thread's flat point index, and the threads of the block.
    unsigned int idx0, stride;
    /// Whether the thread has a point.
    bool active;
    /// The point's index in each direction.
    unsigned int i, j, k;
};

/**
 * @brief The calling thread's point (AdvectionFusedElmtThread).
 */
template <unsigned int NDIM, typename TTraceSize, typename TthreadBlock>
NEK_DEVICE_INLINE static AdvectionFusedElmtThread AdvectionFusedElmtThreadOf(
    const TTraceSize sizeParam, const TthreadBlock &threadBlock)
{
    const unsigned int nm0 = sizeParam.nm0();
    unsigned int nm1       = 1u;
    if constexpr (NDIM >= 2)
    {
        nm1 = sizeParam.nm1();
    }
    AdvectionFusedElmtThread thread;
    thread.idx0   = getLocalIdx<0>(threadBlock);
    thread.stride = getLocalRange<0>(threadBlock);
    thread.active = thread.idx0 < AdvectionFusedElmtPoints(sizeParam);
    thread.i      = thread.idx0 % nm0;
    thread.j      = (thread.idx0 / nm0) % nm1;
    thread.k      = thread.idx0 / (nm0 * nm1);
    return thread;
}

/**
 * @brief Load the transposed differentiation matrices into shared memory,
 * once per launch, and meet at the barrier.
 */
template <unsigned int NDIM, typename TTraceSize, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtLoadMatrices(
    const TTraceSize sizeParam,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const AdvectionFusedElmtSharedMemoryData<TData> &sharedMemoryData,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm0 = sizeParam.nm0();
    LoadTransposeToSharedMemory(nm0, nm0, basisData.derivMatrix[0],
                                sharedMemoryData.DT[0], threadBlock);
    if constexpr (NDIM >= 2)
    {
        const unsigned int nm1 = sizeParam.nm1();
        LoadTransposeToSharedMemory(nm1, nm1, basisData.derivMatrix[1],
                                    sharedMemoryData.DT[1], threadBlock);
    }
    if constexpr (NDIM == 3)
    {
        const unsigned int nm2 = sizeParam.nm2();
        LoadTransposeToSharedMemory(nm2, nm2, basisData.derivMatrix[2],
                                    sharedMemoryData.DT[2], threadBlock);
    }
    localBarrier(threadBlock);
}

/**
 * @brief Zero the outputs of the padding elements, `fields.nelmt` to
 * `fields.nelmtPad`, the thread blocks striding over them as over the
 * elements. Unlike a linear operator's, the arithmetic on a padding
 * element's zero state need not give zero: a flux divides by the density.
 */
template <unsigned int NVAR, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtZeroPadding(
    const AdvectionFusedElmtDeviceFields<TData> &fields,
    const unsigned int nqTot, const unsigned int nqTotTrace,
    const AdvectionFusedElmtThread &thread, const TthreadBlock &threadBlock)
{
    for (size_t e = fields.nelmt + getBlockIdx<0>(threadBlock);
         e < fields.nelmtPad; e += getBlockRange<0>(threadBlock))
    {
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            for (unsigned int p = thread.idx0; p < nqTot; p += thread.stride)
            {
                fields.out[n * fields.volCompSize + e * nqTot + p] = 0.0;
            }
            for (unsigned int p = thread.idx0; p < nqTotTrace;
                 p += thread.stride)
            {
                fields.trace[n * fields.traceCompSize + e * nqTotTrace + p] =
                    0.0;
            }
        }
    }
}

/**
 * @brief The state and the auxiliary values at the calling thread's point
 * of element @p e; zero for a thread without a point.
 */
template <unsigned int NVAR, unsigned int NAUX, typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtLoadPoint(
    const AdvectionFusedElmtDeviceFields<TData> &fields, const size_t e,
    const unsigned int nqTot, const AdvectionFusedElmtThread &thread,
    TData (&state)[NVAR], TData (&aux)[NAUX > 0 ? NAUX : 1])
{
    if (thread.active)
    {
        const size_t volOffset = e * nqTot;
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            state[n] =
                fields.state[n * fields.volCompSize + volOffset + thread.idx0];
        }
        if constexpr (NAUX > 0)
        {
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                aux[a] =
                    fields
                        .aux[a * fields.volCompSize + volOffset + thread.idx0];
            }
        }
    }
    else
    {
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            state[n] = 0.0;
        }
        if constexpr (NAUX > 0)
        {
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                aux[a] = 0.0;
            }
        }
    }
}

/**
 * @brief The metric alignment of one variable's flux vector at one point:
 * the flux contracted with the derivative factors, the quadrature weight
 * and Jacobian applied, and moved from standard to collapsed coordinates,
 * as StdAlignDerivBase{1,2,3}DSumFacTOPKernel() do at every point of an
 * element.
 *
 * @tparam SHAPE     Shape of the element.
 * @tparam DEFORMED  Geometric factors vary point by point.
 * @tparam NDIM      Dimension, and the coordinate dimension.
 *
 * @param   i,j,k    The point's index in each direction.
 * @param   idx      The point's flat index.
 * @param   dfsize   Points per factor plane: the element's point count when
 *                   deformed, one otherwise.
 * @param   basisData  The block's basis data.
 * @param   df       The element's derivative factors, transposed.
 * @param   jac      The element's Jacobian(s).
 * @param   flux     The flux vector at the point.
 * @param   aligned  Out: its direction fields at the point.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtAlignPoint(
    const unsigned int i, [[maybe_unused]] const unsigned int j,
    [[maybe_unused]] const unsigned int k, const unsigned int idx,
    const unsigned int dfsize,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData (&flux)[NDIM], TData (&aligned)[NDIM])
{
    const unsigned int dfindex = DEFORMED ? idx : 0u;
    const TData J              = DEFORMED ? jac[idx] : jac[0];

    // sum[d] = sum_c dxi_d/dx_c flux_c, factor (c, d) in plane NDIM c + d.
    TData sum[NDIM];
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        sum[d] = 0.0;
        for (unsigned int c = 0; c < NDIM; ++c)
        {
            sum[d] += df[(c * NDIM + d) * dfsize + dfindex] * flux[c];
        }
    }

    if constexpr (NDIM == 1)
    {
        aligned[0] = sum[0] * J * basisData.weights[0][i];
    }
    else if constexpr (NDIM == 2)
    {
        const TData wJ = basisData.weights[0][i] * basisData.weights[1][j] * J;
        if constexpr (SHAPE == LibUtilities::Tri)
        {
            const TData *f0 = basisData.factors[0];
            const TData *f1 = basisData.factors[1];
            aligned[0]      = (sum[0] + sum[1] * f0[i]) * f1[j] * wJ;
        }
        else if constexpr (SHAPE == LibUtilities::Quad)
        {
            aligned[0] = sum[0] * wJ;
        }
        aligned[1] = sum[1] * wJ;
    }
    else if constexpr (NDIM == 3)
    {
        const TData wJ = basisData.weights[0][i] * basisData.weights[1][j] *
                         basisData.weights[2][k] * J;
        const TData *f0  = basisData.factors[0];
        const TData *f1  = basisData.factors[1];
        const TData *f1m = basisData.factors[2];
        const TData *f2  = basisData.factors[3];
        if constexpr (SHAPE == LibUtilities::Hex)
        {
            aligned[0] = sum[0] * wJ;
            aligned[1] = sum[1] * wJ;
        }
        else if constexpr (SHAPE == LibUtilities::Tet)
        {
            const TData tmp = f2[k] * wJ;
            aligned[0] = (sum[0] + (sum[1] + sum[2]) * f0[i]) * f1m[j] * tmp;
            aligned[1] = (sum[1] + sum[2] * f1[j]) * tmp;
        }
        else if constexpr (SHAPE == LibUtilities::Prism)
        {
            aligned[0] = (sum[0] + sum[2] * f0[i]) * f2[k] * wJ;
            aligned[1] = sum[1] * wJ;
        }
        else if constexpr (SHAPE == LibUtilities::Pyr)
        {
            const TData tmp = f2[k] * wJ;
            aligned[0]      = (sum[0] + sum[2] * f0[i]) * tmp;
            aligned[1]      = (sum[1] + sum[2] * f1[j]) * tmp;
        }
        aligned[2] = sum[2] * wJ;
    }
}

/**
 * @brief One staged value of one element onto all its traces, in packed
 * trace order, with the calling thread block. Ends with a block barrier,
 * so the caller may restage at once.
 *
 * @param   staged  The value at every point of the element, in shared
 *                  memory.
 * @param   wsp     The trace extraction's workspace in shared memory.
 * @param   out     The element's trace values.
 */
template <LibUtilities::ShapeType SHAPE, unsigned int NDIM, typename TTraceSize,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtExtract(
    const TTraceSize sizeParam,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const TData *staged, TData *wsp, TData *out,
    const TthreadBlock &threadBlock)
{
    const TData *const *B = basisData.traceBasis;
    if constexpr (NDIM == 1)
    {
        const unsigned int nm0    = sizeParam.nm0();
        const unsigned int idx0   = getLocalIdx<0>(threadBlock);
        const unsigned int stride = getLocalRange<0>(threadBlock);
        for (unsigned int t = idx0; t < 2u; t += stride)
        {
            if (basisData.endPts[0])
            {
                out[t] = staged[(t == 0u) ? 0u : (nm0 - 1u)];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += staged[p] * B[0][p * 2u + t];
                }
                out[t] = tmp;
            }
        }
    }
    else if constexpr (NDIM == 2)
    {
        MultiRegions::detail::PhysTraceExtract2DSumFacTOPKernelCore<SHAPE>(
            sizeParam.nm0(), sizeParam.nm1(), sizeParam.nq00(),
            sizeParam.nq10(), B[0], B[1], B[2], B[3], wsp, staged, out,
            basisData.colloc[0], basisData.colloc[1], basisData.endPts[0],
            basisData.endPts[1], threadBlock);
    }
    else if constexpr (NDIM == 3)
    {
        const unsigned int blockSize =
            MultiRegions::detail::PhysTraceExtractBlockSize(sizeParam);
        MultiRegions::detail::PhysTraceExtract3DSumFacTOPKernelCore<SHAPE>(
            sizeParam.nm0(), sizeParam.nm1(), sizeParam.nm2(), sizeParam.nq00(),
            sizeParam.nq01(), sizeParam.nq10(), sizeParam.nq11(),
            sizeParam.nq20(), sizeParam.nq21(), B[0], B[1], B[2], B[3], B[4],
            B[5], B[6], B[7], B[8], wsp, wsp + 2u * blockSize, staged, out,
            basisData.colloc[0], basisData.colloc[1], basisData.colloc[2],
            basisData.colloc[3], basisData.colloc[4], basisData.colloc[5],
            basisData.endPts[0], basisData.endPts[1], basisData.endPts[2],
            threadBlock);
    }
    localBarrier(threadBlock);
}

/**
 * @brief Stage one field for the block's sum-factorised contractions: each
 * thread writes its point's value to shared memory, and the block meets at
 * the barrier before any thread reads the element's values.
 *
 * A thread block is whole warps, so it can hold more threads than the
 * element has points. Those threads have nothing to write, but every thread
 * of the block has to reach the barrier, which is why the kernel guards its
 * point work with @p active rather than returning early.
 *
 * @param   active       The calling thread has a point.
 * @param   idx          The thread's point.
 * @param   value        The field's value at it.
 * @param   staged       The element's values of the field, in shared memory.
 * @param   threadBlock  Thread-block handle supplied by the launch macro.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtStagePoint(
    const bool active, const unsigned int idx, const TData value,
    TData *NEK_RESTRICT staged, const TthreadBlock &threadBlock)
{
    if (active)
    {
        staged[idx] = value;
    }
    localBarrier(threadBlock);
}

/**
 * @brief The state of element @p e on its traces, a variable at a time: each
 * variable staged in shared memory (AdvectionFusedElmtStagePoint()) and
 * extracted (AdvectionFusedElmtExtract()). The advection-diffusion kernel,
 * which forms the gradient while a variable is staged, has its own
 * (AdvDiffFusedElmtStateTrace()).
 */
template <LibUtilities::ShapeType SHAPE, unsigned int NDIM, unsigned int NVAR,
          typename TTraceSize, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtStateTrace(
    const TTraceSize sizeParam,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const AdvectionFusedElmtDeviceFields<TData> &fields, const size_t e,
    const unsigned int nqTotTrace,
    const AdvectionFusedElmtSharedMemoryData<TData> &sharedMemoryData,
    const AdvectionFusedElmtThread &thread, const TData (&state)[NVAR],
    const TthreadBlock &threadBlock)
{
    for (unsigned int n = 0; n < NVAR; ++n)
    {
        AdvectionFusedElmtStagePoint(thread.active, thread.idx0, state[n],
                                     sharedMemoryData.staged, threadBlock);
        AdvectionFusedElmtExtract<SHAPE, NDIM>(
            sizeParam, basisData, sharedMemoryData.staged, sharedMemoryData.wsp,
            fields.trace + n * fields.traceCompSize + e * nqTotTrace,
            threadBlock);
    }
}

/**
 * @brief The negated derivative inner product of the flux vector of element
 * @p e, a variable at a time: the flux vector aligned with the metric at
 * each point into shared memory, then contracted with the transposed
 * derivative matrices into `fields.out`.
 *
 * @param   df,jac  The element's derivative factors and Jacobian(s).
 * @param   flux    The flux vector at the calling thread's point.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          unsigned int NVAR, typename TTraceSize, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void AdvectionFusedElmtFluxDerivative(
    const TTraceSize sizeParam,
    const AdvectionFusedElmtDeviceBasisData<TData> &basisData,
    const AdvectionFusedElmtDeviceFields<TData> &fields, const size_t e,
    const AdvectionFusedElmtSharedMemoryData<TData> &sharedMemoryData,
    const AdvectionFusedElmtThread &thread, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData (&flux)[NDIM][NVAR],
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = AdvectionFusedElmtPoints(sizeParam);
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;
    for (unsigned int n = 0; n < NVAR; ++n)
    {
        if (thread.active)
        {
            TData fluxPt[NDIM], alignedPt[NDIM];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                fluxPt[d] = flux[d][n];
            }
            AdvectionFusedElmtAlignPoint<SHAPE, DEFORMED, NDIM>(
                thread.i, thread.j, thread.k, thread.idx0, dfsize, basisData,
                df, jac, fluxPt, alignedPt);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                sharedMemoryData.aligned[d * nqTot + thread.idx0] =
                    alignedPt[d];
            }
        }
        localBarrier(threadBlock);

        TData *out = fields.out + n * fields.volCompSize + e * nqTot;
        if constexpr (NDIM == 1)
        {
            MultiRegions::detail::SumDerivTensor1DSumFacTOPKernel<true, false>(
                sizeParam.nm0(), sharedMemoryData.DT[0],
                sharedMemoryData.aligned, out, TData(-1.0), threadBlock);
        }
        else if constexpr (NDIM == 2)
        {
            MultiRegions::detail::SumDerivTensor2DSumFacTOPKernel<true, false>(
                sizeParam.nm0(), sizeParam.nm1(), sharedMemoryData.DT[0],
                sharedMemoryData.DT[1], sharedMemoryData.aligned,
                sharedMemoryData.aligned + nqTot, out, TData(-1.0),
                threadBlock);
        }
        else if constexpr (NDIM == 3)
        {
            MultiRegions::detail::SumDerivTensor3DSumFacTOPKernel<true, false>(
                sizeParam.nm0(), sizeParam.nm1(), sizeParam.nm2(),
                sharedMemoryData.DT[0], sharedMemoryData.DT[1],
                sharedMemoryData.DT[2], sharedMemoryData.aligned,
                sharedMemoryData.aligned + nqTot,
                sharedMemoryData.aligned + 2u * nqTot, out, TData(-1.0),
                threadBlock);
        }
    }
}

/**
 * @brief The fused advection pass over a block's elements: the state on the
 * traces and the negated derivative inner products of `-advFlux`.
 *
 * The thread blocks stride over the elements; each takes one element at a
 * time with one thread per point, `threadBlock` holding at least as many
 * threads as the element has points (MultiRegions::GetDeviceKernelParameters()
 * with the maximum block size). The stages are those of the file description.
 * The padding elements, beyond `fields.nelmt`, then have their outputs
 * zeroed (AdvectionFusedElmtZeroPadding()).
 *
 * @tparam SHAPE       Shape of the block's elements.
 * @tparam DEFORMED    Geometric factors vary point by point.
 * @tparam Physics     The flux policy (AdvectionFusedElmtKernels.hpp).
 * @tparam TTraceSize  Trace size parameter, compile-time or not.
 *
 * @param   sizeParam    The volume and trace point counts.
 * @param   physics      The flux policy.
 * @param   basisData    The block's basis data.
 * @param   fields       The block's fields.
 * @param   shmemptr     Dynamic shared memory,
 *                       AdvectionFusedElmtSharedMemorySize() entries.
 * @param   threadBlock  Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, typename Physics,
          typename TTraceSize, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (AdvectionFusedElmtMaxThreads<TTraceSize>()))
    AdvectionFusedElmtKernel(
        const TTraceSize sizeParam, const Physics physics,
        const AdvectionFusedElmtDeviceBasisData<TData> basisData,
        const AdvectionFusedElmtDeviceFields<TData> fields,
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
    const auto thread =
        AdvectionFusedElmtThreadOf<NDIM>(sizeParam, threadBlock);

    AdvectionFusedElmtLoadMatrices<NDIM>(sizeParam, basisData, sharedMemoryData,
                                         threadBlock);

    for (size_t e = getBlockIdx<0>(threadBlock); e < fields.nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const TData *df  = fields.df + NDIM * NDIM * dfsize * e;
        const TData *jac = fields.jac + dfsize * e;

        TData state[NVAR], aux[NAUX > 0 ? NAUX : 1];
        AdvectionFusedElmtLoadPoint<NVAR, NAUX>(fields, e, nqTot, thread, state,
                                                aux);

        // 1. The state on the traces.
        AdvectionFusedElmtStateTrace<SHAPE, NDIM, NVAR>(
            sizeParam, basisData, fields, e, nqTotTrace, sharedMemoryData,
            thread, state, threadBlock);

        // 2. The flux vector and its derivative inner product, negated.
        TData flux[NDIM][NVAR] = {};
        if (thread.active)
        {
            AdvectionFusedElmtFlux(physics, state, NAUX > 0 ? aux : nullptr,
                                   flux);
        }
        AdvectionFusedElmtFluxDerivative<SHAPE, DEFORMED, NDIM, NVAR>(
            sizeParam, basisData, fields, e, sharedMemoryData, thread, df, jac,
            flux, threadBlock);
    }

    AdvectionFusedElmtZeroPadding<NVAR>(fields, nqTot, nqTotTrace, thread,
                                        threadBlock);
}

#endif

} // namespace Nektar::SolverCore::detail
