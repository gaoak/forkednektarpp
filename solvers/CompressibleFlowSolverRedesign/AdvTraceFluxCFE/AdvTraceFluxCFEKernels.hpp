///////////////////////////////////////////////////////////////////////////////
//
// File: AdvTraceFluxCFEKernels.hpp
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
// Description: Launcher for the Riemann solve of the compressible Euler
// trace flux: rotates the two states into the trace-normal frame, evaluates
// the flux and rotates it back.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdlib>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <SolverCore/RiemannSolver/RiemannSolverKernels.hpp>

#include <type_traits>

namespace Nektar::detail
{

/**
 * @brief Solve the Riemann problem on one padded trace block, in Cartesian
 * components through the trace normal.
 *
 * The Euler equations are rotationally invariant, and the tangential
 * velocity enters the interface flux only through invariant combinations,
 * so the solvers are written against the normal directly (as the legacy
 * compressible Riemann solvers are) and need no rotation into a
 * normal-aligned frame and back.
 *
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension.
 * @tparam EoSParamType  Equation-of-state parameter pack.
 * @tparam ExecSpace     Execution space; selects the SIMD width.
 * @tparam NDIM          Spatial dimension.
 * @tparam TData         Floating-point representation.
 *
 * @param EoS        - Equation-of-state parameters.
 * @param blksize    - Padded number of points in the block.
 * @param normalPtr  - Trace normals, @p NDIM components.
 * @param fwdPtr     - Forward-side conserved state.
 * @param bwdPtr     - Backward-side conserved state.
 * @param fluxPtr    - Output flux in Cartesian coordinates.
 * @param streamID   - Stream the launches run in; reset to the default
 *                     afterwards.
 */
template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, unsigned int NDIM,
          typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const EoSParamType &EoS, const size_t blksize, const TData *normalPtr,
    const TData *fwdPtr, const TData *bwdPtr, TData *fluxPtr,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // Explicit vectorisation for AVX backend, vec_t =
    // tinysimd::simd<TData> for AVX, vec_t = TData otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    // Checked here, on the host: the kernels below assume it and may not
    // assert.
    ASSERTL1(blksize % vec_width == 0,
             "The block size is not a multiple of the vector width.");
    const size_t groupsize = blksize / vec_width;

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            RiemannKernel<ExecSpace, EoSParamType, NDIM>()(
                EoS, groupsize, reinterpret_cast<const vec_t *>(fwdPtr) + i,
                reinterpret_cast<const vec_t *>(bwdPtr) + i,
                reinterpret_cast<const vec_t *>(normalPtr) + i,
                reinterpret_cast<vec_t *>(fluxPtr) + i);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief The inviscid flux of one trace point, as the in-place trace flux
 * kernels take it (the point flux of
 * SolverCore/TraceFlux/TraceFluxKernels.hpp): the Riemann flux through the
 * normal, with no second input and no penalty.
 *
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension.
 * @tparam ExecSpace     Execution space; selects the SIMD width.
 * @tparam EoSParamType  Equation-of-state parameter pack.
 * @tparam NDIM          Spatial dimension.
 */
template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename ExecSpace, typename EoSParamType, unsigned int NDIM>
struct InviscidPointFlux
{
    static constexpr unsigned numVar       = NDIM + 2;
    static constexpr unsigned secondPerVar = 0;
    static constexpr bool usesPenalty      = false;

    EoSParamType EoS;

    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const TData *fwd, const TData *bwd,
                                          [[maybe_unused]] const TData *second0,
                                          [[maybe_unused]] const TData *second1,
                                          const TData *normal,
                                          [[maybe_unused]] const TData penalty,
                                          TData *flux) const
    {
        RiemannKernel<ExecSpace, EoSParamType, NDIM>()(EoS, 1, fwd, bwd, normal,
                                                       flux);
    }
};

} // namespace Nektar::detail
