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

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include "SolverCore/RiemannSolver/RiemannSolverKernels.hpp"

namespace Nektar::detail
{

/**
 * @brief Solve the Riemann problem on one padded trace block, in the frame
 * aligned with the trace normal.
 *
 * Compressible-flow Riemann solvers are formulated for a one-dimensional
 * problem normal to the interface, so the conserved state is rotated into that
 * frame, solved there, and the resulting flux rotated back to Cartesian
 * coordinates. Only the momentum components are affected by the rotation;
 * density and total energy are invariant.
 *
 * In 1D and 2D the rotation is cheap enough to rebuild from the normals every
 * time, and callers pass the normals array itself as @p rotMatPtr. In 3D a
 * full 3x3 matrix per point is cached across calls, and @p ROTMAT selects
 * whether this call rebuilds it or reuses what is already there.
 *
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension.
 * @tparam EoSParamType  Equation-of-state parameter pack.
 * @tparam ExecSpace     Execution space; selects the SIMD width.
 * @tparam NDIM          Spatial dimension.
 * @tparam ROTMAT        Regenerate the rotation matrices before use. Only has
 *                       an effect when @p NDIM is 3.
 * @tparam TData         Floating-point representation.
 *
 * @param EoS             - Equation-of-state parameters.
 * @param blksize         - Padded number of points in the block.
 * @param normalPtr       - Trace normals, @p NDIM components.
 * @param rotMatPtr       - Rotation matrices, 9 entries per point in 3D; the
 *                          normals array in 1D and 2D. Written through when
 *                          @p ROTMAT is set, despite being declared const.
 * @param rotStorage1Ptr  - Scratch for the rotated forward state.
 * @param rotStorage2Ptr  - Scratch for the rotated backward state.
 * @param rotStorage3Ptr  - Scratch for the flux in the rotated frame.
 * @param fwdPtr          - Forward-side conserved state.
 * @param bwdPtr          - Backward-side conserved state.
 * @param fluxPtr         - Output flux in Cartesian coordinates.
 * @param streamID        - Stream the launches run in; reset to the default
 *                          afterwards.
 */
template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, unsigned int NDIM,
          bool ROTMAT, typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const EoSParamType &EoS, const size_t blksize, const TData *normalPtr,
    const TData *rotMatPtr, TData *rotStorage1Ptr, TData *rotStorage2Ptr,
    TData *rotStorage3Ptr, const TData *fwdPtr, const TData *bwdPtr,
    TData *fluxPtr, const unsigned int streamID = 0)
{
    using namespace Nektar::detail;

    Nektar::LoopExecutionSetStreamID(streamID);

    // Explicit vectorisation for AVX backend,
    // vec_t = tinysimd::simd<TData> for AVX,
    // vec_t = TData otherwise.
    // using vec_t = typename data_type_if<
    //    std::is_same_v<ExecSpace, NektarSpaces::AVX>, TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    // Checked here, on the host: the kernels below assume it and may not
    // assert.
    ASSERTL1(blksize % vec_width == 0,
             "The block size is not a multiple of the vector width.");
    const size_t groupsize = blksize / vec_width;

    if (NDIM == 3 && ROTMAT)
    {
        Nektar::parallel_for<ExecSpace>(
            0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
                // Only update rotation matrice when necessary.
                SolverCore::detail::GenerateRotationMatrices<ExecSpace>(
                    blksize, normalPtr + i * vec_width,
                    (TData *)rotMatPtr + i * vec_width);

                // Rotate velocity to normal.
                SolverCore::detail::RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, fwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage1Ptr + i * vec_width);
                SolverCore::detail::RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, bwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage2Ptr + i * vec_width);

                // Compute flux in rotated frame.
                RiemannKernel<ExecSpace, EoSParamType, NDIM>()(
                    EoS, blksize, rotStorage1Ptr + i * vec_width,
                    rotStorage2Ptr + i * vec_width,
                    rotStorage3Ptr + i * vec_width);

                // Rotate flux back to Cartesian frame.
                SolverCore::detail::RotateFromNormalKernel<ExecSpace, NDIM>(
                    blksize, rotStorage3Ptr + i * vec_width,
                    rotMatPtr + i * vec_width, fluxPtr + i * vec_width);
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
                // Rotate velocity to normal.
                SolverCore::detail::RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, fwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage1Ptr + i * vec_width);
                SolverCore::detail::RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, bwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage2Ptr + i * vec_width);

                // Compute flux in rotated frame.
                RiemannKernel<ExecSpace, EoSParamType, NDIM>()(
                    EoS, blksize, rotStorage1Ptr + i * vec_width,
                    rotStorage2Ptr + i * vec_width,
                    rotStorage3Ptr + i * vec_width);

                // Rotate flux back to Cartesian frame.
                SolverCore::detail::RotateFromNormalKernel<ExecSpace, NDIM>(
                    blksize, rotStorage3Ptr + i * vec_width,
                    rotMatPtr + i * vec_width, fluxPtr + i * vec_width);
            });
    }

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
