///////////////////////////////////////////////////////////////////////////////
//
// File: CFLVelocityCFEKernels.hpp
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
// Description: Kernel converting conserved variables to velocity and sound
// speed.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

// header should appear after loop execution.hpp
#include "EquationOfState/VariableConverters.hpp"

namespace Nektar::detail
{

/**
 * @brief Fill @p outbase with the @p ndim velocity components followed by
 * the speed of sound.
 *
 * The layout mirrors what MaxStdVelocityOp expects when its sound speed
 * factor is one: velocity in the leading components, wave speed in the one
 * after them.
 */
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void CFLVelocityCFEKernel(
    const EqnOfSParams EoS, const unsigned int npts, const unsigned int ndim,
    const unsigned int inStride, const unsigned int outStride,
    const TData *inbase, TData *outbase, const unsigned int streamID)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize    = npts / vec_width;
    const size_t inVecStride  = inStride / vec_width;
    const size_t outVecStride = outStride / vec_width;
    const auto invec          = reinterpret_cast<const vec_t *>(inbase);
    auto outvec               = reinterpret_cast<vec_t *>(outbase);

    Nektar::LoopExecutionSetStreamID(streamID);

    // Parallelize over point groups; each i is independent.
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            const vec_t rho    = invec[0 * inVecStride + i];
            const vec_t invRho = vec_t(TData(1)) / rho;

            vec_t mom[3] = {vec_t(TData(0)), vec_t(TData(0)), vec_t(TData(0))};

            for (unsigned int d = 0; d < ndim; ++d)
            {
                mom[d] = invec[(1u + d) * inVecStride + i]; // rho*u_d
                outvec[d * outVecStride + i] = mom[d] * invRho;
            }

            const vec_t E = invec[(ndim + 1u) * inVecStride + i];
            const vec_t e = GetInternalEnergy(ndim, rho, mom, E);

            outvec[ndim * outVecStride + i] = GetSoundSpeed(EoS, rho, e);
        });
}

} // namespace Nektar::detail
