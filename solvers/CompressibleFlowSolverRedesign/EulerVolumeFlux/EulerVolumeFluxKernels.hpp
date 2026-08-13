///////////////////////////////////////////////////////////////////////////////
//
// File: EulerVolumeFluxKernels.hpp
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
// Description: Euler volume flux kernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

// header should appear after loop execution.hpp
#include "EquationOfState/VariableConverters.hpp"

namespace Nektar::detail
{
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void EulerVolumeFluxKernel(
    const EqnOfSParams EoS, const unsigned int npts, const unsigned int ndim,
    const unsigned int nvarComps, const unsigned int inStride,
    const unsigned int outStride, const TData *inbase, TData *outbase,
    const unsigned int streamID)
{
    // Check that we have the right number of variables for Euler
    if (nvarComps != ndim + 2)
    {
        return;
    }

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
            // ---- Load conservative variables U at point i ----
            const vec_t rho    = invec[0 * inVecStride + i];
            const vec_t invRho = vec_t(TData(1)) / rho;

            vec_t mom[3] = {vec_t(TData(0)), vec_t(TData(0)), vec_t(TData(0))};
            vec_t vel[3] = {vec_t(TData(0)), vec_t(TData(0)), vec_t(TData(0))};

            for (unsigned int d = 0; d < ndim; ++d)
            {
                mom[d] = invec[(1u + d) * inVecStride + i]; // rho*u_d
                vel[d] = mom[d] * invRho;                   // u_d
            }

            const vec_t E = invec[(ndim + 1u) * inVecStride + i];

            // ---- Pressure via EoS, explicitly vectorized over lanes ----
            const vec_t e      = GetInternalEnergy(ndim, rho, mom, E);
            const vec_t p      = GetPressure(EoS, rho, e);
            const vec_t ePlusP = E + p;

            // ---- Fluxes ----
            // rho equation: F_rho,d = rho*u_d = mom[d]
            for (unsigned int d = 0; d < ndim; ++d)
            {
                outvec[(0u * ndim + d) * outVecStride + i] = mom[d];
            }

            // momentum equations:
            // F_{mom_a,d} = (rho*u_a)*u_d + p*delta_{a,d}
            for (unsigned a = 0; a < ndim; ++a)
            {
                for (unsigned d = 0; d < ndim; ++d)
                {
                    vec_t val = mom[a] * vel[d];
                    if (a == d)
                    {
                        val += p;
                    }
                    outvec[((1u + a) * ndim + d) * outVecStride + i] = val;
                }
            }

            // energy equation: F_E,d = (E+p)*u_d
            for (unsigned d = 0; d < ndim; ++d)
            {
                outvec[((ndim + 1u) * ndim + d) * outVecStride + i] =
                    ePlusP * vel[d];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
