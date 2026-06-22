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

#include "Operators/LoopExecution/LoopExecution.hpp"

// header should appear after loop execution.hpp
#include "EquationOfState/VariableConverters.hpp"

namespace Nektar::Operators::detail
{
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void EulerVolumeFluxKernel(
    const EqnOfSParams EoS, const unsigned int npts, const unsigned int ndim,
    const unsigned int nvarComps, const unsigned int inStride,
    const unsigned int outStride, const TData *inbase, TData *outbase)
{
    // Check that we have the right number of variables for Euler
    if (nvarComps != ndim + 2)
    {
        return;
    }

    // Parallelize over points; each i is independent
    Nektar::parallel_for<ExecSpace>(
        0u, npts, NEKTAR_LAMBDA(const size_t i) {
            // ---- Load conservative variables U at point i ----
            const TData rho    = inbase[0 * inStride + i];
            const TData invRho = TData(1) / rho;

            TData mom[3] = {TData(0), TData(0), TData(0)};
            TData vel[3] = {TData(0), TData(0), TData(0)};

            for (unsigned int d = 0; d < ndim; ++d)
            {
                mom[d] = inbase[(1u + d) * inStride + i]; // rho*u_d
                vel[d] = mom[d] * invRho;                 // u_d
            }

            const TData E = inbase[(ndim + 1u) * inStride + i];

            TData e            = GetInternalEnergy(ndim, rho, mom, E);
            TData p            = GetPressure(EoS, rho, e);
            const TData ePlusP = E + p;

            // ---- Fluxes ----
            // rho equation: F_rho,d = rho*u_d = mom[d]
            for (unsigned int d = 0; d < ndim; ++d)
            {
                outbase[d * outStride + i] = mom[d];
            }

            // momentum equations:
            // F_{mom_a,d} = (rho*u_a)*u_d + p*delta_{a,d}
            for (unsigned a = 0; a < ndim; ++a)
            {
                for (unsigned d = 0; d < ndim; ++d)
                {
                    TData val = mom[a] * vel[d];
                    if (a == d)
                    {
                        val += p;
                    }
                    outbase[((1u + a) * ndim + d) * outStride + i] = val;
                }
            }

            // energy equation: F_E,d = (E+p)*u_d
            for (unsigned d = 0; d < ndim; ++d)
            {
                outbase[((ndim + 1u) * ndim + d) * outStride + i] =
                    ePlusP * vel[d];
            }
        });
}

} // namespace Nektar::Operators::detail
