///////////////////////////////////////////////////////////////////////////////
//
// File: VolumeFluxKernels.hpp
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
// Description: Volume flux kernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Operators/Common/Spaces.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/NekInline.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AdvectVolumeFluxKernel(
    const unsigned int npts, const unsigned int velComps,
    const unsigned int nvarComps, const unsigned int velStride,
    const unsigned int inStride, const unsigned int outStride,
    const TData *velbase, const TData *inbase, TData *outbase)
{
    // Parallelize over points; each i is independent
    Nektar::parallel_for<ExecSpace>(
        0u, npts, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned int nvar = 0; nvar < nvarComps; ++nvar)
            {
                const size_t in_off = (nvar)*inStride + i;
                const TData in_val  = inbase[in_off];

                for (unsigned int ndim = 0; ndim < velComps; ++ndim)
                {
                    const size_t vel_off = ndim * velStride + i;
                    const size_t out_off =
                        (nvar * velComps + ndim) * outStride + i;

                    outbase[out_off] = in_val * velbase[vel_off];
                }
            }
        });
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void EulerVolumeFluxKernel(
    const unsigned int npts, const unsigned int ndim,
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

            // ---- Pressure (ideal gas): p = (gamma-1)*(E - 0.5*rho*|u|^2) ----
            TData u2sum = TData(0);
            for (unsigned int d = 0; d < ndim; ++d)
            {
                u2sum += vel[d] * vel[d];
            }

            const TData p      = (1.4 - 1) * (E - TData(0.5) * rho * u2sum);
            const TData ePlusP = E + p;

            // ---- Fluxes ----
            // rho equation: F_rho,d = rho*u_d = mom[d]
            for (unsigned int d = 0; d < ndim; ++d)
            {
                outbase[(0u * ndim + d) * outStride + i] = mom[d];
            }

            // momentum equations:
            // F_{mom_a,d} = (rho*u_a)*u_d + p*delta_{a,d}
            for (unsigned int a = 0; a < ndim; ++a)
            {
                for (unsigned int d = 0; d < ndim; ++d)
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
            for (unsigned int d = 0; d < ndim; ++d)
            {
                outbase[((ndim + 1u) * ndim + d) * outStride + i] =
                    ePlusP * vel[d];
            }
        });
}

} // namespace Nektar::Operators::detail