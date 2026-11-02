///////////////////////////////////////////////////////////////////////////////
//
// File: AUSM2SolverKernels.hpp
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

#include "RiemannSolvers/AUSMSolverKernels.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

struct AUSM2Upwinding
{
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const TData &cA, const TData &rhoL,
                                          const TData &rhoR, const TData &pL,
                                          const TData &pR, const TData &ML,
                                          const TData &MR, TData &pbar,
                                          TData &Mbar)
    {
        using std::max;

        using scalar_t = tinysimd::scalar_type_t<TData>;

        // Parameters for specify the upwinding
        constexpr scalar_t beta  = 0.125;
        constexpr scalar_t alpha = 0.1875;
        constexpr scalar_t sigma = 1.0;
        constexpr scalar_t Kp    = 0.25;
        constexpr scalar_t Ku    = 0.75;
        const TData Mtilde       = TData(0.5) * (ML * ML + MR * MR);
        const TData rhoA         = TData(0.5) * (rhoL + rhoR);
        const TData Mp           = TData(-Kp) * ((pR - pL) / (rhoA * cA * cA)) *
                         max(TData(1.0) - TData(sigma) * Mtilde, TData(0.0));

        Mbar = M4Function(0, TData(beta), ML) + M4Function(1, TData(beta), MR) +
               Mp;

        const TData pu = TData(-2.0 * Ku) * rhoA * cA * cA * (MR - ML) *
                         P5Function(0, TData(alpha), ML) *
                         P5Function(1, TData(alpha), MR);

        pbar = pL * P5Function(0, TData(alpha), ML) +
               pR * P5Function(1, TData(alpha), MR) + pu;
    }
};

template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct AUSM2SolverKernel
{
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TData *fwd, const TData *bwd,
                                          const TData *normal, TData *flux)
    {
        AUSMSolverKernel<ExecSpace, EqnOfStParams, NDIM, AUSM2Upwinding>()(
            EoS, blksize, fwd, bwd, normal, flux);
    }
};

} // namespace Nektar::detail
