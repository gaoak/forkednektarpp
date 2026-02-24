///////////////////////////////////////////////////////////////////////////////
//
// File: AUSM3SolverKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

#include "RiemannSolvers/AUSMSolver/AUSMSolverKernels.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

struct AUSM3Upwinding
{
    template <typename TData>
    NEK_DEVICE_INLINE void operator()(const TData &cA, const TData &rhoL,
                                      const TData &rhoR, const TData &pL,
                                      const TData &pR, const TData &ML,
                                      const TData &MR, TData &pbar, TData &Mbar)
    {
        // Parameters for specify the upwinding
        // Note: if fa = 1 then AUSM3 = AUSM2
        TData Mco    = 0.01;
        TData Mtilde = 0.5 * (ML * ML + MR * MR);
        TData Mo     = std::sqrt(std::min(1.0, std::max(Mtilde, Mco * Mco)));
        TData fa     = Mo * (2.0 - Mo);
        TData beta   = 0.125;
        TData alpha  = 0.1875;
        TData sigma  = 1.0;
        TData Kp     = 0.25;
        TData Ku     = 0.75;
        TData rhoA   = 0.5 * (rhoL + rhoR);
        TData Mp     = -(Kp / fa) * ((pR - pL) / (rhoA * cA * cA)) *
                   std::max(1.0 - sigma * Mtilde, 0.0);

        Mbar = M4Function(0, beta, ML) + M4Function(1, beta, MR) + Mp;

        TData pu = -2.0 * Ku * rhoA * cA * cA * (MR - ML) *
                   P5Function(0, alpha, ML) * P5Function(1, alpha, MR);

        pbar =
            pL * P5Function(0, alpha, ML) + pR * P5Function(1, alpha, MR) + pu;
    }
};

template <typename ExecSpace, unsigned int NDIM> struct AUSM3SolverKernel
{
    template <typename TData>
    NEK_FORCE_INLINE void operator()(const size_t blksize, const TData *fwd,
                                     const TData *bwd, TData *flux)
    {
        AUSMSolverKernel<ExecSpace, NDIM, AUSM3Upwinding>()(blksize, fwd, bwd,
                                                            flux);
    }
};

} // namespace Nektar::Operators::detail
