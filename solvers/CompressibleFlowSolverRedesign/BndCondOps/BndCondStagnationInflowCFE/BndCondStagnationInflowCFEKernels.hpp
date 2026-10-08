///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondStagnationInflowCFEKernels.hpp
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
// Description: Block kernels of the stagnation inflow boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Reduce a block's session-evaluated storage to the stagnation density,
 * the stagnation energy and a unit inflow direction, taking the inward
 * normal where the session gives no direction.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ReduceStagnationStateBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const TData *bndPtr, const TData *norms, TData *rhoStag, TData *EStag,
    TData *dir, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            using std::sqrt;

            rhoStag[i] = bndPtr[i];
            EStag[i]   = bndPtr[eng * stride + i];

            // The momentum entries are a direction; only their ratio
            // matters, so normalise. Zero means normal to the
            // boundary, and the normals here are already in the
            // storage layout, so the inward normal is simply -n at
            // this point.
            TData norm = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData c = bndPtr[(d + 1) * stride + i];
                norm += c * c;
            }
            norm = sqrt(norm);

            for (unsigned d = 0; d < coordDim; ++d)
            {
                dir[d * stride + i] = (norm > TData(1.0e-8))
                                          ? bndPtr[(d + 1) * stride + i] / norm
                                          : -norms[d * stride + i];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Build the inflow state on a block from the stagnation state and the
 * speed of the seeded interior state, along the isentrope.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ApplyStagnationInflowBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const TData gamma, const TData *rhoStag, const TData *EStag,
    const TData *dir, TData *bndPtr, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const TData gamM1 = gamma - TData(1.0);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            using std::pow;
            using std::sqrt;

            // Speed from the interior state the caller seeded.
            const TData rhoInt = bndPtr[i];
            TData momSq        = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData m = bndPtr[(d + 1) * stride + i];
                momSq += m * m;
            }
            const TData absVel = sqrt(momSq) / rhoInt;

            // Stagnation enthalpy, then the static one by a
            // one-dimensional energy balance.
            const TData hStag = gamma * EStag[i] / rhoStag[i];
            const TData hStat = hStag - TData(0.5) * absVel * absVel;

            // Density along the isentrope from the stagnation state.
            const TData rho =
                rhoStag[i] * pow(rhoStag[i] * hStat / (EStag[i] * gamma),
                                 TData(1.0) / gamM1);

            for (unsigned d = 0; d < coordDim; ++d)
            {
                bndPtr[(d + 1) * stride + i] =
                    rho * absVel * dir[d * stride + i];
            }

            bndPtr[i] = rho;
            bndPtr[eng * stride + i] =
                rho * (hStat / gamma + TData(0.5) * absVel * absVel);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
