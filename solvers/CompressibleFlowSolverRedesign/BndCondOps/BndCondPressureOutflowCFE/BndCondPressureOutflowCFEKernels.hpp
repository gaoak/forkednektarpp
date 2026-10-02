///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondPressureOutflowCFEKernels.hpp
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
// Description: Block kernels of the pressure outflow boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Copy the prescribed pressure out of the energy slot of a block's
 * session-evaluated storage.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyPrescribedPressureBlock(
    const size_t stride, const size_t eng, const TData *bndPtr, TData *pOut,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, stride,
        NEKTAR_LAMBDA(const size_t i) { pOut[i] = bndPtr[eng * stride + i]; });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Impose the prescribed pressure on a block's seeded interior state
 * where the outflow is subsonic; supersonic points keep the extrapolated
 * state.
 */
template <typename ExecSpace, typename TData, typename EoSParamType>
NEK_FORCE_INLINE static void ApplyPressureOutflowBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const EoSParamType EoS, const TData *pOut, const TData *norms,
    TData *bndPtr, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            using std::abs;
            using std::sqrt;

            const TData rho = bndPtr[i];

            // Kinetic energy and normal momentum from the extrapolated
            // state. The normals share the storage layout, so they are
            // indexed exactly as the momentum components are.
            TData momSq   = TData(0.0);
            TData momDotN = TData(0.0);
            for (unsigned d = 1; d <= coordDim; ++d)
            {
                const TData m = bndPtr[d * stride + i];
                momSq += m * m;
                momDotN += m * norms[(d - 1) * stride + i];
            }
            const TData Ek = TData(0.5) * momSq / rho;

            const TData eInt = (bndPtr[eng * stride + i] - Ek) / rho;
            const TData c    = GetSoundSpeed(EoS, rho, eInt);

            // Normal Mach number: it is the characteristic crossing the
            // boundary that decides whether anything may be imposed,
            // so the test is on n.u and not on the speed.
            const TData MachN = abs(momDotN) / (rho * c);

            if (MachN < TData(0.99))
            {
                // Subsonic: one characteristic enters, so impose the
                // pressure through the energy and leave rho and
                // momentum extrapolated.
                const TData eOut = GetIntEnergyFromPressure(EoS, rho, pOut[i]);
                bndPtr[eng * stride + i] = rho * eOut + Ek;
            }
            // Supersonic: no characteristic enters, nothing may be
            // imposed, and the whole state is extrapolated. The seed is
            // already that state, so there is deliberately nothing to
            // do - the branch is kept so that the choice is visible.
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
