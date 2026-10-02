///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondWallCFEKernels.hpp
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
// Description: Block kernels of the viscous wall boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Set the exterior energy on a block so that the trace average of the
 * internal energy sits at the wall temperature.
 */
template <typename ExecSpace, typename TData, typename EoSParamType>
NEK_FORCE_INLINE static void ImposeIsothermalWallEnergyBlock(
    const size_t nPts, const size_t compStride, const unsigned coordDim,
    const size_t eng, const EoSParamType EoS, const TData Twall,
    const TData *interior, TData *ext, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nPts, NEKTAR_LAMBDA(const size_t i) {
            const TData rhoFwd = interior[i];
            const TData rhoBwd = ext[i];
            const TData rhoAve = TData(0.5) * (rhoFwd + rhoBwd);

            TData momSqFwd = TData(0.0);
            TData momSqBwd = TData(0.0);
            for (unsigned d = 1; d <= coordDim; ++d)
            {
                const size_t od = d * compStride + i;
                momSqFwd += interior[od] * interior[od];
                momSqBwd += ext[od] * ext[od];
            }

            // Interior internal energy, as the kernel forms it.
            const TData eIntFwd =
                interior[eng * compStride + i] - TData(0.5) * momSqFwd / rhoFwd;

            const TData eWall = GetIntEnergyFromTemperature(EoS, rhoAve, Twall);

            // Exterior internal energy that puts the average on the wall
            // value, then back to a total energy.
            const TData eIntBwd = TData(2.0) * rhoAve * eWall - eIntFwd;

            ext[eng * compStride + i] =
                eIntBwd + TData(0.5) * momSqBwd / rhoBwd;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Reverse the momentum of a block's seeded interior state, leaving
 * density and energy as they are.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ReverseMomentumBlock(
    const size_t stride, const unsigned coordDim, TData *bndPtr,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned d = 1; d <= coordDim; ++d)
            {
                bndPtr[d * stride + i] = -bndPtr[d * stride + i];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
