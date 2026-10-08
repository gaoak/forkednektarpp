///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondRiemannInvariantCFEKernels.hpp
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
// Description: Block kernels of the Riemann invariant farfield boundary
// condition.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::detail
{

/**
 * @brief Project the freestream velocity onto a block's outward normals.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ProjectFreestreamVelocityBlock(
    const size_t stride, const unsigned coordDim, const TData *velInfHost,
    const TData *norms, TData *VnInf, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // A host array of coordDim entries, copied into a fixed-size local so
    // that the kernel captures the values rather than a host address.
    TData velInf[3] = {TData(0.0), TData(0.0), TData(0.0)};
    for (unsigned d = 0; d < coordDim; ++d)
    {
        velInf[d] = velInfHost[d];
    }

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            TData vn = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                vn += velInf[d] * norms[d * stride + i];
            }
            VnInf[i] = vn;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Reconstruct the boundary state on a block from the Riemann invariants,
 * each taken from the side that carries it.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ApplyRiemannInvariantBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const TData gamma, const TData rhoInf, const TData pInf,
    const TData *velInfHost, const TData *VnInf, const TData *norms,
    TData *bndPtr, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const TData gamM1        = gamma - TData(1.0);
    const TData twoOverGamM1 = TData(2.0) / gamM1;

    // A host array of coordDim entries, copied into a fixed-size local so
    // that the kernel captures the values rather than a host address.
    TData velInf[3] = {TData(0.0), TData(0.0), TData(0.0)};
    for (unsigned d = 0; d < coordDim; ++d)
    {
        velInf[d] = velInfHost[d];
    }

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
#if defined(__SYCL_DEVICE_ONLY__)
            using sycl::pow;
            using sycl::sqrt;
#else
    using std::pow;
    using std::sqrt;
#endif

            // The storage holds the interior state.
            const TData rhoInt = bndPtr[i];

            TData momSq = TData(0.0);
            TData vn    = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData mom = bndPtr[(d + 1) * stride + i];
                momSq += mom * mom;
                vn += mom * norms[d * stride + i];
            }
            vn /= rhoInt;

            const TData pInt = gamM1 * (bndPtr[eng * stride + i] -
                                        TData(0.5) * momSq / rhoInt);
            const TData cInt = sqrt(gamma * pInt / rhoInt);
            const TData cInf = sqrt(gamma * pInf / rhoInf);

            const bool inflow = (vn <= TData(0.0));
            const bool sonic  = (std::abs(vn) / cInt >= TData(1.0));

            // Each invariant comes from the side it is carried from.
            // Supersonic takes both from the upstream side, which
            // makes the reconstruction below return that side's state
            // unchanged - no special case needed.
            const bool plusFromInterior  = !(inflow && sonic);
            const bool minusFromInterior = (!inflow && sonic);

            const TData rPlus  = plusFromInterior
                                     ? vn + cInt * twoOverGamM1
                                     : VnInf[i] + cInf * twoOverGamM1;
            const TData rMinus = minusFromInterior
                                     ? vn - cInt * twoOverGamM1
                                     : VnInf[i] - cInf * twoOverGamM1;

            const TData vnBC = TData(0.5) * (rPlus + rMinus);
            const TData cBC  = TData(0.25) * gamM1 * (rPlus - rMinus);

            // The entropy comes from upstream, and with it the
            // velocity vector whose normal component is corrected.
            const TData sBC =
                inflow ? pInf / pow(rhoInf, gamma) : pInt / pow(rhoInt, gamma);
            const TData vnUp = inflow ? VnInf[i] : vn;

            const TData rhoBC =
                pow(cBC * cBC / (gamma * sBC), TData(1.0) / gamM1);
            const TData pBC = rhoBC * cBC * cBC / gamma;

            const TData vDiff = vnBC - vnUp;

            TData EkBC = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData n = norms[d * stride + i];
                const TData up =
                    inflow ? velInf[d] : bndPtr[(d + 1) * stride + i] / rhoInt;
                const TData v = up + vDiff * n;

                bndPtr[(d + 1) * stride + i] = rhoBC * v;
                EkBC += TData(0.5) * rhoBC * v * v;
            }

            bndPtr[i]                = rhoBC;
            bndPtr[eng * stride + i] = pBC / gamM1 + EkBC;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
