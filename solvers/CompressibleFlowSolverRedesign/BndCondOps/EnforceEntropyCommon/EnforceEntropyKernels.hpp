///////////////////////////////////////////////////////////////////////////////
//
// File: EnforceEntropyKernels.hpp
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
// Description: The subsonic star states of the entropy Riemann boundary
// conditions.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>

#include <LibUtilities/LoopExecution/LoopExecution.hpp>
#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::detail
{

/**
 * @brief The state an entropy Riemann boundary condition imposes at a point.
 *
 * Density, pressure and the *inward* normal velocity. The caller turns the
 * last into a velocity vector, which is where the three conditions agree
 * again.
 */
template <typename TData> struct EntropyStarState
{
    TData rho;
    TData p;
    TData u;
};

/**
 * @brief The incoming Riemann invariant of the interior state.
 *
 * \f$R^- = -u_n - 2c/(\gamma-1)\f$ with \f$u_n\f$ the *outward* normal
 * velocity, so this is the invariant carried along the characteristic
 * entering the domain. Two of the three conditions solve for the star state
 * against it; the third replaces it with the interior sound speed directly.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE TData EntropyRiemannInvariant(const TData vn,
                                                    const TData c,
                                                    const TData twoOverGamM1)
{
    return -vn - c * twoOverGamM1;
}

/**
 * @brief Subsonic inflow holding the entropy and the velocity.
 *
 * The sound speed is carried over from the interior unchanged and the entropy
 * comes from the prescribed state; density and pressure follow from the pair.
 * The normal velocity is the prescribed one reflected about the interior,
 * \f$2(-V_n^\infty) + u_n\f$, which is what leaves the tangential components
 * of the prescribed velocity untouched once the caller applies the mismatch.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE EntropyStarState<TData> EntropyVelocityStar(
    const TData vn, const TData cInt, [[maybe_unused]] const TData pInt,
    [[maybe_unused]] const TData rhoInt, const TData rhoBC, const TData pBC,
    const TData VnInf, const TData gamma, const TData gamM1,
    [[maybe_unused]] const TData twoOverGamM1)
{
#if defined(__SYCL_DEVICE_ONLY__)
    using sycl::pow;
#else
    using std::pow;
#endif

    const TData s = pBC / pow(rhoBC, gamma);
    const TData c = cInt;

    EntropyStarState<TData> star;
    star.rho = pow(c * c / (gamma * s), TData(1.0) / gamM1);
    star.p   = c * c * star.rho / gamma;
    star.u   = TData(2.0) * (-VnInf) + vn;

    return star;
}

/**
 * @brief Subsonic inflow holding the entropy and the pressure.
 *
 * Density and pressure are taken from the prescribed state - which fixes the
 * entropy with them - and the normal velocity is whatever the incoming
 * invariant then requires. This is the combination the reference singles out
 * as the stable way to impose a known pressure.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE EntropyStarState<TData> EntropyPressureStar(
    const TData vn, const TData cInt, [[maybe_unused]] const TData pInt,
    [[maybe_unused]] const TData rhoInt, const TData rhoBC, const TData pBC,
    [[maybe_unused]] const TData VnInf, const TData gamma,
    [[maybe_unused]] const TData gamM1, const TData twoOverGamM1)
{
    using std::sqrt;

    const TData rR = EntropyRiemannInvariant(vn, cInt, twoOverGamM1);

    EntropyStarState<TData> star;
    star.rho = rhoBC;
    star.p   = pBC;

    const TData c = sqrt(gamma * star.p / star.rho);
    star.u        = rR + c * twoOverGamM1;

    return star;
}

/**
 * @brief Subsonic inflow holding the entropy and the total enthalpy.
 *
 * The remaining freedom is filled by the stagnation state rather than by a
 * static quantity, so the sound speed is no longer available directly and
 * comes from a quadratic: with \f$k = 2/(\gamma-1)\f$,
 *
 * \f[ (k^2 + k)\,c_*^2 + 2k\,R^-c_* + \left(R^{-2} - v_n^2
 *     - k\,\gamma p_{BC}/\rho_{BC}\right)(k^2+k) = 0 \f]
 *
 * solved for c_*. The entropy of the prescribed state
 * then gives the density, and the pressure follows.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE EntropyStarState<TData> EntropyTotalEnthalpyStar(
    const TData vn, const TData cInt, [[maybe_unused]] const TData pInt,
    [[maybe_unused]] const TData rhoInt, const TData rhoBC, const TData pBC,
    const TData VnInf, const TData gamma, [[maybe_unused]] const TData gamM1,
    const TData twoOverGamM1)
{
#if defined(__SYCL_DEVICE_ONLY__)
    using sycl::pow;
    using sycl::sqrt;
#else
    using std::pow;
    using std::sqrt;
#endif

    const TData rR   = EntropyRiemannInvariant(vn, cInt, twoOverGamM1);
    const TData vnBC = -VnInf;

    TData tmp1 = twoOverGamM1 * rR;
    tmp1       = tmp1 * tmp1;

    TData tmp2 = rR * rR - vnBC * vnBC - twoOverGamM1 * gamma * pBC / rhoBC;
    tmp2       = (twoOverGamM1 * twoOverGamM1 + twoOverGamM1) * tmp2;

    TData c = -twoOverGamM1 * rR + sqrt(tmp1 - tmp2);
    c       = c / (twoOverGamM1 * twoOverGamM1 + twoOverGamM1);

    const TData s = pBC / pow(rhoBC, gamma);

    EntropyStarState<TData> star;
    star.u   = rR + twoOverGamM1 * c;
    star.rho = pow(c * c / (gamma * s), TData(0.5) * twoOverGamM1);
    star.p   = star.rho * c * c / gamma;

    return star;
}

/**
 * @brief Subsonic outflow imposing the prescribed pressure isentropically.
 *
 * Shared by the entropy-pressure and entropy-total-enthalpy conditions. The
 * density comes from the interior along an isentrope to the imposed pressure,
 * and the normal velocity from the incoming invariant. The entropy-velocity
 * condition has no such branch - it imposes the prescribed state at an outflow
 * instead - so it does not use this.
 */
template <typename TData>
NEK_HOSTDEVICE_INLINE EntropyStarState<TData> EntropyPressureOutflowStar(
    const TData vn, const TData cInt, const TData pInt, const TData rhoInt,
    [[maybe_unused]] const TData rhoBC, const TData pBC, const TData gamma,
    const TData twoOverGamM1)
{
#if defined(__SYCL_DEVICE_ONLY__)
    using sycl::pow;
    using sycl::sqrt;
#else
    using std::pow;
    using std::sqrt;
#endif

    const TData rR = EntropyRiemannInvariant(vn, cInt, twoOverGamM1);

    EntropyStarState<TData> star;
    star.p   = pBC;
    star.rho = rhoInt * pow(star.p / pInt, TData(1.0) / gamma);

    const TData c = sqrt(gamma * star.p / star.rho);
    star.u        = rR + c * twoOverGamM1;

    return star;
}

/**
 * @brief Reduce a block's session-evaluated conserved state to the density,
 * pressure, velocity and normal velocity the entropy conditions prescribe.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ReduceEntropyStateBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const TData gamma, const TData *bndPtr, const TData *norms, TData *rhoBC,
    TData *pBC, TData *velBC, TData *VnInf, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const TData gamM1 = gamma - TData(1.0);

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            const TData rho = bndPtr[i];

            TData momSq = TData(0.0);
            TData vn    = TData(0.0);

            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData mom = bndPtr[(d + 1) * stride + i];
                const TData vel = mom / rho;

                velBC[d * stride + i] = vel;
                momSq += mom * mom;
                vn += vel * norms[d * stride + i];
            }

            rhoBC[i] = rho;
            VnInf[i] = vn;
            pBC[i] =
                gamM1 * (bndPtr[eng * stride + i] - TData(0.5) * momSq / rho);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief Build the boundary state on a block from the policy's star state,
 * chosen by the direction and normal Mach number of the seeded interior
 * state.
 */
template <typename ExecSpace, typename Policy, typename TData>
NEK_FORCE_INLINE static void ApplyEnforceEntropyBlock(
    const size_t stride, const unsigned coordDim, const size_t eng,
    const TData gamma, const TData *rhoBC, const TData *pBC, const TData *velBC,
    const TData *VnInf, const TData *norms, TData *bndPtr,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const TData gamM1        = gamma - TData(1.0);
    const TData twoOverGamM1 = TData(2.0) / gamM1;

    Nektar::parallel_for<ExecSpace>(
        0u, stride, NEKTAR_LAMBDA(const size_t i) {
            using std::abs;
            using std::sqrt;

            // The storage holds the interior state, seeded by the
            // caller before this runs.
            const TData rhoInt = bndPtr[i];

            TData momSq = TData(0.0);
            TData vn    = TData(0.0);
            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData mom = bndPtr[(d + 1) * stride + i];
                momSq += mom * mom;
                vn += mom * norms[d * stride + i];
            }
            // Outward normal velocity of the interior state.
            vn /= rhoInt;

            const TData pInt = gamM1 * (bndPtr[eng * stride + i] -
                                        TData(0.5) * momSq / rhoInt);
            const TData cInt = sqrt(gamma * pInt / rhoInt);

            // It is the normal Mach number that decides how many
            // characteristics cross, so the test is on n.u rather than
            // on the speed.
            const TData machN   = abs(vn) / cInt;
            const bool inflow   = (vn <= TData(0.0));
            const bool subsonic = (machN < TData(1.0));

            EntropyStarState<TData> star;

            if (inflow && subsonic)
            {
                star =
                    Policy::InflowStar(vn, cInt, pInt, rhoInt, rhoBC[i], pBC[i],
                                       VnInf[i], gamma, gamM1, twoOverGamM1);
            }
            else if (!inflow && subsonic && Policy::hasSubsonicOutflow)
            {
                star = Policy::OutflowStar(vn, cInt, pInt, rhoInt, rhoBC[i],
                                           pBC[i], gamma, twoOverGamM1);
            }
            else
            {
                // Supersonic carries nothing back into the domain, so
                // the whole prescribed state stands. The conditions
                // without a subsonic outflow branch land here too,
                // since switching on the outflow Mach number was found
                // to hurt convergence.
                star.rho = rhoBC[i];
                star.p   = pBC[i];
                star.u   = -VnInf[i];
            }

            // Internal energy from the star pressure; the kinetic part
            // comes with the velocity below.
            TData EBC = star.p / gamM1;

            // On inflow the prescribed velocity vector is kept and
            // only its normal component shifted by the Riemann
            // mismatch, so the tangential components survive. On
            // outflow the velocity is purely normal.
            const TData vnDiff = star.u + VnInf[i];

            for (unsigned d = 0; d < coordDim; ++d)
            {
                const TData n = norms[d * stride + i];
                const TData v =
                    inflow ? velBC[d * stride + i] - vnDiff * n : -star.u * n;

                bndPtr[(d + 1) * stride + i] = star.rho * v;
                EBC += TData(0.5) * star.rho * v * v;
            }

            bndPtr[i]                = star.rho;
            bndPtr[eng * stride + i] = EBC;
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
