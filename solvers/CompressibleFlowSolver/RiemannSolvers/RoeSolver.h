///////////////////////////////////////////////////////////////////////////////
//
// File: RoeSolver.h
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
// Description: Roe Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_COMPRESSIBLEFLOWSOLVER_RIEMANNSOLVER_ROESOLVER
#define NEKTAR_SOLVERS_COMPRESSIBLEFLOWSOLVER_RIEMANNSOLVER_ROESOLVER

#include <CompressibleFlowSolver/RiemannSolvers/CompressibleSolver.h>

namespace Nektar
{

class RoeSolver : public CompressibleSolver
{
public:
    static RiemannSolverSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession)
    {
        return RiemannSolverSharedPtr(new RoeSolver(pSession));
    }

    static std::string solverName;
    /// Registration of the "RoeOpt" alias, kept for backwards compatibility.
    static std::string solverNameOpt;

    /// programmatic ctor
    RoeSolver();

protected:
    RoeSolver(const LibUtilities::SessionReaderSharedPtr &pSession);

    using ND = NekDouble;

    void v_Solve(const int nDim, const Array<OneD, const Array<OneD, ND>> &Fwd,
                 const Array<OneD, const Array<OneD, ND>> &Bwd,
                 Array<OneD, Array<OneD, ND>> &flux) final;
};

/**
 * @brief Roe Riemann solver, in Cartesian components.
 *
 * Stated equation numbers are from:
 *
 *   "Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical
 *   Introduction", E. F. Toro (3rd edition, 2009).
 *
 * The algorithm is the one prescribed following equation 11.70, but written
 * against the trace normal @p normal rather than against a rotated frame whose
 * first axis is the normal. The two are algebraically identical: in the
 * rotated frame the tangential components enter only through the kinetic
 * energy and through the pair of shear waves, and the latter contribute
 * \f$\alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3\f$, which is the
 * momentum jump projected onto the tangent plane. That projection needs no
 * basis for the tangent plane, which is why the rotation -- whose tangential
 * axes were always an arbitrary choice -- is not required.
 *
 * Components of @p momL, @p momR and @p normal above the problem dimension are
 * expected to be zero; the corresponding entries of @p momf are then zero too.
 */
template <class T, std::enable_if_t<std::is_floating_point_v<T> ||
                                        tinysimd::is_vector_floating_point_v<T>,
                                    bool>
                       Enable = true>
inline void RoeKernel(T &rhoL, T *momL, T &EL, T &rhoR, T *momR, T &ER,
                      T *normal, T &rhof, T *momf, T &Ef, NekDouble gamma)
{
    // Left and right velocities
    T uL[3], uR[3];
    for (size_t d = 0; d < 3; ++d)
    {
        uL[d] = momL[d] / rhoL;
        uR[d] = momR[d] / rhoR;
    }

    // Left and right pressures
    T pL = (gamma - 1.0) * (EL - 0.5 * RiemannDot(momL, uL));
    T pR = (gamma - 1.0) * (ER - 0.5 * RiemannDot(momR, uR));

    // Left and right enthalpy
    T hL = (EL + pL) / rhoL;
    T hR = (ER + pR) / rhoR;

    // Left and right normal velocities
    T unL = RiemannDot(uL, normal);
    T unR = RiemannDot(uR, normal);

    // Square root of rhoL and rhoR.
    T srL  = sqrt(rhoL);
    T srR  = sqrt(rhoR);
    T srLR = srL + srR;

    // Velocity, enthalpy and sound speed Roe averages (equation 11.60).
    T uRoe[3];
    for (size_t d = 0; d < 3; ++d)
    {
        uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
    }
    T hRoe  = (srL * hL + srR * hR) / srLR;
    T URoe  = RiemannDot(uRoe, uRoe);
    T cRoe  = sqrt((gamma - 1.0) * (hRoe - 0.5 * URoe));
    T unRoe = RiemannDot(uRoe, normal);

    // Calculate jumps \Delta u_i (defined preceding equation 11.67).
    T jumpRho = rhoR - rhoL;
    T jumpE   = ER - EL;
    T jumpMom[3];
    for (size_t d = 0; d < 3; ++d)
    {
        jumpMom[d] = momR[d] - momL[d];
    }
    T jumpMn = RiemannDot(jumpMom, normal);

    // The two shear waves of the rotated system combine into the part of the
    // momentum jump lying in the tangent plane. This is the only place the
    // tangential directions appear, and projecting them out requires no basis.
    T shear[3];
    {
        T q[3];
        for (size_t d = 0; d < 3; ++d)
        {
            q[d] = jumpMom[d] - jumpRho * uRoe[d];
        }
        T qn = RiemannDot(q, normal);
        for (size_t d = 0; d < 3; ++d)
        {
            shear[d] = q[d] - qn * normal[d];
        }
    }

    // Define \Delta u_5 (equation 11.70).
    T jumpbar = jumpE - RiemannDot(shear, uRoe);

    // Compute wave amplitudes (equations 11.68, 11.69).
    T alpha1 = (gamma - 1.0) *
               (jumpRho * (hRoe - unRoe * unRoe) + unRoe * jumpMn - jumpbar) /
               (cRoe * cRoe);
    T alpha0 =
        (jumpRho * (unRoe + cRoe) - jumpMn - cRoe * alpha1) / (2.0 * cRoe);
    T alpha4 = jumpRho - (alpha0 + alpha1);

    // Needed to get right overload resolution for std::abs
    using std::abs;

    // Wave strengths scaled by the eigenvalues \lambda_i (equation 11.58).
    // The two shear waves share the eigenvalue |u.n|.
    T unRoeAbs = abs(unRoe);
    T s0       = 0.5 * alpha0 * abs(unRoe - cRoe);
    T s1       = 0.5 * alpha1 * unRoeAbs;
    T s4       = 0.5 * alpha4 * abs(unRoe + cRoe);
    T ss       = 0.5 * unRoeAbs;

    // Average of the left and right directional fluxes F.n needed for
    // equation 11.29.
    rhof = 0.5 * (rhoL * unL + rhoR * unR);
    for (size_t d = 0; d < 3; ++d)
    {
        momf[d] = 0.5 * (rhoL * unL * uL[d] + rhoR * unR * uR[d] +
                         (pL + pR) * normal[d]);
    }
    Ef = 0.5 * (unL * (EL + pL) + unR * (ER + pR));

    // Finally perform the summation (11.29). Rotated back to Cartesian
    // components the acoustic eigenvectors carry momentum u -/+ c n and the
    // entropy wave carries u, while the shear waves carry the tangential jump.
    rhof -= s0 + s1 + s4;
    for (size_t d = 0; d < 3; ++d)
    {
        momf[d] -= s0 * (uRoe[d] - cRoe * normal[d]) + s1 * uRoe[d] +
                   s4 * (uRoe[d] + cRoe * normal[d]) + ss * shear[d];
    }
    Ef -= s0 * (hRoe - unRoe * cRoe) + s1 * 0.5 * URoe +
          s4 * (hRoe + unRoe * cRoe) + ss * RiemannDot(shear, uRoe);
}

} // namespace Nektar

#endif
