///////////////////////////////////////////////////////////////////////////////
//
// File: CompressibleSolver.h
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
// Description: Compressible Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_COMPRESSIBLEFLOWSOLVER_RIEMANNSOLVER_COMPRESSIBLESOLVER
#define NEKTAR_SOLVERS_COMPRESSIBLEFLOWSOLVER_RIEMANNSOLVER_COMPRESSIBLESOLVER

#include <CompressibleFlowSolver/Misc/EquationOfState.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

using namespace Nektar::SolverUtils;

namespace Nektar
{

/**
 * @brief Dot product of two three-component arrays.
 *
 * Defined for both scalar and SIMD value types so that the Riemann kernels can
 * be written in vector notation without a rotation to the trace normal.
 */
template <class T> inline T RiemannDot(const T *a, const T *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

class CompressibleSolver : public RiemannSolver
{
protected:
    EquationOfStateSharedPtr m_eos;
    bool m_idealGas;

    /// Session ctor
    CompressibleSolver(const LibUtilities::SessionReaderSharedPtr &pSession);

    /// Programmatic ctor
    CompressibleSolver();

    using ND = NekDouble;

    /**
     * @brief Default implementation: apply v_PointSolve at each trace point.
     *
     * Solvers that can work on whole arrays at a time, and so vectorise,
     * override this directly instead of implementing v_PointSolve.
     */
    void v_Solve(const int nDim, const Array<OneD, const Array<OneD, ND>> &Fwd,
                 const Array<OneD, const Array<OneD, ND>> &Bwd,
                 Array<OneD, Array<OneD, ND>> &flux) override;

    /**
     * @brief Solve the Riemann problem across a trace point directly in
     * Cartesian components.
     *
     * The Euler equations are rotationally invariant, so the interface flux
     * could equivalently be obtained by rotating the states onto the trace
     * normal, applying a one-dimensional solver and rotating back. That
     * rotation is unnecessary: the tangential velocity components only ever
     * enter through rotationally-invariant combinations (the kinetic energy
     * and the shear jump projected onto the tangent plane), which is exactly
     * why the tangential basis could be chosen arbitrarily. Working in
     * Cartesian components throughout avoids building, storing and applying a
     * rotation matrix per quadrature point.
     *
     * @param rhoL      Density, left state.
     * @param momL      Momentum vector (3 components), left state.
     * @param EL        Energy, left state.
     * @param rhoR      Density, right state.
     * @param momR      Momentum vector (3 components), right state.
     * @param ER        Energy, right state.
     * @param normal    Unit trace normal (3 components).
     * @param rhof      Computed Riemann flux for density.
     * @param momf      Computed Riemann flux for momentum (3 components).
     * @param Ef        Computed Riemann flux for energy.
     *
     * Components of @p momL, @p momR and @p normal above the problem dimension
     * are zero on entry; the corresponding entries of @p momf are ignored.
     */
    virtual void v_PointSolve(
        [[maybe_unused]] ND rhoL, [[maybe_unused]] const ND *momL,
        [[maybe_unused]] ND EL, [[maybe_unused]] ND rhoR,
        [[maybe_unused]] const ND *momR, [[maybe_unused]] ND ER,
        [[maybe_unused]] const ND *normal, [[maybe_unused]] ND &rhof,
        [[maybe_unused]] ND *momf, [[maybe_unused]] ND &Ef)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This function should be defined by subclasses.");
    }

    ND GetRoeSoundSpeed(ND rhoL, ND pL, ND eL, ND HL, ND srL, ND rhoR, ND pR,
                        ND eR, ND HR, ND srR, ND HRoe, ND URoe2, ND srLR);
};

} // namespace Nektar

#endif
