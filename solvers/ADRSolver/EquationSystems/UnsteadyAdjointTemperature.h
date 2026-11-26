///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyAdjointTemperature.h
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
// Description: Unsteady temperature adjoint equation advanced in pseudo-time
//
//              tau = t1 - t,  phi(x,tau) = psi(x,t1-tau),  U(x,tau)=u(x,t1-tau)
//              d_tau phi - U.grad(phi) - nu laplacian(phi)
//                  = w_T(x,t1-tau)/(rho cp),
//              phi(tau=0) = 0,   nu = k/(rho cp).
//
//              Implemented with V = -U as the advection velocity of the
//              standard unsteady advection-diffusion equation.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ADRSOLVER_EQUATIONSYSTEMS_UNSTEADYADJOINTTEMPERATURE_H
#define NEKTAR_SOLVERS_ADRSOLVER_EQUATIONSYSTEMS_UNSTEADYADJOINTTEMPERATURE_H

#include <ADRSolver/Adjoint/AdjointBaseFlow.h>
#include <ADRSolver/EquationSystems/UnsteadyAdvectionDiffusion.h>

namespace Nektar
{

/**
 * @brief Unsteady continuous temperature adjoint, marched forward in
 *        pseudo-time tau = t1 - t.
 *
 * The physical adjoint equation is
 * \f[
 *   -\partial_t\psi - u\cdot\nabla\psi - \nu\nabla^2\psi
 *       = \frac{w_T}{\rho c_p}, \qquad \psi(t_1)=0,
 * \f]
 * with Robin conditions
 * \f$\nu\partial_n\psi + [\beta/(\rho c_p)+u_n]\psi
 *     = \beta w_q/(\rho c_p)@f$.
 *
 * With \f$\tau=t_1-t@f$, \f$\phi(\tau)=\psi(t_1-\tau)@f$ and
 * \f$V=-u(t_1-\tau)@f$ this is the standard advection-diffusion equation
 * advanced forward in pseudo-time.  The time integrator, output and
 * checkpoint machinery is inherited from UnsteadyAdvectionDiffusion.
 *
 * Required session entries
 * ------------------------
 *  - A placeholder function @c AdvectionVelocity with zero components
 *    (@c Vx,@c Vy[,@c Vz]); the base class needs it to allocate the velocity
 *    arrays and it is overwritten by the base flow at every RHS evaluation.
 *  - A session function @c AdjointSource giving @f$w_T/(\rho c_p)@f$ for
 *    @c psi.  The solver evaluates it itself at the physical time
 *    t = t1 - tau and adds it to the right-hand side; do not also place it
 *    in a @c FORCING block (that would count it twice).  The direct
 *    evaluation is used because the cached expression evaluation in
 *    ForcingBody assumes time increases, whereas the physical time
 *    decreases during the pseudo-time march.
 *  - Neumann/Robin boundary conditions for @c psi.  Note that the Robin
 *    PRIMCOEFF is evaluated at t=0 by this Nektar version, so it must be
 *    time-independent (use the VALUE for time-dependent b).
 *
 * Time parameters
 * ---------------
 *  - @c AdjointStartTime (default 0) and @c AdjointEndTime (default
 *    @c FinalTime): physical window [t0, t1].  Pseudo-time runs from 0 to
 *    t1-t0 with @c TimeStep.
 *  - Output/checkpoint metadata keeps @c Time = tau (required for restart)
 *    and adds @c PhysicalTime = t1-tau.
 *
 * Not supported (asserted): sub-stepping time integration, steady-state
 * stopping, and time-dependent Robin PRIMCOEFF.
 */
class UnsteadyAdjointTemperature : public UnsteadyAdvectionDiffusion
{
public:
    friend class MemoryManager<UnsteadyAdjointTemperature>;

    /// Creates an instance of this class
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<UnsteadyAdjointTemperature>::AllocateSharedPtr(
                pSession, pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

protected:
    UnsteadyAdjointTemperature(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~UnsteadyAdjointTemperature() override = default;

    void v_InitObject(bool DeclareFields = true) override;
    void v_DoInitialise(bool dumpInitialConditions = true) override;
    bool v_PostIntegrate(int step) override;
    void v_GenerateSummary(SolverUtils::SummaryList &s) override;

    /// RHS in pseudo-time: update V=-u(t1-tau), then call the base RHS with
    /// the physical time so that the source is evaluated correctly.
    void DoOdeRhs(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time);

    /// Projection: evaluate boundary conditions at the physical time.
    void DoOdeProjection(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time);

    /// Implicit diffusion: evaluate boundary conditions at the physical time.
    void DoImplicitSolve(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray, NekDouble time,
        NekDouble lambda);

private:
    void CopyVelocityMinusToFields();

    AdjointBaseFlow m_baseFlow;

    /// Workspace for evaluating AdjointSource at the physical time.
    Array<OneD, NekDouble> m_source;

    NekDouble m_rho;
    NekDouble m_cp;
    NekDouble m_conductivity;
    NekDouble m_nu;

    /// Physical observation window [m_t0, m_t1]; pseudo-time is t1-t.
    NekDouble m_t0;
    NekDouble m_t1;
};

} // namespace Nektar

#endif
