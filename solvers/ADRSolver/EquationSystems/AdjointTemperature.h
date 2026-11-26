///////////////////////////////////////////////////////////////////////////////
//
// File: AdjointTemperature.h
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
// Description: Steady temperature adjoint equation
//
//              -u.grad(psi) - nu laplacian(psi) = s,
//              nu = k/(rho cp),  s = w_T/(rho cp)
//
//              solved with the forward velocity reversed, c = -u.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ADRSOLVER_EQUATIONSYSTEMS_ADJOINTTEMPERATURE_H
#define NEKTAR_SOLVERS_ADRSOLVER_EQUATIONSYSTEMS_ADJOINTTEMPERATURE_H

#include <ADRSolver/Adjoint/AdjointBaseFlow.h>
#include <ADRSolver/EquationSystems/SteadyAdvectionDiffusion.h>

namespace Nektar
{

/**
 * @brief Steady continuous adjoint of the temperature advection-diffusion
 *        equation.
 *
 * Solves
 * \f[
 *    -u\cdot\nabla\psi - \nu\nabla^2\psi = \frac{w_T}{\rho c_p},
 *    \qquad \nu = \frac{k}{\rho c_p},
 * \f]
 * by reusing the SteadyAdvectionDiffusion assembly with @f$c=-u@f$ and
 * @f$\epsilon=\nu@f$.  Dirichlet/Neumann/Robin conditions are supplied in the
 * usual Nektar form; the adjoint Robin condition is
 * \f$\partial_n\psi + a\psi = b@f$ with
 * \f$a=(\beta+\rho c_p u_n)/k@f$, \f$b=\beta w_q/k@f$.
 *
 * Parameters
 * ----------
 *  - @c Rho, @c Cp, @c Conductivity : physical properties (default 1).
 *  - @c epsilon : optional direct override of nu (takes precedence).
 *  - @c lambda  : optional linear reaction coefficient (default 0).
 *  - @c BaseFlowTime : physical time at which the base flow is taken when a
 *    time series is supplied (default: first slice).
 *
 * A session function named @c AdjointSource (falling back to @c Forcing) must
 * provide the volumetric source @f$s=w_T/(\rho c_p)@f$ for the variable
 * @c psi.  Its value is divided internally by @f$\epsilon@f$ by the inherited
 * assembly, exactly as in SteadyAdvectionDiffusion.
 */
class AdjointTemperature : public SteadyAdvectionDiffusion
{
public:
    friend class MemoryManager<AdjointTemperature>;

    /// Creates an instance of this class
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<AdjointTemperature>::AllocateSharedPtr(pSession,
                                                                 pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

protected:
    AdjointTemperature(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~AdjointTemperature() override = default;

    void v_InitObject(bool DeclareFields = true) override;
    void v_DoInitialise(bool dumpInitialConditions = true) override;
    void v_DoSolve() override;
    void v_GenerateSummary(SolverUtils::SummaryList &s) override;

private:
    AdjointBaseFlow m_baseFlow;

    /// Volumetric source s = w_T/(rho cp), cached at v_DoInitialise so that
    /// repeated calls to v_DoSolve do not reuse the previous solution.
    Array<OneD, Array<OneD, NekDouble>> m_source;

    NekDouble m_rho;
    NekDouble m_cp;
    NekDouble m_conductivity;

    std::string m_sourceFunction;
};

} // namespace Nektar

#endif
