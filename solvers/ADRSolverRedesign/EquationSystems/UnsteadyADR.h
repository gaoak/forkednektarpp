///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyADR.h
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
// Description: Unsteady Advection Diffusion problem solve routines for new
// operators
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Operators/ElmtOps/Advection/AdvectionOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>
#include <Operators/SolverUtilsOps/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp>
#include <Operators/SolverUtilsOps/Advection/VolumeFluxOp.hpp>
#include <Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp>
#include <SolverCore/EquationSystems/UnsteadySystem.h>
#include <SolverCore/Forcing/Forcing.h>

namespace Nektar
{
using namespace SolverCore;
using namespace Operators;

class UnsteadyADR : public UnsteadySystem
{
public:
    friend class MemoryManager<UnsteadyADR>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr equ =
            MemoryManager<UnsteadyADR>::AllocateSharedPtr(pSession, pGraph);
        equ->InitObject();
        return equ;
    }

    /// Name of class
    static std::string className0;
    static std::string className1;
    static std::string className2;

protected:
    // Diffusion coefficient
    double m_epsilon;
    std::vector<double> m_diffCoeff;

    // Time stepping coefficient
    double m_lambda;
    bool m_advection         = false;
    bool m_diffusion         = false;
    bool m_explicitAdvection = false;
    bool m_explicitDiffusion = false;
    bool m_implicitAdvection = false;
    bool m_implicitDiffusion = false;

    // Setup workspaces
    Field<double, FieldState::Phys> m_advectionVel;
    Field<double, FieldState::Phys> m_traceAdvectionVel;

    // Initialise operators
    std::shared_ptr<AdvectionOp<double>> m_advectionCGOp;
    std::shared_ptr<AdvectionWeakDGOp<double>> m_advectionWeakDGOp;
    std::shared_ptr<VolumeFluxOp<double>> m_volumeFluxOp;
    std::shared_ptr<RiemannSolverOp<double>> m_riemannSolverOp;
    std::shared_ptr<GetFwdBwdTracePhysOp<double>> m_getFwdBwdTracePhysOp;
    std::shared_ptr<LinearSystemOp<double>> m_linearSystemOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverOp;
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconOp;
    std::vector<ForcingSharedPtr> m_forcing;

    UnsteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
                const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~UnsteadyADR() override = default;

    void DoImplicit(Field<double, FieldState::Phys> &inout,
                    Field<double, FieldState::Phys> &out,
                    [[maybe_unused]] const double &time, const double &lambda);

    void DoExplicitRhs(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out, const double &time,
                       const double &dt);

    void v_InitObject(bool declareExpansionLists = true) override;

    void v_GenerateSummary(SummaryList &s) override;

    void InitialiseParameters();

    void v_InitialiseFields() override;

    void v_InitialiseOperators() override;

    void SetDiffusionCoeff();

    void SetAdvectionVel();
};

} // namespace Nektar
