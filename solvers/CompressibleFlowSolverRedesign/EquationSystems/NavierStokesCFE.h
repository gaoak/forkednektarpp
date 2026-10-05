///////////////////////////////////////////////////////////////////////////////
//
// File: NavierStokesCFE.h
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
// Description: Navier-Stokes equations in conservative variables without
// artificial diffusion
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <CompressibleFlowSolverRedesign/CFLVelocityCFE/CFLVelocityCFEOp.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <SolverCore/AdvDiffusion/AdvWeakDGDiffusionIP/AdvWeakDGDiffusionIPOp.hpp>
#include <SolverCore/EquationSystems/UnsteadySystem.h>

namespace Nektar
{
using namespace SolverCore;
using namespace Operators;

class NavierStokesCFE : public UnsteadySystem
{
public:
    friend class MemoryManager<NavierStokesCFE>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr equ =
            MemoryManager<NavierStokesCFE>::AllocateSharedPtr(pSession, pGraph);
        equ->InitObject();
        return equ;
    }

    /// Name of class
    static std::string className;

protected:
    // Initialise operators
    std::shared_ptr<AdvDiffusionOp<double>> m_advDiffusionOp;
    std::shared_ptr<ExpressionOp<double>> m_initialOp;
    std::shared_ptr<ExpressionOp<double>> m_velOp;

    NavierStokesCFE(const LibUtilities::SessionReaderSharedPtr &pSession,
                    const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~NavierStokesCFE() override = default;

    void DoOdeRhs(LibUtilities::Field<double, FieldState::Phys> &in,
                  LibUtilities::Field<double, FieldState::Phys> &out,
                  [[maybe_unused]] const double &time,
                  [[maybe_unused]] const double &factor);

    void v_InitObject(bool declareExpansionLists = true) override;

    /// Attach the boundary conditions this system supports.
    void SetUpBoundaryConditions();

    void v_GenerateSummary(SummaryList &s) override;

    void v_InitialiseOperators() override;

    void InitialiseParameters();

    LibUtilities::Field<double, FieldState::Phys> &v_GetCFLVelocityField()
        override;

    double v_GetSoundSpeedFactor() override;

    /// Velocity components followed by the speed of sound, filled from the
    /// conserved variables only when a Courant estimate is asked for.
    std::shared_ptr<CFLVelocityCFEOp<double>> m_cflVelocityOp;
    LibUtilities::Field<double, FieldState::Phys> m_cflVelocity;
};

} // namespace Nektar
