///////////////////////////////////////////////////////////////////////////////
//
// File: EulerCFE.h
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
// Description: Euler equations in consƒervative variables without artificial
// diffusion
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <CompressibleFlowSolverRedesign/EulerVolumeFlux/EulerVolumeFluxOp.hpp>
#include <CompressibleFlowSolverRedesign/RiemannSolvers/CompressibleSolverOp.hpp>
#include <SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp>
#include <SolverCore/EquationSystems/UnsteadySystem.h>

namespace Nektar
{
using namespace SolverCore;
using namespace Operators;

class EulerCFE : public UnsteadySystem
{
public:
    friend class MemoryManager<EulerCFE>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr equ =
            MemoryManager<EulerCFE>::AllocateSharedPtr(pSession, pGraph);
        equ->InitObject();
        return equ;
    }

    /// Name of class
    static std::string className;

protected:
    // Parameters for CFE
    double m_gamma;
    double m_gasConstant;

    // Initialise operators
    std::shared_ptr<AdvectionWeakDGOp<double>> m_advectionWeakDGOp;
    std::shared_ptr<CompressibleSolverOp<double>> m_riemannSolverOp;
    std::shared_ptr<EulerVolumeFluxOp<double>> m_volumeFluxOp;
    std::shared_ptr<ExpressionOp<double>> m_initialOp;
    std::shared_ptr<ExpressionOp<double>> m_velOp;

    EulerCFE(const LibUtilities::SessionReaderSharedPtr &pSession,
             const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~EulerCFE() override = default;

    void DoAdvection(LibUtilities::Field<double, FieldState::Phys> &in,
                     LibUtilities::Field<double, FieldState::Phys> &out,
                     [[maybe_unused]] const double &time,
                     [[maybe_unused]] const double &factor);

    void v_InitObject(bool declareExpansionLists = true) override;

    void v_GenerateSummary(SummaryList &s) override;

    void v_InitialiseOperators() override;

    void InitialiseParameters();
};

} // namespace Nektar
