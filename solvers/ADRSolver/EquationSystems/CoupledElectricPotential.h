///////////////////////////////////////////////////////////////////////////////
//
// File: CoupledElectricPotential.h
//
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_ADRSOLVER_COUPLEDELECTRICPOTENTIAL_H
#define NEKTAR_ADRSOLVER_COUPLEDELECTRICPOTENTIAL_H

#include <SolverUtils/Core/Coupling.h>
#include <SolverUtils/EquationSystem.h>

namespace Nektar
{

/// Quasi-static electric potential on a conforming fluid/solid mesh.
/// Velocity is received into auxiliary arrays, not additional solution fields.
class CoupledElectricPotential : public SolverUtils::EquationSystem
{
public:
    friend class MemoryManager<CoupledElectricPotential>;
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph)
    {
        auto p = MemoryManager<CoupledElectricPotential>::AllocateSharedPtr(
            session, graph);
        p->InitObject();
        return p;
    }
    static std::string className;

protected:
    CoupledElectricPotential(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph);
    void v_InitObject(bool declareFields = true) override;
    void v_DoInitialise(bool dumpInitialConditions = true) override;
    void v_DoSolve() override;
    void v_GenerateSummary(SolverUtils::SummaryList &summary) override;
    void v_ExtraFldOutput(std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
                          std::vector<std::string> &variables) override;
    Array<OneD, bool> v_GetSystemSingularChecks() override;

private:
    void SolvePotential();
    SolverUtils::CouplingSharedPtr m_coupling;
    Array<OneD, Array<OneD, NekDouble>> m_received;
    std::vector<std::string> m_recvNames, m_sendNames;
    Array<OneD, NekDouble> m_sigma;
    Array<OneD, int> m_fluid;
    Array<OneD, NekDouble> m_electric, m_magnetic;
    StdRegions::VarCoeffMap m_varcoeff;
    int m_dimension;
};

} // namespace Nektar
#endif
