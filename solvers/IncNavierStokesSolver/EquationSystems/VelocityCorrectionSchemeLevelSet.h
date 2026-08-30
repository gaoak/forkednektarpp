///////////////////////////////////////////////////////////////////////////////
//
// File: VelocityCorrectionSchemeLevelSet.h
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
// Description: Level Set Velocity Correction Scheme header
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_VELOCITYCORRECTIONSCHEMELEVELSET_H
#define NEKTAR_SOLVERS_VELOCITYCORRECTIONSCHEMELEVELSET_H

#include <IncNavierStokesSolver/EquationSystems/VelocityCorrectionScheme.h>

namespace Nektar
{
class VCSLevelSet : public VelocityCorrectionScheme
{
public:
    friend class MemoryManager<VCSLevelSet>;

    /// Creates an instance of this class
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<VCSLevelSet>::AllocateSharedPtr(pSession, pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

protected:
    // Variables for Level Set
    NekDouble m_epsilon;
    NekDouble m_rhol;
    NekDouble m_rhof;
    NekDouble m_viscl;
    NekDouble m_viscf;
    NekDouble m_sigma;
    NekDouble m_gravity;
    Array<OneD, Array<OneD, NekDouble>> m_IntNorm;
    Array<OneD, Array<OneD, NekDouble>> m_SurfTen;

    static std::string solverTypeLookupId;

    VCSLevelSet(const LibUtilities::SessionReaderSharedPtr &pSession,
                const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~VCSLevelSet() override = default;

    void v_InitObject(bool DeclareField = true) override;

    // Virtual functions

    void v_GenerateSummary(SolverUtils::SummaryList &s) override;

    void v_SolveUnsteadyStokesSystem(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time,
        const NekDouble a_iixDt) override;

    void SetUpLevelSetForcing(const Array<OneD, NekDouble> &fields,
                              Array<OneD, NekDouble> &Forcing,
                              const NekDouble aii_Dt);

    void v_SetUpPressureForcing(
        const Array<OneD, const Array<OneD, NekDouble>> &fields,
        Array<OneD, Array<OneD, NekDouble>> &Forcing,
        const NekDouble aii_Dt) override;

    void v_SetUpViscousForcing(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &Forcing,
        const NekDouble aii_Dt) override;

    void SolveLevelSet(const Array<OneD, NekDouble> &Forcing,
                       const Array<OneD, NekDouble> &inarray,
                       Array<OneD, NekDouble> &outarrayPhi,
                       Array<OneD, NekDouble> &outarrayRho,
                       Array<OneD, NekDouble> &outarrayVisc,
                       const NekDouble aii_Dt);

    void v_SolvePressure(const Array<OneD, NekDouble> &Forcing) override;

    void v_SolveViscous(
        const Array<OneD, const Array<OneD, NekDouble>> &Forcing,
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray,
        const NekDouble aii_Dt) override;

private:
};

typedef std::shared_ptr<VCSLevelSet> VCSLevelSetSharedPtr;

} // namespace Nektar

#endif // VELOCITY_CORRECTION_SCHEME_LEVELSET_H
