///////////////////////////////////////////////////////////////////////////////
//
// File: VCSQuasiStaticMHD.h
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
// Description: Velocity Correction Scheme for fluid-structure interaction
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_VCSQUASISTATICMHD_H
#define NEKTAR_SOLVERS_VCSQUASISTATICMHDH

#include <IncNavierStokesSolver/EquationSystems/VelocityCorrectionScheme.h>
namespace Nektar
{
class VCSQuasiStaticMHD : public VelocityCorrectionScheme
{
public:
    /// Creates an instance of this class
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<VCSQuasiStaticMHD>::AllocateSharedPtr(pSession,
                                                                pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

    /// Constructor.
    VCSQuasiStaticMHD(const LibUtilities::SessionReaderSharedPtr &pSession,
                      const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~VCSQuasiStaticMHD() override;

    void v_InitObject(bool DeclareField = true) override;

    void SolveEfield(const Array<OneD, Array<OneD, NekDouble>> &movEfield,
                     Array<OneD, Array<OneD, NekDouble>> &totEfield);

protected:
    void SetInsulatorBCs(const Array<OneD, Array<OneD, NekDouble>> &movEfield);

    static std::string solverTypeLookupId;

    /// Pointer to field holding electric potential field
    MultiRegions::ExpListSharedPtr m_potential;
};

typedef std::shared_ptr<VCSQuasiStaticMHD> VCSQuasiStaticMHDSharedPtr;

} // namespace Nektar

#endif // VCSQUASISTATICMHD_H
