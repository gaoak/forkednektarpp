///////////////////////////////////////////////////////////////////////////////
//
// File: AdjointBoundaryCheck.h
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
// Description: Shared boundary-condition / linear-solver checks for the
//              temperature adjoint solvers.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ADRSOLVER_ADJOINT_ADJOINTBOUNDARYCHECK_H
#define NEKTAR_SOLVERS_ADRSOLVER_ADJOINT_ADJOINTBOUNDARYCHECK_H

#include <string>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SpatialDomains/Conditions.h>

namespace Nektar
{

/// Returns true if the session imposes any boundary condition that is neither
/// Dirichlet nor periodic.
inline bool AdjointHasNonDirichletBoundary(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
{
    SpatialDomains::BoundaryConditions bcs(session, graph);
    const SpatialDomains::BoundaryConditionCollection &bconditions =
        bcs.GetBoundaryConditions();

    for (const auto &region : bconditions)
    {
        if (!region.second)
        {
            continue;
        }
        for (const auto &cond : *region.second)
        {
            const SpatialDomains::BoundaryConditionType type =
                cond.second->GetBoundaryConditionType();
            if (type != SpatialDomains::eDirichlet &&
                type != SpatialDomains::ePeriodic)
            {
                return true;
            }
        }
    }
    return false;
}

/**
 * Xxt performs a sparse Cholesky factorisation and therefore requires a
 * symmetric global system.  The boundary (static-condensation) system of the
 * advection-diffusion(-reaction) operator is non-symmetric whenever a base
 * flow is present; Neumann and Robin conditions put boundary degrees of
 * freedom into that system, and Xxt then silently returns a wrong solution.
 * (Reproduced with the stock SteadyAdvectionDiffusion solver as well, so this
 * is a library limitation rather than an adjoint issue.)
 *
 * Such sessions are switched to DirectStaticCond.  The GMRES-based
 * IterativeStaticCond was tested too and is wrong for mixed Dirichlet +
 * Neumann boundaries (it returns the exact solution minus a uniform constant
 * in the interior), so it is not used as the fallback either.
 *
 * Sessions that explicitly request DirectStaticCond or DirectFull are left
 * untouched.
 */
inline void AdjointGuardXxtGlobalSysSoln(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
{
    std::string globalSysSoln = "DirectStaticCond";
    if (session->DefinesSolverInfo("GlobalSysSoln"))
    {
        globalSysSoln = session->GetSolverInfo("GlobalSysSoln");
    }
    if (globalSysSoln.compare(0, 3, "Xxt") == 0 &&
        AdjointHasNonDirichletBoundary(session, graph))
    {
        WARNINGL0(false,
                  "GlobalSysSoln = '" + globalSysSoln +
                      "' uses a sparse Cholesky factorisation, which is "
                      "only valid for symmetric systems.  With Neumann/Robin "
                      "boundary conditions the advection-diffusion boundary "
                      "system is non-symmetric when a base flow is present, "
                      "and Xxt would return a wrong solution.  Switching to "
                      "DirectStaticCond.  Set GlobalSysSoln = "
                      "'DirectStaticCond' explicitly to suppress this "
                      "warning.");
        session->SetSolverInfo("GlobalSysSoln", "DirectStaticCond");
    }
}

} // namespace Nektar

#endif
