///////////////////////////////////////////////////////////////////////////////
//
// File: AverageSolver.cpp
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
// Description: Average Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <CompressibleFlowSolver/RiemannSolvers/AverageSolver.h>

namespace Nektar
{

std::string AverageSolver::solverName =
    SolverUtils::GetRiemannSolverFactory().RegisterCreatorFunction(
        "Average", AverageSolver::create, "Average Riemann solver");

AverageSolver::AverageSolver(
    const LibUtilities::SessionReaderSharedPtr &pSession)
    : CompressibleSolver(pSession)
{
}

/**
 * @brief Average Riemann solver.
 *
 * Written directly in Cartesian components against the trace normal; see
 * CompressibleSolver::v_PointSolve for why no rotation is needed.
 *
 * @param Fwd   Forwards trace space.
 * @param Bwd   Backwards trace space.
 * @param flux  Resultant flux along trace space.
 */
void AverageSolver::v_Solve(
    [[maybe_unused]] const int nDim,
    const Array<OneD, const Array<OneD, NekDouble>> &Fwd,
    const Array<OneD, const Array<OneD, NekDouble>> &Bwd,
    Array<OneD, Array<OneD, NekDouble>> &flux)
{
    size_t expDim = Fwd.size() - 2;
    size_t i, j;

    ASSERTL1(CheckVectors("N"), "N not defined.");
    const Array<OneD, const Array<OneD, NekDouble>> normals = m_vectors["N"]();

    Array<OneD, NekDouble> Ufwd(expDim);
    Array<OneD, NekDouble> Ubwd(expDim);

    for (j = 0; j < Fwd[0].size(); ++j)
    {
        NekDouble tmp1 = 0.0, tmp2 = 0.0;
        NekDouble unFwd = 0.0, unBwd = 0.0;

        for (i = 0; i < expDim; ++i)
        {
            Ufwd[i] = Fwd[i + 1][j] / Fwd[0][j];
            Ubwd[i] = Bwd[i + 1][j] / Bwd[0][j];
            tmp1 += Ufwd[i] * Fwd[i + 1][j];
            tmp2 += Ubwd[i] * Bwd[i + 1][j];
            unFwd += Ufwd[i] * normals[i][j];
            unBwd += Ubwd[i] * normals[i][j];
        }

        // Internal energy
        NekDouble eFwd = (Fwd[expDim + 1][j] - 0.5 * tmp1) / Fwd[0][j];
        NekDouble eBwd = (Bwd[expDim + 1][j] - 0.5 * tmp2) / Bwd[0][j];
        // Pressure
        NekDouble Pfwd = m_eos->GetPressure(Fwd[0][j], eFwd);
        NekDouble Pbwd = m_eos->GetPressure(Bwd[0][j], eBwd);

        // Mass flux through the interface from each side
        NekDouble mnFwd = Fwd[0][j] * unFwd;
        NekDouble mnBwd = Bwd[0][j] * unBwd;

        // Compute the average flux
        flux[0][j]          = 0.5 * (mnFwd + mnBwd);
        flux[expDim + 1][j] = 0.5 * (unFwd * (Fwd[expDim + 1][j] + Pfwd) +
                                     unBwd * (Bwd[expDim + 1][j] + Pbwd));

        // Momentum flux, including the pressure contribution along the normal
        for (i = 0; i < expDim; ++i)
        {
            flux[i + 1][j] = 0.5 * (mnFwd * Ufwd[i] + mnBwd * Ubwd[i] +
                                    (Pfwd + Pbwd) * normals[i][j]);
        }
    }
}

} // namespace Nektar
