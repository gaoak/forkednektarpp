///////////////////////////////////////////////////////////////////////////////
//
// File: RoeSolver.cpp
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
// Description: Roe Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <CompressibleFlowSolver/RiemannSolvers/RoeSolver.h>

#include <LibUtilities/SimdLib/tinysimd.hpp>

namespace Nektar
{

std::string RoeSolver::solverName =
    SolverUtils::GetRiemannSolverFactory().RegisterCreatorFunction(
        "Roe", RoeSolver::create, "Roe Riemann solver");

/// Retained so that sessions selecting the former RoeSolverSIMD keep working.
/// That solver existed only to apply the rotation onto the trace normal
/// pointwise in registers rather than through array operations; with the
/// rotation gone it was the same calculation as this one.
std::string RoeSolver::solverNameOpt =
    SolverUtils::GetRiemannSolverFactory().RegisterCreatorFunction(
        "RoeOpt", RoeSolver::create, "Roe Riemann solver (alias of Roe)");

RoeSolver::RoeSolver(const LibUtilities::SessionReaderSharedPtr &pSession)
    : CompressibleSolver(pSession)
{
}

/// programmatic ctor
RoeSolver::RoeSolver() : CompressibleSolver()
{
}

namespace
{

/**
 * @brief Vectorised Roe flux across a whole trace.
 *
 * The spatial dimension is a template parameter rather than a run-time value
 * so that the per-point loops over components unroll completely and the
 * three-component arrays stay in registers. With a run-time bound the compiler
 * cannot establish the trip count, the loops are left rolled and the arrays
 * spill to the stack, which costs around a sixth of the run time of this
 * routine.
 *
 * Components above @p DIM are left zero throughout, so one kernel serves 1D,
 * 2D and 3D.
 *
 * @param gamma     Ratio of specific heats.
 * @param normals   Unit trace normals.
 * @param fwd       Forwards trace space.
 * @param bwd       Backwards trace space.
 * @param flux      Resultant flux along trace space.
 */
template <size_t DIM>
void RoeArraySolveImpl(NekDouble gamma,
                       const Array<OneD, const Array<OneD, NekDouble>> &normals,
                       const Array<OneD, const Array<OneD, NekDouble>> &fwd,
                       const Array<OneD, const Array<OneD, NekDouble>> &bwd,
                       Array<OneD, Array<OneD, NekDouble>> &flux)
{
    constexpr size_t nVars = DIM + 2;

    using namespace tinysimd;
    using vec_t = simd<NekDouble>;

    // get limit of vectorizable chunk
    size_t sizeScalar = fwd[0].size();
    size_t sizeVec    = (sizeScalar / vec_t::width) * vec_t::width;

    // SIMD loop
    size_t i = 0;
    for (; i < sizeVec; i += vec_t::width)
    {
        vec_t rhoL{}, EL{}, rhoR{}, ER{};

        // load
        rhoL.load(&(fwd[0][i]), is_not_aligned);
        EL.load(&(fwd[DIM + 1][i]), is_not_aligned);
        rhoR.load(&(bwd[0][i]), is_not_aligned);
        ER.load(&(bwd[DIM + 1][i]), is_not_aligned);

        // components above DIM stay zero
        vec_t momL[3]{}, momR[3]{}, n[3]{};
        for (size_t d = 0; d < DIM; ++d)
        {
            momL[d].load(&(fwd[d + 1][i]), is_not_aligned);
            momR[d].load(&(bwd[d + 1][i]), is_not_aligned);
            n[d].load(&(normals[d][i]), is_not_aligned);
        }

        vec_t rhof{}, momf[3]{}, Ef{};

        RoeKernel(rhoL, momL, EL, rhoR, momR, ER, n, rhof, momf, Ef, gamma);

        // store
        rhof.store(&(flux[0][i]), is_not_aligned);
        Ef.store(&(flux[nVars - 1][i]), is_not_aligned);
        for (size_t d = 0; d < DIM; ++d)
        {
            momf[d].store(&(flux[d + 1][i]), is_not_aligned);
        }

    } // avx loop

    // spillover loop
    for (; i < sizeScalar; ++i)
    {
        // load
        NekDouble rhoL = fwd[0][i];
        NekDouble EL   = fwd[DIM + 1][i];
        NekDouble rhoR = bwd[0][i];
        NekDouble ER   = bwd[DIM + 1][i];

        NekDouble momL[3] = {0.0, 0.0, 0.0};
        NekDouble momR[3] = {0.0, 0.0, 0.0};
        NekDouble n[3]    = {0.0, 0.0, 0.0};
        for (size_t d = 0; d < DIM; ++d)
        {
            momL[d] = fwd[d + 1][i];
            momR[d] = bwd[d + 1][i];
            n[d]    = normals[d][i];
        }

        NekDouble rhof{}, momf[3]{}, Ef{};

        RoeKernel(rhoL, momL, EL, rhoR, momR, ER, n, rhof, momf, Ef, gamma);

        // store
        flux[0][i]         = rhof;
        flux[nVars - 1][i] = Ef;
        for (size_t d = 0; d < DIM; ++d)
        {
            flux[d + 1][i] = momf[d];
        }

    } // loop
}

} // namespace

/**
 * @brief Dispatch to the vectorised kernel for the dimension of this trace.
 *
 * The session-dependent quantities are fetched once here rather than being
 * cached in function-local statics, which were shared across every instance of
 * the solver and only ever initialised by whichever one ran first.
 */
void RoeSolver::v_Solve([[maybe_unused]] const int nDim,
                        const Array<OneD, const Array<OneD, NekDouble>> &fwd,
                        const Array<OneD, const Array<OneD, NekDouble>> &bwd,
                        Array<OneD, Array<OneD, NekDouble>> &flux)
{
    NekDouble gamma = m_params["gamma"]();

    ASSERTL1(CheckVectors("N"), "N not defined.");
    const Array<OneD, const Array<OneD, NekDouble>> normals = m_vectors["N"]();

    switch (fwd.size() - 2)
    {
        case 1:
            RoeArraySolveImpl<1>(gamma, normals, fwd, bwd, flux);
            break;
        case 2:
            RoeArraySolveImpl<2>(gamma, normals, fwd, bwd, flux);
            break;
        case 3:
            RoeArraySolveImpl<3>(gamma, normals, fwd, bwd, flux);
            break;
        default:
            NEKERROR(ErrorUtil::efatal, "Invalid space dimension.");
            break;
    }
}

} // namespace Nektar
