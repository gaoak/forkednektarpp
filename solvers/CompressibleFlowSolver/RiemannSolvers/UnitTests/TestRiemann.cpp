///////////////////////////////////////////////////////////////////////////////
//
// File: TestRiemann.cpp
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
/// The above copyright notice and this permission notice shall be included
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

// #include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include "../RoeSolver.h"

namespace Nektar::RiemannTests
{
BOOST_AUTO_TEST_CASE(RoeAlongXconstSolution)
{
    // LibUtilities::SessionReaderSharedPtr dummySession;
    // SolverUtils::RiemannSolverSharedPtr riemannSolver;
    // std::string riemannName = "Roe";
    // riemannSolver = SolverUtils::GetRiemannSolverFactory()
    // .CreateInstance(riemName, m_session);
    // auto riemannSolver = RoeSolver();
    auto riemannSolver = RoeSolver();
    // Setting up parameters for Riemann solver
    NekDouble gamma = 1.4;
    riemannSolver.SetParam("gamma",
                           [&gamma]() -> NekDouble & { return gamma; });

    size_t spaceDim = 3;
    size_t npts     = 5; // so that avx spillover loop is engaged

    // Set up locations of velocity vector.
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1);
    vecLocs[0] = Array<OneD, NekDouble>(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        vecLocs[0][i] = 1 + i;
    }
    riemannSolver.SetAuxVec(
        "vecLocs",
        [&vecLocs]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return vecLocs;
        });

    // setup normals
    Array<OneD, Array<OneD, NekDouble>> normals(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        normals[i] = Array<OneD, NekDouble>(npts, 0.0);
    }
    riemannSolver.SetVector(
        "N", [&normals]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return normals;
        });

    size_t nFields = spaceDim + 2;
    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields), flxRef(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i]    = Array<OneD, NekDouble>(npts);
        bwd[i]    = Array<OneD, NekDouble>(npts);
        flx[i]    = Array<OneD, NekDouble>(npts);
        flxRef[i] = Array<OneD, NekDouble>(npts);
    }

    // up to now it is "boiler plate" code to set up the test
    // below are the conditions for the test

    for (size_t i = 0; i < npts; ++i)
    {
        // density
        NekDouble rho = 0.9;
        fwd[0][i]     = rho;
        bwd[0][i]     = rho;
        // x-momentum
        NekDouble rhou = rho * 1.0;
        fwd[1][i]      = rhou;
        bwd[1][i]      = rhou;
        // y-momentum
        NekDouble rhov = rho * 2.0;
        fwd[2][i]      = rhov;
        bwd[2][i]      = rhov;
        // z-momentum
        NekDouble rhow = rho * 3.0;
        fwd[3][i]      = rhow;
        bwd[3][i]      = rhow;
        // energy
        NekDouble p    = 1.0;
        NekDouble rhoe = p / (gamma - 1.0);
        NekDouble E =
            rhoe + 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
        fwd[nFields - 1][i] = E;
        bwd[nFields - 1][i] = E;
        // set face normal along x
        normals[0][i] = 1.0;
        // Ref solution
        flxRef[0][i]           = rhou;
        flxRef[1][i]           = rhou * rhou / rho + p;
        flxRef[2][i]           = rhou * rhov / rho;
        flxRef[3][i]           = rhou * rhow / rho;
        flxRef[nFields - 1][i] = (E + p) * rhou / rho;
    }

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    // check fluxes
    for (size_t i = 0; i < npts; ++i)
    {
        BOOST_CHECK_CLOSE(flxRef[0][i], flx[0][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[1][i], flx[1][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[2][i], flx[2][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[3][i], flx[3][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[4][i], flx[4][i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(RoeAlongYconstSolution)
{
    // LibUtilities::SessionReaderSharedPtr dummySession;
    // SolverUtils::RiemannSolverSharedPtr riemannSolver;
    // std::string riemannName = "Roe";
    // riemannSolver = SolverUtils::GetRiemannSolverFactory()
    // .CreateInstance(riemName, m_session);
    // auto riemannSolver = RoeSolver();
    auto riemannSolver = RoeSolver();
    // Setting up parameters for Riemann solver
    NekDouble gamma = 1.4;
    riemannSolver.SetParam("gamma",
                           [&gamma]() -> NekDouble & { return gamma; });

    size_t spaceDim = 3;
    size_t npts     = 5;

    // Set up locations of velocity vector.
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1);
    vecLocs[0] = Array<OneD, NekDouble>(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        vecLocs[0][i] = 1 + i;
    }
    riemannSolver.SetAuxVec(
        "vecLocs",
        [&vecLocs]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return vecLocs;
        });

    // setup normals
    Array<OneD, Array<OneD, NekDouble>> normals(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        normals[i] = Array<OneD, NekDouble>(npts, 0.0);
    }
    riemannSolver.SetVector(
        "N", [&normals]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return normals;
        });

    size_t nFields = spaceDim + 2;
    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields), flxRef(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i]    = Array<OneD, NekDouble>(npts);
        bwd[i]    = Array<OneD, NekDouble>(npts);
        flx[i]    = Array<OneD, NekDouble>(npts);
        flxRef[i] = Array<OneD, NekDouble>(npts);
    }

    // up to now it is "boiler plate" code to set up the test
    // below are the conditions for the test

    // density
    NekDouble rho = 0.9;
    fwd[0][0]     = rho;
    bwd[0][0]     = rho;
    // x-momentum
    NekDouble rhou = rho * 1.0;
    fwd[1][0]      = rhou;
    bwd[1][0]      = rhou;
    // y-momentum
    NekDouble rhov = rho * 2.0;
    fwd[2][0]      = rhov;
    bwd[2][0]      = rhov;
    // z-momentum
    NekDouble rhow = rho * 3.0;
    fwd[3][0]      = rhow;
    bwd[3][0]      = rhow;
    // energy
    NekDouble p    = 1.0;
    NekDouble rhoe = p / (gamma - 1.0);
    NekDouble E = rhoe + 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
    fwd[nFields - 1][0] = E;
    bwd[nFields - 1][0] = E;
    // set face normal along y
    normals[1][0] = 1.0;
    // Ref solution
    flxRef[0][0]           = rhov;
    flxRef[1][0]           = rhov * rhou / rho;
    flxRef[2][0]           = rhov * rhov / rho + p;
    flxRef[3][0]           = rhov * rhow / rho;
    flxRef[nFields - 1][0] = (E + p) * rhov / rho;

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    // check fluxes
    BOOST_CHECK_CLOSE(flxRef[0][0], flx[0][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[1][0], flx[1][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[2][0], flx[2][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[3][0], flx[3][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[4][0], flx[4][0], 1e-10);
}

BOOST_AUTO_TEST_CASE(RoeAlongZconstSolution)
{
    // LibUtilities::SessionReaderSharedPtr dummySession;
    // SolverUtils::RiemannSolverSharedPtr riemannSolver;
    // std::string riemannName = "Roe";
    // riemannSolver = SolverUtils::GetRiemannSolverFactory()
    // .CreateInstance(riemName, m_session);
    // auto riemannSolver = RoeSolver();
    auto riemannSolver = RoeSolver();
    // Setting up parameters for Riemann solver
    NekDouble gamma = 1.4;
    riemannSolver.SetParam("gamma",
                           [&gamma]() -> NekDouble & { return gamma; });

    size_t spaceDim = 3;
    size_t npts     = 1;

    // Set up locations of velocity vector.
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1);
    vecLocs[0] = Array<OneD, NekDouble>(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        vecLocs[0][i] = 1 + i;
    }
    riemannSolver.SetAuxVec(
        "vecLocs",
        [&vecLocs]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return vecLocs;
        });

    // setup normals
    Array<OneD, Array<OneD, NekDouble>> normals(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        normals[i] = Array<OneD, NekDouble>(npts, 0.0);
    }
    riemannSolver.SetVector(
        "N", [&normals]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return normals;
        });

    size_t nFields = spaceDim + 2;
    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields), flxRef(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i]    = Array<OneD, NekDouble>(npts);
        bwd[i]    = Array<OneD, NekDouble>(npts);
        flx[i]    = Array<OneD, NekDouble>(npts);
        flxRef[i] = Array<OneD, NekDouble>(npts);
    }

    // up to now it is "boiler plate" code to set up the test
    // below are the conditions for the test

    // density
    NekDouble rho = 0.9;
    fwd[0][0]     = rho;
    bwd[0][0]     = rho;
    // x-momentum
    NekDouble rhou = rho * 1.0;
    fwd[1][0]      = rhou;
    bwd[1][0]      = rhou;
    // y-momentum
    NekDouble rhov = rho * 2.0;
    fwd[2][0]      = rhov;
    bwd[2][0]      = rhov;
    // z-momentum
    NekDouble rhow = rho * 3.0;
    fwd[3][0]      = rhow;
    bwd[3][0]      = rhow;
    // energy
    NekDouble p    = 1.0;
    NekDouble rhoe = p / (gamma - 1.0);
    NekDouble E = rhoe + 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
    fwd[nFields - 1][0] = E;
    bwd[nFields - 1][0] = E;
    // set face normal along y
    normals[2][0] = 1.0;
    // Ref solution
    flxRef[0][0]           = rhow;
    flxRef[1][0]           = rhow * rhou / rho;
    flxRef[2][0]           = rhow * rhov / rho;
    flxRef[3][0]           = rhow * rhow / rho + p;
    flxRef[nFields - 1][0] = (E + p) * rhow / rho;

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    // check fluxes
    BOOST_CHECK_CLOSE(flxRef[0][0], flx[0][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[1][0], flx[1][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[2][0], flx[2][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[3][0], flx[3][0], 1e-10);
    BOOST_CHECK_CLOSE(flxRef[4][0], flx[4][0], 1e-10);
}

BOOST_AUTO_TEST_CASE(RoeAlongXdensityJump)
{
    // LibUtilities::SessionReaderSharedPtr dummySession;
    // SolverUtils::RiemannSolverSharedPtr riemannSolver;
    // std::string riemannName = "Roe";
    // riemannSolver = SolverUtils::GetRiemannSolverFactory()
    // .CreateInstance(riemName, m_session);
    // auto riemannSolver = RoeSolver();
    auto riemannSolver = RoeSolver();
    // Setting up parameters for Riemann solver
    NekDouble gamma = 1.4;
    riemannSolver.SetParam("gamma",
                           [&gamma]() -> NekDouble & { return gamma; });

    size_t spaceDim = 3;
    size_t npts     = 11; // so that avx spillover loop is engaged

    // Set up locations of velocity vector.
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1);
    vecLocs[0] = Array<OneD, NekDouble>(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        vecLocs[0][i] = 1 + i;
    }
    riemannSolver.SetAuxVec(
        "vecLocs",
        [&vecLocs]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return vecLocs;
        });

    // setup normals
    Array<OneD, Array<OneD, NekDouble>> normals(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        normals[i] = Array<OneD, NekDouble>(npts, 0.0);
    }
    riemannSolver.SetVector(
        "N", [&normals]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return normals;
        });

    size_t nFields = spaceDim + 2;
    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields), flxRef(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i]    = Array<OneD, NekDouble>(npts);
        bwd[i]    = Array<OneD, NekDouble>(npts);
        flx[i]    = Array<OneD, NekDouble>(npts);
        flxRef[i] = Array<OneD, NekDouble>(npts);
    }

    // up to now it is "boiler plate" code to set up the test
    // below are the conditions for the test

    for (size_t i = 0; i < npts; ++i)
    {
        // density
        NekDouble rhoL = 1.0;
        NekDouble rhoR = 2 * rhoL;
        fwd[0][i]      = rhoL;
        bwd[0][i]      = rhoR;
        // x-momentum
        NekDouble rhou = rhoL * 1.0;
        fwd[1][i]      = rhou;
        bwd[1][i]      = rhou;
        // y-momentum
        NekDouble rhov = rhoL * 2.0;
        fwd[2][i]      = rhov;
        bwd[2][i]      = rhov;
        // z-momentum
        NekDouble rhow = rhoL * 3.0;
        fwd[3][i]      = rhow;
        bwd[3][i]      = rhow;
        // energy
        NekDouble p    = 1.0;
        NekDouble rhoe = p / (gamma - 1.0);
        NekDouble EL =
            rhoe + 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoL;
        NekDouble ER =
            rhoe + 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoR;
        fwd[nFields - 1][i] = EL;
        bwd[nFields - 1][i] = ER;
        // set face normal along x
        normals[0][i] = 1.0;
        // Ref solution (from point solve)
        flxRef[0][i]           = 0.87858599768171342;
        flxRef[1][i]           = 2.0449028304431223;
        flxRef[2][i]           = 1.8282946712594808;
        flxRef[3][i]           = 2.7424420068892208;
        flxRef[nFields - 1][i] = 9.8154698039903128;
    }

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    // check fluxes
    for (size_t i = 0; i < npts; ++i)
    {
        BOOST_CHECK_CLOSE(flxRef[0][i], flx[0][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[1][i], flx[1][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[2][i], flx[2][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[3][i], flx[3][i], 1e-10);
        BOOST_CHECK_CLOSE(flxRef[4][i], flx[4][i], 1e-10);
    }
}

namespace
{

/// Wire up @p riemannSolver for a 3D problem over @p npts trace points, using
/// the caller's storage for gamma, vecLocs and the trace normals.
void SetupRoeSolver(SolverUtils::RiemannSolver &riemannSolver, NekDouble &gamma,
                    Array<OneD, Array<OneD, NekDouble>> &vecLocs,
                    Array<OneD, Array<OneD, NekDouble>> &normals,
                    size_t spaceDim, size_t npts)
{
    riemannSolver.SetParam("gamma",
                           [&gamma]() -> NekDouble & { return gamma; });

    vecLocs[0] = Array<OneD, NekDouble>(spaceDim);
    for (size_t i = 0; i < spaceDim; ++i)
    {
        vecLocs[0][i] = 1 + i;
    }
    riemannSolver.SetAuxVec(
        "vecLocs",
        [&vecLocs]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return vecLocs;
        });

    for (size_t i = 0; i < spaceDim; ++i)
    {
        normals[i] = Array<OneD, NekDouble>(npts, 0.0);
    }
    riemannSolver.SetVector(
        "N", [&normals]() -> const Array<OneD, const Array<OneD, NekDouble>> & {
            return normals;
        });
}

/// Rotation by @p angle about the unit axis @p axis, as a row-major 3x3.
void RodriguesRotation(const NekDouble *axis, NekDouble angle, NekDouble *R)
{
    NekDouble c = cos(angle), s = sin(angle), t = 1.0 - c;
    R[0] = t * axis[0] * axis[0] + c;
    R[1] = t * axis[0] * axis[1] - s * axis[2];
    R[2] = t * axis[0] * axis[2] + s * axis[1];
    R[3] = t * axis[0] * axis[1] + s * axis[2];
    R[4] = t * axis[1] * axis[1] + c;
    R[5] = t * axis[1] * axis[2] - s * axis[0];
    R[6] = t * axis[0] * axis[2] - s * axis[1];
    R[7] = t * axis[1] * axis[2] + s * axis[0];
    R[8] = t * axis[2] * axis[2] + c;
}

/**
 * @brief Check that a Riemann solver is equivariant under a rigid rotation
 * applied to both the left and right states and to the trace normal, i.e. that
 *
 *     F(R U_L, R U_R, R n) = R F(U_L, U_R, n).
 *
 * This is the property that makes a rotation onto the trace normal
 * unnecessary. The old implementation built such a rotation with an arbitrary
 * choice of tangential axes, so any dependence on that choice would show up
 * here as a failure.
 *
 * Five distinct (state, normal) configurations are placed in the first half of
 * the trace and their rotated counterparts in the second half, so that both
 * the vectorised and the scalar spillover paths are exercised.
 */
void CheckRotationalInvariance(SolverUtils::RiemannSolver &riemannSolver)
{
    const size_t spaceDim = 3;
    const size_t nCase    = 5;
    const size_t npts     = 2 * nCase;
    const size_t nFields  = spaceDim + 2;

    NekDouble gamma = 1.4;
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1), normals(spaceDim);
    SetupRoeSolver(riemannSolver, gamma, vecLocs, normals, spaceDim, npts);

    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i] = Array<OneD, NekDouble>(npts, 0.0);
        bwd[i] = Array<OneD, NekDouble>(npts, 0.0);
        flx[i] = Array<OneD, NekDouble>(npts, 0.0);
    }

    // An arbitrary rotation, about an axis deliberately unrelated to any of
    // the normals below.
    NekDouble axis[3] = {1.0, 1.0, 1.0};
    NekDouble anorm   = sqrt(3.0);
    for (size_t d = 0; d < 3; ++d)
    {
        axis[d] /= anorm;
    }
    NekDouble R[9];
    RodriguesRotation(axis, 0.7, R);

    // Five left/right states with genuine jumps, and five oblique normals.
    NekDouble rhoL[nCase]    = {0.9, 1.3, 0.5, 2.0, 1.0};
    NekDouble rhoR[nCase]    = {1.4, 0.7, 1.1, 0.8, 1.0};
    NekDouble pL[nCase]      = {1.0, 2.5, 0.4, 1.7, 1.0};
    NekDouble pR[nCase]      = {0.6, 0.9, 1.3, 2.2, 1.0};
    NekDouble velL[nCase][3] = {{1.0, 2.0, 3.0},
                                {-0.5, 1.2, 0.3},
                                {2.1, -1.4, 0.7},
                                {0.0, 0.0, 1.5},
                                {1.0, 1.0, 1.0}};
    NekDouble velR[nCase][3] = {{0.4, -1.1, 2.2},
                                {1.8, 0.6, -2.0},
                                {-0.9, 0.5, 1.3},
                                {1.1, -0.3, 0.0},
                                {1.0, 1.0, 1.0}};
    NekDouble rawN[nCase][3] = {{1.0, 2.0, 3.0},
                                {-2.0, 1.0, 4.0},
                                {3.0, -3.0, 1.0},
                                {1.0, 1.0, 1.0},
                                {0.0, 2.0, -5.0}};

    for (size_t k = 0; k < nCase; ++k)
    {
        NekDouble n[3];
        NekDouble nn = sqrt(rawN[k][0] * rawN[k][0] + rawN[k][1] * rawN[k][1] +
                            rawN[k][2] * rawN[k][2]);
        for (size_t d = 0; d < 3; ++d)
        {
            n[d] = rawN[k][d] / nn;
        }

        NekDouble EL = pL[k] / (gamma - 1.0) +
                       0.5 * rhoL[k] *
                           (velL[k][0] * velL[k][0] + velL[k][1] * velL[k][1] +
                            velL[k][2] * velL[k][2]);
        NekDouble ER = pR[k] / (gamma - 1.0) +
                       0.5 * rhoR[k] *
                           (velR[k][0] * velR[k][0] + velR[k][1] * velR[k][1] +
                            velR[k][2] * velR[k][2]);

        // unrotated configuration
        fwd[0][k]           = rhoL[k];
        bwd[0][k]           = rhoR[k];
        fwd[nFields - 1][k] = EL;
        bwd[nFields - 1][k] = ER;
        for (size_t d = 0; d < 3; ++d)
        {
            fwd[d + 1][k] = rhoL[k] * velL[k][d];
            bwd[d + 1][k] = rhoR[k] * velR[k][d];
            normals[d][k] = n[d];
        }

        // the same configuration, rigidly rotated
        size_t m            = k + nCase;
        fwd[0][m]           = rhoL[k];
        bwd[0][m]           = rhoR[k];
        fwd[nFields - 1][m] = EL;
        bwd[nFields - 1][m] = ER;
        for (size_t d = 0; d < 3; ++d)
        {
            NekDouble rvL = 0.0, rvR = 0.0, rn = 0.0;
            for (size_t e = 0; e < 3; ++e)
            {
                rvL += R[3 * d + e] * velL[k][e];
                rvR += R[3 * d + e] * velR[k][e];
                rn += R[3 * d + e] * n[e];
            }
            fwd[d + 1][m] = rhoL[k] * rvL;
            bwd[d + 1][m] = rhoR[k] * rvR;
            normals[d][m] = rn;
        }
    }

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    for (size_t k = 0; k < nCase; ++k)
    {
        size_t m = k + nCase;

        // scalar fluxes are invariant
        BOOST_CHECK_SMALL(flx[0][m] - flx[0][k], 1e-12);
        BOOST_CHECK_SMALL(flx[nFields - 1][m] - flx[nFields - 1][k], 1e-12);

        // the momentum flux must have rotated with the configuration
        for (size_t d = 0; d < 3; ++d)
        {
            NekDouble expected = 0.0;
            for (size_t e = 0; e < 3; ++e)
            {
                expected += R[3 * d + e] * flx[e + 1][k];
            }
            BOOST_CHECK_SMALL(flx[d + 1][m] - expected, 1e-12);
        }
    }
}

/**
 * @brief Check that a uniform state produces exactly the physical flux F.n for
 * oblique trace normals.
 */
void CheckObliqueConstSolution(SolverUtils::RiemannSolver &riemannSolver)
{
    const size_t spaceDim = 3;
    const size_t npts     = 5;
    const size_t nFields  = spaceDim + 2;

    NekDouble gamma = 1.4;
    Array<OneD, Array<OneD, NekDouble>> vecLocs(1), normals(spaceDim);
    SetupRoeSolver(riemannSolver, gamma, vecLocs, normals, spaceDim, npts);

    Array<OneD, Array<OneD, NekDouble>> fwd(nFields), bwd(nFields),
        flx(nFields), flxRef(nFields);
    for (size_t i = 0; i < nFields; ++i)
    {
        fwd[i]    = Array<OneD, NekDouble>(npts, 0.0);
        bwd[i]    = Array<OneD, NekDouble>(npts, 0.0);
        flx[i]    = Array<OneD, NekDouble>(npts, 0.0);
        flxRef[i] = Array<OneD, NekDouble>(npts, 0.0);
    }

    NekDouble rawN[npts][3] = {{1.0, 2.0, 3.0},
                               {-2.0, 1.0, 4.0},
                               {3.0, -3.0, 1.0},
                               {1.0, 1.0, 1.0},
                               {0.0, 2.0, -5.0}};

    NekDouble rho = 0.9, p = 1.0;
    NekDouble vel[3] = {1.0, 2.0, 3.0};
    NekDouble E =
        p / (gamma - 1.0) +
        0.5 * rho * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);

    for (size_t i = 0; i < npts; ++i)
    {
        NekDouble n[3];
        NekDouble nn = sqrt(rawN[i][0] * rawN[i][0] + rawN[i][1] * rawN[i][1] +
                            rawN[i][2] * rawN[i][2]);
        NekDouble un = 0.0;
        for (size_t d = 0; d < 3; ++d)
        {
            n[d] = rawN[i][d] / nn;
            un += vel[d] * n[d];
        }

        fwd[0][i]           = rho;
        bwd[0][i]           = rho;
        fwd[nFields - 1][i] = E;
        bwd[nFields - 1][i] = E;
        for (size_t d = 0; d < 3; ++d)
        {
            fwd[d + 1][i]    = rho * vel[d];
            bwd[d + 1][i]    = rho * vel[d];
            normals[d][i]    = n[d];
            flxRef[d + 1][i] = rho * un * vel[d] + p * n[d];
        }
        flxRef[0][i]           = rho * un;
        flxRef[nFields - 1][i] = (E + p) * un;
    }

    riemannSolver.Solve(spaceDim, fwd, bwd, flx);

    for (size_t i = 0; i < npts; ++i)
    {
        for (size_t j = 0; j < nFields; ++j)
        {
            BOOST_CHECK_SMALL(flxRef[j][i] - flx[j][i], 1e-12);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(RoeObliqueNormalConstSolution)
{
    auto riemannSolver = RoeSolver();
    CheckObliqueConstSolution(riemannSolver);
}

BOOST_AUTO_TEST_CASE(RoeRotationalInvariance)
{
    auto riemannSolver = RoeSolver();
    CheckRotationalInvariance(riemannSolver);
}

} // namespace Nektar::RiemannTests
