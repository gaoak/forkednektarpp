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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestRiemann

#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <MultiRegions/DisContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

#include "../CompressibleSolverOp.hpp"

// Currently the BOOST_TEST_DYN_LINK is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_DYN_LINK)
#if !defined(BOOST_TEST_DYN_LINK)
#define LOCALLY_DEFINED_BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#endif
#endif

// Currently the BOOST_TEST_NO_MAIN is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_NO_MAIN)
#if !defined(BOOST_TEST_NO_MAIN)
#define LOCALLY_DEFINED_BOOST_TEST_NO_MAIN
#define BOOST_TEST_NO_MAIN
#endif
#endif

#if defined(BOOST_TEST_DYN_LINK) || defined(BOOST_TEST_NO_MAIN)
#define BOOST_TEST_ALTERNATIVE_INIT_API
#endif

#if defined(BOOST_TEST_DYN_LINK)
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/test/unit_test_log.hpp>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

struct GlobalConfiguration
{
    GlobalConfiguration()
    {
        [[maybe_unused]] int argc =
            boost::unit_test::framework::master_test_suite().argc;
        [[maybe_unused]] char **argv =
            boost::unit_test::framework::master_test_suite().argv;

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

#if defined(LOCALLY_DEFINED_BOOST_TEST_DYN_LINK)
#undef BOOST_TEST_DYN_LINK
#endif

#if defined(LOCALLY_DEFINED_BOOST_TEST_NO_MAIN)
#undef BOOST_TEST_NO_MAIN
#endif

// Helpers: flatten/unflatten component-major
template <typename TData>
static Array<OneD, TData> FlattenCompMajor(
    const Array<OneD, Array<OneD, TData>> &a, unsigned int ncomp, size_t npts)
{
    Array<OneD, TData> flat(ncomp * npts, 0.0);
    for (unsigned int c = 0; c < ncomp; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            flat[c * npts + i] = a[c][i];
        }
    }
    return flat;
}

template <typename TData>
static void UnflattenCompMajor(const Array<OneD, TData> &flat,
                               Array<OneD, Array<OneD, TData>> &a,
                               unsigned int ncomp, size_t npts)
{
    for (unsigned int c = 0; c < ncomp; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            a[c][i] = flat[c * npts + i];
        }
    }
}

// A primitive Euler state (for building left/right Riemann states).
//
// Fields:
//   rho : density
//   p   : pressure
//   u   : velocity vector (only first spaceDim entries used; others ignored)
struct PrimState
{
    double rho;
    double p;
    double u[3];
};

// Convert primitive state W = (rho, u, p) to conserved Euler state U = (rho,
// rhou..., E).
static inline void PrimToCons(const PrimState &W, unsigned int spaceDim,
                              Array<OneD, double> &U, double gamma)
{
    // U layout: [rho, rhou..., E] length = spaceDim + 2
    const unsigned int nFields = spaceDim + 2;
    ASSERTL0(U.size() == nFields, "PrimToCons: wrong U size");

    U[0] = W.rho;

    double kin = 0.0;
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        U[1 + d] = W.rho * W.u[d];
        kin += 0.5 * W.rho * W.u[d] * W.u[d];
    }

    const double rhoe = W.p / (gamma - 1.0);
    U[nFields - 1]    = rhoe + kin;
}

static inline double ConsToPressure(const Array<OneD, const double> &U,
                                    unsigned int spaceDim, double gamma)
{
    const unsigned int nFields = spaceDim + 2;
    const double rho           = U[0];

    double mom2_over_rho = 0.0;
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        const double m = U[1 + d];
        mom2_over_rho += (m * m) / rho;
    }

    const double E = U[nFields - 1];
    const double p = (gamma - 1.0) * (E - 0.5 * mom2_over_rho);
    return p;
}

// Compute the physical flux dotted with a unit (or non-unit) normal n.
//
// Inputs:
//   U        : conserved state [rho, rhou..., E]
//   n        : normal vector length = spaceDim
//   spaceDim : spatial dimension
//   gamma    : ratio of specific heats
//
// Output:
//   fluxDotN : length = spaceDim + 2, holding F(U)·n:
static inline void PhysicalFluxDotN(const Array<OneD, const double> &U,
                                    const Array<OneD, const double> &n,
                                    unsigned int spaceDim, double gamma,
                                    Array<OneD, double> &fluxDotN)
{
    // Euler physical flux in normal direction
    // layout: [rho, rhou..., E] length = spaceDim + 2
    const unsigned int nFields = spaceDim + 2;
    ASSERTL0(U.size() == nFields, "PhysicalFluxDotN: wrong U size");
    ASSERTL0(n.size() == spaceDim, "PhysicalFluxDotN: wrong n size");
    ASSERTL0(fluxDotN.size() == nFields,
             "PhysicalFluxDotN: wrong fluxDotN size");

    const double rho = U[0];
    double u[3]      = {0.0, 0.0, 0.0};

    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        u[d] = U[1 + d] / rho;
    }

    double un = 0.0;
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        un += u[d] * n[d];
    }

    const double p = ConsToPressure(U, spaceDim, gamma);
    const double E = U[nFields - 1];

    // mass
    fluxDotN[0] = rho * un;

    // momentum
    for (unsigned int k = 0; k < spaceDim; ++k)
    {
        const double momk = U[1 + k];
        fluxDotN[1 + k]   = momk * un + p * n[k];
    }

    // energy
    fluxDotN[nFields - 1] = (E + p) * un;
}

static void FillTraceFromPrimStates(unsigned int spaceDim, size_t npts,
                                    const PrimState &WL, const PrimState &WR,
                                    Array<OneD, Array<OneD, double>> &fwd,
                                    Array<OneD, Array<OneD, double>> &bwd,
                                    double gamma)
{
    const unsigned int nFields = spaceDim + 2;

    Array<OneD, double> UL(nFields, 0.0), UR(nFields, 0.0);
    PrimToCons(WL, spaceDim, UL, gamma);
    PrimToCons(WR, spaceDim, UR, gamma);

    for (size_t i = 0; i < npts; ++i)
    {
        for (unsigned int c = 0; c < nFields; ++c)
        {
            fwd[c][i] = UL[c];
            bwd[c][i] = UR[c];
        }
    }
}

static void FillTraceNormals(unsigned int spaceDim, size_t npts,
                             const Array<OneD, const double> &n,
                             Array<OneD, Array<OneD, double>> &normals)
{
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        Vmath::Fill((int)npts, n[d], normals[d], 1);
    }
}

static inline double SoundSpeedFromRhoP(double rho, double p, double gamma)
{
    return std::sqrt(gamma * p / rho);
}

// Build primitive state using Mach definition.
static inline PrimState MakeStateFromMach(double rho, double p, double M,
                                          double cRef)
{
    PrimState W;
    W.rho = rho;
    W.p   = p;
    // x-normal velocity only
    W.u[0] = M * cRef;
    W.u[1] = 0.0;
    W.u[2] = 0.0;
    return W;
}

static inline Array<OneD, double> PhysicalFlux(const PrimState &W,
                                               unsigned int spaceDim,
                                               double gamma)
{
    Array<OneD, double> U(spaceDim + 2, 0.0), F(spaceDim + 2, 0.0);
    PrimToCons(W, spaceDim, U, gamma);

    Array<OneD, double> normals(spaceDim, 0.0);
    // x-normal
    normals[0] = 1.0;
    PhysicalFluxDotN(U, normals, spaceDim, gamma, F);
    return F;
}

static void CheckCloseRel(const Array<OneD, const double> &a,
                          const Array<OneD, const double> &b, double tol)
{
    BOOST_REQUIRE_EQUAL(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double denom = std::max({1.0, std::abs(a[i]), std::abs(b[i])});
        BOOST_CHECK_SMALL((a[i] - b[i]) / denom, tol);
    }
}

static void CheckFinite(const Array<OneD, const double> &f)
{
    for (size_t i = 0; i < f.size(); ++i)
    {
        BOOST_CHECK(std::isfinite(f[i]));
    }
}

// Session + DG setup
static LibUtilities::SessionReaderSharedPtr SetSession(
    const std::string &xml, const std::string &exec = "Serial")
{
    BOOST_TEST_MESSAGE("Creating input and output fields");

    int argc    = 3;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestRiemannOperator");
    argv[1]     = strdup(xml.c_str());
    argv[2]     = strdup(("--opExecSpace=" + exec).c_str());

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}

static MultiRegions::DisContFieldSharedPtr SetExpList(
    const LibUtilities::SessionReaderSharedPtr &session)
{
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    return MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
        session, graph, "DefaultVar", true, true, Collections::eNoCollection);
}

// Apply operator to fwd/bwd arrays, producing flux arrays
template <typename TData>
static void ApplyRiemannOperator(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &method, const std::string &execStr,
    const MultiRegions::DisContFieldSharedPtr &dg, unsigned int spaceDim,
    size_t npts, unsigned int nFields,
    const Array<OneD, Array<OneD, TData>> &normals,
    const Array<OneD, Array<OneD, TData>> &fwd,
    const Array<OneD, Array<OneD, TData>> &bwd,
    Array<OneD, Array<OneD, TData>> &flx)
{
    auto traceAttr =
        GetBlockAttributes<TData, FieldState::Phys>(dg->GetTrace());

    LibUtilities::Field<TData, FieldState::Phys> fwdField("fwd", traceAttr,
                                                          nFields, 1);
    LibUtilities::Field<TData, FieldState::Phys> bwdField("bwd", traceAttr,
                                                          nFields, 1);
    LibUtilities::Field<TData, FieldState::Phys> flxField("flux", traceAttr,
                                                          nFields, 1);

    LibUtilities::Field<TData, FieldState::Phys> normalsField(
        "traceNormals", traceAttr, spaceDim, 1);

    fwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(fwd, nFields, npts));
    bwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(bwd, nFields, npts));
    normalsField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(normals, spaceDim, npts));

    auto vars = session->GetVariables();

    auto op = CompressibleSolverOp<double>::Create(dg, vars, method, execStr);
    op->SetTraceNormals(normalsField);
    op->Apply(fwdField, bwdField, flxField);

    UnflattenCompMajor(flxField.ToArray(), flx, nFields, npts);
}

// Build constant Euler state and reference physical flux F·n for any normal
// Conserved layout: [rho, rhou, rhov, rhow, E] = spaceDim + 2
static void FillConstStateAndReferenceFlux(
    unsigned int spaceDim, size_t npts,
    const Array<OneD, Array<OneD, double>> &normals,
    Array<OneD, Array<OneD, double>> &fwd,
    Array<OneD, Array<OneD, double>> &bwd,
    Array<OneD, Array<OneD, double>> &fluxRef, double gamma)
{
    const unsigned int nFields = spaceDim + 2;

    const double rho = 0.9;
    const double p   = 1.0;
    double u[3]      = {1.0, 2.0, 3.0};

    for (size_t i = 0; i < npts; ++i)
    {
        // density
        fwd[0][i] = rho;
        bwd[0][i] = rho;

        // momentum
        double mom[3] = {0, 0, 0};
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            mom[d]        = rho * u[d];
            fwd[1 + d][i] = mom[d];
            bwd[1 + d][i] = mom[d];
        }

        // energy
        const double rhoe = p / (gamma - 1.0);
        double kin        = 0.0;
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            kin += 0.5 * mom[d] * mom[d] / rho;
        }
        const double E      = rhoe + kin;
        fwd[nFields - 1][i] = E;
        bwd[nFields - 1][i] = E;

        // normal velocity u_n = u dot n
        double un = 0.0;
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            un += u[d] * normals[d][i];
        }

        // reference flux F·n
        // mass flux is rho * u_n
        fluxRef[0][i] = rho * un; // mass

        // momentum flux is rho * u * u_n + p * n
        for (unsigned int k = 0; k < spaceDim; ++k)
        {
            fluxRef[1 + k][i] = mom[k] * un + p * normals[k][i];
        }

        // energy flux is (E + p) * u_n
        fluxRef[nFields - 1][i] = (E + p) * un;
    }
}

// Runner: setup session/dg, make normals, fill states, apply op, check flux
// normalDir: 0=x, 1=y, 2=z (valid if normalDir < spaceDim)
// If normalDir is invalid for that mesh dimension, we SKIP.
static void RunConstStateRiemannTest(const std::string &xml,
                                     const std::string &method,
                                     const std::string &execStr,
                                     unsigned int normalDir, double gamma = 1.4)
{
    auto session = SetSession(xml, execStr);
    auto dg      = SetExpList(session);

    dg->SetDataWarehouse();
    dg->GetTrace()->SetDataWarehouse();

    const unsigned int spaceDim = dg->GetExp(0)->GetShapeDimension();
    const size_t npts           = dg->GetTrace()->GetTotPoints();
    const unsigned int nFields  = spaceDim + 2;

    // If user listed a normalDir that doesn't exist for this mesh dimension,
    // just skip (use ASSERT if you prefer strict behaviour).
    if (normalDir >= spaceDim)
    {
        return;
    }

    // normals: unit along normalDir
    Array<OneD, Array<OneD, double>> normals(spaceDim);
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        normals[d] = Array<OneD, double>(npts, 0.0);
    }
    for (size_t i = 0; i < npts; ++i)
    {
        normals[normalDir][i] = 1.0;
    }

    // arrays
    Array<OneD, Array<OneD, double>> fwd(nFields), bwd(nFields), flux(nFields),
        fluxRef(nFields);
    for (unsigned int c = 0; c < nFields; ++c)
    {
        fwd[c]     = Array<OneD, double>(npts, 0.0);
        bwd[c]     = Array<OneD, double>(npts, 0.0);
        flux[c]    = Array<OneD, double>(npts, 0.0);
        fluxRef[c] = Array<OneD, double>(npts, 0.0);
    }

    FillConstStateAndReferenceFlux(spaceDim, npts, normals, fwd, bwd, fluxRef,
                                   gamma);

    ApplyRiemannOperator<double>(session, method, execStr, dg, spaceDim, npts,
                                 nFields, normals, fwd, bwd, flux);

    // check
    for (unsigned int c = 0; c < nFields; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            BOOST_CHECK_CLOSE(fluxRef[c][i], flux[c][i], 1e-10);
        }
    }
}

static void RunOneRiemannCase(const std::string &xml, const std::string &method,
                              const std::string &execStr, const PrimState &WL,
                              const PrimState &WR,
                              const Array<OneD, const double> &n,
                              Array<OneD, double> &flux, double gamma = 1.4)
{
    auto session = SetSession(xml, execStr);
    auto dg      = SetExpList(session);

    dg->SetDataWarehouse();
    dg->GetTrace()->SetDataWarehouse();

    const unsigned int spaceDim = dg->GetExp(0)->GetShapeDimension();
    const size_t npts           = dg->GetTrace()->GetTotPoints();
    const unsigned int nFields  = spaceDim + 2;

    ASSERTL0(n.size() == spaceDim, "RunOneRiemannCase: wrong normal size");

    Array<OneD, Array<OneD, double>> normals(spaceDim);
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        normals[d] = Array<OneD, double>(npts, 0.0);
    }
    FillTraceNormals(spaceDim, npts, n, normals);

    Array<OneD, Array<OneD, double>> fwdArr(nFields), bwdArr(nFields),
        fluxArr(nFields);
    for (unsigned int c = 0; c < nFields; ++c)
    {
        fwdArr[c]  = Array<OneD, double>(npts, 0.0);
        bwdArr[c]  = Array<OneD, double>(npts, 0.0);
        fluxArr[c] = Array<OneD, double>(npts, 0.0);
    }

    FillTraceFromPrimStates(spaceDim, npts, WL, WR, fwdArr, bwdArr, gamma);

    ApplyRiemannOperator<double>(session, method, execStr, dg, spaceDim, npts,
                                 nFields, normals, fwdArr, bwdArr, fluxArr);

    flux = Array<OneD, double>(nFields, 0.0);
    for (unsigned int c = 0; c < nFields; ++c)
    {
        flux[c] = fluxArr[c][0];
    }
}

// Antisymmetry with normal flip:
// F(L,R;n=+x) = - F(R,L;n=-x).
static void CheckNormalFlipAntisymmetry(
    const std::string &xml, const std::string &method,
    const std::string &execStr, const PrimState &WL, const PrimState &WR,
    unsigned int spaceDim, double gamma, double tol)
{
    Array<OneD, double> normals(spaceDim, 0.0);
    normals[0] = 1.0;

    Array<OneD, double> negNormals(spaceDim, 0.0);
    negNormals[0] = -1.0;

    Array<OneD, double> f_plus, f_minus;

    // F(L,R; +n)
    RunOneRiemannCase(xml, method, execStr, WL, WR, normals, f_plus, gamma);

    // F(R,L; -n)
    RunOneRiemannCase(xml, method, execStr, WR, WL, negNormals, f_minus, gamma);

    // Check f_plus + f_minus == 0
    BOOST_REQUIRE_EQUAL(f_plus.size(), f_minus.size());

    Array<OneD, double> sum(f_plus.size(), 0.0);
    for (size_t i = 0; i < f_plus.size(); ++i)
    {
        sum[i] = f_plus[i] + f_minus[i];
    }

    Array<OneD, double> zero(f_plus.size(), 0.0);
    CheckCloseRel(sum, zero, tol);
}

[[maybe_unused]] static void RunMachTest(const std::string &xml,
                                         const std::vector<std::string> &ops,
                                         const std::string &execStr,
                                         double gamma = 1.4)
{
    // Determine dimension from the mesh.
    auto session                = SetSession(xml, execStr);
    auto dg                     = SetExpList(session);
    const unsigned int spaceDim = dg->GetExp(0)->GetShapeDimension();
    BOOST_REQUIRE(spaceDim == 2 || spaceDim == 3);

    const double tol = 1e-10;

    Array<OneD, double> n(spaceDim, 0.0);
    n[0] = 1.0;

    // ---------------------------------------------------------------------
    // Case family 1: Consistency for WL==WR across Mach list (ALL ops).
    // ---------------------------------------------------------------------
    const std::vector<double> Machs = {-3.0, -1.01, -0.99, -0.2, 0.0,
                                       0.2,  0.99,  1.01,  3.0};

    // Pick a base state.
    const double rho0 = 1.0;
    const double p0   = 1.0;
    const double a0   = SoundSpeedFromRhoP(rho0, p0, gamma);

    for (double M : Machs)
    {
        PrimState W                 = MakeStateFromMach(rho0, p0, M, a0);
        Array<OneD, double> fluxRef = PhysicalFlux(W, spaceDim, gamma);

        for (auto &method : ops)
        {
            BOOST_TEST_CONTEXT("Case family 1 Riemann method: "
                               << method << ", SpaceDim=" << spaceDim
                               << ", Mach=" << M)
            {
                Array<OneD, double> flux;
                RunOneRiemannCase(xml, method, execStr, W, W, n, flux, gamma);
                CheckFinite(flux);

                CheckCloseRel(flux, fluxRef, tol);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Case family 2: Supersonic upwind limit (upwind solvers only).
    // Exclude "Average". LaxFriedrichs might not be exact depending on alpha.
    // ---------------------------------------------------------------------
    const std::vector<std::string> upwindOps = {
        "HLL", "HLLC", "AUSM0", "AUSM1", "AUSM2", "AUSM3", "Roe"};

    // Make different left/right states, but same-sign supersonic
    // Mach. Use cRef based on code-consistent cA for AUSM to hit |M|>=1
    // reliably:
    const double rhoL = 1.0, pL = 1.0;
    const double rhoR = 0.4, pR = 0.7;
    const double cL = SoundSpeedFromRhoP(rhoL, pL, gamma);
    const double cR = SoundSpeedFromRhoP(rhoR, pR, gamma);
    const double cA = 0.5 * (cL + cR);

    auto DoSupersonic = [&](double ML, double MR) {
        PrimState WL = MakeStateFromMach(rhoL, pL, ML, cA);
        PrimState WR = MakeStateFromMach(rhoR, pR, MR, cA);

        // Expected: + => left physical flux, - => right physical flux
        Array<OneD, double> fluxRef = (ML > 0.0 && MR > 0.0)
                                          ? PhysicalFlux(WL, spaceDim, gamma)
                                          : PhysicalFlux(WR, spaceDim, gamma);

        for (auto &method : upwindOps)
        {
            BOOST_TEST_CONTEXT("Case family 2 Riemann method: "
                               << method << ", SpaceDim=" << spaceDim
                               << ", ML=" << ML << ", MR=" << MR)
            {
                Array<OneD, double> flux;
                RunOneRiemannCase(xml, method, execStr, WL, WR, n, flux, gamma);
                CheckFinite(flux);
                CheckCloseRel(flux, fluxRef, tol);
            }
        }
    };

    DoSupersonic(+2.5, +2.0);
    DoSupersonic(-2.5, -2.0);

    // ---------------------------------------------------------------------
    // Case family 3: Near-sonic / subsonic mixed cases: verify antisymmetry
    // ---------------------------------------------------------------------
    struct Pair
    {
        double ML, MR;
    };
    const std::vector<Pair> transonicPairs = {{+0.99, +1.01},
                                              {-0.99, -1.01},
                                              {+0.2, -0.2},
                                              {-0.2, +0.2},
                                              {+0.8, +0.3}};

    for (auto pr : transonicPairs)
    {
        PrimState WL = MakeStateFromMach(rhoL, pL, pr.ML, cA);
        PrimState WR = MakeStateFromMach(rhoR, pR, pr.MR, cA);

        for (auto &method : ops)
        {
            BOOST_TEST_CONTEXT("Case family 3 Riemann method: "
                               << method << ", SpaceDim=" << spaceDim
                               << ", ML=" << pr.ML << ", MR=" << pr.MR)
            {
                Array<OneD, double> flux;
                RunOneRiemannCase(xml, method, execStr, WL, WR, n, flux, gamma);
                CheckFinite(flux);
                CheckNormalFlipAntisymmetry(xml, method, execStr, WL, WR,
                                            spaceDim, gamma, tol);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Case family 4: M=0 with pressure jump
    // AUSM0/1: pressure-only (mass=0, energy=0, mom=pbar)
    // AUSM2/3: include Mp correction (nonzero mass & energy possible)
    // ---------------------------------------------------------------------
    const std::vector<std::string> ausmOps = {"AUSM0", "AUSM1", "AUSM2",
                                              "AUSM3"};

    {
        auto ExpectAUSM0or1M0 = [&]() {
            const unsigned int nFields = spaceDim + 2;
            Array<OneD, double> fluxRef(nFields, 0.0);
            const double pbar = 0.5 * (pL + pR);
            fluxRef[0]        = 0.0;  // mass
            fluxRef[1]        = pbar; // x-momentum
            if (spaceDim > 1)
            {
                fluxRef[2] = 0.0;
            }
            if (spaceDim > 2)
            {
                fluxRef[3] = 0.0;
            }
            fluxRef[nFields - 1] = 0.0; // energy
            return fluxRef;
        };

        auto ExpectAUSM2or3M0 = [&](bool ausm3) {
            const double rhoA = 0.5 * (rhoL + rhoR);
            const double pbar = 0.5 * (pL + pR);

            const double Kp = 0.25;
            double fa       = 1.0;
            if (ausm3)
            {
                const double Mco = 0.01;
                fa               = Mco * (2.0 - Mco);
            }

            const double Mp = -(Kp / fa) * ((pR - pL) / (rhoA * cA * cA));
            const bool cond = (Mp >= 0.0);

            const double rhoUp = cond ? rhoL : rhoR;
            const double pUp   = cond ? pL : pR;

            const unsigned int nFields = spaceDim + 2;
            Array<OneD, double> fluxRef(nFields, 0.0);

            // uUp=0 so momentum convective part is 0
            fluxRef[0] = cA * Mp * rhoUp; // mass
            fluxRef[1] = pbar;            // x-momentum
            if (spaceDim > 1)
            {
                fluxRef[2] = 0.0;
            }
            if (spaceDim > 2)
            {
                fluxRef[3] = 0.0;
            }

            // EUp + pUp = gamma/(gamma-1) * pUp (since u=0)
            fluxRef[nFields - 1] =
                cA * Mp * (gamma / (gamma - 1.0)) * pUp; // energy
            return fluxRef;
        };

        PrimState WL = MakeStateFromMach(rhoL, pL, 0.0, cA);
        PrimState WR = MakeStateFromMach(rhoR, pR, 0.0, cA);

        for (auto &method : ausmOps)
        {
            BOOST_TEST_CONTEXT("Case family 4 Riemann method: "
                               << method << ", SpaceDim=" << spaceDim)
            {
                Array<OneD, double> flux;
                RunOneRiemannCase(xml, method, execStr, WL, WR, n, flux, gamma);
                CheckFinite(flux);

                Array<OneD, double> fluxRef =
                    (method == "AUSM0" || method == "AUSM1")
                        ? ExpectAUSM0or1M0()
                        : (method == "AUSM2" ? ExpectAUSM2or3M0(false)
                                             : ExpectAUSM2or3M0(true));
                CheckCloseRel(flux, fluxRef, tol);
            }
        }
    }
}

// Case list and helper to run all cases for one operator.
// Adding a new op = add one RunCasesForOp(...) block.
struct Case
{
    std::string xml;
    unsigned int normalDir;
};

[[maybe_unused]] static void RunCasesForOp(const std::string &method,
                                           const std::string &execStr,
                                           const std::vector<Case> &cases)
{
    for (const auto &tc : cases)
    {
        BOOST_TEST_CONTEXT("method=" << method << " exec=" << execStr << " xml="
                                     << tc.xml << " normalDir=" << tc.normalDir)
        {
            RunConstStateRiemannTest(tc.xml, method, execStr, tc.normalDir);
        }
    }
}

BOOST_AUTO_TEST_SUITE(Riemann_ConstState_AllOps_AllCases)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_Average)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("Average", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_LaxFriedrichs)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("LaxFriedrichs", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_HLL)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("HLL", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_HLLC)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("HLLC", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_AUSM0)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("AUSM0", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_AUSM1)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("AUSM1", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_AUSM2)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("AUSM2", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_AUSM3)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("AUSM3", execStr, cases);
}

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_Roe)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square_Euler.xml", 0}, {"run/square_Euler.xml", 1},
        {"run/hex_Euler.xml", 0},    {"run/hex_Euler.xml", 1},
        {"run/hex_Euler.xml", 2},
    };

    RunCasesForOp("Roe", execStr, cases);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Riemann_MachTest)

BOOST_AUTO_TEST_CASE(Riemann_MachTest_2D_square)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<std::string> ops = {"Average", "LaxFriedrichs", "HLL",
                                          "HLLC",    "AUSM0",         "AUSM1",
                                          "AUSM2",   "AUSM3",         "Roe"};

    RunMachTest("run/square_Euler.xml", ops, execStr, 1.4);
}

BOOST_AUTO_TEST_CASE(Riemann_MachTest_3D_hex)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<std::string> ops = {"Average", "LaxFriedrichs", "HLL",
                                          "HLLC",    "AUSM0",         "AUSM1",
                                          "AUSM2",   "AUSM3",         "Roe"};

    RunMachTest("run/hex_Euler.xml", ops, execStr, 1.4);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
