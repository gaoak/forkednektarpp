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

#include <MultiRegions/DisContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <Operators/Field/Field.hpp>
#include <Operators/Utils/UtilsKernels.hpp>

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

using namespace Nektar;
using namespace Nektar::Operators;

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

    Field<TData, FieldState::Phys> fwdField("fwd", traceAttr, nFields, 1);
    Field<TData, FieldState::Phys> bwdField("bwd", traceAttr, nFields, 1);
    Field<TData, FieldState::Phys> flxField("flux", traceAttr, nFields, 1);

    Field<TData, FieldState::Phys> normalsField("traceNormals", traceAttr,
                                                spaceDim, 1);

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
    Array<OneD, Array<OneD, double>> &flxRef, double gamma)
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
        flxRef[0][i] = rho * un; // mass

        // momentum flux is rho * u * u_n + p * n
        for (unsigned int k = 0; k < spaceDim; ++k)
        {
            flxRef[1 + k][i] = mom[k] * un + p * normals[k][i];
        }

        // energy flux is (E + p) * u_n
        flxRef[nFields - 1][i] = (E + p) * un;
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
    Array<OneD, Array<OneD, double>> fwd(nFields), bwd(nFields), flx(nFields),
        flxRef(nFields);
    for (unsigned int c = 0; c < nFields; ++c)
    {
        fwd[c]    = Array<OneD, double>(npts, 0.0);
        bwd[c]    = Array<OneD, double>(npts, 0.0);
        flx[c]    = Array<OneD, double>(npts, 0.0);
        flxRef[c] = Array<OneD, double>(npts, 0.0);
    }

    FillConstStateAndReferenceFlux(spaceDim, npts, normals, fwd, bwd, flxRef,
                                   gamma);

    ApplyRiemannOperator<double>(session, method, execStr, dg, spaceDim, npts,
                                 nFields, normals, fwd, bwd, flx);

    // check
    for (unsigned int c = 0; c < nFields; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            BOOST_CHECK_CLOSE(flxRef[c][i], flx[c][i], 1e-10);
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

static void RunCasesForOp(const std::string &method, const std::string &execStr,
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
