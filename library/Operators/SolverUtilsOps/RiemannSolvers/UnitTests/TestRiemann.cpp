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

#include "../UpwindSolver/UpwindSolverOp.hpp"

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
        session, graph, "u", true, true, Collections::eNoCollection);
}

// Apply operator to fwd/bwd arrays, producing flux arrays
template <typename TData>
static void ApplyRiemannOperator(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &method, const std::string &execStr,
    const MultiRegions::DisContFieldSharedPtr &dg, unsigned int spaceDim,
    size_t npts, unsigned int nFields,
    const Array<OneD, Array<OneD, TData>> &normals,
    const Array<OneD, Array<OneD, TData>> &traceAdvVel,
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
    Field<TData, FieldState::Phys> traceAdvVelField("traceAdvVel", traceAttr,
                                                    spaceDim, 1);

    fwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(fwd, nFields, npts));
    bwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(bwd, nFields, npts));
    normalsField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(normals, spaceDim, npts));
    traceAdvVelField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(traceAdvVel, spaceDim, npts));

    auto vars = session->GetVariables();

    auto op = RiemannSolverOp<double>::Create(dg, vars, method, execStr);
    op->SetTraceNormals(normalsField);
    op->SetTraceAdvVel(traceAdvVelField);
    op->Apply(fwdField, bwdField, flxField);

    UnflattenCompMajor(flxField.ToArray(), flx, nFields, npts);
}

// Build constant Euler state and reference physical flux F·n for any normal
// Conserved layout: [rho, rhou, rhov, rhow, E] = spaceDim + 2
static void FillConstStateAndReferenceFlux(
    unsigned int spaceDim, unsigned int nFields, size_t npts,
    Array<OneD, Array<OneD, double>> &normals,
    Array<OneD, Array<OneD, double>> &traceAdvVel,
    Array<OneD, Array<OneD, double>> &fwd,
    Array<OneD, Array<OneD, double>> &bwd,
    Array<OneD, Array<OneD, double>> &flxRef)
{
    // Constant advection velocity a
    const double a[3] = {1.0, 2.0, 3.0};

    // Use +/- x direction normals so that un=a·n flips sign across trace
    // points.
    for (size_t i = 0; i < npts; ++i)
    {
        normals[0][i] = (i < npts / 2) ? 1.0 : -1.0;
    }

    for (size_t i = 0; i < npts; ++i)
    {
        // Compute un = a·n
        double un = 0.0;
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            un += a[d] * normals[d][i];
            traceAdvVel[d][i] = a[d];
        }

        for (unsigned int c = 0; c < nFields; ++c)
        {
            // Distinct values per component
            // e.g. 2,3,4,...
            fwd[c][i] = 2.0 + c;
            // e.g. -1,-2,-3,...
            bwd[c][i] = -1.0 - c;

            const double up = (un > 0.0) ? fwd[c][i] : bwd[c][i];

            flxRef[c][i] = un * up;
        }
    }
}

// Runner: setup session/dg, make normals, fill states, apply op, check flux
static void RunConstStateRiemannTest(const std::string &xml,
                                     const std::string &method,
                                     const std::string &execStr)
{
    auto session = SetSession(xml, execStr);
    auto dg      = SetExpList(session);

    dg->SetDataWarehouse();
    dg->GetTrace()->SetDataWarehouse();

    const unsigned int spaceDim = dg->GetExp(0)->GetShapeDimension();
    const size_t npts           = dg->GetTrace()->GetTotPoints();
    const unsigned int nFields  = session->GetVariables().size();

    // normals: unit along x
    Array<OneD, Array<OneD, double>> normals(spaceDim);
    Array<OneD, Array<OneD, double>> traceAdvVel(spaceDim);
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        normals[d]     = Array<OneD, double>(npts, 0.0);
        traceAdvVel[d] = Array<OneD, double>(npts, 0.0);
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

    FillConstStateAndReferenceFlux(spaceDim, nFields, npts, normals,
                                   traceAdvVel, fwd, bwd, flxRef);

    ApplyRiemannOperator<double>(session, method, execStr, dg, spaceDim, npts,
                                 nFields, normals, traceAdvVel, fwd, bwd, flx);

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
};

static void RunCasesForOp(const std::string &method, const std::string &execStr,
                          const std::vector<Case> &cases)
{
    for (const auto &tc : cases)
    {
        BOOST_TEST_CONTEXT("method=" << method << " exec=" << execStr
                                     << " xml=" << tc.xml)
        {
            RunConstStateRiemannTest(tc.xml, method, execStr);
        }
    }
}

BOOST_AUTO_TEST_SUITE(Riemann_ConstState_AllOps_AllCases)

BOOST_AUTO_TEST_CASE(Riemann_ConstState_AllOps_Upwind)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square.xml"},
        {"run/hex.xml"},
    };

    RunCasesForOp("UpwindSolver", execStr, cases);
}

BOOST_AUTO_TEST_SUITE_END()
