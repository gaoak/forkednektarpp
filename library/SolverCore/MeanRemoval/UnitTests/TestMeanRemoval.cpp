///////////////////////////////////////////////////////////////////////////////
//
// File: TestMeanRemoval.cpp
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
// Description: Unit test for SolverCore MeanRemovalOp. Mirrors the setup used
// by the RiemannSolver unit test (TestRiemann.cpp): a hand-rolled session +
// ExpList, analytic input fields, and an independent oracle for the expected
// result.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestMeanRemoval

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

#include "../MeanRemovalOp.hpp"

#include <UnitTests/TestBoostSetup.hpp>

using namespace Nektar;
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

#include <UnitTests/TestBoostTeardown.hpp>

// Session + volume ExpList setup (same pattern as TestRiemann.cpp).
static LibUtilities::SessionReaderSharedPtr SetSession(
    const std::string &xml, const std::string &exec = "Serial")
{
    int argc    = 3;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestMeanRemovalOperator");
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

static MultiRegions::ExpListSharedPtr SetExpList(
    const LibUtilities::SessionReaderSharedPtr &session)
{
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    return MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
        session, graph, true, "u", Collections::eNoCollection);
}

// Analytic, per-component input field, flattened component-major. Deliberately
// carries a non-zero volume-averaged mean per component (the leading constant).
static void FillField(const MultiRegions::ExpListSharedPtr &explist,
                      unsigned int nComp, size_t npts,
                      Array<OneD, double> &flat)
{
    Array<OneD, double> x(npts, 0.0), y(npts, 0.0), z(npts, 0.0);
    explist->GetCoords(x, y, z);

    for (unsigned int c = 0; c < nComp; ++c)
    {
        const double s = 1.0 + static_cast<double>(c);
        for (size_t i = 0; i < npts; ++i)
        {
            flat[c * npts + i] = 0.7 * s + 0.30 * s * x[i] +
                                 0.15 * (c + 2) * y[i] * y[i] -
                                 0.05 * s * z[i] + 0.02 * s * x[i] * y[i];
        }
    }
}

// Runner: build session/ExpList, fill fields, apply MeanRemovalOp, and check
// against an independent oracle (mean computed from ExpList's own quadrature).
static void RunMeanRemovalTest(const std::string &xml,
                               const std::string &execStr)
{
    auto session = SetSession(xml, execStr);
    auto explist = SetExpList(session);
    explist->SetDataWarehouse();

    const size_t npts        = explist->GetTotPoints();
    const auto vars          = session->GetVariables();
    const unsigned int nComp = static_cast<unsigned int>(vars.size());

    // Reference volume from the ExpList's own integration (independent of the
    // operator's VolumeKernel).
    Array<OneD, double> ones(npts, 1.0);
    const double vol = explist->Integral(ones);
    BOOST_REQUIRE(vol > 0.0);

    Array<OneD, double> orig(nComp * npts, 0.0);
    FillField(explist, nComp, npts, orig);

    // Oracle: expected[c][i] = orig[c][i] - mean_c, with
    // mean_c = Integral(orig_c) / vol computed via ExpList quadrature.
    std::vector<double> mean(nComp);
    Array<OneD, double> expected(nComp * npts, 0.0);
    for (unsigned int c = 0; c < nComp; ++c)
    {
        Array<OneD, double> comp(npts);
        for (size_t i = 0; i < npts; ++i)
        {
            comp[i] = orig[c * npts + i];
        }
        mean[c] = explist->Integral(comp) / vol;
        for (size_t i = 0; i < npts; ++i)
        {
            expected[c * npts + i] = orig[c * npts + i] - mean[c];
        }
    }

    auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);

    auto op = MeanRemovalOp<double>::Create(explist, vars, execStr);

    const double tol = 1.0e-10;

    // --- Non-interleaved storage ---
    {
        LibUtilities::Field<double, FieldState::Phys> f("u", blockAttr, nComp,
                                                        1);
        f.template CopyArray<NektarSpaces::HostSpace>(orig);

        op->Apply(f);

        auto res = f.template ToArray<double>();
        for (unsigned int c = 0; c < nComp; ++c)
        {
            for (size_t i = 0; i < npts; ++i)
            {
                BOOST_CHECK_SMALL(res[c * npts + i] - expected[c * npts + i],
                                  tol);
            }

            // The volume-averaged mean is genuinely gone.
            Array<OneD, double> rc(npts);
            for (size_t i = 0; i < npts; ++i)
            {
                rc[i] = res[c * npts + i];
            }
            BOOST_CHECK_SMALL(explist->Integral(rc) / vol, tol);
        }
    }

    // --- Interleaved storage (exercises the SumFac / warp-interleaved path)
    // ---
    {
        LibUtilities::Field<double, FieldState::Phys> f("u", blockAttr, nComp,
                                                        1);
        f.template CopyArray<NektarSpaces::HostSpace>(orig);

        const unsigned int width =
            NektarSpaces::GetVectorWidth<double>(execStr);
        f.ReshapeStorage(width, execStr);

        op->Apply(f);

        f.ReshapeStorage(1, execStr);
        auto res = f.template ToArray<double>();
        for (unsigned int c = 0; c < nComp; ++c)
        {
            for (size_t i = 0; i < npts; ++i)
            {
                BOOST_CHECK_SMALL(res[c * npts + i] - expected[c * npts + i],
                                  tol);
            }
        }
    }
}

struct Case
{
    std::string xml;
};

[[maybe_unused]] static void RunCases(const std::string &execStr,
                                      const std::vector<Case> &cases)
{
    for (const auto &tc : cases)
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << tc.xml)
        {
            RunMeanRemovalTest(tc.xml, execStr);
        }
    }
}

BOOST_AUTO_TEST_SUITE(MeanRemoval_MeanSubtraction_AllCases)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(MeanRemoval_MeanSubtraction)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    const std::vector<Case> cases = {
        {"run/square.xml"}, // 2D quads, regular
        {"run/hex.xml"},    // 3D hexes, variable order + one deformed element
    };

    RunCases(execStr, cases);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
