///////////////////////////////////////////////////////////////////////////////
//
// File: TestForcingNoise.cpp
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
// Description: Unit test for the SolverCore white noise forcing, loaded from
// a session through Forcing::Load and driven through a sequence of Apply()
// calls covering its statistics, UPDATEFREQ, append and NSTEPS.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestForcingNoise

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/Forcing/Forcing.h>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <cmath>

using namespace Nektar;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
// The mesh and variables of run/square.xml, with the FORCING section of
// run/ForcingNoise.xml: WHITENOISE 0.5, UPDATEFREQ 2, NSTEPS 5.
static LibUtilities::SessionReaderSharedPtr SetSession(const std::string &exec)
{
    int argc    = 4;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestForcingNoise");
    argv[1]     = strdup("run/square.xml");
    argv[2]     = strdup("run/ForcingNoise.xml");
    argv[3]     = strdup(("--opExecSpace=" + exec).c_str());

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteForcingNoise)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(ForcingNoise_Apply)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    auto session = SetSession(execStr);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
        session, graph, true, "u", Collections::eNoCollection);
    explist->SetDataWarehouse();

    const auto vars = session->GetVariables();
    auto forcings   = Forcing::Load(session, explist, vars);
    BOOST_REQUIRE(forcings.size() == 1);
    auto &forcing = forcings[0];

    const double sigma = 0.5;
    const double tol   = 1.0e-14;

    auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    LibUtilities::Field<double, FieldState::Phys> in("in", blockAttr, vars, 1);
    LibUtilities::Field<double, FieldState::Phys> out("out", blockAttr, vars,
                                                      1);

    const size_t n = vars.size() * explist->GetTotPoints();
    Array<OneD, double> ones(n, 1.0);

    // Apply 0, assigned at scale 2: normal noise of zero mean and standard
    // deviation 2 sigma, to six standard errors.
    forcing->SetAppend(false);
    forcing->Apply(in, out, 0.0, 2.0);
    auto noise0 = out.template ToArray<double>();
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        mean += noise0[i];
    }
    mean /= n;
    double variance = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        variance += (noise0[i] - mean) * (noise0[i] - mean);
    }
    variance /= n - 1.0;
    const double sigma2 = 4.0 * sigma * sigma;
    BOOST_TEST(std::abs(mean) < 6.0 * 2.0 * sigma / std::sqrt(n));
    BOOST_TEST(std::abs(variance - sigma2) < 6.0 * sigma2 * std::sqrt(2.0 / n));

    // Apply 1 keeps the noise of apply 0.
    forcing->Apply(in, out, 0.0, 1.0);
    auto noise1 = out.template ToArray<double>();
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(noise1[i] == 0.5 * noise0[i]);
    }

    // Apply 2 redraws it.
    forcing->Apply(in, out, 0.0, 1.0);
    auto noise2  = out.template ToArray<double>();
    bool redrawn = false;
    for (size_t i = 0; i < n; ++i)
    {
        redrawn = redrawn || noise2[i] != noise1[i];
    }
    BOOST_TEST(redrawn);

    // Apply 3, appended, adds the noise of apply 2.
    out.template CopyArray<NektarSpaces::HostSpace>(ones);
    forcing->SetAppend(true);
    forcing->Apply(in, out, 0.0, 1.0);
    auto noise3 = out.template ToArray<double>();
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(std::abs(noise3[i] - 1.0 - noise2[i]) < tol);
    }

    // Apply 4 redraws the noise and is the last of NSTEPS.
    forcing->SetAppend(false);
    forcing->Apply(in, out, 0.0, 1.0);
    auto noise4 = out.template ToArray<double>();
    redrawn     = false;
    for (size_t i = 0; i < n; ++i)
    {
        redrawn = redrawn || noise4[i] != noise2[i];
    }
    BOOST_TEST(redrawn);

    // Beyond NSTEPS, assigning gives zero and appending leaves out as it is.
    forcing->Apply(in, out, 0.0, 1.0);
    auto zero = out.template ToArray<double>();
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(zero[i] == 0.0);
    }

    out.template CopyArray<NektarSpaces::HostSpace>(ones);
    forcing->SetAppend(true);
    forcing->Apply(in, out, 0.0, 1.0);
    auto unchanged = out.template ToArray<double>();
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(unchanged[i] == 1.0);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
