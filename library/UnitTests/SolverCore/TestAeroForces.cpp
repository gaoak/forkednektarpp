///////////////////////////////////////////////////////////////////////////////
//
// File: TestAeroForces.cpp
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
// Description: Unit test for the SolverCore AeroForcesOp.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestAeroForces

#include <array>
#include <cmath>
#include <numeric>

#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/AeroForces/AeroForcesOp.hpp>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

using namespace Nektar;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
static LibUtilities::SessionReaderSharedPtr SetSession(const std::string &xml,
                                                       const std::string &exec)
{
    int argc    = 3;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestAeroForcesOperator");
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

// A continuous field, which holds the boundary expansions the operator
// integrates over.
static MultiRegions::ExpListSharedPtr SetExpList(
    const LibUtilities::SessionReaderSharedPtr &session)
{
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto field = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        session, graph, "u", true, false, Collections::eNoCollection);
    field->SetDataWarehouse();
    return field;
}

static std::array<double, 3> Cross(const std::array<double, 3> &a,
                                   const std::array<double, 3> &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

// Integrates the tractions of an affine pressure and an affine velocity
// gradient over every boundary region, which together close the domain, with
// the pressure and the gradient each laid out with and without interleaving.
// By the divergence theorem, with V the volume and X the first moment of the
// domain about the pivot c:
//
// - the pressure p = a + b.x gives the force rho V b and the moment
//   rho X x b;
// - the gradient G, with symmetric part S = G + G^T, gives the force
//   -mu V div(S) and the moment -mu X x div(S), div(S) being constant.
static void RunClosedBoundaryTest(const std::string &xml,
                                  const std::string &execStr)
{
    auto session          = SetSession(xml, execStr);
    auto explist          = SetExpList(session);
    const unsigned int nD = explist->GetCoordim(0);
    const size_t nPoints  = explist->GetTotPoints();
    const double rho      = 1.3;
    const double mu       = 0.07;
    const std::array<double, 3> pivot{0.3, -0.2, nD == 3 ? 0.45 : 0.0};
    const std::array<double, 3> b{0.8, -1.1, nD == 3 ? 0.6 : 0.0};

    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);

    // Derivative of velocity component c in direction d, gradient component
    // c * nD + d: G_cd = A_cd + sum_e B_cde x_e.
    auto slope = [](const unsigned int c, const unsigned int d,
                    const unsigned int e) {
        return 0.1 * (c + 1) - 0.2 * d + 0.15 * e * (c + d + 1) - 0.05 * c * e;
    };
    Array<OneD, double> pressure(nPoints, 0.0);
    Array<OneD, double> gradient(nD * nD * nPoints, 0.0);
    for (size_t i = 0; i < nPoints; ++i)
    {
        const double xi[3] = {x[i], y[i], z[i]};
        pressure[i]        = 0.4 + b[0] * xi[0] + b[1] * xi[1] + b[2] * xi[2];
        for (unsigned int c = 0; c < nD; ++c)
        {
            for (unsigned int d = 0; d < nD; ++d)
            {
                double value = 0.3 * c - 0.7 * d + 0.2;
                for (unsigned int e = 0; e < nD; ++e)
                {
                    value += slope(c, d, e) * xi[e];
                }
                gradient[(c * nD + d) * nPoints + i] = value;
            }
        }
    }

    // div(S)_j = sum_k (B_jkk + B_kjk).
    std::array<double, 3> divS{};
    for (unsigned int j = 0; j < nD; ++j)
    {
        for (unsigned int k = 0; k < nD; ++k)
        {
            divS[j] += slope(j, k, k) + slope(k, j, k);
        }
    }

    Array<OneD, double> ones(nPoints, 1.0);
    const double volume = explist->Integral(ones);
    BOOST_REQUIRE(volume > 0.0);
    const std::array<Array<OneD, double>, 3> coords{x, y, z};
    std::array<double, 3> moment{};
    for (unsigned int d = 0; d < nD; ++d)
    {
        moment[d] = explist->Integral(coords[d]) - pivot[d] * volume;
    }

    std::array<double, 3> expectedPressureForce{};
    std::array<double, 3> expectedViscousForce{};
    for (unsigned int d = 0; d < 3; ++d)
    {
        expectedPressureForce[d] = rho * volume * b[d];
        expectedViscousForce[d]  = -mu * volume * divS[d];
    }
    std::array<double, 3> expectedPressureMoment = Cross(moment, b);
    std::array<double, 3> expectedViscousMoment  = Cross(moment, divS);
    for (unsigned int d = 0; d < 3; ++d)
    {
        expectedPressureMoment[d] *= rho;
        expectedViscousMoment[d] *= -mu;
    }

    const std::vector<std::string> names =
        nD == 2 ? std::vector<std::string>{"u", "v"}
                : std::vector<std::string>{"u", "v", "w"};
    std::vector<unsigned int> velocityComponents(nD);
    std::iota(velocityComponents.begin(), velocityComponents.end(), 0u);
    const std::vector<bool> selected(explist->GetBndCondExpansions().size(),
                                     true);
    auto op = AeroForcesOp<double>::Create(explist, names, execStr);
    op->Configure(velocityComponents, explist, 0, selected, rho, mu, pivot);

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    for (const unsigned int pressureInterleaveWidth : {1u, implInterleaveWidth})
    {
        for (const unsigned int gradientInterleaveWidth :
             {1u, implInterleaveWidth})
        {
            BOOST_TEST_CONTEXT("pressure interleave="
                               << pressureInterleaveWidth
                               << " gradient interleave="
                               << gradientInterleaveWidth)
            {
                LibUtilities::Field<double, FieldState::Phys> pressureField(
                    "p", blockAttr, 1, 1);
                LibUtilities::Field<double, FieldState::Phys> gradientField(
                    "gradient", blockAttr, nD * nD, 1);
                pressureField.template CopyArray<NektarSpaces::HostSpace>(
                    pressure);
                gradientField.template CopyArray<NektarSpaces::HostSpace>(
                    gradient);
                pressureField.ReshapeStorage(pressureInterleaveWidth, execStr);
                gradientField.ReshapeStorage(gradientInterleaveWidth, execStr);

                (*op)(pressureField, gradientField);

                const auto check = [](const std::array<double, 3> &actual,
                                      const std::array<double, 3> &expected) {
                    for (unsigned int d = 0; d < 3; ++d)
                    {
                        BOOST_TEST_CONTEXT("component " << d)
                        {
                            BOOST_CHECK_SMALL(
                                actual[d] - expected[d],
                                1.0e-10 * (1.0 + std::abs(expected[d])));
                        }
                    }
                };
                check(op->GetPressureForce(), expectedPressureForce);
                check(op->GetViscousForce(), expectedViscousForce);
                check(op->GetPressureMoment(), expectedPressureMoment);
                check(op->GetViscousMoment(), expectedViscousMoment);
            }
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteAeroForces)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(AeroForces_ClosedBoundary)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml : {"run/TriQuad_varP_Dir.xml", "run/Hex_Dir.xml",
                                  "run/cube_all_elements_dir.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunClosedBoundaryTest(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
