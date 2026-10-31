///////////////////////////////////////////////////////////////////////////////
//
// File: TestEnergy.cpp
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
// Description: Unit test for the SolverCore EnergyOp.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestEnergy

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/Energy/EnergyOp.hpp>

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
    argv[0]     = strdup("TestEnergyOperator");
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
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
        session, graph, true, "u", Collections::eNoCollection);
    explist->SetDataWarehouse();
    return explist;
}

// Integrates the energy of a polynomial velocity, with a gradient given
// independently of it, laying the velocity and its gradient out with and
// without interleaving, and compares with the expansion list's own
// quadrature. In 2D the velocity is the last two of three components, in
// reverse order, so that the component selection is exercised.
static void RunEnergyTest(const std::string &xml, const std::string &execStr)
{
    auto session          = SetSession(xml, execStr);
    auto explist          = SetExpList(session);
    const unsigned int nD = explist->GetCoordim(0);
    const size_t nPoints  = explist->GetTotPoints();

    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);

    const unsigned int nComp = 3;
    const std::vector<unsigned int> velocityComponents =
        nD == 2 ? std::vector<unsigned int>{2, 1}
                : std::vector<unsigned int>{0, 1, 2};

    // Velocity component c and gradient component c * nD + d, the latter
    // standing for the derivative of component c in direction d.
    Array<OneD, double> velocity(nComp * nPoints, 0.0);
    Array<OneD, double> gradient(nComp * nD * nPoints, 0.0);
    for (size_t i = 0; i < nPoints; ++i)
    {
        const double xi[3] = {x[i], y[i], z[i]};
        for (unsigned int c = 0; c < nComp; ++c)
        {
            velocity[c * nPoints + i] =
                0.5 + c - 0.3 * xi[0] + 0.2 * (c + 1) * xi[1] + 0.1 * xi[2];
            for (unsigned int d = 0; d < nD; ++d)
            {
                gradient[(c * nD + d) * nPoints + i] =
                    0.25 * (c + 1) - 0.5 * d + 0.4 * xi[(c + d) % nD] -
                    0.15 * c * xi[d];
            }
        }
    }

    // Reference integrands, from the selected components.
    Array<OneD, double> velocitySquared(nPoints, 0.0);
    Array<OneD, double> vorticitySquared(nPoints, 0.0);
    for (size_t i = 0; i < nPoints; ++i)
    {
        auto derivative = [&](const unsigned int c, const unsigned int d) {
            return gradient[(velocityComponents[c] * nD + d) * nPoints + i];
        };
        for (unsigned int c = 0; c < nD; ++c)
        {
            const double u = velocity[velocityComponents[c] * nPoints + i];
            velocitySquared[i] += u * u;
        }
        if (nD == 2)
        {
            const double curl   = derivative(1, 0) - derivative(0, 1);
            vorticitySquared[i] = curl * curl;
        }
        else
        {
            const double curlX  = derivative(2, 1) - derivative(1, 2);
            const double curlY  = derivative(0, 2) - derivative(2, 0);
            const double curlZ  = derivative(1, 0) - derivative(0, 1);
            vorticitySquared[i] = curlX * curlX + curlY * curlY + curlZ * curlZ;
        }
    }
    Array<OneD, double> ones(nPoints, 1.0);
    const double expectedVolume    = explist->Integral(ones);
    const double expectedKinetic   = explist->Integral(velocitySquared);
    const double expectedEnstrophy = explist->Integral(vorticitySquared);
    BOOST_REQUIRE(expectedVolume > 0.0);

    std::vector<std::string> names(nComp);
    for (unsigned int c = 0; c < nComp; ++c)
    {
        names[c] = "component" + std::to_string(c);
    }
    auto op = EnergyOp<double>::Create(explist, names, execStr);
    op->Configure(velocityComponents);

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    for (const unsigned int interleaveWidth : {1u, implInterleaveWidth})
    {
        BOOST_TEST_CONTEXT("interleave=" << interleaveWidth)
        {
            LibUtilities::Field<double, FieldState::Phys> velocityField(
                "velocity", blockAttr, nComp, 1);
            LibUtilities::Field<double, FieldState::Phys> gradientField(
                "gradient", blockAttr, nComp * nD, 1);
            velocityField.template CopyArray<NektarSpaces::HostSpace>(velocity);
            gradientField.template CopyArray<NektarSpaces::HostSpace>(gradient);
            velocityField.ReshapeStorage(interleaveWidth, execStr);
            gradientField.ReshapeStorage(interleaveWidth, execStr);

            (*op)(velocityField, gradientField);

            const double tol = 1.0e-10;
            BOOST_CHECK_SMALL(op->GetVolume() - expectedVolume,
                              tol * (1.0 + expectedVolume));
            BOOST_CHECK_SMALL(op->GetKineticEnergy() - expectedKinetic,
                              tol * (1.0 + expectedKinetic));
            BOOST_CHECK_SMALL(op->GetEnstrophy() - expectedEnstrophy,
                              tol * (1.0 + expectedEnstrophy));
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteEnergy)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(Energy_Integral)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml :
         {"run/square.xml", "run/hex.xml", "run/square_all_elements.xml",
          "run/cube_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunEnergyTest(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
