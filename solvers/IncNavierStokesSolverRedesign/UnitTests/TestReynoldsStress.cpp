///////////////////////////////////////////////////////////////////////////////
//
// File: TestReynoldsStress.cpp
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
// Description: Unit test for the ReynoldsStressOp.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestReynoldsStress

#include <cmath>
#include <functional>
#include <vector>

#include <MultiRegions/ExpList.h>
#include <MultiRegions/ExpList3DHomogeneous1D.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <IncNavierStokesSolverRedesign/Utilities/ReynoldsStress/ReynoldsStressOp.hpp>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

using namespace Nektar;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
static LibUtilities::SessionReaderSharedPtr SetSession(const std::string &xml,
                                                       const std::string &exec)
{
    int argc    = 3;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestReynoldsStressOperator");
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

// Homogeneous length of the 3DH1 cases, other than 1 so that it shows.
static constexpr double homoLen = 2.0;

// A 2D or 3D expansion list without a homogeneous direction, or a 3DH1 one
// with nHomo planes.
static MultiRegions::ExpListSharedPtr SetExpList(
    const LibUtilities::SessionReaderSharedPtr &session,
    const unsigned int nHomo = 1)
{
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    MultiRegions::ExpListSharedPtr explist;
    if (nHomo == 1)
    {
        explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            session, graph, true, "u", Collections::eNoCollection);
    }
    else
    {
        const LibUtilities::PointsKey pkey(nHomo,
                                           LibUtilities::eFourierEvenlySpaced);
        const LibUtilities::BasisKey bkey(LibUtilities::eFourier, nHomo, pkey);
        explist = MemoryManager<MultiRegions::ExpList3DHomogeneous1D>::
            AllocateSharedPtr(session, bkey, homoLen, false, false, graph, "u",
                              Collections::eNoCollection);
    }
    explist->SetDataWarehouse();
    return explist;
}

// Evaluates one velocity component at a point.
using ComponentFunction =
    std::function<double(unsigned int, double, double, double)>;

// Applies ReynoldsStressOp to a field holding a passive scalar followed by
// the velocity, first with unit scale and then with scale 0.5 on the same
// product field, and checks that every product holds 1.5 u_a u_b and that
// the velocity is unchanged. The velocity and the products are laid out with
// and without interleaving, at the same width or at different widths on
// entry, and the products leave at the width of the velocity. A 3DH1 case,
// with nHomo planes, is 3D, the fields flattened plane-major within a
// component.
static void CheckProducts(const std::string &xml, const std::string &execStr,
                          const ComponentFunction &function,
                          const unsigned int nHomo = 1)
{
    const double tol      = 1.0e-12;
    auto session          = SetSession(xml, execStr);
    auto explist          = SetExpList(session, nHomo);
    const unsigned int nD = nHomo == 1 ? explist->GetCoordim(0) : 3;
    const size_t nPoints  = explist->GetTotPoints();

    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);

    // Component 0 is a passive scalar, the velocity follows.
    const unsigned int nComp = nD + 1;
    Array<OneD, double> values(nComp * nPoints, 0.0);
    for (size_t i = 0; i < nPoints; ++i)
    {
        values[i] = 7.0;
        for (unsigned int a = 0; a < nD; ++a)
        {
            values[(a + 1) * nPoints + i] = function(a, x[i], y[i], z[i]);
        }
    }

    const std::vector<std::string> allNames = {"s", "u", "v", "w"};
    const std::vector<std::string> names(allNames.begin(),
                                         allNames.begin() + nComp);
    const std::vector<std::string> velocityNames(names.begin() + 1,
                                                 names.end());
    std::vector<unsigned int> velocityComponents(nD);
    for (unsigned int a = 0; a < nD; ++a)
    {
        velocityComponents[a] = a + 1;
    }
    const auto productNames =
        ReynoldsStressOp<double>::ProductNames(velocityNames);
    BOOST_REQUIRE_EQUAL(productNames.size(), nD * (nD + 1) / 2);

    auto op = ReynoldsStressOp<double>::Create(explist, names, execStr);
    op->Configure(velocityComponents);

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    std::vector<unsigned int> interleaveWidths = {1u};
    if (implInterleaveWidth != 1u)
    {
        interleaveWidths.push_back(implInterleaveWidth);
    }

    for (const unsigned int velocityWidth : interleaveWidths)
    {
        for (const unsigned int productWidth : interleaveWidths)
        {
            BOOST_TEST_CONTEXT("velocity interleave=" << velocityWidth
                                                      << " product interleave="
                                                      << productWidth)
            {
                LibUtilities::Field<double, FieldState::Phys> velocity(
                    "velocity", blockAttr, nComp, nHomo);
                velocity.template CopyArray<NektarSpaces::HostSpace>(values);
                velocity.ReshapeStorage(velocityWidth, execStr);

                LibUtilities::Field<double, FieldState::Phys> products(
                    "products", blockAttr, productNames, nHomo);
                products.template Initialize<NektarSpaces::HostSpace>(0.0);
                products.ReshapeStorage(productWidth, execStr);

                op->SetScale(1.0);
                op->Apply(velocity, products);
                op->SetScale(0.5);
                op->Apply(velocity, products);

                for (const auto &block : products.GetBlocks())
                {
                    BOOST_CHECK_EQUAL(block.GetInterleaveWidth(),
                                      velocityWidth);
                }

                products.ReshapeStorage(1, execStr);
                const auto result = products.template ToArray<double>();
                BOOST_REQUIRE_EQUAL(result.size(),
                                    productNames.size() * nPoints);
                unsigned int product = 0;
                for (unsigned int a = 0; a < nD; ++a)
                {
                    for (unsigned int b = a; b < nD; ++b, ++product)
                    {
                        for (size_t i = 0; i < nPoints; ++i)
                        {
                            const double expected =
                                1.5 * values[(a + 1) * nPoints + i] *
                                values[(b + 1) * nPoints + i];
                            BOOST_CHECK_SMALL(
                                result[product * nPoints + i] - expected, tol);
                        }
                    }
                }

                velocity.ReshapeStorage(1, execStr);
                const auto unchanged = velocity.template ToArray<double>();
                for (size_t i = 0; i < values.size(); ++i)
                {
                    BOOST_CHECK_EQUAL(unchanged[i], values[i]);
                }
            }
        }
    }
}

// A constant velocity u=2, v=3, w=0 gives uu=4, uv=6, uw=0, vv=9, vw=0,
// ww=0 at every point.
static void RunConstantTest(const std::string &execStr)
{
    auto session         = SetSession("run/hex.xml", execStr);
    auto explist         = SetExpList(session);
    const size_t nPoints = explist->GetTotPoints();

    Array<OneD, double> values(3 * nPoints, 0.0);
    for (size_t i = 0; i < nPoints; ++i)
    {
        values[i]           = 2.0;
        values[nPoints + i] = 3.0;
    }

    const std::vector<std::string> names = {"u", "v", "w"};
    const auto productNames = ReynoldsStressOp<double>::ProductNames(names);
    const std::vector<std::string> expectedNames = {"uu", "uv", "uw",
                                                    "vv", "vw", "ww"};
    BOOST_CHECK(productNames == expectedNames);

    auto op = ReynoldsStressOp<double>::Create(explist, names, execStr);
    op->Configure({0, 1, 2});

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    LibUtilities::Field<double, FieldState::Phys> velocity("velocity",
                                                           blockAttr, 3, 1);
    velocity.template CopyArray<NektarSpaces::HostSpace>(values);
    LibUtilities::Field<double, FieldState::Phys> products(
        "products", blockAttr, productNames, 1);
    products.template Initialize<NektarSpaces::HostSpace>(0.0);

    op->Apply(velocity, products);

    const auto result                  = products.template ToArray<double>();
    const std::vector<double> expected = {4.0, 6.0, 0.0, 9.0, 0.0, 0.0};
    for (size_t product = 0; product < expected.size(); ++product)
    {
        for (size_t i = 0; i < nPoints; ++i)
        {
            BOOST_CHECK_EQUAL(result[product * nPoints + i], expected[product]);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteReynoldsStress)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(ReynoldsStress_Constant)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_TEST_CONTEXT("exec=" << execStr)
    {
        RunConstantTest(execStr);
    }
}

BOOST_AUTO_TEST_CASE(ReynoldsStress_Varying3D)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    const ComponentFunction velocity = [](unsigned int a, double x, double y,
                                          double z) {
        switch (a)
        {
            case 0:
                return 1.0 + x - 2.0 * y * z;
            case 1:
                return std::sin(2.0 * x) + 0.5 * z;
            default:
                return -0.3 + x * y + std::cos(z);
        }
    };
    for (const std::string xml : {"run/hex.xml", "run/cube_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            CheckProducts(xml, execStr, velocity);
        }
    }
}

BOOST_AUTO_TEST_CASE(ReynoldsStress_Varying2D)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    const ComponentFunction velocity = [](unsigned int a, double x, double y,
                                          double) {
        return a == 0 ? 0.25 + x * y : std::exp(-x) - y;
    };
    for (const std::string xml :
         {"run/square.xml", "run/square_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            CheckProducts(xml, execStr, velocity);
        }
    }
}

// 3DH1: a 2D mesh with four planes, the velocity varying in z.
BOOST_AUTO_TEST_CASE(ReynoldsStress_3DH1)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    const ComponentFunction velocity = [](unsigned int a, double x, double y,
                                          double z) {
        switch (a)
        {
            case 0:
                return 1.0 + x - 2.0 * y * z;
            case 1:
                return std::sin(2.0 * x) + 0.5 * z;
            default:
                return -0.3 + x * y + std::cos(z);
        }
    };
    for (const std::string xml :
         {"run/square.xml", "run/square_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            CheckProducts(xml, execStr, velocity, 4);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
