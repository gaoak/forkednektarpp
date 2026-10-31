///////////////////////////////////////////////////////////////////////////////
//
// File: TestMean.cpp
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
// Description: Unit test for the SolverCore MeanOp.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestMean

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/Mean/MeanOp.hpp>

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
    argv[0]     = strdup("TestMeanOperator");
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

// Evaluates one analytic function per component at the quadrature points,
// flattened component-major.
using ComponentFunction =
    std::function<double(unsigned int, double, double, double)>;

static Array<OneD, double> FillField(
    const MultiRegions::ExpListSharedPtr &explist, const unsigned int nComp,
    const ComponentFunction &function)
{
    const size_t nPoints = explist->GetTotPoints();
    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);

    Array<OneD, double> flat(nComp * nPoints, 0.0);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        for (size_t i = 0; i < nPoints; ++i)
        {
            flat[comp * nPoints + i] = function(comp, x[i], y[i], z[i]);
        }
    }
    return flat;
}

// Applies MeanOp to the given values with non-interleaved and with
// implementation-width interleaved storage, checking the domain volume, the
// per-component averages and that the field values are left unchanged.
static void CheckMeans(const MultiRegions::ExpListSharedPtr &explist,
                       const std::string &execStr,
                       const std::vector<std::string> &variables,
                       const Array<OneD, double> &values,
                       const std::vector<double> &expectedMeans,
                       const double expectedVolume)
{
    const double tol = 1.0e-12;
    const auto nComp = static_cast<unsigned int>(variables.size());
    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);

    auto op = MeanOp<double>::Create(explist, variables, execStr);

    for (const unsigned int interleaveWidth : {1u, implInterleaveWidth})
    {
        BOOST_TEST_CONTEXT("interleave=" << interleaveWidth)
        {
            LibUtilities::Field<double, FieldState::Phys> field("u", blockAttr,
                                                                nComp, 1);
            field.template CopyArray<NektarSpaces::HostSpace>(values);
            field.ReshapeStorage(interleaveWidth, execStr);

            (*op)(field);

            BOOST_CHECK_CLOSE(op->GetVolume(), expectedVolume, 1.0e-10);
            const auto means = op->GetMeans();
            BOOST_REQUIRE_EQUAL(means.size(), nComp);
            for (unsigned int comp = 0; comp < nComp; ++comp)
            {
                BOOST_CHECK_SMALL(means[comp] - expectedMeans[comp], tol);
            }

            field.ReshapeStorage(1, execStr);
            const auto fieldValues = field.template ToArray<double>();
            for (size_t i = 0; i < values.size(); ++i)
            {
                BOOST_CHECK_EQUAL(fieldValues[i], values[i]);
            }
        }
    }
}

// A constant field averages to that constant on any domain.
static void RunConstantTest(const std::string &xml, const std::string &execStr)
{
    auto session     = SetSession(xml, execStr);
    auto explist     = SetExpList(session);
    const auto vars  = session->GetVariables();
    const auto nComp = static_cast<unsigned int>(vars.size());

    Array<OneD, double> ones(explist->GetTotPoints(), 1.0);
    const double volume = explist->Integral(ones);
    BOOST_REQUIRE(volume > 0.0);

    std::vector<double> expected(nComp);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        expected[comp] = 1.5 - 2.25 * comp;
    }
    const auto values = FillField(explist, nComp,
                                  [&](unsigned int comp, double, double,
                                      double) { return expected[comp]; });

    CheckMeans(explist, execStr, vars, values, expected, volume);
}

// A linear field on the unit square centred at the origin averages to its
// constant term, and the square has unit area.
static void RunLinearTest(const std::string &execStr)
{
    auto session     = SetSession("run/square.xml", execStr);
    auto explist     = SetExpList(session);
    const auto vars  = session->GetVariables();
    const auto nComp = static_cast<unsigned int>(vars.size());

    std::vector<double> expected(nComp);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        expected[comp] = 0.25 + comp;
    }
    const auto values = FillField(
        explist, nComp, [&](unsigned int comp, double x, double y, double) {
            return expected[comp] + (1.0 + comp) * x - 0.5 * y;
        });

    CheckMeans(explist, execStr, vars, values, expected, 1.0);
}

// A general polynomial field is compared with the expansion list's own
// quadrature, on a mesh with several element types and orders.
static void RunIntegralTest(const std::string &xml, const std::string &execStr)
{
    auto session         = SetSession(xml, execStr);
    auto explist         = SetExpList(session);
    const auto vars      = session->GetVariables();
    const auto nComp     = static_cast<unsigned int>(vars.size());
    const size_t nPoints = explist->GetTotPoints();

    const auto values = FillField(
        explist, nComp, [](unsigned int comp, double x, double y, double z) {
            const double s = 1.0 + comp;
            return 0.7 * s + 0.3 * s * x + 0.15 * (comp + 2) * y * y -
                   0.05 * s * z + 0.02 * s * x * y * z;
        });

    Array<OneD, double> ones(nPoints, 1.0);
    const double volume = explist->Integral(ones);
    BOOST_REQUIRE(volume > 0.0);

    std::vector<double> expected(nComp);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        Array<OneD, double> component(nPoints);
        for (size_t i = 0; i < nPoints; ++i)
        {
            component[i] = values[comp * nPoints + i];
        }
        expected[comp] = explist->Integral(component) / volume;
    }

    CheckMeans(explist, execStr, vars, values, expected, volume);
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteMean)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(Mean_Constant)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml : {"run/square.xml", "run/hex.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunConstantTest(xml, execStr);
        }
    }
}

BOOST_AUTO_TEST_CASE(Mean_Linear)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_TEST_CONTEXT("exec=" << execStr)
    {
        RunLinearTest(execStr);
    }
}

BOOST_AUTO_TEST_CASE(Mean_Integral)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml :
         {"run/square.xml", "run/hex.xml", "run/square_all_elements.xml",
          "run/cube_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunIntegralTest(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
