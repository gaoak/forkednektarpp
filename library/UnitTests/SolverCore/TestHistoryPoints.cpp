///////////////////////////////////////////////////////////////////////////////
//
// File: TestHistoryPoints.cpp
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
// Description: Unit test for the SolverCore HistoryPointsOp.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestHistoryPoints

#include <algorithm>
#include <array>

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/HistoryPoints/HistoryPointsOp.hpp>

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
    argv[0]     = strdup("TestHistoryPointsOperator");
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

// A polynomial each element's quadrature interpolates exactly.
static double Polynomial(const unsigned int comp, const double x,
                         const double y, const double z)
{
    return 1.0 + comp + 0.5 * x - 0.25 * (comp + 1) * y + 0.1 * z +
           0.2 * comp * x * y;
}

// Evaluates a polynomial field at points in a spread of elements, two in each,
// laying the field out with and without interleaving.
static void RunPointsTest(const std::string &xml, const std::string &execStr)
{
    auto session          = SetSession(xml, execStr);
    auto explist          = SetExpList(session);
    const unsigned int nD = explist->GetCoordim(0);
    const size_t nPoints  = explist->GetTotPoints();
    const auto vars       = session->GetVariables();
    const auto nComp      = static_cast<unsigned int>(vars.size());

    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);

    Array<OneD, double> values(nComp * nPoints, 0.0);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        for (size_t i = 0; i < nPoints; ++i)
        {
            values[comp * nPoints + i] = Polynomial(comp, x[i], y[i], z[i]);
        }
    }

    // Two points in each of a spread of elements, each between the centroid
    // of the element's quadrature points and one of them, so that they lie
    // inside the element and away from its quadrature points.
    const std::array<Array<OneD, double>, 3> coords{x, y, z};
    const size_t nElmts = explist->GetExpSize();
    const size_t stride = std::max<size_t>(1, nElmts / 24);
    std::vector<std::array<double, 3>> points;
    for (size_t e = 0; e < nElmts; e += stride)
    {
        const size_t offset = explist->GetPhys_Offset(e);
        const size_t nq     = explist->GetExp(e)->GetTotPoints();
        std::array<double, 3> centroid{};
        for (unsigned int d = 0; d < nD; ++d)
        {
            for (size_t q = 0; q < nq; ++q)
            {
                centroid[d] += coords[d][offset + q] / nq;
            }
        }
        for (const size_t q : {e % nq, (3 * e + nq / 2) % nq})
        {
            std::array<double, 3> point{};
            for (unsigned int d = 0; d < nD; ++d)
            {
                point[d] = 0.7 * centroid[d] + 0.3 * coords[d][offset + q];
            }
            points.push_back(point);
        }
    }
    const size_t nHistoryPoints = points.size();

    auto op = HistoryPointsOp<double>::Create(explist, vars, execStr);
    op->Configure(points);

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(explist);
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    for (const unsigned int interleaveWidth : {1u, implInterleaveWidth})
    {
        BOOST_TEST_CONTEXT("interleave=" << interleaveWidth)
        {
            LibUtilities::Field<double, FieldState::Phys> field("u", blockAttr,
                                                                nComp, 1);
            field.template CopyArray<NektarSpaces::HostSpace>(values);
            field.ReshapeStorage(interleaveWidth, execStr);

            (*op)(field);

            const auto result = op->GetValues();
            BOOST_REQUIRE_EQUAL(result.size(), nHistoryPoints * nComp);
            for (size_t p = 0; p < nHistoryPoints; ++p)
            {
                for (unsigned int comp = 0; comp < nComp; ++comp)
                {
                    const double expected = Polynomial(
                        comp, points[p][0], points[p][1], points[p][2]);
                    BOOST_CHECK_SMALL(result[p * nComp + comp] - expected,
                                      1.0e-10);
                }
            }
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteHistoryPoints)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(HistoryPoints_Polynomial)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml :
         {"run/square.xml", "run/hex.xml", "run/square_all_elements.xml",
          "run/cube_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunPointsTest(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
