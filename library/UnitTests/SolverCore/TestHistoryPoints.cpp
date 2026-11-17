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
#include <cmath>

#include <MultiRegions/ExpList.h>
#include <MultiRegions/ExpList3DHomogeneous1D.h>
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

// Homogeneous length of the 3DH1 case, other than 1 so that it shows.
static constexpr double homoLen = 2.0;

// A 3DH1 expansion list with nHomo planes over a 2D mesh.
static MultiRegions::ExpListSharedPtr SetExpList3DH1(
    const LibUtilities::SessionReaderSharedPtr &session,
    const unsigned int nHomo)
{
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    const LibUtilities::PointsKey pkey(nHomo,
                                       LibUtilities::eFourierEvenlySpaced);
    const LibUtilities::BasisKey bkey(LibUtilities::eFourier, nHomo, pkey);
    auto explist =
        MemoryManager<MultiRegions::ExpList3DHomogeneous1D>::AllocateSharedPtr(
            session, bkey, homoLen, false, false, graph, "u",
            Collections::eNoCollection);
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
// laying the field out with and without interleaving and collecting the
// values on every rank and on rank zero alone.
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
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            explist, MultiRegions::UnitInterleaveWidths(explist));
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    for (const bool rootOnly : {false, true})
    {
        for (const unsigned int interleaveWidth : {1u, implInterleaveWidth})
        {
            BOOST_TEST_CONTEXT("rootOnly=" << rootOnly
                                           << " interleave=" << interleaveWidth)
            {
                LibUtilities::Field<double, FieldState::Phys> field(
                    "u", blockAttr, nComp, 1);
                field.template CopyArray<NektarSpaces::HostSpace>(values);
                field.ReshapeStorage(interleaveWidth, execStr);

                op->SetRootOnly(rootOnly);
                op->Apply(field);

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
}

// A field polynomial in x and y and made of Fourier modes in z below the
// Nyquist mode of the 3DH1 case, which the planes interpolate exactly.
static double Periodic(const unsigned int comp, const double x, const double y,
                       const double z)
{
    const double theta = 2.0 * M_PI * z / homoLen;
    return Polynomial(comp, x, y, 0.0) * (1.0 + 0.3 * std::cos(theta)) +
           0.2 * (comp + 1) * x * std::sin(2.0 * theta);
}

// As RunPointsTest(), on a 3DH1 field with eight planes, at points between
// the planes, so that their value is interpolated in z from all of them.
static void RunPoints3DH1Test(const std::string &xml,
                              const std::string &execStr)
{
    const unsigned int nHomo = 8;
    auto session             = SetSession(xml, execStr);
    auto explist             = SetExpList3DH1(session, nHomo);
    auto plane               = explist->GetPlane(0);
    const unsigned int nD    = plane->GetCoordim(0);
    const size_t nPoints     = explist->GetTotPoints();
    const auto vars          = session->GetVariables();
    const auto nComp         = static_cast<unsigned int>(vars.size());

    // Flattened component-major, and plane-major within a component.
    Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
    explist->GetCoords(x, y, z);
    Array<OneD, double> values(nComp * nPoints, 0.0);
    for (unsigned int comp = 0; comp < nComp; ++comp)
    {
        for (size_t i = 0; i < nPoints; ++i)
        {
            values[comp * nPoints + i] = Periodic(comp, x[i], y[i], z[i]);
        }
    }

    // Two points in each of a spread of elements of the plane, as in
    // RunPointsTest(), each at a z between the planes.
    const size_t nPlanePoints = plane->GetTotPoints();
    Array<OneD, double> px(nPlanePoints, 0.0), py(nPlanePoints, 0.0),
        pz(nPlanePoints, 0.0);
    plane->GetCoords(px, py, pz);
    const std::array<Array<OneD, double>, 2> coords{px, py};
    const size_t nElmts = plane->GetExpSize();
    const size_t stride = std::max<size_t>(1, nElmts / 24);
    std::vector<std::array<double, 3>> points;
    for (size_t e = 0; e < nElmts; e += stride)
    {
        const size_t offset = plane->GetPhys_Offset(e);
        const size_t nq     = plane->GetExp(e)->GetTotPoints();
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
            point[2] = homoLen * (0.13 + 0.29 * (points.size() % 3));
            points.push_back(point);
        }
    }
    const size_t nHistoryPoints = points.size();

    auto op = HistoryPointsOp<double>::Create(explist, vars, execStr);
    op->Configure(points);

    const auto blockAttr =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            explist, MultiRegions::UnitInterleaveWidths(explist));
    const unsigned int implInterleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execStr);
    for (const bool rootOnly : {false, true})
    {
        for (const unsigned int interleaveWidth : {1u, implInterleaveWidth})
        {
            BOOST_TEST_CONTEXT("rootOnly=" << rootOnly
                                           << " interleave=" << interleaveWidth)
            {
                LibUtilities::Field<double, FieldState::Phys> field(
                    "u", blockAttr, nComp, nHomo);
                field.template CopyArray<NektarSpaces::HostSpace>(values);
                field.ReshapeStorage(interleaveWidth, execStr);

                op->SetRootOnly(rootOnly);
                op->Apply(field);

                const auto result = op->GetValues();
                BOOST_REQUIRE_EQUAL(result.size(), nHistoryPoints * nComp);
                for (size_t p = 0; p < nHistoryPoints; ++p)
                {
                    for (unsigned int comp = 0; comp < nComp; ++comp)
                    {
                        const double expected = Periodic(
                            comp, points[p][0], points[p][1], points[p][2]);
                        BOOST_CHECK_SMALL(result[p * nComp + comp] - expected,
                                          1.0e-10);
                    }
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

// 3DH1: a 2D mesh with eight planes, the points between them.
BOOST_AUTO_TEST_CASE(HistoryPoints_3DH1)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml :
         {"run/square.xml", "run/square_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunPoints3DH1Test(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
