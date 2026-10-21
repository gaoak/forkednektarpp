///////////////////////////////////////////////////////////////////////////////
//
// File: TestHOutflowBndCond.cpp
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
// Description: Unit tests of the high-order outflow boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestHOutflowBndCond

// Boost.Test pulls in windows.h, whose min/max macros break tinysimd.
#define NOMINMAX
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOutflowBndCond/HOutflowBndCondKernels.hpp>
#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOutflowBndCond/HOutflowBndCondOp.hpp>
#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>

using namespace Nektar;
using namespace Nektar::detail;

struct GlobalConfiguration
{
    GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;
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

BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);

namespace
{
template <unsigned int DIM>
void CheckPointFormula(const std::array<double, DIM> &u,
                       const std::array<double, DIM * DIM> &gradient,
                       const std::array<double, DIM> &normal,
                       const std::array<double, DIM> &fb, const double kinvis,
                       const double theta, const double alpha1,
                       const double alpha2, const double U0, const double delta)
{
    double pressure = 0.0, divergence = 0.0;
    std::array<double, DIM> energyMinusFb{}, velocityFlux{};

    HOutflowPressureKernel<NektarSpaces::Serial, DIM>(
        1, 1, u.data(), gradient.data(), normal.data(), fb.data(), kinvis, 1.0,
        theta, alpha1, alpha2, U0, delta, &pressure, energyMinusFb.data(),
        &divergence, 0);

    double un = 0.0, u2 = 0.0, nGradUn = 0.0, expectedDiv = 0.0;
    for (unsigned int d = 0; d < DIM; ++d)
    {
        un += normal[d] * u[d];
        u2 += u[d] * u[d];
        expectedDiv += gradient[d * DIM + d];
        for (unsigned int j = 0; j < DIM; ++j)
        {
            nGradUn += normal[d] * gradient[d * DIM + j] * normal[j];
        }
    }
    const double S0     = 0.5 * (1.0 - std::tanh(un / (U0 * delta)));
    const double k1     = 0.5 * (theta + alpha2);
    const double k2     = 0.5 * (1.0 - theta + alpha1);
    double energyNormal = 0.0;
    for (unsigned int d = 0; d < DIM; ++d)
    {
        const double expectedEnergy =
            (k1 * u2 * normal[d] + k2 * un * u[d]) * S0 - fb[d];
        BOOST_TEST(energyMinusFb[d] == expectedEnergy,
                   boost::test_tools::tolerance(1.0e-13));
        energyNormal += normal[d] * expectedEnergy;
    }
    const double expectedPressure = kinvis * nGradUn - energyNormal;
    BOOST_TEST(divergence == expectedDiv,
               boost::test_tools::tolerance(1.0e-13));
    BOOST_TEST(pressure == expectedPressure,
               boost::test_tools::tolerance(1.0e-13));

    HOutflowVelocityKernel<NektarSpaces::Serial, DIM>(
        1, 1, &pressure, normal.data(), energyMinusFb.data(), &divergence,
        kinvis, velocityFlux.data(), 0);
    for (unsigned int d = 0; d < DIM; ++d)
    {
        const double expected =
            ((pressure - kinvis * expectedDiv) * normal[d] + energyMinusFb[d]) /
            kinvis;
        BOOST_TEST(velocityFlux[d] == expected,
                   boost::test_tools::tolerance(1.0e-13));
    }
}

// Whether a boundary condition is part of the high-order outflow condition.
bool IsHOutflow(const SpatialDomains::BoundaryConditionShPtr &condition)
{
    return StringUtils::IEquals(condition->GetUserDefined(), "HOutflow");
}

// Load the session given after `--` on the command line, as
// `-- <ExecSpace> <Implementation> <Session.xml>`.
LibUtilities::SessionReaderSharedPtr CreateSession()
{
    const auto &suite = boost::unit_test::framework::master_test_suite();

    const std::string execStr(suite.argv[1]);
    const std::string implStr(suite.argv[2]);
    const std::string meshName(suite.argv[3]);

    int argc    = 4;
    char **argv = new char *[argc + 1];
    argv[0]     = strdup("TestHOutflowBndCond");
    argv[1]     = strdup(meshName.c_str());
    argv[2]     = strdup(("--opExecSpace=" + execStr).c_str());
    argv[3]     = strdup(("--opImpl=" + implStr).c_str());
    argv[argc]  = nullptr;

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    session->InitSession();

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}

// Set @p reference to interleave width 1 and @p interleaved to @p
// interleaveWidth, and fill both with the same non-zero values, which are
// returned in the layout of ToVector().
std::vector<double> FillAtTwoWidths(
    LibUtilities::Field<double, FieldState::Coeff> &reference,
    LibUtilities::Field<double, FieldState::Coeff> &interleaved,
    const unsigned int interleaveWidth)
{
    reference.Initialize<NektarSpaces::HostSpace>(0.0);
    interleaved.Initialize<NektarSpaces::HostSpace>(0.0);
    for (auto &block : interleaved.GetBlocks())
    {
        // The storage is zero, so it need not be reshaped.
        block.SetInterleaveWidth<double>(interleaveWidth);
    }

    std::vector<double> initialValues(reference.ToVector<double>().size());
    for (std::size_t i = 0; i < initialValues.size(); ++i)
    {
        initialValues[i] = 1.0 + 1.0e-3 * i;
    }
    reference.CopyVector<NektarSpaces::HostSpace>(initialValues);
    interleaved.CopyVector<NektarSpaces::HostSpace>(initialValues);
    return initialValues;
}

// Check that @p interleaved is still at @p interleaveWidth and holds the values
// of
// @p reference to within @p tolerance, and that the operator changed them
// from @p initialValues.
void CheckAtTwoWidths(
    LibUtilities::Field<double, FieldState::Coeff> &reference,
    LibUtilities::Field<double, FieldState::Coeff> &interleaved,
    const unsigned int interleaveWidth,
    const std::vector<double> &initialValues, const double tolerance)
{
    for (auto &block : interleaved.GetBlocks())
    {
        BOOST_TEST(block.GetInterleaveWidth() == interleaveWidth);
    }

    const auto referenceValues   = reference.ToVector<double>();
    const auto interleavedValues = interleaved.ToVector<double>();
    BOOST_REQUIRE_EQUAL(interleavedValues.size(), referenceValues.size());
    double maxChange = 0.0;
    for (std::size_t i = 0; i < referenceValues.size(); ++i)
    {
        maxChange = std::max(maxChange,
                             std::abs(referenceValues[i] - initialValues[i]));
        BOOST_TEST(std::abs(interleavedValues[i] - referenceValues[i]) <=
                   tolerance);
    }
    BOOST_TEST(maxChange > 1.0e-2);
}
} // namespace

BOOST_AUTO_TEST_SUITE(TestHOutflowBndCond)

BOOST_AUTO_TEST_CASE(outgoing_2d_default_parameters_nonzero_fb)
{
    CheckPointFormula<2>({1.25, -0.2}, {0.3, -0.4, 0.1, -0.15}, {1.0, 0.0},
                         {0.17, -0.08}, 0.025, 1.0, 0.0, 0.0, 1.0, 1.0 / 20.0);
}

BOOST_AUTO_TEST_CASE(backflow_2d_nondefault_energy_parameters)
{
    CheckPointFormula<2>({-0.6, 0.7}, {-0.2, 0.35, -0.45, 0.25}, {1.0, 0.0},
                         {-0.11, 0.09}, 0.04, 0.35, 0.4, 0.2, 1.7, 0.12);
}

BOOST_AUTO_TEST_CASE(backflow_3d_divergence_term)
{
    CheckPointFormula<3>({-0.35, 0.4, -0.15},
                         {0.2, -0.1, 0.3, -0.25, 0.15, 0.05, 0.4, -0.2, -0.3},
                         {0.0, 1.0, 0.0}, {0.03, -0.07, 0.12}, 0.015, 0.8, 0.1,
                         0.25, 0.9, 0.08);
}

// The pressure condition on a boundary region of two elements, through the
// operator: a linear velocity field, whose gradient is exact, is extracted
// onto the region and the Dirichlet pressure coefficients are compared with
// the Dong formula evaluated from the analytic velocity at the boundary
// points. Unlike the tests above, this checks the extraction and the block
// layout of the boundary data, with the velocity at interleave width 1 and at
// the interleave width of the execution space.
BOOST_AUTO_TEST_CASE(build_boundary_element_data_manufactured_multi_element)
{
    const auto &suite = boost::unit_test::framework::master_test_suite();
    if (suite.argc < 4)
    {
        // Run without a session, for the tests above only.
        return;
    }

    auto session = CreateSession();
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const std::vector<std::string> velocityComponents = {"u", "v"};
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);

    // A linear velocity field, whose gradient PhysDerivOp reproduces to
    // rounding error.
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "manufactured velocity", velocityPhysBlocks, velocityComponents, 1);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < velocity.GetBlocks().size(); ++blk)
    {
        auto &velocityBlock = velocity.GetBlocks()[blk];
        auto velocityPtr =
            velocityBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto collection        = GetCollection(velocityExpList, blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            Array<OneD, double> x(velocityBlock.GetNumData());
            Array<OneD, double> y(velocityBlock.GetNumData());
            Array<OneD, double> z(velocityBlock.GetNumData(), 0.0);
            expansions[e]->GetCoords(x, y, z);
            for (unsigned int p = 0; p < velocityBlock.GetNumData(); ++p)
            {
                const auto id   = e * velocityBlock.GetNumData() + p;
                velocityPtr[id] = 2.0 * x[p] + 3.0 * y[p] + 1.0;
                velocityPtr[velocityBlock.CompSize() + id] =
                    -x[p] + 4.0 * y[p] - 2.0;
            }
        }
    }

    constexpr double kinvis = 0.3;
    constexpr double scale  = 1.0;
    constexpr double fb0 = 0.1, fb1 = -0.2;

    auto op =
        HOutflowBndCondOp<double>::Create(pressureExpList, pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(kinvis);
    op->SetScale(scale);

    // The Dong pressure condition from the analytic velocity at the points
    // of the region, projected and mapped onto the pressure coefficients.
    const auto &bndExpansions = pressureExpList->GetBndCondExpansions();
    const auto &bndConditions = pressureExpList->GetBndConditions();
    const auto &localToGlobal = pressureExpList->GetLocalToGlobalMap();
    const auto &coeffMap  = localToGlobal->GetBndCondCoeffsToLocalCoeffsMap();
    const auto &coeffSign = localToGlobal->GetBndCondCoeffsToLocalCoeffsSign();
    const bool signChange = localToGlobal->GetSignChange();

    Array<OneD, double> expectedCoeffs(pressureExpList->GetNcoeffs(), 0.0);
    unsigned int bndCoeffOffset = 0;
    bool foundRegion            = false;
    for (unsigned int n = 0; n < bndExpansions.size(); ++n)
    {
        auto bndExp = bndExpansions[n];
        if (bndConditions[n]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet &&
            IsHOutflow(bndConditions[n]))
        {
            foundRegion     = true;
            const auto npts = bndExp->GetTotPoints();
            Array<OneD, double> x(npts), y(npts), z(npts, 0.0);
            bndExp->GetCoords(x, y, z);
            Array<OneD, Array<OneD, double>> normals;
            pressureExpList->GetBoundaryNormals(n, normals);

            Array<OneD, double> expectedPhys(npts, 0.0);
            for (unsigned int p = 0; p < npts; ++p)
            {
                const double ux         = 2.0 * x[p] + 3.0 * y[p] + 1.0;
                const double uy         = -x[p] + 4.0 * y[p] - 2.0;
                const double u[2]       = {ux, uy};
                const double grad[2][2] = {{2.0, 3.0}, {-1.0, 4.0}};
                const double nrm[2]     = {normals[0][p], normals[1][p]};
                const double fb[2]      = {fb0, fb1};

                double un = 0.0, u2 = 0.0, nGradUn = 0.0;
                for (unsigned int d = 0; d < 2; ++d)
                {
                    un += nrm[d] * u[d];
                    u2 += u[d] * u[d];
                    for (unsigned int j = 0; j < 2; ++j)
                    {
                        nGradUn += nrm[d] * grad[d][j] * nrm[j];
                    }
                }
                const double S0 = 0.5 * (1.0 - std::tanh(un / (1.0 * 0.05)));
                const double k1 = 0.5 * (1.0 + 0.0);
                const double k2 = 0.5 * (1.0 - 1.0 + 0.0);
                double energyNormal = 0.0;
                for (unsigned int d = 0; d < 2; ++d)
                {
                    const double value =
                        (k1 * u2 * nrm[d] + k2 * un * u[d]) * S0 - fb[d];
                    energyNormal += nrm[d] * value;
                }
                expectedPhys[p] = kinvis * nGradUn - energyNormal;
            }

            Array<OneD, double> bndCoeffsOut(bndExp->GetNcoeffs(), 0.0);
            bndExp->FwdTrans(expectedPhys, bndCoeffsOut);
            for (unsigned int c = 0; c < bndExp->GetNcoeffs(); ++c)
            {
                const auto bndCoeff   = bndCoeffOffset + c;
                const auto localCoeff = coeffMap[bndCoeff];
                expectedCoeffs[localCoeff] =
                    (signChange ? coeffSign[bndCoeff] : 1.0) * bndCoeffsOut[c];
            }
        }
        bndCoeffOffset += bndExp->GetNcoeffs();
    }
    BOOST_REQUIRE_MESSAGE(foundRegion,
                          "No HOutflow pressure Dirichlet region found in "
                          "the extraction test fixture.");

    LibUtilities::Field<double, FieldState::Coeff> expected(
        "expected pressure coeffs",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    expected.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < expected.GetBlocks().size(); ++blk)
    {
        auto &expectedBlock = expected.GetBlocks()[blk];
        auto expectedPtr =
            expectedBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto collection        = GetCollection(pressureExpList, blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            const auto offset =
                pressureExpList->GetCoeff_Offset(expansions[e]->GetElmtId());
            std::copy(expectedCoeffs.begin() + offset,
                      expectedCoeffs.begin() + offset +
                          expectedBlock.GetNumData(),
                      expectedPtr + e * expectedBlock.GetNumData());
        }
    }

    const auto expectedValues = expected.ToVector<double>();
    const auto maxExpected =
        *std::max_element(expectedValues.begin(), expectedValues.end(),
                          [](const double lhs, const double rhs) {
                              return std::abs(lhs) < std::abs(rhs);
                          });
    BOOST_TEST(std::abs(maxExpected) > 0.1);

    // The velocity at interleave width 1 and at the interleave width of the
    // execution space. The operator gathers it into fields at its width,
    // whose width changes from one update to the next.
    const auto execSpace =
        MultiRegions::Operator<double>::GetOpExecSpace(session);
    const unsigned int interleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execSpace);
    LibUtilities::Field<double, FieldState::Coeff> pressureCoeffs(
        "pressure coeffs",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    for (const unsigned int velocityWidth : {1u, interleaveWidth, 1u})
    {
        velocity.ReshapeStorage(velocityWidth, execSpace);
        op->UpdatePressure(velocity);
        pressureCoeffs.Initialize<NektarSpaces::HostSpace>(0.0);
        op->ApplyPressure(pressureCoeffs);

        const auto actualValues = pressureCoeffs.ToVector<double>();
        BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());
        for (std::size_t i = 0; i < actualValues.size(); ++i)
        {
            BOOST_CHECK_MESSAGE(std::abs(actualValues[i] - expectedValues[i]) <
                                    2.5e-10,
                                "velocity at width " << velocityWidth);
        }
    }
}

// The pressure coefficients set, and the velocity forcing added, in a field at
// the interleave width of the execution space match those in the same field at
// interleave width 1, and the fields keep their widths. Each field is applied
// to twice, the widths alternating. The velocity forcing follows the pressure
// as it stands when the velocity condition is applied.
BOOST_AUTO_TEST_CASE(apply_interleaved_matches_width_one)
{
    const auto &suite = boost::unit_test::framework::master_test_suite();
    if (suite.argc < 4)
    {
        // Run without a session, for the tests above only.
        return;
    }

    auto session = CreateSession();
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const std::vector<std::string> velocityComponents = {"u", "v"};
    const std::vector<std::string> pressureComponents = {"p"};

    // An outflow velocity, u . n = 2 on the regions, so that S0 = 0, and a
    // solved pressure varying along them.
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "outflow velocity",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList),
        velocityComponents, 1);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    for (auto &velocityBlock : velocity.GetBlocks())
    {
        auto velocityPtr =
            velocityBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        const size_t nPoints =
            velocityBlock.GetNumElements() * velocityBlock.GetNumData();
        for (unsigned int d = 0; d < velocityComponents.size(); ++d)
        {
            std::fill(velocityPtr + d * velocityBlock.CompSize(),
                      velocityPtr + d * velocityBlock.CompSize() + nPoints,
                      -2.0);
        }
    }
    LibUtilities::Field<double, FieldState::Phys> pressure(
        "solved pressure",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            pressureExpList),
        pressureComponents, 1);
    pressure.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < pressure.GetBlocks().size(); ++blk)
    {
        auto &pressureBlock = pressure.GetBlocks()[blk];
        auto pressurePtr =
            pressureBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto collection        = GetCollection(pressureExpList, blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            Array<OneD, double> x(pressureBlock.GetNumData());
            Array<OneD, double> y(pressureBlock.GetNumData());
            Array<OneD, double> z(pressureBlock.GetNumData(), 0.0);
            expansions[e]->GetCoords(x, y, z);
            for (unsigned int p = 0; p < pressureBlock.GetNumData(); ++p)
            {
                pressurePtr[e * pressureBlock.GetNumData() + p] =
                    1.0 + x[p] + 2.0 * y[p];
            }
        }
    }

    constexpr double kinvis = 0.3;
    auto op =
        HOutflowBndCondOp<double>::Create(pressureExpList, pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(kinvis);
    op->UpdatePressure(velocity);

    const unsigned int interleaveWidth = NektarSpaces::GetVectorWidth<double>(
        MultiRegions::Operator<double>::GetOpExecSpace(session));

    // The vertex shared by the two regions is set from both, on the device
    // in either order, with values equal up to the rounding of the velocity
    // gradient.
    const auto pressureCoeffBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList);
    LibUtilities::Field<double, FieldState::Coeff> pressureReference(
        "pressure coeffs at width 1", pressureCoeffBlocks, pressureComponents,
        1);
    LibUtilities::Field<double, FieldState::Coeff> pressureInterleaved(
        "interleaved pressure coeffs", pressureCoeffBlocks, pressureComponents,
        1);
    const auto pressureInitialValues = FillAtTwoWidths(
        pressureReference, pressureInterleaved, interleaveWidth);
    auto pressureOp = op->GetPressureDirichletOp();
    for (unsigned int pass = 0; pass < 2; ++pass)
    {
        pressureOp->Apply(pressureReference);
        pressureOp->Apply(pressureInterleaved);
    }
    CheckAtTwoWidths(pressureReference, pressureInterleaved, interleaveWidth,
                     pressureInitialValues, 1.0e-12);

    // The velocity forcing sums the contributions to the shared vertex
    // before adding them, so both widths agree exactly.
    op->SetPressure(pressure);
    const auto velocityCoeffBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            velocityExpList);
    LibUtilities::Field<double, FieldState::Coeff> velocityReference(
        "velocity coeffs at width 1", velocityCoeffBlocks, velocityComponents,
        1);
    LibUtilities::Field<double, FieldState::Coeff> velocityInterleaved(
        "interleaved velocity coeffs", velocityCoeffBlocks, velocityComponents,
        1);
    const auto velocityInitialValues = FillAtTwoWidths(
        velocityReference, velocityInterleaved, interleaveWidth);
    auto velocityOp = op->GetVelocityNeumannOp();
    for (unsigned int pass = 0; pass < 2; ++pass)
    {
        velocityOp->Apply(velocityReference);
        velocityOp->Apply(velocityInterleaved);
    }
    CheckAtTwoWidths(velocityReference, velocityInterleaved, interleaveWidth,
                     velocityInitialValues, 0.0);

    // The velocity condition computes the flux from the pressure as it
    // stands when it is applied: refilling the pressure in place, without
    // setting it again, changes the forcing, and restoring the pressure
    // restores the forcing.
    auto addToPressure = [&](const double shift) {
        for (auto &pressureBlock : pressure.GetBlocks())
        {
            auto pressurePtr =
                pressureBlock
                    .template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
            const size_t nPoints =
                pressureBlock.GetNumElements() * pressureBlock.GetNumData();
            for (size_t p = 0; p < nPoints; ++p)
            {
                pressurePtr[p] += shift;
            }
        }
    };
    auto applyVelocity = [&]() {
        LibUtilities::Field<double, FieldState::Coeff> forcing(
            "velocity forcing", velocityCoeffBlocks, velocityComponents, 1);
        forcing.Initialize<NektarSpaces::HostSpace>(0.0);
        velocityOp->Apply(forcing);
        return forcing.ToVector<double>();
    };
    const auto forcing = applyVelocity();
    addToPressure(0.5);
    const auto shiftedForcing = applyVelocity();
    addToPressure(-0.5);
    const auto restoredForcing = applyVelocity();
    BOOST_REQUIRE_EQUAL(shiftedForcing.size(), forcing.size());
    BOOST_REQUIRE_EQUAL(restoredForcing.size(), forcing.size());
    double maxShiftChange = 0.0, maxRestoredChange = 0.0;
    for (std::size_t i = 0; i < forcing.size(); ++i)
    {
        maxShiftChange =
            std::max(maxShiftChange, std::abs(shiftedForcing[i] - forcing[i]));
        maxRestoredChange = std::max(maxRestoredChange,
                                     std::abs(restoredForcing[i] - forcing[i]));
    }
    BOOST_TEST(maxShiftChange > 1.0e-2);
    BOOST_TEST(maxRestoredChange <= 1.0e-12);
}

BOOST_AUTO_TEST_SUITE_END()
