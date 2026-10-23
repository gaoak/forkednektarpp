///////////////////////////////////////////////////////////////////////////////
//
// File: TestHOPressureBndCond.cpp
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Unit tests of the higher-order pressure boundary condition.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestHOPressureBndCond

// Boost.Test pulls in windows.h, whose min/max macros break tinysimd.
#define NOMINMAX
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOPressureBndCond/HOPressureBndCondOp.hpp>
#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/StringUtils.hpp>

using namespace Nektar;

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
LibUtilities::SessionReaderSharedPtr CreateSession()
{
    const auto &suite = boost::unit_test::framework::master_test_suite();
    BOOST_REQUIRE_MESSAGE(suite.argc >= 4,
                          "Usage: TestHOPressureBndCond -- <ExecSpace> "
                          "<Implementation> <Session.xml> [<Session.xml>...]");

    const std::string execStr(suite.argv[1]);
    const std::string implStr(suite.argv[2]);
    const int nSessionFiles = suite.argc - 3;

    int argc    = nSessionFiles + 3;
    char **argv = new char *[argc + 1];
    argv[0]     = strdup("TestHOPressureBndCond");
    for (int i = 0; i < nSessionFiles; ++i)
    {
        argv[i + 1] = strdup(suite.argv[i + 3]);
    }
    argv[nSessionFiles + 1] = strdup(("--opExecSpace=" + execStr).c_str());
    argv[nSessionFiles + 2] = strdup(("--opImpl=" + implStr).c_str());
    argv[argc]              = nullptr;

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    session->InitSession();

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}

std::vector<std::string> GetVelocityComponents(const unsigned int nDim)
{
    const std::vector<std::string> allComponents = {"u", "v", "w"};
    BOOST_REQUIRE(nDim >= 2 && nDim <= allComponents.size());
    return {allComponents.begin(), allComponents.begin() + nDim};
}

// Whether a pressure region takes the higher-order pressure flux: a Neumann
// condition with USERDEFINEDTYPE="H".
bool IsTaggedHOPressure(const SpatialDomains::BoundaryConditionShPtr &condition)
{
    return condition->GetBoundaryConditionType() == SpatialDomains::eNeumann &&
           StringUtils::IEquals(condition->GetUserDefined(), "H");
}

// The expected pressure forcing of the flux gradient(x, y, z) . n on the
// Neumann pressure regions. With requireHOPTag, only the regions with
// USERDEFINEDTYPE="H" contribute, for sessions mixing them with other
// Neumann regions; otherwise every Neumann region does.
template <typename Gradient>
std::vector<double> BuildExpectedBoundaryContribution(
    const MultiRegions::ExpListSharedPtr &pressureExpList,
    const unsigned int nDim, Gradient gradient,
    const bool requireHOPTag = false)
{
    Array<OneD, double> expectedCoeffs(pressureExpList->GetNcoeffs(), 0.0);
    const auto &bndExpansions = pressureExpList->GetBndCondExpansions();
    const auto &bndConditions = pressureExpList->GetBndConditions();
    auto pressureContField =
        std::dynamic_pointer_cast<MultiRegions::ContField>(pressureExpList);
    BOOST_REQUIRE(pressureContField);
    const auto &localToGlobal = pressureContField->GetLocalToGlobalMap();
    const auto &coeffMap  = localToGlobal->GetBndCondCoeffsToLocalCoeffsMap();
    const auto &coeffSign = localToGlobal->GetBndCondCoeffsToLocalCoeffsSign();
    const bool signChange = localToGlobal->GetSignChange();

    unsigned int bndCoeffOffset = 0;
    for (unsigned int n = 0; n < bndExpansions.size(); ++n)
    {
        auto bndExp = bndExpansions[n];
        const bool selected =
            requireHOPTag ? IsTaggedHOPressure(bndConditions[n])
                          : bndConditions[n]->GetBoundaryConditionType() ==
                                SpatialDomains::eNeumann;
        if (selected)
        {
            const auto npts = bndExp->GetTotPoints();
            Array<OneD, double> x(npts), y(npts), z(npts, 0.0);
            bndExp->GetCoords(x, y, z);
            Array<OneD, Array<OneD, double>> vectorFlux(nDim);
            for (unsigned int d = 0; d < nDim; ++d)
            {
                vectorFlux[d] = Array<OneD, double>(npts, 0.0);
            }
            for (unsigned int p = 0; p < npts; ++p)
            {
                const auto value = gradient(x[p], y[p], z[p]);
                for (unsigned int d = 0; d < nDim; ++d)
                {
                    vectorFlux[d][p] = value[d];
                }
            }

            Array<OneD, double> bndCoeffs(bndExp->GetNcoeffs(), 0.0);
            bndExp->NormVectorIProductWRTBase(vectorFlux, bndCoeffs);
            for (unsigned int c = 0; c < bndExp->GetNcoeffs(); ++c)
            {
                const auto bndCoeff   = bndCoeffOffset + c;
                const auto localCoeff = coeffMap[bndCoeff];
                expectedCoeffs[localCoeff] +=
                    (signChange ? coeffSign[bndCoeff] : 1.0) * bndCoeffs[c];
            }
        }
        bndCoeffOffset += bndExp->GetNcoeffs();
    }

    const std::vector<std::string> pressureComponents = {"p"};
    LibUtilities::Field<double, FieldState::Coeff> expected(
        "expected Poisson RHS",
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
    return expected.ToVector<double>();
}

// The flux on a region whose velocity is Dirichlet in u and Neumann in v,
// from UpdateWithAcceleration(), or from UpdateWeak() with isWeak.
void CheckNeumannVelocityBoundaryFlux(const bool isWeak)
{
    auto session = CreateSession();
    if (!session->DefinesParameter("HOPNeumannVelocityTest"))
    {
        return;
    }
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const unsigned int nDim       = graph->GetMeshDimension();
    const auto velocityComponents = GetVelocityComponents(nDim);
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);
    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> rhs(
        "explicit contribution", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "velocity", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> predictor(
        "implicit predictor", velocityPhysBlocks, velocityComponents, 1);

    poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    rhs.Initialize<NektarSpaces::HostSpace>(0.0);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    predictor.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < predictor.GetBlocks().size(); ++blk)
    {
        auto &rhsBlock       = rhs.GetBlocks()[blk];
        auto &predictorBlock = predictor.GetBlocks()[blk];
        auto rhsPtr =
            rhsBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto predictorPtr =
            predictorBlock
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int d = 0; d < nDim; ++d)
        {
            std::fill(rhsPtr + d * rhsBlock.CompSize(),
                      rhsPtr + (d + 1) * rhsBlock.CompSize(), 0.03 * (d + 1));
            std::fill(predictorPtr + d * predictorBlock.CompSize(),
                      predictorPtr + (d + 1) * predictorBlock.CompSize(),
                      0.1 * (d + 1));
        }
    }

    static constexpr double time  = 0.4;
    static constexpr double scale = 2.5;
    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    BOOST_TEST(op->HasNonDirichletVelocity());
    op->SetKinvis(0.0);
    op->SetTime(time);
    op->SetScale(scale);
    // u: scale * (predictor - u_D), with u_D = time; v: scale * rhs. The
    // weak form leaves out scale * predictor: u: -scale * u_D; v:
    // -scale * (predictor - rhs).
    std::array<double, 3> expectedFlux = {scale * (0.1 - time), scale * 0.06,
                                          0.0};
    if (isWeak)
    {
        op->UpdateWeak(rhs, velocity, predictor);
        expectedFlux = {-scale * time, -scale * (0.2 - 0.06), 0.0};
    }
    else
    {
        op->UpdateWithAcceleration(rhs, velocity, predictor);
    }
    op->Apply(poissonRhs);

    const auto actualValues   = poissonRhs.ToVector<double>();
    const auto expectedValues = BuildExpectedBoundaryContribution(
        pressureExpList, nDim,
        [&expectedFlux](const double, const double, const double) {
            return expectedFlux;
        });
    BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());
    double maxDifference = 0.0;
    for (std::size_t i = 0; i < actualValues.size(); ++i)
    {
        maxDifference = std::max(maxDifference,
                                 std::abs(actualValues[i] - expectedValues[i]));
    }
    BOOST_TEST(maxDifference < 2.5e-11);

    LibUtilities::Field<double, FieldState::Phys> noRhs;
    if (isWeak)
    {
        BOOST_CHECK_THROW(op->UpdateWeak(noRhs, velocity, predictor),
                          Nektar::ErrorUtil::NekError);
    }
    else
    {
        BOOST_CHECK_THROW(
            op->UpdateWithAcceleration(noRhs, velocity, predictor),
            Nektar::ErrorUtil::NekError);
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(TestHOPressureBndCond)

BOOST_AUTO_TEST_CASE(factory_create_and_empty_apply_preserves_rhs)
{
    auto session = CreateSession();
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    pressureExpList->SetDataWarehouse();

    const std::vector<std::string> pressureComponents = {"p"};
    LibUtilities::Field<double, FieldState::Coeff> rhs(
        "rhs",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    rhs.Initialize<NektarSpaces::HostSpace>(1.25);

    auto before = rhs.ToVector<double>();

    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->Apply(rhs);

    auto after = rhs.ToVector<double>();
    BOOST_REQUIRE_EQUAL(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        BOOST_TEST(after[i] == before[i]);
    }
}

BOOST_AUTO_TEST_CASE(update_then_apply_zero_contribution_preserves_rhs)
{
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

    const auto velocityComponents =
        GetVelocityComponents(graph->GetMeshDimension());
    const std::vector<std::string> pressureComponents = {"p"};

    LibUtilities::Field<double, FieldState::Coeff> rhs(
        "rhs",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> boundaryRhs(
        "boundary rhs",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList),
        velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "velocity",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList),
        velocityComponents, 1);

    rhs.Initialize<NektarSpaces::HostSpace>(1.25);
    boundaryRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);

    auto before = rhs.ToVector<double>();

    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(1.0);
    op->Update(boundaryRhs, velocity);
    op->Apply(rhs);

    auto after = rhs.ToVector<double>();
    BOOST_REQUIRE_EQUAL(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        BOOST_TEST(after[i] == before[i]);
    }
}

BOOST_AUTO_TEST_CASE(update_then_apply_manufactured_boundary_flux)
{
    auto session = CreateSession();
    if (session->DefinesParameter("KovaLam"))
    {
        return;
    }
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const unsigned int nDim       = graph->GetMeshDimension();
    const auto velocityComponents = GetVelocityComponents(nDim);
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);

    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> boundaryRhs(
        "HOPressure boundary RHS", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "manufactured velocity", velocityPhysBlocks, velocityComponents, 1);

    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    boundaryRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < velocity.GetBlocks().size(); ++blk)
    {
        auto &velocityBlock = velocity.GetBlocks()[blk];
        auto &rhsBlock      = boundaryRhs.GetBlocks()[blk];
        auto velocityPtr =
            velocityBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto rhsPtr =
            rhsBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
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
                const double x2 = x[p] * x[p];
                const double y2 = y[p] * y[p];
                if (nDim == 2)
                {
                    // This field is divergence-free and has a non-constant
                    // CurlCurl. The resulting flux is grad(p), where
                    // p = x^3 + x*y^2 + 2*x^2*y.
                    velocityPtr[id] = 2.0 * x[p] * x2 * y[p];
                    velocityPtr[velocityBlock.CompSize() + id] = -3.0 * x2 * y2;
                    rhsPtr[id] = 3.0 * x2 + y2 + x[p] * y[p];
                    rhsPtr[rhsBlock.CompSize() + id] =
                        3.5 * x2 + 2.0 * x[p] * y[p] + 1.5 * y2;
                }
                else
                {
                    // u=(y^2,z^2,x^2) is divergence-free and CurlCurl(u) is
                    // (-2,-2,-2). For nu=1/4 this RHS produces grad(p), where
                    // p=x^2+2*y^2+3*z^2+x*y+y*z+z*x.
                    const double z2                            = z[p] * z[p];
                    velocityPtr[id]                            = y2;
                    velocityPtr[velocityBlock.CompSize() + id] = z2;
                    velocityPtr[2 * velocityBlock.CompSize() + id] = x2;
                    rhsPtr[id] = 2.0 * x[p] + y[p] + z[p] - 0.5;
                    rhsPtr[rhsBlock.CompSize() + id] =
                        x[p] + 4.0 * y[p] + z[p] - 0.5;
                    rhsPtr[2 * rhsBlock.CompSize() + id] =
                        x[p] + y[p] + 6.0 * z[p] - 0.5;
                }
            }
        }
    }

    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    static constexpr double contributionScale = 2.5;
    op->SetKinvis(0.25);
    op->SetScale(contributionScale);

    const auto expectedValues = BuildExpectedBoundaryContribution(
        pressureExpList, nDim,
        [nDim](const double x, const double y, const double z) {
            if (nDim == 2)
            {
                return std::array<double, 3>{
                    contributionScale * (3.0 * x * x + y * y + 4.0 * x * y),
                    contributionScale * (2.0 * x * y + 2.0 * x * x), 0.0};
            }
            return std::array<double, 3>{contributionScale * (2.0 * x + y + z),
                                         contributionScale * (x + 4.0 * y + z),
                                         contributionScale * (x + y + 6.0 * z)};
        });
    const auto maxExpected =
        *std::max_element(expectedValues.begin(), expectedValues.end(),
                          [](const double lhs, const double rhs) {
                              return std::abs(lhs) < std::abs(rhs);
                          });
    BOOST_TEST(std::abs(maxExpected) > 0.1);

    // The velocity and the right-hand side at interleave width 1, at the
    // interleave width of the execution space, and one at each. The
    // operator gathers them, each at its own width, into fields at the
    // width of the velocity, whose width changes from one update to the
    // next.
    const auto execSpace =
        MultiRegions::Operator<double>::GetOpExecSpace(session);
    const unsigned int interleaveWidth =
        NektarSpaces::GetVectorWidth<double>(execSpace);
    const std::vector<std::pair<unsigned int, unsigned int>> widths = {
        {1, 1},
        {interleaveWidth, interleaveWidth},
        {interleaveWidth, 1},
        {1, interleaveWidth},
        {1, 1}};
    for (const auto &[velocityWidth, rhsWidth] : widths)
    {
        velocity.ReshapeStorage(velocityWidth, execSpace);
        boundaryRhs.ReshapeStorage(rhsWidth, execSpace);
        poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);
        op->Update(boundaryRhs, velocity);
        op->Apply(poissonRhs);

        const auto actualValues = poissonRhs.ToVector<double>();
        BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());
        for (std::size_t i = 0; i < actualValues.size(); ++i)
        {
            BOOST_CHECK_MESSAGE(
                std::abs(actualValues[i] - expectedValues[i]) < 2.5e-10,
                "velocity at width " << velocityWidth << ", rhs at width "
                                     << rhsWidth);
        }
    }
}

BOOST_AUTO_TEST_CASE(update_then_apply_acceleration_boundary_flux)
{
    auto session = CreateSession();
    if (session->DefinesParameter("KovaLam"))
    {
        return;
    }
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const unsigned int nDim       = graph->GetMeshDimension();
    const auto velocityComponents = GetVelocityComponents(nDim);
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);
    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "velocity", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> predictor(
        "implicit predictor", velocityPhysBlocks, velocityComponents, 1);

    poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    predictor.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < predictor.GetBlocks().size(); ++blk)
    {
        auto &predictorBlock = predictor.GetBlocks()[blk];
        auto predictorPtr =
            predictorBlock
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int d = 0; d < nDim; ++d)
        {
            std::fill(predictorPtr + d * predictorBlock.CompSize(),
                      predictorPtr + (d + 1) * predictorBlock.CompSize(),
                      0.1 * (d + 1));
        }
    }

    static constexpr double time  = 0.4;
    static constexpr double scale = 2.5;
    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(0.0);
    op->SetTime(time);
    op->SetScale(scale);
    // Every velocity component is Dirichlet, so rhs is not read.
    LibUtilities::Field<double, FieldState::Phys> rhs;
    op->UpdateWithAcceleration(rhs, velocity, predictor);
    op->Apply(poissonRhs);

    // With the acceleration added, the vector flux is
    // scale * (predictor - u_D) before projection onto the boundary normal.
    const auto actualValues   = poissonRhs.ToVector<double>();
    const auto expectedValues = BuildExpectedBoundaryContribution(
        pressureExpList, nDim, [](const double, const double, const double) {
            std::array<double, 3> value = {0.0, 0.0, 0.0};
            for (unsigned int d = 0; d < value.size(); ++d)
            {
                value[d] = scale * (d + 1) * (0.1 - time);
            }
            return value;
        });
    BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());
    double maxDifference = 0.0;
    for (std::size_t i = 0; i < actualValues.size(); ++i)
    {
        maxDifference = std::max(maxDifference,
                                 std::abs(actualValues[i] - expectedValues[i]));
    }
    BOOST_TEST(maxDifference < 2.5e-11);
}

// The forcing added to a field at the interleave width of the execution space
// matches that added to the same field at interleave width 1, and the field
// keeps its width. Each field takes the forcing twice, the widths alternating.
BOOST_AUTO_TEST_CASE(update_then_apply_interleaved_matches_width_one)
{
    auto session = CreateSession();
    if (session->DefinesParameter("KovaLam"))
    {
        return;
    }
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const unsigned int nDim       = graph->GetMeshDimension();
    const auto velocityComponents = GetVelocityComponents(nDim);
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);
    const auto pressureCoeffBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "velocity", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> predictor(
        "implicit predictor", velocityPhysBlocks, velocityComponents, 1);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    predictor.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < predictor.GetBlocks().size(); ++blk)
    {
        auto &predictorBlock = predictor.GetBlocks()[blk];
        auto predictorPtr =
            predictorBlock
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        const size_t nPoints =
            predictorBlock.GetNumElements() * predictorBlock.GetNumData();
        for (unsigned int d = 0; d < nDim; ++d)
        {
            std::fill(predictorPtr + d * predictorBlock.CompSize(),
                      predictorPtr + d * predictorBlock.CompSize() + nPoints,
                      0.1 * (d + 1));
        }
    }

    // The same non-zero right-hand side at interleave width 1 and at the
    // interleave width of the execution space.
    LibUtilities::Field<double, FieldState::Coeff> reference(
        "Poisson RHS at width 1", pressureCoeffBlocks, pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Coeff> interleaved(
        "interleaved Poisson RHS", pressureCoeffBlocks, pressureComponents, 1);
    reference.Initialize<NektarSpaces::HostSpace>(0.0);
    interleaved.Initialize<NektarSpaces::HostSpace>(0.0);
    const unsigned int interleaveWidth = NektarSpaces::GetVectorWidth<double>(
        MultiRegions::Operator<double>::GetOpExecSpace(session));
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

    constexpr double time  = 0.4;
    constexpr double scale = 2.5;
    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(0.0);
    op->SetTime(time);
    op->SetScale(scale);
    // Every velocity component is Dirichlet, so rhs is not read.
    LibUtilities::Field<double, FieldState::Phys> rhs;
    op->UpdateWithAcceleration(rhs, velocity, predictor);
    for (unsigned int pass = 0; pass < 2; ++pass)
    {
        op->Apply(reference);
        op->Apply(interleaved);
    }

    for (auto &block : interleaved.GetBlocks())
    {
        BOOST_TEST(block.GetInterleaveWidth() == interleaveWidth);
    }

    const auto referenceValues   = reference.ToVector<double>();
    const auto interleavedValues = interleaved.ToVector<double>();
    BOOST_REQUIRE_EQUAL(interleavedValues.size(), referenceValues.size());
    double maxForcing = 0.0;
    for (std::size_t i = 0; i < referenceValues.size(); ++i)
    {
        maxForcing = std::max(maxForcing,
                              std::abs(referenceValues[i] - initialValues[i]));
        BOOST_TEST(interleavedValues[i] == referenceValues[i]);
    }
    BOOST_TEST(maxForcing > 1.0e-2);
}

BOOST_AUTO_TEST_CASE(update_then_apply_kovasznay_boundary_flux)
{
    auto session = CreateSession();
    if (!session->DefinesParameter("KovaLam"))
    {
        return;
    }

    auto graph = SpatialDomains::MeshGraphIO::Read(session);
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

    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> boundaryRhs(
        "Kovasznay boundary RHS", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "Kovasznay velocity", velocityPhysBlocks, velocityComponents, 1);

    const double lambda     = session->GetParameter("KovaLam");
    const double kinvis     = session->GetParameter("Kinvis");
    const double waveNumber = 2.0 * std::acos(-1.0);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    boundaryRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < velocity.GetBlocks().size(); ++blk)
    {
        auto &velocityBlock = velocity.GetBlocks()[blk];
        auto &rhsBlock      = boundaryRhs.GetBlocks()[blk];
        auto velocityPtr =
            velocityBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto rhsPtr =
            rhsBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
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
                const auto id            = e * velocityBlock.GetNumData() + p;
                const double exponential = std::exp(-lambda * x[p]);
                const double sine        = std::sin(waveNumber * y[p]);
                const double cosine      = std::cos(waveNumber * y[p]);

                velocityPtr[id] = 1.0 - exponential * cosine;
                velocityPtr[velocityBlock.CompSize() + id] =
                    -(lambda / waveNumber) * exponential * sine;

                // Physical explicit RHS is -u.grad(u). Together with
                // -nu*CurlCurl(u), this equals grad(p) for Kovasznay flow.
                rhsPtr[id] = -lambda * exponential * cosine +
                             lambda * exponential * exponential;
                rhsPtr[rhsBlock.CompSize() + id] =
                    -(lambda * lambda / waveNumber) * exponential * sine;
            }
        }
    }

    poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpList, velocityComponents);
    op->SetKinvis(kinvis);
    op->Update(boundaryRhs, velocity);
    op->Apply(poissonRhs);

    Array<OneD, double> expectedCoeffs(pressureExpList->GetNcoeffs(), 0.0);
    const auto &bndExpansions = pressureExpList->GetBndCondExpansions();
    const auto &bndConditions = pressureExpList->GetBndConditions();
    const auto &localToGlobal = pressureExpList->GetLocalToGlobalMap();
    const auto &coeffMap  = localToGlobal->GetBndCondCoeffsToLocalCoeffsMap();
    const auto &coeffSign = localToGlobal->GetBndCondCoeffsToLocalCoeffsSign();
    const bool signChange = localToGlobal->GetSignChange();
    unsigned int bndCoeffOffset = 0;
    for (unsigned int n = 0; n < bndExpansions.size(); ++n)
    {
        auto bndExp = bndExpansions[n];
        if (bndConditions[n]->GetBoundaryConditionType() ==
            SpatialDomains::eNeumann)
        {
            const auto npts = bndExp->GetTotPoints();
            Array<OneD, double> x(npts), y(npts), z(npts, 0.0);
            bndExp->GetCoords(x, y, z);
            Array<OneD, Array<OneD, double>> pressureGradient(2);
            pressureGradient[0] = Array<OneD, double>(npts);
            pressureGradient[1] = Array<OneD, double>(npts, 0.0);
            for (unsigned int p = 0; p < npts; ++p)
            {
                pressureGradient[0][p] =
                    lambda * std::exp(-2.0 * lambda * x[p]);
            }

            Array<OneD, double> bndCoeffs(bndExp->GetNcoeffs(), 0.0);
            bndExp->NormVectorIProductWRTBase(pressureGradient, bndCoeffs);
            for (unsigned int c = 0; c < bndExp->GetNcoeffs(); ++c)
            {
                const auto bndCoeff   = bndCoeffOffset + c;
                const auto localCoeff = coeffMap[bndCoeff];
                expectedCoeffs[localCoeff] +=
                    (signChange ? coeffSign[bndCoeff] : 1.0) * bndCoeffs[c];
            }
        }
        bndCoeffOffset += bndExp->GetNcoeffs();
    }

    LibUtilities::Field<double, FieldState::Coeff> expected(
        "expected Kovasznay Poisson RHS",
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

    const auto actualValues   = poissonRhs.ToVector<double>();
    const auto expectedValues = expected.ToVector<double>();
    BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());
    double maxDifference = 0.0;
    for (std::size_t i = 0; i < actualValues.size(); ++i)
    {
        maxDifference = std::max(maxDifference,
                                 std::abs(actualValues[i] - expectedValues[i]));
    }
    BOOST_TEST(maxDifference < 1.0e-6);
}

// The Kovasznay velocity is not representable on the boundary expansion of
// KovasznayFlowHOPBC2D_SmallTimeStep.xml. A predictor holding its projection
// onto the continuous expansion, with the Dirichlet velocity imposed as the
// velocity solve imposes it, is a steady state of the boundary acceleration:
// without an explicit contribution or viscosity, the flux vanishes.
// Subtracting the expression of the Dirichlet velocity instead leaves
// scale * (u_D - u_D,exact) . n, large at the scale of a small time step.
BOOST_AUTO_TEST_CASE(update_then_apply_kovasznay_discrete_boundary_velocity)
{
    auto session = CreateSession();
    if (!session->DefinesParameter("KovaLam"))
    {
        return;
    }

    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    const std::vector<std::string> velocityComponents = {"u", "v"};
    const std::vector<std::string> pressureComponents = {"p"};
    std::vector<MultiRegions::ContFieldSharedPtr> velocityExpLists;
    for (const auto &component : velocityComponents)
    {
        velocityExpLists.push_back(
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                session, graph, component, true, false,
                Collections::eNoCollection));
    }
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpLists[0]->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    // The projection of the exact velocity onto the continuous expansion,
    // whose Dirichlet coefficients are those of the projected Dirichlet
    // velocity.
    const double lambda     = session->GetParameter("KovaLam");
    const double waveNumber = 2.0 * std::acos(-1.0);
    const size_t nq         = velocityExpLists[0]->GetTotPoints();
    Array<OneD, double> x(nq), y(nq), z(nq, 0.0);
    velocityExpLists[0]->GetCoords(x, y, z);
    std::vector<Array<OneD, double>> discreteVelocity;
    for (unsigned int d = 0; d < velocityComponents.size(); ++d)
    {
        Array<OneD, double> exact(nq);
        for (size_t p = 0; p < nq; ++p)
        {
            const double exponential = std::exp(-lambda * x[p]);
            exact[p] = d == 0 ? 1.0 - exponential * std::cos(waveNumber * y[p])
                              : -(lambda / waveNumber) * exponential *
                                    std::sin(waveNumber * y[p]);
        }
        auto &expList = velocityExpLists[d];
        expList->EvaluateBoundaryConditions(0.0);
        expList->FwdTrans(exact, expList->UpdateCoeffs());
        discreteVelocity.emplace_back(nq, 0.0);
        expList->BwdTrans(expList->GetCoeffs(), discreteVelocity.back());
    }

    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpLists[0]);
    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "velocity", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> predictor(
        "implicit predictor", velocityPhysBlocks, velocityComponents, 1);
    poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    predictor.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < predictor.GetBlocks().size(); ++blk)
    {
        auto &predictorBlock = predictor.GetBlocks()[blk];
        auto predictorPtr =
            predictorBlock
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto collection        = GetCollection(velocityExpLists[0], blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            const auto offset =
                velocityExpLists[0]->GetPhys_Offset(expansions[e]->GetElmtId());
            for (unsigned int d = 0; d < velocityComponents.size(); ++d)
            {
                std::copy_n(discreteVelocity[d].begin() + offset,
                            predictorBlock.GetNumData(),
                            predictorPtr + d * predictorBlock.CompSize() +
                                e * predictorBlock.GetNumData());
            }
        }
    }

    // 1 / (gamma dt) of a second-order step of dt = 1e-4.
    static constexpr double scale = 1.5e4;
    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    op->SetVelocityContext(velocityExpLists[0], velocityComponents);
    op->SetKinvis(0.0);
    op->SetScale(scale);
    // Every velocity component is Dirichlet, so rhs is not read.
    LibUtilities::Field<double, FieldState::Phys> rhs;
    op->UpdateWithAcceleration(rhs, velocity, predictor);
    op->Apply(poissonRhs);

    double maxValue = 0.0;
    for (const auto value : poissonRhs.ToVector<double>())
    {
        maxValue = std::max(maxValue, std::abs(value));
    }
    BOOST_TEST(maxValue < 1.0e-8);
}

BOOST_AUTO_TEST_CASE(update_then_apply_selection_respects_hop_tag)
{
    auto session = CreateSession();
    if (!session->DefinesParameter("HOPSelectionTest"))
    {
        return;
    }
    auto graph = SpatialDomains::MeshGraphIO::Read(session);
    auto velocityExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "u", true, false, Collections::eNoCollection);
    auto pressureExpList =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, "p", true, false, Collections::eNoCollection);
    velocityExpList->SetDataWarehouse();
    pressureExpList->SetDataWarehouse();

    const unsigned int nDim       = graph->GetMeshDimension();
    const auto velocityComponents = GetVelocityComponents(nDim);
    const std::vector<std::string> pressureComponents = {"p"};
    const auto velocityPhysBlocks =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            velocityExpList);

    LibUtilities::Field<double, FieldState::Coeff> poissonRhs(
        "Poisson RHS",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            pressureExpList),
        pressureComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> boundaryRhs(
        "HOPressure boundary RHS", velocityPhysBlocks, velocityComponents, 1);
    LibUtilities::Field<double, FieldState::Phys> velocity(
        "manufactured velocity", velocityPhysBlocks, velocityComponents, 1);

    velocity.Initialize<NektarSpaces::HostSpace>(0.0);
    boundaryRhs.Initialize<NektarSpaces::HostSpace>(0.0);
    for (unsigned int blk = 0; blk < velocity.GetBlocks().size(); ++blk)
    {
        auto &velocityBlock = velocity.GetBlocks()[blk];
        auto &rhsBlock      = boundaryRhs.GetBlocks()[blk];
        auto velocityPtr =
            velocityBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        auto rhsPtr =
            rhsBlock.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
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
                // Same divergence-free, non-constant-CurlCurl field as
                // update_then_apply_manufactured_boundary_flux's 2D branch:
                // grad(p) for p = x^3 + x*y^2 + 2*x^2*y.
                const auto id   = e * velocityBlock.GetNumData() + p;
                const double x2 = x[p] * x[p];
                const double y2 = y[p] * y[p];
                velocityPtr[id] = 2.0 * x[p] * x2 * y[p];
                velocityPtr[velocityBlock.CompSize() + id] = -3.0 * x2 * y2;
                rhsPtr[id] = 3.0 * x2 + y2 + x[p] * y[p];
                rhsPtr[rhsBlock.CompSize() + id] =
                    3.5 * x2 + 2.0 * x[p] * y[p] + 1.5 * y2;
            }
        }
    }

    poissonRhs.Initialize<NektarSpaces::HostSpace>(0.0);

    auto op = HOPressureBndCondOp<double>::Create(pressureExpList,
                                                  pressureComponents);
    // Region 1 is a Neumann pressure region without USERDEFINEDTYPE="H",
    // with a non-Dirichlet velocity: it does not take the higher-order flux,
    // so this does not throw.
    op->SetVelocityContext(velocityExpList, velocityComponents);
    static constexpr double contributionScale = 2.5;
    op->SetKinvis(0.25);
    op->SetScale(contributionScale);
    op->Update(boundaryRhs, velocity);
    op->Apply(poissonRhs);

    const auto actualValues = poissonRhs.ToVector<double>();
    // Only regions 0 and 2, with USERDEFINEDTYPE="H", contribute; region 1,
    // also a Neumann region, contributes nothing.
    const auto expectedValues = BuildExpectedBoundaryContribution(
        pressureExpList, nDim,
        [](const double x, const double y, const double) {
            return std::array<double, 3>{
                contributionScale * (3.0 * x * x + y * y + 4.0 * x * y),
                contributionScale * (2.0 * x * y + 2.0 * x * x), 0.0};
        },
        /* requireHOPTag */ true);
    BOOST_REQUIRE_EQUAL(actualValues.size(), expectedValues.size());

    const auto maxExpected =
        *std::max_element(expectedValues.begin(), expectedValues.end(),
                          [](const double lhs, const double rhs) {
                              return std::abs(lhs) < std::abs(rhs);
                          });
    BOOST_TEST(std::abs(maxExpected) > 0.1);

    for (std::size_t i = 0; i < actualValues.size(); ++i)
    {
        BOOST_TEST(std::abs(actualValues[i] - expectedValues[i]) < 2.5e-10);
    }
}

// On a region whose velocity is Dirichlet in u and Neumann in v, u takes the
// acceleration and v none: its flux is formed from rhs. Without rhs, the
// update is refused.
BOOST_AUTO_TEST_CASE(update_then_apply_neumann_velocity_boundary_flux)
{
    CheckNeumannVelocityBoundaryFlux(false);
}

// The same region in the weak pressure form, whose flux leaves out
// scale * predictor: u takes the acceleration alone, and v the velocity
// predictor - rhs as its boundary velocity. Without rhs, the update is
// refused.
BOOST_AUTO_TEST_CASE(update_weak_then_apply_neumann_velocity_boundary_flux)
{
    CheckNeumannVelocityBoundaryFlux(true);
}

BOOST_AUTO_TEST_SUITE_END()
