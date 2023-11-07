#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestHelmSolve
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorHelmSolve.hpp"
#include "init_helmsolvefields.hpp"

BOOST_AUTO_TEST_SUITE(TestHelmSolve)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE(helmsolve_seg, Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_quad, Quad)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_tri, Tri)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_square_all_elements, SquareAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_hex, Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_prism, Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_pyr, Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_tet, Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_cube_prism_hex, CubePrismHex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve_cube_all_elements, CubeAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(1.0);
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(helmsolve2d_p7_allbcs, Helmholtz2D_P7_AllBCs)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(fixt_explist->GetSession()->GetParameter("Lambda"));
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

/*BOOST_FIXTURE_TEST_CASE(helmsolve3d_hex_allbcs_p6, Helmholtz3D_Hex_AllBCs_P6)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmSolveOp  = HelmSolve<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    HelmSolveOp->setPrecon(DiagPreconOp);
    HelmSolveOp->setLambda(fixt_explist->GetSession()->GetParameter("Lambda"));
    HelmSolveOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}*/

BOOST_AUTO_TEST_SUITE_END()
