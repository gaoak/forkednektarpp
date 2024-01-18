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

BOOST_FIXTURE_TEST_CASE(helmsolve_seg, Helmholtz1D_Seg)
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

BOOST_FIXTURE_TEST_CASE(helmsolve_tri_quad, Helmholtz2D_Tri_Quad)
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

BOOST_FIXTURE_TEST_CASE(helmsolve_hex, Helmholtz3D_Hex)
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

BOOST_FIXTURE_TEST_CASE(helmsolve_prism, Helmholtz3D_Prism)
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

BOOST_FIXTURE_TEST_CASE(helmsolve_pyr, Helmholtz3D_Pyr)
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

BOOST_FIXTURE_TEST_CASE(helmsolve_tet, Helmholtz3D_Tet)
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

BOOST_AUTO_TEST_SUITE_END()
