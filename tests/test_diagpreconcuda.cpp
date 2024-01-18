#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDiagPreconCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "init_diagpreconfields.hpp"

BOOST_AUTO_TEST_SUITE(TestDiagPreconCUDA)

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_seg, Helmholtz1D_Seg)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_hex, Helmholtz3D_Hex)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_prism, Helmholtz3D_Prism)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_pyr, Helmholtz3D_Pyr)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagpreconcuda_tet, Helmholtz3D_Tet)
{
    Configure();
    SetTestCase(fixtcuda_in->GetBlocks(), fixtcuda_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_AUTO_TEST_SUITE_END()
