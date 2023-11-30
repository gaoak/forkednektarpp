#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDiagPrecon
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "init_diagpreconfields.hpp"

BOOST_AUTO_TEST_SUITE(TestDiagPrecon)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE(diagprecon_seg, Helmholtz1D_Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(diagprecon_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(diagprecon_hex, Helmholtz3D_Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagprecon_prism, Helmholtz3D_Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagprecon_pyr, Helmholtz3D_Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-10);
    }
}

BOOST_FIXTURE_TEST_CASE(diagprecon_tet, Helmholtz3D_Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto HelmholtzOp  = Helmholtz<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "");
    HelmholtzOp->setLambda(1.0);
    DiagPreconOp->configure(HelmholtzOp);
    DiagPreconOp->apply(*fixt_in, *fixt_out);
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
