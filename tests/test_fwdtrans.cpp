#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestFwdTrans
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "init_fwdtransfields.hpp"

BOOST_AUTO_TEST_SUITE(TestFwdTrans)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE(fwdtrans_seg, Helmholtz1D_Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtrans_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtrans_hex, Helmholtz3D_Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtrans_prism, Helmholtz3D_Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtrans_pyr, Helmholtz3D_Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-09));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-09);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtrans_tet, Helmholtz3D_Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "StdMat");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist);
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-09));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-09);
    }
}

BOOST_AUTO_TEST_SUITE_END()
