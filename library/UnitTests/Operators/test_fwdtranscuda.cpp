#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestFwdTransCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "init_fwdtransfields.hpp"

BOOST_AUTO_TEST_SUITE(TestFwdTransCUDA)

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_seg, Helmholtz1D_Seg)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_hex, Helmholtz3D_Hex)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_prism, Helmholtz3D_Prism)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_pyr, Helmholtz3D_Pyr)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-09));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-09);
    }
}

BOOST_FIXTURE_TEST_CASE(fwdtranscuda_tet, Helmholtz3D_Tet)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    auto FwdTransOp   = FwdTrans<>::create(fixt_explist, "CUDA");
    auto DiagPreconOp = DiagPrecon<>::create(fixt_explist, "CUDA");
    FwdTransOp->setPrecon(DiagPreconOp);
    FwdTransOp->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-08));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(
            fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr(),
            fixt_expected->GetStorage().GetCPUPtr(), 1.0E-08);
    }
}

BOOST_AUTO_TEST_SUITE_END()
