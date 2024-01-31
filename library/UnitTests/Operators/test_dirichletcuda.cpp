#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDirichletCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDirBndCond.hpp"
#include "init_dirichletfields.hpp"

BOOST_AUTO_TEST_SUITE(TestDirichletCUDA)

BOOST_FIXTURE_TEST_CASE(dirichlet1dcuda_seg, Helmholtz1D_Seg)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(dirichlet2dcuda_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_hex, Helmholtz3D_Hex)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_prism, Helmholtz3D_Prism)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_pyr, Helmholtz3D_Pyr)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_tet, Helmholtz3D_Tet)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "CUDA");
    DirBndCondOp->apply(*fixtcuda_out);
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

BOOST_AUTO_TEST_SUITE_END()
