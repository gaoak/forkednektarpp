#define BOOST_TEST_MODULE TestIProductWRTDerivBaseCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorIProductWRTDerivBase.hpp"
#include "init_ipwrtderivbasefields.hpp"

BOOST_AUTO_TEST_SUITE(TestIProductWRTDerivBaseCUDA)

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_seg, Seg)
{
    Configure(1, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_quad, Quad)
{
    Configure(2, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_tri, Tri)
{
    Configure(2, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_square_all_elements,
                        SquareAllElements)
{
    Configure(2, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_hex, Hex)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_prism, Prism)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_pyr, Pyr)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_tet, Tet)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_cube_prism_hex, CubePrismHex)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(ipwrtderivbasecuda_cube_all_elements, CubeAllElements)
{
    Configure(3, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    IProductWRTDerivBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);
    IProductWRTDerivBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()
