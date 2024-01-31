#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestAssmbScatrCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorAssmbScatr.hpp"
#include "init_assmbscatrfields.hpp"

BOOST_AUTO_TEST_SUITE(TestAssmbScatrCUDA)

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_seg, Seg)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_quad, Quad)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_tri, Tri)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_square_all_elements, SquareAllElements)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_hex, Hex)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_prism, Prism)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_pyr, Pyr)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_tet, Tet)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_cube_prism_hex, CubePrismHex)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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

BOOST_FIXTURE_TEST_CASE(assmbscatrcuda_cube_all_elements, CubeAllElements)
{
    Configure();
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
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
