#define BOOST_TEST_MODULE TestIdentityCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorIdentity.hpp"
#include "init_identityphysfields.hpp"

BOOST_AUTO_TEST_SUITE(TestIdentityCUDA)

BOOST_FIXTURE_TEST_CASE(identitycuda_seg, Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_quad, Quad)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_tri, Tri)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_square_all_elements, SquareAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_hex, Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_prism, Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_pyr, Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_tet, Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_cube_prism_hex, CubePrismHex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(identitycuda_cube_all_elements, CubeAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    Identity<double, FieldState::Phys>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()
