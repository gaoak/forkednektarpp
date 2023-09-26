#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorPhysDeriv.hpp"
#include "init_physderivfields.hpp"

BOOST_FIXTURE_TEST_CASE(physderiv_seg, Seg)
{
    Configure(1, 1);
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
}

BOOST_FIXTURE_TEST_CASE(physderiv_quad, Quad)
{
    Configure(1, 2);
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
}

/*BOOST_FIXTURE_TEST_CASE(physderiv_hex, Hex)
{
    Configure(1, 3);
    SetTestCase(
        fixtcuda_in->GetBlocks(),
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, 1.0E-12));
}*/
