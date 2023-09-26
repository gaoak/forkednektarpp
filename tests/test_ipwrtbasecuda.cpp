#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestIPWRTBaseCUDA
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

BOOST_AUTO_TEST_SUITE(TestIPWRTBaseCUDA)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class Line : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    Line() : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
        meshName = "line.xml";
    }
};

class Square : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    Square() : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
        meshName = "square.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(ipwrtbasecuda_line, Line)
{
    Configure();

    double *x = fixt_in->GetStorage().GetCPUPtr();
    double *y =
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixtcuda_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t phys = 0; phys < block.num_pts; ++phys)
            {
                *(x++) = phys;
                *(y++) = phys;
            }
        }
    }

    IProductWRTBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);

    // Generate the expected results as reference to be compared
    IProductWRTBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);

    double TOL = 1e-12;

    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, TOL));
}

BOOST_FIXTURE_TEST_CASE(ipwrtbasecuda_square, Square)
{
    Configure();

    double *x = fixt_in->GetStorage().GetCPUPtr();
    double *y =
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixtcuda_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t phys = 0; phys < block.num_pts; ++phys)
            {
                *(x++) = phys;
                *(y++) = phys;
            }
        }
    }

    IProductWRTBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);

    // Generate the expected results as reference to be compared
    IProductWRTBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_expected);

    double TOL = 1e-12;

    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, TOL));
}

BOOST_AUTO_TEST_SUITE_END()
