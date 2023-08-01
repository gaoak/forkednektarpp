#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

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

BOOST_FIXTURE_TEST_CASE(ipwrtbasecuda, Line)
{
    Configure();

    static double *x =
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixtcuda_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t phys = 0; phys < block.num_pts; ++phys)
            {
                *(x++) = phys;
            }
        }
    }

    IProductWRTBase<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);

    static double *y =
        fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
    double TOL = 1e-12;
    BOOST_CHECK_CLOSE(y[0], 1.097532178406115, TOL);
    BOOST_CHECK_CLOSE(y[1], 1.902467821593885, TOL);
    BOOST_CHECK_CLOSE(y[2], 0.500000000000000, TOL);
    BOOST_CHECK_CLOSE(y[3], 0.150377322580158, TOL);
    BOOST_TEST(std::abs(y[4] - 1.387778780781446e-17) < TOL);
    BOOST_CHECK_CLOSE(y[5], 0.0074684742369482, TOL);
}
