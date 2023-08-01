#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorPhysDeriv.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class Line : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    Line() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
        meshName = "line.xml";
    }
};

class Square : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    Square() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
        meshName = "square.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(physderiv, Line)
{

    Configure();

    double *inptr =
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
    double *exptr = fixt_expected->GetStorage().GetCPUPtr();

    size_t order = 6, pts = 0;
    Array<OneD, double> x(fixt_explist->GetTotPoints());
    fixt_explist->GetCoords(x);
    for (auto const &block : fixtcuda_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t phys = 0; phys < block.num_pts; ++phys)
            {
                double tmp1 = 0.0, tmp2 = 0.0;
                for (size_t k = 0; k < order; k++)
                {
                    tmp1 += std::pow(x[pts], k);
                    tmp2 += k * std::pow(x[pts], k - 1);
                }
                pts++;
                *(inptr++) = tmp1;
                *(exptr++) = tmp2;
            }
        }
    }

    PhysDeriv<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out, *fixtcuda_out, *fixtcuda_out);
    double TOL{1.0E-12};
    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, TOL));
}
