#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>
#include <cmath>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorFwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>
#include <MultiRegions/ContField.h>

#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;
using namespace Nektar::MultiRegions;

/*
class Line : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    Line() : InitFields<double, FieldState::Coeff, FieldState::Phys>() {
        meshName = "line.xml";
    }
};
*/
class Square : public InitFields<double, FieldState::Phys, FieldState::Coeff, MultiRegions::ContField>
{
public:
    Square() : InitFields<double, FieldState::Phys, FieldState::Coeff, MultiRegions::ContField>() {
        meshName = "square_fwdtrans.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(fwdtrans, Square)
{
    Configure();

    //explist = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(session, graph);

    auto blocks_phys = GetBlockAttributes(FieldState::Phys, fixt_explist);
    auto blocks_coeff = GetBlockAttributes(FieldState::Coeff, fixt_explist);

    auto in = Field<double, FieldState::Phys>::create(blocks_phys);
    auto out = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto correct = Field<double, FieldState::Phys>::create(blocks_phys);

    std::shared_ptr<ContField> explistCF = std::dynamic_pointer_cast<ContField>(fixt_explist);
    std::shared_ptr<AssemblyMapCG> assMap = explistCF->GetLocalToGlobalMap(); // error here

    // get size of global/local systems
    size_t Nglobal = assMap->GetNumGlobalCoeffs();
    size_t Nlocal = assMap->GetNumLocalCoeffs();

    auto *input_iter = in.GetStorage().GetCPUPtr();

    int CG_test = 1;
    
    if (CG_test == 0)
    {
        for (int i = 0; i < fixt_explist->GetTotPoints(); ++i)
            *(input_iter++) = 1;
    }
    else if (CG_test == 1)
    {    
        int np = fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> x(np), y(np), z(np);
        fixt_explist->GetCoords(x, y, z);
        for (int i = 0; i < fixt_explist->GetTotPoints(); ++i)
            *(input_iter++) = x[i]*x[i] + y[i]*y[i] + 20;
    }

    // Get output of fwd trans of input and bwd trans of this output to recover input
    FwdTrans<double>::create(fixt_explist)->apply(in, out);
    BwdTrans<double>::create(fixt_explist)->apply(out, correct);

    // calculate magnitude of error between BwdTrans of output and input
    double eps = 0.;
    input_iter = in.GetStorage().GetCPUPtr();
    auto *correct_iter = correct.GetStorage().GetCPUPtr();
    for (int i = 0; i < fixt_explist->GetTotPoints(); ++i)
    {
        double diff = (*(input_iter++)) - (*(correct_iter++));
        eps += std::pow(diff*diff, 2);
    }
    eps = std::sqrt(eps);

    double TOL{1e-6};

    bool pass = eps < TOL;

    BOOST_TEST(pass);
}
