#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

using namespace Nektar::Operators;

int main()
{
    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 10},
        {ShapeType::eTriangle, {4, 4, 1}, 20}};

    Field<double, FieldState::Coeff> in(blocks);
    Field<double, FieldState::Phys> out(blocks);

    GetOperatorFactory<double>().PrintAvailableClasses();

    double* x = in.GetStorage().GetPtr();
    const size_t n = in.GetStorage().size();
    for (size_t i = 0; i < n; ++i) {
        x[i] = i;
    }

    BwdTrans<>::create()->apply(in, out);
   // BwdTrans<>::create("SumFac")->apply(in, out);
    
    double* y = out.GetStorage().GetPtr();
    for (size_t i = 0; i < n; ++i) {
        std::cout << y[i] << std::endl;
    }

    return 0;
}
