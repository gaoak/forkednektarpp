#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

using namespace Nektar::Operators;

int main()
{
    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 100},
        {ShapeType::eTriangle, {4, 4, 1}, 200}};

    Field<double, FieldState::Coeff> in(blocks);
    Field<double, FieldState::Phys> out(blocks);

    GetOperatorFactory<double>().PrintAvailableClasses();

    BwdTrans<>::create()->apply(in, out);
    BwdTrans<>::create("SumFac")->apply(in, out);

    return 0;
}
