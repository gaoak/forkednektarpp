#include <iostream>
#include <memory>

#include "Field.hpp"
#include "NekFactory.hpp"
#include "OperatorBwdTrans.hpp"

template <OpMethod T> using OpBwdTransd = OperatorBwdTrans<double, T>;

int main()
{

    auto baseMatFree =
        GetOpBwdTransFactory().CreateInstance("OperatorBwdTransDoubleMatFree");

    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 100},
        {ShapeType::eTriangle, {4, 4, 1}, 200}};

    Field<double, FieldState::Coeff> in(blocks);
    Field<double, FieldState::Phys> out(blocks);

    baseMatFree->apply(in, out);

    auto baseSumFac =
        GetOpBwdTransFactory().CreateInstance("OperatorBwdTransDoubleSumFac");

    baseSumFac->apply(in, out);

    return 0;
}
