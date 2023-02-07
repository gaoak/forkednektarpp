#include <iostream>
#include <memory>

#include "Field.hpp"
#include "OperatorBwdTrans.hpp"
// #include "NekFactory.hpp"

template <OpMethod T> using OpBwdTransd = OperatorBwdTrans<double, T>;

int main()
{

    OpBwdTransd<OpMethod::MatFree> deriv;
    OpBwdTransd<OpMethod::MatFree>::OpBaseT *base = &deriv;

    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 100},
        {ShapeType::eTriangle, {4, 4, 1}, 200}};

    Field<double, FieldState::Coeff> in(blocks);
    Field<double, FieldState::Phys> out(blocks);

    base->apply(in, out);

    OpBwdTransd<OpMethod::SumFac> derivSumFac;
    OpBwdTransd<OpMethod::SumFac>::OpBaseT *baseSumFac = &derivSumFac;

    baseSumFac->apply(in, out);

    return 0;
}
