#include <iostream>
#include <memory>

#include "Field.hpp"
#include "OperatorBwdTrans.hpp"
// #include "NekFactory.hpp"

template <typename T> using OpBwdTransd = OperatorBwdTrans<double, T>;

int main()
{

    OpBwdTransd<MethodMatFree> deriv;
    OpBwdTransd<MethodMatFree>::OpBaseT *base = &deriv;

    std::vector<BlockAttributes> blocks = {{eQuadrilateral, {4, 4, 1}, 100},
                                           {eTriangle, {4, 4, 1}, 200}};

    Field<double, StateCoeff> in(blocks);
    Field<double, StatePhys> out(blocks);

    base->apply(in, out);

    OpBwdTransd<MethodSumFac> derivSumFac;
    OpBwdTransd<MethodSumFac>::OpBaseT *baseSumFac = &derivSumFac;

    baseSumFac->apply(in, out);

    return 0;
}
