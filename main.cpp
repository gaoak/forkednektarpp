
#include <boost/core/demangle.hpp>

#include <iostream>
#include <typeinfo>
using namespace std;

#include "OperatorBwdTrans.hpp"
//#include "OperatorIProduct.hpp"

#include "Operator.hpp"

int main() {
    using DefaultMethod = MethodLocMat;
    using TData = double;

    std::vector<BlockAttributes> blocks = {
        {eQuadrilateral, {4, 4, 1}, 100},
        {eTriangle,      {4, 4, 1}, 200}
    };

    Field<double, StateCoeff> in(blocks);
    Field<double, StatePhys> out(blocks);

    auto test2 = std::make_shared<Operator<double, OpBwdTrans, MethodLocMat, BackendCPU>>();

/*
    auto test = std::dynamic_pointer_cast<OperatorBase<Operator<double, OpBwdTrans>, double, StateCoeff, StatePhys>>(test2);
    test->apply(in, out);

    auto test3 = std::make_shared<Operator<double, OpIProduct>>();
    auto test4 = std::dynamic_pointer_cast<OperatorBase<Operator<double, OpIProduct>, double, StatePhys, StateCoeff>>(test3);
    test4->apply(out, in);

/*
    Operator<TData, OpBwdTrans>::create()->apply(in, out);

    auto o = Operator<TData, OpBwdTrans>::create();
    o->apply(in, out);
    cout << boost::core::demangle(typeid(o).name()) << endl;
*/
}
