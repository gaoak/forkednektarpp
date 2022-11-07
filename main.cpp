#include <boost/core/demangle.hpp>

#include <iostream>
#include <typeinfo>
using namespace std;

#include "OperatorBwdTrans.hpp"

int main() {
    using DefaultMethod = MethodLocMat;
    using TData = double;

    Field<double, StateCoeff> in;
    Field<double, StatePhys> out;

    auto test2 = std::make_shared<Operator<double, OpBwdTrans>>();
    auto test = std::dynamic_pointer_cast<OperatorBase<Operator<double, OpBwdTrans>, double, StateCoeff, StatePhys>>(test2);
    test->apply(in, out);

/*
    Operator<TData, OpBwdTrans>::create()->apply(in, out);

    auto o = Operator<TData, OpBwdTrans>::create();
    o->apply(in, out);
    cout << boost::core::demangle(typeid(o).name()) << endl;
*/
}
