#include <boost/core/demangle.hpp>

#include <iostream>
#include <typeinfo>
using namespace std;

#include "Operator.hpp"

int main() {
    using DefaultMethod = MethodLocMat;
    using TData = double;

    Operator<TData, OpBwdTrans>::create()->apply(5.0, 6.0);

    auto o = Operator<TData, OpBwdTrans>::create();
    o->apply(5.0, 6.0);
    cout << boost::core::demangle(typeid(o).name()) << endl;
}
