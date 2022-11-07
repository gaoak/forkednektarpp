
#include <boost/core/demangle.hpp>

#include <iostream>
#include <typeinfo>
using namespace std;

#include "OperatorBwdTrans.hpp"

#ifdef NEKTAR_USE_CUDA
#include "OperatorBwdTransCUDA.hpp"
#endif

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

    auto &inptr = in.GetStorage();
    for (int i = 0; i < inptr.m_size; ++i)
    {
        inptr.m_host[i] = 2.0;
    }

    auto test2 = std::make_shared<Operator<double, OpBwdTrans, MethodLocMat, BackendCUDA>>();
    auto test = std::dynamic_pointer_cast<OperatorBase<Operator<double, OpBwdTrans, MethodLocMat, BackendCUDA>, double, StateCoeff, StatePhys>>(test2);
    test->apply(in, out);

    auto &outptr = out.GetStorage();
    outptr.DeviceToHost();

    bool isClose = true;
    for (int i = 0; i < inptr.m_size; ++i)
    {
        if (abs(2.0*inptr.m_host[i] - outptr.m_host[i]) > 1e-10)
        {
            std::cout << 2.0*inptr.m_host[i] << " " << outptr.m_host[i] << std::endl;
            isClose = false;
            break;
        }
    }
    std::cout << "Vectors are close: " << (isClose ? "yes" : "no") << std::endl;

/*
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
