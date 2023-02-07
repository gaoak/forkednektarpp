#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

#ifdef NEKTAR_USE_CUDA
#include "MemoryRegionCUDA.hpp"
#endif

using namespace Nektar::Operators;

int main()
{
    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 10},
        {ShapeType::eTriangle, {4, 4, 1}, 20}};

    auto in  = Field<double, FieldState::Coeff>::create(blocks);
    auto out = Field<double, FieldState::Phys>::create(blocks);

    GetOperatorFactory<double>().PrintAvailableClasses();

    double *x      = in.GetStorage().GetCPUPtr();
    const size_t n = in.GetStorage().size();
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = i;
    }

    BwdTrans<>::create("")->apply(in, out);
    // BwdTrans<>::create("SumFac")->apply(in, out);

    double *y = out.GetStorage().GetCPUPtr();
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << y[i] << std::endl;
    }

#ifdef NEKTAR_USE_CUDA

    // Test CUDA call
    in  = Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(blocks);
    out = Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(blocks);
    BwdTrans<>::create("CUDA")->apply(in, out);

    // Using a CPU operator with a CUDA mem region should also work
    BwdTrans<>::create("MatFree")->apply(in, out);

    try
    {
        // It does not work the other way around though
        in  = Field<double, FieldState::Coeff>::create(blocks);
        out = Field<double, FieldState::Phys>::create(blocks);
        BwdTrans<>::create("CUDA")->apply(in, out);
    }
    catch (std::exception const &e)
    {
        std::cout << e.what() << std::endl;
    }
#endif

    return 0;
}
