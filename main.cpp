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

    // Create two Field objects with a MemoryRegionCPU backend
    auto in  = Field<double, FieldState::Coeff>::create(blocks);
    auto out = Field<double, FieldState::Phys>::create(blocks);

    GetOperatorFactory<double>().PrintAvailableClasses();

    // Fill up the memory with data
    double *x      = in.GetStorage().GetCPUPtr();
    const size_t n = in.GetStorage().size();
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = i;
    }

    // Instantiate and apply a backward transform operator
    BwdTrans<>::create("")->apply(in, out);
    // BwdTrans<>::create("SumFac")->apply(in, out);

    double *y = out.GetStorage().GetCPUPtr();
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << y[i] << std::endl;
    }

#ifdef NEKTAR_USE_CUDA

    // Test CUDA MemoryRegion

    // Create two Fields with memory on the GPU
    in  = Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(blocks);
    out = Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(blocks);
    BwdTrans<>::create("CUDA")->apply(in, out);

    // Test the GPU-backed fields with a CPU operator
    // This should show a debug warning due to the implicit conversion
    BwdTrans<>::create("MatFree")->apply(in, out);

    std::cout << std::endl;
    // Create two CPU backed fields and use them with a GPU operator
    // This call implicitly converts the fields to a GPU backend and warns the
    // user about that fact
    in  = Field<double, FieldState::Coeff>::create(blocks);
    out = Field<double, FieldState::Phys>::create(blocks);
    BwdTrans<>::create("CUDA")->apply(in, out);
#endif

    return 0;
}
