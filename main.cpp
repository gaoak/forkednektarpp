/**
 * @file main.cpp
 * @author Nektar++ Development Team
 * @brief Demonstrator program for the new Field class.
 * @version 0.1
 * @date 2023-02-13
 * 
 * @copyright Copyright (c) 2023 Imperial College London, University of Utah,
 * Kings College London
 * 
 */
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
    // Define the structure of the computational domain. In this case, we
    // suppose it consists of two composites: 10 quadrilaterals and 20
    // triangles, each with 4 modes / 4 quadrature points.
    std::vector<BlockAttributes> blocks = {
        {ShapeType::eQuadrilateral, {4, 4, 1}, 10},
        {ShapeType::eTriangle, {4, 4, 1}, 20}};

    // Create two Field objects with a MemoryRegionCPU backend by default
    auto in  = Field<double, FieldState::Coeff>::create(blocks);
    auto out = Field<double, FieldState::Phys >::create(blocks);

    // Populate the field with some data. In this case, we just grab a pointer
    // to the memory on the CPU and populate the array with values. Operators,
    // which will have access to the MeshGraph/ExpList data structures, will be
    // able to populate this array in a more meaningful way eventually.
    //
    // Note that GetCPUPtr() states that the memory needs to be on the host. If
    // we were doing this with in_cuda declared above, then this would
    // implicitly transfer the data from the GPU to the host. The intention is
    // that all operators support multiple implementations and the appropriate
    // choice is made based on the backend selected.
    double *x      = in.GetStorage().GetCPUPtr();
    const size_t n = in.GetStorage().size();
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = i;
    }

    // Operators are instantiated using a Factory pattern. First lets check
    // which operators have been registered with the Factory.
    GetOperatorFactory<double>().PrintAvailableClasses();

    // We can create a BwdTrans operator (default implementation)
    // Default implementation might be provided through configuration options,
    // benchmarking, etc eventually.
    auto bt = BwdTrans<>::create();
    // ...and then apply it
    bt->apply(in, out);

    // We can combine these for one-shot operations
    BwdTrans<>::create()->apply(in, out);
    // We can also explicitly select the implementation of the operator to use
    BwdTrans<>::create("SumFac")->apply(in, out);

    // Let's display the result
    double *y = out.GetStorage().GetCPUPtr();
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << y[i] << std::endl;
    }

#ifdef NEKTAR_USE_CUDA

    // Test CUDA MemoryRegion

    // Create two Fields with memory on the GPU
    in  = Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(blocks);
    out = Field<double, FieldState::Phys >::create<MemoryRegionCUDA>(blocks);

    // Perform the BwdTrans on the fields using the CUDA implementation
    // Since this is a CUDA operator, acting on CUDA fields, everything happens
    // on the GPU.
    BwdTrans<>::create("CUDA")->apply(in, out);

    // Test the GPU-backed fields with a CPU operator
    // This should show a debug warning due to the implicit conversion. The
    // purpose of this is to allow us to transition the code to the new
    // infrastructure and add CUDA operators, without the need to add ALL CUDA
    // operators before we can test anything.
    BwdTrans<>::create("MatFree")->apply(in, out);

    // Create two CPU backed fields and use them with a GPU operator
    // This call implicitly converts the fields to a GPU backend and warns the
    // user about that fact
    in  = Field<double, FieldState::Coeff>::create(blocks);
    out = Field<double, FieldState::Phys>::create(blocks);
    BwdTrans<>::create("CUDA")->apply(in, out);
#endif

    return 0;
}
