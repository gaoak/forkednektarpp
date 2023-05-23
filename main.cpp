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

#include <LibUtilities/BasicUtils/SessionReader.h>
//#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <SpatialDomains/MeshGraph.h>
#include <MultiRegions/ExpList.h>

#ifdef NEKTAR_USE_CUDA
#include "MemoryRegionCUDA.hpp"
#endif

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TType, FieldState State>
std::ostream &operator<<(std::ostream &stream, Field<TType, State> &f)
{
    auto *x = f.GetStorage().GetCPUPtr();
    for (size_t i = 0; i < f.GetStorage().size(); ++i)
    {
        stream << x[i] << ' ';
    }
    stream << std::endl;

    return stream;
}


int main(int argc, char *argv[])
{
    // Initialise a session, graph and create an expansion list
    LibUtilities::SessionReaderSharedPtr session;
    SpatialDomains::MeshGraphSharedPtr   graph;
    MultiRegions::ExpListSharedPtr       explist;

    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph   = SpatialDomains::MeshGraph::Read(session);
    explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr
                    (session, graph);

    // Generate a blocks definition from the expansion list for each state
    auto blocks_phys  = GetBlockAttributes(FieldState::Phys,  explist);
    auto blocks_coeff = GetBlockAttributes(FieldState::Coeff, explist);

    // Create two Field objects with a MemoryRegionCPU backend by default
    auto in  = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto in2 = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto out = Field<double, FieldState::Phys >::create(blocks_phys);

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
    double *x = in.GetStorage().GetCPUPtr();
    for (auto const &block : blocks_coeff)
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                // Each element is the index of the degree of freedom
                // this is useful for testing reshapes
                *(x++) = coeff;
            }
        }
    }

    memset(out.GetStorage().GetCPUPtr(), 0, out.GetStorage().size()*sizeof(double));

    std::cout << "Initial shape:\n" << in << std::endl << std::endl;

    // Test out field reshaping
    in.ReshapeStorage<4>();
    std::cout << "Reshaped to 4:\n" << in << std::endl << std::endl;

#ifdef NEKTAR_ENABLE_SIMD_AVX2
    // Test out SIMD instructions
    using vec_t = tinysimd::simd<double>;

    // Reshape into vec_t::width, with the right memory alignment requirements
    in.ReshapeStorage<vec_t::width, vec_t::alignment>();
    std::cout << "SIMD In:\n" << in << std::endl << std::endl;

    in2.ReshapeStorage<vec_t::width, vec_t::alignment>();
    std::cout << "SIMD Out (before):\n" << out << std::endl << std::endl;

    // Reinterpret casting from double to vector type allows for efficient
    // conversion to SIMD intrinsics
    vec_t::vectorType *inptr =
        reinterpret_cast<vec_t::vectorType *>(in.GetStorage().GetCPUPtr());
    vec_t::scalarType *in2ptr = in2.GetStorage().GetCPUPtr();

    // Loop over each block in the field and square each element
    for (auto const &block : blocks_coeff)
    {
        const size_t numMetaBlocks = block.num_elements / vec_t::width;
        for (size_t metaBlock = 0; metaBlock < numMetaBlocks * block.num_pts;
             ++metaBlock)
        {
            (vec_t(inptr[metaBlock]) * vec_t(inptr[metaBlock])).store(in2ptr);
            in2ptr += vec_t::width;
        }

        inptr += numMetaBlocks * block.num_pts;
    }

    // Back to non-interleaved for non-SIMD Operators
    in.ReshapeStorage<1>();
    in2.ReshapeStorage<1>();

    std::cout << "Out:\n" << in2 << std::endl;
#endif

    // Operators are instantiated using a Factory pattern. First lets check
    // which operators have been registered with the Factory.
    GetOperatorFactory<double>().PrintAvailableClasses();

    // We can create a BwdTrans operator (default implementation)
    // Default implementation might be provided through configuration options,
    // benchmarking, etc eventually.
    auto bt = BwdTrans<>::create(explist);
    // ...and then apply it
    bt->apply(in, out);

    // We can combine these for one-shot operations
    BwdTrans<>::create(explist)->apply(in, out);
    // We can also explicitly select the implementation of the operator to use
    BwdTrans<>::create(explist, "SumFac")->apply(in, out);

    // Let's display the result
    std::cout << out << std::endl;

    in              = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto &inStorage = in.GetStorage();
    std::fill(inStorage.GetCPUPtr(), inStorage.GetCPUPtr() + inStorage.size(),
              0);
    out = Field<double, FieldState::Phys>::create(blocks_phys);

    auto &outStorage = out.GetStorage();
    std::fill(outStorage.GetCPUPtr(),
              outStorage.GetCPUPtr() + outStorage.size(), 0);

    for (auto const &block : in.GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            in.GetStorage().GetCPUPtr()[el * block.num_pts] = 1;
        }
    }

    std::cout << in << std::endl;

    BwdTrans<>::create(explist, "StdMat")->apply(in, out);

    std::cout << out << std::endl;


#ifdef NEKTAR_USE_CUDA

    // Test CUDA MemoryRegion

    // Create two Fields with memory on the GPU
    in  = Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(blocks_coeff);
    out = Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(blocks_phys);

    // Perform the BwdTrans on the fields using the CUDA implementation
    // Since this is a CUDA operator, acting on CUDA fields, everything happens
    // on the GPU.
    BwdTrans<>::create(explist, "CUDA")->apply(in, out);

    // Test the GPU-backed fields with a CPU operator
    // This should show a debug warning due to the implicit conversion. The
    // purpose of this is to allow us to transition the code to the new
    // infrastructure and add CUDA operators, without the need to add ALL CUDA
    // operators before we can test anything.
    BwdTrans<>::create(explist, "MatFree")->apply(in, out);

    // Create two CPU backed fields and use them with a GPU operator
    // This call implicitly converts the fields to a GPU backend and warns the
    // user about that fact
    in  = Field<double, FieldState::Coeff>::create(blocks);
    out = Field<double, FieldState::Phys>::create(blocks);
    BwdTrans<>::create(explist, "CUDA")->apply(in, out);
#endif

    std::cout << "END" << std::endl;
    return 0;
}
