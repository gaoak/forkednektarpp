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
#include <cmath>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorIdentity.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorFwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
//#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraph.h>

#ifdef NEKTAR_USE_CUDA
#include "MemoryRegionCUDA.hpp"
#endif

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;
using namespace Nektar::MultiRegions;

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
    SpatialDomains::MeshGraphSharedPtr graph;
    MultiRegions::ExpListSharedPtr explist;

    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph   = SpatialDomains::MeshGraph::Read(session);
    explist =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(session, graph);

    // Generate a blocks definition from the expansion list for each state
    auto blocks_phys  = GetBlockAttributes(FieldState::Phys, explist);
    auto blocks_coeff = GetBlockAttributes(FieldState::Coeff, explist);

    // Create two Field objects with a MemoryRegionCPU backend by default
    auto in  = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto in2 = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto out = Field<double, FieldState::Phys>::create(blocks_phys);

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

    memset(out.GetStorage().GetCPUPtr(), 0,
           out.GetStorage().size() * sizeof(double));

    std::cout << "Initial shape:\n" << in << std::endl << std::endl;

    // ****************************************************************************

    // (willdenny) Check identity operator
    //std::cout << "in2 before identity with in\n" << in2 << "\n\n";
    //Identity<double, FieldState::Coeff>::create(explist, "")->apply(in, in2);
    //std::cout << "in2 after identity with in\n"  << in2 << "\n\n";

    // ****************************************************************************

    // (willdenny) Check CG operator
    
    std::shared_ptr<ContField> explistCF = std::dynamic_pointer_cast<ContField>(explist);
    std::shared_ptr<AssemblyMapCG> assMap = explistCF->GetLocalToGlobalMap();

    // get size of global/local systems
    size_t Nglobal = assMap->GetNumGlobalCoeffs();
    size_t Nlocal = assMap->GetNumLocalCoeffs();

    auto CG_f     = Field<double, FieldState::Phys>::create(blocks_phys);
    auto CG_u_hat = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto CG_f2    = Field<double, FieldState::Phys >::create(blocks_phys);

    auto *iter = CG_f.GetStorage().GetCPUPtr();

    int CG_test = 1;
    
    if (CG_test == 0)
    {
        for (int i = 0; i < explist->GetTotPoints(); ++i)
            *(iter++) = 1;
    }
    else if (CG_test == 1)
    {    
        int np = explist->GetTotPoints();
        Array<OneD, NekDouble> x(np), y(np), z(np);
        explist->GetCoords(x, y, z);
        for (int i = 0; i < explist->GetTotPoints(); ++i)
            *(iter++) = x[i]*x[i] + y[i]*y[i] + 20;
    }

    FwdTrans<double>::create(explist)->apply(CG_f, CG_u_hat);
    BwdTrans<double>::create(explist)->apply(CG_u_hat, CG_f2);

    std::cout << "f:\n" << CG_f << "\n";
    std::cout << "u hat:\n" << CG_u_hat << "\n";
    std::cout << "f2:\n" << CG_f2 << "\n";

    // ****************************************************************************

    // Test SIMD Implementation
#ifdef NEKTAR_ENABLE_SIMD_AVX2
    {
        // Test out field reshaping
        in.ReshapeStorage<4>();
        std::cout << "Reshaped to 4:\n" << in << std::endl << std::endl;

        // Test out SIMD instructions
        using vec_t = tinysimd::simd<double>;

        // Reshape into vec_t::width, with the right memory alignment
        // requirements
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
            for (size_t metaBlock = 0;
                 metaBlock < numMetaBlocks * block.num_pts; ++metaBlock)
            {
                (vec_t(inptr[metaBlock]) * vec_t(inptr[metaBlock]))
                    .store(in2ptr);
                in2ptr += vec_t::width;
            }

            inptr += numMetaBlocks * block.num_pts;
        }

        // Back to non-interleaved for non-SIMD Operators
        in.ReshapeStorage<1>();
        in2.ReshapeStorage<1>();

        std::cout << "Out:\n" << in2 << std::endl;
    }
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

    // Test BwdTrans Implementation
    {
        std::cout << "BwdTrans (StdMat) test starts." << std::endl;

        // Create two Fields with memory on the CPU.
        auto inCoeff = Field<double, FieldState::Coeff>::create(blocks_coeff);
        auto outPhys = Field<double, FieldState::Phys>::create(blocks_phys);

        // Assign input values from the CPU.
        auto *inptr = inCoeff.GetStorage().GetCPUPtr();
        for (auto const &block : inCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    *(inptr++) = coeff + 1;
                }
            }
        }

        std::cout << "Initial shape:\n" << inCoeff << std::endl;

        // Perform the BwdTrans on the fields.
        BwdTrans<>::create(explist, "StdMat")->apply(inCoeff, outPhys);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto outptr = outPhys.GetStorage().GetCPUPtr();
        for (auto const &block : out.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    std::cout << *(outptr++) << ' ';
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    // Test BwdTrans (CUDA) Implementation
#ifdef NEKTAR_USE_CUDA
    {
        std::cout << "BwdTrans (CUDA) test starts." << std::endl;

        // Create two Fields with memory on the GPU.
        auto inCoeff =
            Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(
                blocks_coeff);
        auto outPhys =
            Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(
                blocks_phys);

        // Assign input values from the CPU.
        std::cout << "Initial shape: " << std::endl;
        auto *inptr =
            inCoeff.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : inCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    *(inptr++) = coeff + 1;
                    std::cout << coeff + 1 << " ";
                }
            }
        }
        std::cout << std::endl << std::endl;

        // Perform the BwdTrans on the fields using the CUDA implementation
        // Since this is a CUDA operator, acting on CUDA fields, everything
        // happens on the GPU.
        BwdTrans<>::create(explist, "CUDA")->apply(inCoeff, outPhys);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto *outptr =
            outPhys.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : outPhys.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    std::cout << *(outptr++) << ' ';
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif
    // Test IProductWRTBase Implementation
    {
        std::cout << "IProductWRTBase (StdMat) test starts." << std::endl;

        // Create two Field objects with a MemoryRegionCPU backend by default
        // for the inner product with respect to base
        auto inPhys   = Field<double, FieldState::Phys>::create(blocks_phys);
        auto outCoeff = Field<double, FieldState::Coeff>::create(blocks_coeff);

        // Assign input values
        auto *inptr = inPhys.GetStorage().GetCPUPtr();
        for (auto const &block : inPhys.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    // Each element is the index of the quadrature points
                    // this is useful for testing reshapes
                    *(inptr++) = phys;
                }
            }
        }

        std::cout << "Initial shape:\n" << inPhys << std::endl;

        // IProductWRTBase
        IProductWRTBase<>::create(explist, "StdMat")->apply(inPhys, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto *outptr = outCoeff.GetStorage().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << ' ';
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "END" << std::endl;
    return 0;
}
