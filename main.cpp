/*i*
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
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorIProductWRTDerivBase.hpp"
#include "Operators/OperatorPhysDeriv.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include <LibUtilities/BasicUtils/Timer.h>

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
        // if (i > 20)
        // {
        //     stream << "...";
        //     break;
        // }
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
        MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(session, graph);

    // Generate a blocks definition from the expansion list for each state
    // Test out SIMD instructions
    using vec_t = tinysimd::simd<double>;
    auto blocks_phys =
        GetBlockAttributes(FieldState::Phys, explist, vec_t::width);
    auto blocks_coeff =
        GetBlockAttributes(FieldState::Coeff, explist, vec_t::width);

    // Create two Field objects with a MemoryRegionCPU backend by default
    auto in  = Field<double, FieldState::Coeff>::create(blocks_coeff, 1,
                                                       vec_t::alignment);
    auto in2 = Field<double, FieldState::Coeff>::create(blocks_coeff, 1,
                                                        vec_t::alignment);
    auto out = Field<double, FieldState::Phys>::create(blocks_phys, 1,
                                                       vec_t::alignment);

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
        for (size_t el = 0; el < block.num_padding_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                *(x++) = 0.0;
            }
        }
    }

    // initialize out to zero
    x = out.GetStorage().GetCPUPtr();
    for (auto const &block : blocks_phys)
    {
        for (size_t pt = 0; pt < block.block_size; ++pt)
        {
            *(x++) = 0.0;
        }
    }

    memset(out.GetStorage().GetCPUPtr(), 0,
           out.GetStorage().size() * sizeof(double));

    std::cout << "Initial In shape:\n" << in << std::endl << std::endl;

    // Test SIMD Implementation
#ifdef NEKTAR_ENABLE_SIMD_AVX2
    {
        // Test out field reshaping
        in.ReshapeStorage<2>();
        std::cout << "Reshape In to 2:\n" << in << std::endl << std::endl;

        // Reshape into vec_t::width, with the right memory alignment
        // requirements
        in.ReshapeStorage<vec_t::width>();
        std::cout << "SIMD In:\n" << in << std::endl << std::endl;

        in2.ReshapeStorage<vec_t::width>();

        // Reinterpret casting from double to vector type allows for efficient
        // conversion to SIMD intrinsics
        vec_t::vectorType *inptr =
            reinterpret_cast<vec_t::vectorType *>(in.GetStorage().GetCPUPtr());

        vec_t::scalarType *in2ptr = in2.GetStorage().GetCPUPtr();
        // Loop over each block in the field and square each element
        for (auto const &block : blocks_coeff)
        {
            if ((block.num_elements + block.num_padding_elements) %
                    vec_t::width !=
                0)
            {
                break; // Now with paddings this should not happen
            }
            else
            {
                const size_t numMetaBlocks =
                    (block.num_elements + block.num_padding_elements) /
                    vec_t::width;
                for (size_t metaBlock = 0;
                     metaBlock < numMetaBlocks * block.num_pts; ++metaBlock)
                {
                    (vec_t(inptr[metaBlock]) * vec_t(inptr[metaBlock]))
                        .store(in2ptr);
                    in2ptr += vec_t::width;
                }
                inptr += numMetaBlocks * block.num_pts;
            }
        }

        // Back to non-interleaved for non-SIMD Operators
        in.ReshapeStorage<1>();
        std::cout << "Reshape In back to 1:\n" << in << std::endl << std::endl;

        in2.ReshapeStorage<1>();
        std::cout << "In2 is the square of In:\n" << in2 << std::endl;
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
    // BwdTrans<>::create(explist, "SumFac")->apply(in, out);

    // Let's display the result
    std::cout << out << std::endl;
    std::cout << std::endl;

    // Test BwdTrans Implementation
    {
        std::cout << "BwdTrans (StdMat) test starts." << std::endl;

        LibUtilities::Timer timer;

        // Create two Fields with memory on the CPU.
        auto inCoeff = Field<double, FieldState::Coeff>::create(
            blocks_coeff, 1, vec_t::alignment);
        auto outPhys = Field<double, FieldState::Phys>::create(
            blocks_phys, 1, vec_t::alignment);

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
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    *(inptr++) = 0.0;
                }
            }
        }
        // initialize out to zero
        double *outptr = outPhys.GetStorage().GetCPUPtr();
        for (auto const &block : outPhys.GetBlocks())
        {
            for (size_t pt = 0; pt < block.block_size; ++pt)
            {
                *(outptr++) = 0.0;
            }
        }

        std::cout << "Initial shape:\n" << inCoeff << std::endl;

        // Perform the BwdTrans on the fields.
        auto op = BwdTrans<>::create(explist, "StdMat");
        timer.Start();
        op->apply(inCoeff, outPhys);
        timer.Stop();
        std::cout << ">>> Time for BwdTrans StdMat: " << timer.TimePerTest(1)
                  << " s \n"<< std::endl;

        // Check output values.
        std::cout << "Out (StdMat):" << std::endl;
        std::cout << outPhys << std::endl;

        // Perform the BwdTrans by matfree
        auto outPhys2 = Field<double, FieldState::Phys>::create(
            blocks_phys, 1, vec_t::alignment);
        // Make sure field is reshaped before calling apply()
        // So that we can get true performance of the operator
        inCoeff.ReshapeStorage<vec_t::width>();
        outPhys2.ReshapeStorage<vec_t::width>();
        auto op2 = BwdTrans<>::create(explist, "MatFree");
        timer.Start();
        op2->apply(inCoeff, outPhys2);
        timer.Stop();
        std::cout << ">>> Time for BwdTrans MatFree: " << timer.TimePerTest(1)
                  << " s \n"<< std::endl;
        
        // Check output values.
        std::cout << "Initial shape (MatFree):\n" << inCoeff << std::endl;
        std::cout << "Out (MatFree):" << std::endl;
        std::cout << outPhys2 << std::endl;
        // We can compare any two field of same storage - not necessary to be
        // scalar
        outPhys.ReshapeStorage<vec_t::width>();
        if (outPhys2.compare(outPhys, 1e-9))
        {
            std::cout << "Results match! (BwdTrans MatFree/StdMat) \n"
                      << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (BwdTrans MatFree/StdMat)\n"
                      << std::endl;
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
                    std::cout << *(outptr++) << " ";
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

        LibUtilities::Timer timer;

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
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    *(inptr++) = 0.0;
                }
            }
        }

        std::cout << "Initial shape:\n" << inPhys << std::endl;

        // IProductWRTBase
        auto op = IProductWRTBase<>::create(explist, "StdMat");
        timer.Start();
        op->apply(inPhys, outCoeff);
        timer.Stop();
        std::cout << ">>> Time for IProductWRTBase StdMat: " << timer.TimePerTest(1)
                  << " s \n"<< std::endl;
        // Check output values.
        std::cout << "Out (StdMat):" << std::endl;
        std::cout << outCoeff << std::endl;

        // IProductWRTBase by MatrixFree
        auto outCoeff2 = Field<double, FieldState::Coeff>::create(
            blocks_coeff, 1, vec_t::alignment);
        // Make sure field is reshaped before calling apply()
        // So that we can get true performance of the operator
        inPhys.ReshapeStorage<vec_t::width>();
        outCoeff2.ReshapeStorage<vec_t::width>();
        auto op2 = IProductWRTBase<>::create(explist, "MatFree");
        timer.Start();
        op2->apply(inPhys, outCoeff2);
        timer.Stop();
        std::cout << ">>> Time for IProductWRTBase MatFree: " << timer.TimePerTest(1)
                  << " s \n"<< std::endl;
        // Check output values.
        std::cout << "Out (MatFree):" << std::endl;
        std::cout << outCoeff2 << std::endl;
        // We can compare any two field of same storage - not necessary to be
        // scalar
        outCoeff2.ReshapeStorage<1>();
        if (outCoeff.compare(outCoeff2, 1e-9))
        {
            std::cout << "Results match! (IProductWRTBase MatFree/StdMat) \n"
                      << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (IProductWRTBase MatFree/StdMat)\n"
                      << std::endl;
        }
        std::cout << std::endl;
    }
    // Test IProductWRTBase (CUDA) Implementation
#ifdef NEKTAR_USE_CUDA
    {
        std::cout << "IProductWRTBase (CUDA) test starts." << std::endl;

        // Create two Fields with memory on the GPU
        auto inPhys = Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(
            blocks_phys);
        auto outCoeff =
            Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(
                blocks_coeff);

        // Assign input values.
        std::cout << "Initial shape: " << std::endl;
        auto *inptr =
            inPhys.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : inPhys.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    // Each element is the index of the quadrature points
                    // this is useful for testing reshapes
                    *(inptr++) = phys;
                    std::cout << phys << " ";
                }
            }
        }
        std::cout << std::endl << std::endl;

        // Perform the IProductWRTBase on the fields using the CUDA
        // implementation Since this is a CUDA operator, acting on CUDA fields,
        // everything happens on the GPU.
        IProductWRTBase<>::create(explist, "CUDA")->apply(inPhys, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto *outptr =
            outCoeff.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif
    // Test PhysDeriv Implementation
    {
        std::cout << "PhysDeriv (StdMat) test starts." << std::endl;

        // Create two Field objects with a MemoryRegionCPU backend by default
        // for the inner product with respect to base
        auto inPhys  = Field<double, FieldState::Phys>::create(blocks_phys);
        auto outPhys = Field<double, FieldState::Phys>::create(
                               blocks_phys, explist->GetCoordim(0));

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

        // PhysDeriv
        PhysDeriv<>::create(explist, "StdMat")->apply(inPhys, outPhys);

        // Check output values.
        auto *outptr = outPhys.GetStorage().GetCPUPtr();
        for (size_t d = 0; d < outPhys.GetNumComponents(); ++d)
        {
            std::cout << "Out" << d << ":" << std::endl;
            for (auto const &block : outPhys.GetBlocks())
            {
                for (size_t el = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        std::cout << *(outptr++) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        std::cout << "IProductWRTDerivBase (StdMat) test starts." << std::endl;

        // Create two Field objects with a MemoryRegionCPU backend by default
        // for the inner product with respect to deriv base
        auto outCoeff = Field<double, FieldState::Coeff>::create(blocks_coeff);

        // IProductWRTDerivBase
        IProductWRTDerivBase<>::create(explist, "StdMat")
            ->apply(outPhys, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        outptr = outCoeff.GetStorage().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    // Test PhysDeriv (CUDA) Implementation
#ifdef NEKTAR_USE_CUDA
    {
        std::cout << "PhysDeriv (CUDA) test starts." << std::endl;

        // Create two Field objects with a MemoryRegionCPU backend by default
        // for the inner product with respect to base
        auto inPhys = Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(
            blocks_phys);
        auto outPhys =
            Field<double, FieldState::Phys>::create<MemoryRegionCUDA>(
                blocks_phys, 3);

        // Assign input values
        std::cout << "Initial shape: " << std::endl;
        auto *inptr =
            inPhys.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : inPhys.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    // Each element is the index of the quadrature points
                    // this is useful for testing reshapes
                    *(inptr++) = phys;
                    std::cout << phys << " ";
                }
            }
        }
        std::cout << std::endl << std::endl;

        // PhysDeriv
        PhysDeriv<>::create(explist, "CUDA")->apply(inPhys, outPhys);

        // Check output values.
        auto *outptr =
            outPhys.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (size_t d = 0; d < outPhys.GetNumComponents(); ++d)
        {
            std::cout << "Out" << d << ":" << std::endl;
            for (auto const &block : outPhys.GetBlocks())
            {
                for (size_t el = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys)
                    {
                        std::cout << *(outptr++) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        std::cout << "IProductWRTDerivBase (CUDA) test starts." << std::endl;

        // Create two Field objects with a MemoryRegionCPU backend by default
        // for the inner product with respect to deriv base
        auto outCoeff =
            Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(
                blocks_coeff);

        // IProductWRTDerivBase
        IProductWRTDerivBase<>::create(explist, "CUDA")
            ->apply(outPhys, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        outptr =
            outCoeff.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif
    // Test Helmholtz Implementation
    {
        std::cout << "Helmholtz (StdMat) test starts." << std::endl;

        // Create two Fields with memory on the CPU.
        auto inCoeff  = Field<double, FieldState::Coeff>::create(blocks_coeff);
        auto outCoeff = Field<double, FieldState::Coeff>::create(blocks_coeff);

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

        // Perform the Helmholtz on the fields.
        Helmholtz<>::create(explist, "StdMat")->apply(inCoeff, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto outptr = outCoeff.GetStorage().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    // Test Helmholtz (CUDA) Implementation
#ifdef NEKTAR_USE_CUDA
    {
        std::cout << "Helmholtz (CUDA) test starts." << std::endl;

        // Create two Fields with memory on the GPU.
        auto inCoeff =
            Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(
                blocks_coeff);
        auto outCoeff =
            Field<double, FieldState::Coeff>::create<MemoryRegionCUDA>(
                blocks_coeff);

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

        // Perform the Helmholtz on the fields using the CUDA implementation
        // Since this is a CUDA operator, acting on CUDA fields, everything
        // happens on the GPU.
        Helmholtz<>::create(explist, "CUDA")->apply(inCoeff, outCoeff);

        // Check output values.
        std::cout << "Out:" << std::endl;
        auto *outptr =
            outCoeff.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
        for (auto const &block : outCoeff.GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
                {
                    std::cout << *(outptr++) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

    std::cout << "END" << std::endl;

    // Finalise session
    session->Finalise();

    return 0;
}
