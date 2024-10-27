///////////////////////////////////////////////////////////////////////////////
//
// File: init_fields.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <Operators/Field/Field.hpp>
#include <Operators/Utils/UtilsKernels.hpp>

// Currently the BOOST_TEST_DYN_LINK is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_DYN_LINK)
#if !defined(BOOST_TEST_DYN_LINK)
#define LOCALLY_DEFINED_BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#endif
#endif

// Currently the BOOST_TEST_NO_MAIN is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_NO_MAIN)
#if !defined(BOOST_TEST_NO_MAIN)
#define LOCALLY_DEFINED_BOOST_TEST_NO_MAIN
#define BOOST_TEST_NO_MAIN
#endif
#endif

#if defined(BOOST_TEST_DYN_LINK) || defined(BOOST_TEST_NO_MAIN)
#define BOOST_TEST_ALTERNATIVE_INIT_API
#endif

#if defined(BOOST_TEST_DYN_LINK)
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/test/unit_test_log.hpp>

#include <string>
#include <type_traits>
#include <vector>

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

using namespace Nektar::LibUtilities;
using namespace Nektar;

struct GlobalConfiguration
{
    std::string testModule{STRVX(BOOST_TEST_MODULE)};

    GlobalConfiguration()
    {
        [[maybe_unused]] int argc =
            boost::unit_test::framework::master_test_suite().argc;
        [[maybe_unused]] char **argv =
            boost::unit_test::framework::master_test_suite().argv;

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#ifdef NEKTAR_ENABLE_CUDA
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cudaSetDevice(rank);
#endif
#endif

#if defined(NEKTAR_ENABLE_KOKKOS)
        if (testModule.find("Kokkos") != std::string::npos ||
            testModule.find("KOKKOS") != std::string::npos)
        {
            Kokkos::initialize(argc, argv);
        }
#endif
    }

    ~GlobalConfiguration()
    {
#if defined(NEKTAR_ENABLE_KOKKOS)
        if (testModule.find("Kokkos") != std::string::npos ||
            testModule.find("KOKKOS") != std::string::npos)
        {
            Kokkos::finalize();
        }
#endif

#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

/**
 * @struct InitFields
 *
 * A test fixture responsible for making available input and output
 * Field objects.  The structure constructor (destructor) is called
 * before (after) each call to BOOST_FIXTURE_TEST_CASE(<test name>,
 * InitFields) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */

template <typename TData, FieldState stateIn = FieldState::Coeff,
          FieldState stateOut = FieldState::Phys,
          typename TExpList   = MultiRegions::ExpList>
class InitFields
{
public:
    InitFields()
    {
    }

    ~InitFields()
    {
        BOOST_TEST_MESSAGE("teardown fixture");

        if (fixt_in)
        {
            delete fixt_in;
        }
        if (fixt_out)
        {
            delete fixt_out;
        }
        if (fixt_expected)
        {
            delete fixt_expected;
        }

        if (session)
        {
            session->Finalise();
        }
    }

    void Configure(size_t nin = 1, size_t nout = 1)
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");
        // Initialise a session, graph and create an expansion list
        SpatialDomains::MeshGraphSharedPtr graph;

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 2;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = meshName.data();

        session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        graph   = SpatialDomains::MeshGraphIO::Read(session);
        if constexpr (std::is_same_v<TExpList, MultiRegions::ContField>)
        {
            fixt_explist =
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    session, graph, "u", true, false,
                    Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::DisContField>)
        {
            fixt_explist =
                MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                    session, graph, "u", true, true,
                    Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::ExpList>)
        {
            fixt_explist =
                MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                    session, graph, true, "u", Collections::eNoCollection);
        }

        // Create two Field objects with a MemoryRegionHost backend by default
        auto blocks_in  = GetBlockAttributes<TData>(stateIn, fixt_explist);
        auto blocks_out = GetBlockAttributes<TData>(stateOut, fixt_explist);
        if (testModule.find("AVX") != std::string::npos)
        {
            alignment = NektarSpaces::AVX::alignment;
        }
        else if (testModule.find("CUDA") != std::string::npos)
        {
            alignment = NektarSpaces::CUDA::alignment;
        }
        else if (testModule.find("SYCL") != std::string::npos)
        {
            alignment = NektarSpaces::SYCL::alignment;
        }
        else if (testModule.find("Kokkos") != std::string::npos ||
                 testModule.find("KOKKOS") != std::string::npos)
        {
            alignment = NektarSpaces::KOKKOS::alignment;
        }
        else
        {
            alignment = NektarSpaces::Serial::alignment;
        }

        auto f_in =
            Field<TData, stateIn>::template create<NektarSpaces::DeviceSpace>(
                "f_in", blocks_in, nin, alignment);
        auto f_out =
            Field<TData, stateOut>::template create<NektarSpaces::DeviceSpace>(
                "f_out", blocks_out, nout, alignment);
        auto f_expected =
            Field<TData, stateOut>::template create<NektarSpaces::HostSpace>(
                "f_expected", blocks_out, nout, alignment);
        fixt_in       = new Field<TData, stateIn>(std::move(f_in));
        fixt_out      = new Field<TData, stateOut>(std::move(f_out));
        fixt_expected = new Field<TData, stateOut>(std::move(f_expected));
    }

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     *
     * @return bool
     */
    bool Compare(TData tol)
    {
        if (fixt_expected->GetNumComponents() != fixt_out->GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (fixt_expected->GetBlocks().size() != fixt_out->GetBlocks().size())
        {
            std::cout << "Mismatch of block size." << std::endl;
            return false;
        }

        ReshapeToScalar(*fixt_out);
        ReshapeToScalar(*fixt_expected);

        const TData *in_store =
            fixt_out->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        const TData *ref_store =
            fixt_expected->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        bool isMatch = true;

        printf(
            "#elm #pts output               expected            difference\n");
        for (size_t bl = 0; bl < fixt_out->GetBlocks().size(); ++bl)
        {
            if ((fixt_out->GetBlocks()[bl].num_elements !=
                 fixt_expected->GetBlocks()[bl].num_elements) ||
                (fixt_out->GetBlocks()[bl].num_pts !=
                 fixt_expected->GetBlocks()[bl].num_pts))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            size_t MisMatchcnt = 0, total = 0;

            for (size_t component = 0; component < fixt_out->GetNumComponents();
                 ++component)
            {
                for (size_t el = 0; el < fixt_out->GetBlocks()[bl].num_elements;
                     ++el)
                {
                    for (size_t pts = 0;
                         pts < fixt_out->GetBlocks()[bl].num_pts; ++pts)
                    {
                        if (std::abs(*in_store - *ref_store) > tol)
                        {
                            printf("%04zu %04zu %20.16f %20.16f %20.16f\n", el,
                                   pts, *in_store, *ref_store,
                                   std::abs(*in_store - *ref_store));
                            MisMatchcnt++;
                        }
                        total++;
                        in_store++;
                        ref_store++;
                    }
                }

                in_store += fixt_out->GetBlocks()[bl].num_pts *
                            fixt_out->GetBlocks()[bl].num_padding_elements;
                ref_store +=
                    fixt_expected->GetBlocks()[bl].num_pts *
                    fixt_expected->GetBlocks()[bl].num_padding_elements;
            }

            if (MisMatchcnt)
            {
                std::cout << "Number of mismatches in block " << bl << " is "
                          << MisMatchcnt << " out of " << total << std::endl;

                isMatch = false;
            }
        }

        if (isMatch)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void ReshapeToScalar(Field<TData, stateOut> &in)
    {
        auto ptr = in.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        for (auto &block : in.GetBlocks())
        {
            int numElmtsPad = block.num_elements + block.num_padding_elements;
            for (int component = 0; component < in.GetNumComponents();
                 component++)
            {
                ReshapeStorage<NektarSpaces::Serial, 1>(
                    block.GetInterleaveWidth(), numElmtsPad, block.num_pts,
                    ptr + component * block.block_size);
            }

            block.SetInterleaveWidth(1);

            // Increment pointer and index for next block.
            ptr += block.block_size * in.GetNumComponents();
        }
    }

protected:
    std::string meshName                  = "";
    Field<TData, stateIn> *fixt_in        = nullptr;
    Field<TData, stateOut> *fixt_out      = nullptr;
    Field<TData, stateOut> *fixt_expected = nullptr;
    std::shared_ptr<TExpList> fixt_explist{nullptr};
    size_t alignment;
    LibUtilities::SessionReaderSharedPtr session;
    std::string testModule{STRVX(BOOST_TEST_MODULE)};
};

#if defined(LOCALLY_DEFINED_BOOST_TEST_DYN_LINK)
#undef BOOST_TEST_DYN_LINK
#endif

#if defined(LOCALLY_DEFINED_BOOST_TEST_NO_MAIN)
#undef BOOST_TEST_NO_MAIN
#endif
