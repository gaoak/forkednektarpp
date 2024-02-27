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
#include <boost/test/unit_test_log.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <MultiRegions/ExpList.h>
#include <Operators/Field.hpp>

#ifdef NEKTAR_ENABLE_CUDA
#include "Operators/MemoryRegionCUDA.hpp"
#endif

#include <boost/test/included/unit_test.hpp>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

#ifdef NEKTAR_USE_MPI
struct InitMPI
{
    InitMPI()
    {
        int argc    = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        MPI_Init(&argc, &argv);
    }
    ~InitMPI()
    {
        MPI_Finalize();
    }
};

BOOST_TEST_GLOBAL_CONFIGURATION(InitMPI);
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
#ifdef NEKTAR_ENABLE_CUDA
        if (fixtcuda_in)
        {
            delete fixtcuda_in;
        }
        if (fixtcuda_out)
        {
            delete fixtcuda_out;
        }
#endif
        if (session)
        {
            session->Finalise();
        }
    }

    InitFields() = default;

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
        graph   = SpatialDomains::MeshGraph::Read(session);
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

        if constexpr (std::is_same_v<TExpList, MultiRegions::ExpList>)
        {
            fixt_explist =
                MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                    session, graph, true, "u", Collections::eNoCollection);
        }

        // Generate a blocks definition from the expansion list for each state
        using vec_t = tinysimd::simd<double>;

        auto blocks_in =
            GetBlockAttributes(stateIn, fixt_explist, vec_t::width);
        auto blocks_out =
            GetBlockAttributes(stateOut, fixt_explist, vec_t::width);

        // Create two Field objects with a MemoryRegionCPU backend by default
        auto f_in =
            Field<TData, stateIn>::create(blocks_in, nin, vec_t::alignment);
        auto f_out =
            Field<TData, stateOut>::create(blocks_out, nout, vec_t::alignment);
        auto f_expected =
            Field<TData, stateOut>::create(blocks_out, nout, vec_t::alignment);
        fixt_in       = new Field<TData, stateIn>(std::move(f_in));
        fixt_out      = new Field<TData, stateOut>(std::move(f_out));
        fixt_expected = new Field<TData, stateOut>(std::move(f_expected));
#ifdef NEKTAR_ENABLE_CUDA
        auto fcuda_in =
            Field<TData, stateIn>::template create<MemoryRegionCUDA>(blocks_in,
                                                                     nin);
        auto fcuda_out =
            Field<TData, stateOut>::template create<MemoryRegionCUDA>(
                blocks_out, nout);
        fixtcuda_in  = new Field<TData, stateIn>(std::move(fcuda_in));
        fixtcuda_out = new Field<TData, stateOut>(std::move(fcuda_out));
#endif
    }

    void OutputIfNotMatch(double *outptr, double *expptr, double tol)
    {
        printf(
            "#elm #pts output               expected            difference\n");
        for (auto const &block : fixt_out->GetBlocks())
        {
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    if (fabs(*outptr - *expptr) > tol)
                    {
                        printf("%04zu %04zu %20.16f %20.16f %20.16f\n", el,
                               phys, *outptr, *expptr, fabs(*outptr - *expptr));
                    }
                    expptr++;
                    outptr++;
                }
            }
            for (size_t el = 0; el < block.num_padding_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys)
                {
                    expptr++;
                    outptr++;
                }
            }
        }
    }

protected:
    std::string meshName                  = "";
    Field<TData, stateIn> *fixt_in        = nullptr;
    Field<TData, stateOut> *fixt_out      = nullptr;
    Field<TData, stateOut> *fixt_expected = nullptr;
#ifdef NEKTAR_ENABLE_CUDA
    Field<TData, stateIn> *fixtcuda_in   = nullptr;
    Field<TData, stateOut> *fixtcuda_out = nullptr;
#endif
    std::shared_ptr<TExpList> fixt_explist{nullptr};

    LibUtilities::SessionReaderSharedPtr session;
};
