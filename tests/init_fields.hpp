#pragma once
#include <boost/test/unit_test_log.hpp>
#include <string>
#include <vector>

#include "Field.hpp"
#include <MultiRegions/ExpList.h>

#ifdef NEKTAR_USE_CUDA
#include "MemoryRegionCUDA.hpp"
#endif

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

/**
 * @struct InitFields
 *
 * A test fixture responsible for making available input and output
 * Field objects.  The structure constructor (destrucor) is called
 * before (after) each call to BOOST_FIXTURE_TEST_CASE(<test name>,
 * InitFields) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */

template <typename TData, FieldState stateIn = FieldState::Coeff,
          FieldState stateOut = FieldState::Phys>
class InitFields
{
public:
    Field<TData, stateIn> *fixt_in        = nullptr;
    Field<TData, stateOut> *fixt_out      = nullptr;
    Field<TData, stateOut> *fixt_expected = nullptr;
#ifdef NEKTAR_USE_CUDA
    Field<TData, stateIn> *fixtcuda_in   = nullptr;
    Field<TData, stateOut> *fixtcuda_out = nullptr;
#endif
    MultiRegions::ExpListSharedPtr fixt_explist{nullptr};

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
#ifdef NEKTAR_USE_CUDA
        if (fixtcuda_in)
        {
            delete fixtcuda_in;
        }
        if (fixtcuda_out)
        {
            delete fixtcuda_out;
        }
#endif
    }

    InitFields() = default;

    void Configure(size_t nin = 1, size_t nout = 1)
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");
        // Initialise a session, graph and create an expansion list
        LibUtilities::SessionReaderSharedPtr session;
        SpatialDomains::MeshGraphSharedPtr graph;

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc     = 2;
        char *argv[] = {(char *)"exe_name", meshName.data()};

        session      = LibUtilities::SessionReader::CreateInstance(argc, argv);
        graph        = SpatialDomains::MeshGraph::Read(session);
        fixt_explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            session, graph);

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
#ifdef NEKTAR_USE_CUDA
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
        }
    }

protected:
    std::string meshName = "";
};
