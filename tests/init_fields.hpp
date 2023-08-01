#pragma once
#include <boost/test/unit_test_log.hpp>
#include <string>
#include <vector>

#include "Field.hpp"
#include <MultiRegions/ExpList.h>
#include <Operators/OperatorBwdTrans.hpp>
#include <Operators/OperatorIProductWRTBase.hpp>

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

    void Configure()
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
        auto blocks_in  = GetBlockAttributes(stateIn, fixt_explist);
        auto blocks_out = GetBlockAttributes(stateOut, fixt_explist);

        // Create two Field objects with a MemoryRegionCPU backend by default
        auto f_in       = Field<TData, stateIn>::create(blocks_in);
        auto f_out      = Field<TData, stateOut>::create(blocks_out);
        auto f_expected = Field<TData, stateOut>::create(blocks_out);
        fixt_in         = new Field<TData, stateIn>(std::move(f_in));
        fixt_out        = new Field<TData, stateOut>(std::move(f_out));
        fixt_expected   = new Field<TData, stateOut>(std::move(f_expected));
#ifdef NEKTAR_USE_CUDA
        auto fcuda_in =
            Field<TData, stateIn>::template create<MemoryRegionCUDA>(blocks_in);
        auto fcuda_out =
            Field<TData, stateOut>::template create<MemoryRegionCUDA>(
                blocks_out);
        fixtcuda_in  = new Field<TData, stateIn>(std::move(fcuda_in));
        fixtcuda_out = new Field<TData, stateOut>(std::move(fcuda_out));
#endif
    }

protected:
    std::string meshName = "";
};
