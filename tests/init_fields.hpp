#include <boost/test/unit_test_log.hpp>
#include <vector>
#include <string>

#include "Field.hpp"
#include <MultiRegions/ExpList.h>
#include <Operators/OperatorBwdTrans.hpp>

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
 * See "Single test case fixture" on the Boost.Test documentation for more details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */
struct InitFields {
  Field<double, FieldState::Coeff> *fixt_in;
  Field<double, FieldState::Phys> *fixt_out;
  MultiRegions::ExpListSharedPtr fixt_explist {nullptr};
  ~InitFields() {BOOST_TEST_MESSAGE("teardown fixture");}

// // This function has been moved to Field.hpp
//   static std::vector<BlockAttributes>
//   GetBlockAttributes(FieldState state,
// 		     const MultiRegions::ExpListSharedPtr explist) {
//         const int n = explist->GetNumElmts();
//     std::map<std::tuple<LibUtilities::ShapeType,unsigned int,unsigned int>,size_t> blockList;
//     for (int i = 0; i < explist->GetNumElmts(); ++i)
//     {
//         auto e = explist->GetExp(i);
//         blockList[{e->DetShapeType(),e->GetNcoeffs(),e->GetTotPoints()}]++;
//     }
//     std::vector<BlockAttributes> blockAttr;
//     for (auto &x : blockList)
//     {
//         auto val = state == FieldState::Phys ? std::get<2>(x.first) : std::get<1>(x.first);
//         blockAttr.push_back( { x.second, val } );
//     }
//     return blockAttr;
//   }

  InitFields() {
    BOOST_TEST_MESSAGE("Creating input and output fields");
    // Initialise a session, graph and create an expansion list
    LibUtilities::SessionReaderSharedPtr session;
    SpatialDomains::MeshGraphSharedPtr   graph;

    // Construct a fake command-line argument array to be fed to
    // Session::Reader::CreateInstance. The first element stands for
    // the name of the executable which, in our case, doesn't matter.
    int argc = 2;
    char *argv[] = {
      (char*)"exe_name", (char*)"square.xml"
    };

    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph   = SpatialDomains::MeshGraph::Read(session);
    fixt_explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr
                    (session, graph);

    // Generate a blocks definition from the expansion list for each state
    auto blocks_phys  = GetBlockAttributes(FieldState::Phys,  fixt_explist);
    auto blocks_coeff = GetBlockAttributes(FieldState::Coeff, fixt_explist);

    // Create two Field objects with a MemoryRegionCPU backend by default
    auto f_in  = Field<double, FieldState::Coeff>::create(blocks_coeff);
    auto f_out = Field<double, FieldState::Phys >::create(blocks_phys);
    fixt_in  = new Field<double, FieldState::Coeff>(std::move(f_in));
    fixt_out = new Field<double, FieldState::Phys>(std::move(f_out));
  }
};
