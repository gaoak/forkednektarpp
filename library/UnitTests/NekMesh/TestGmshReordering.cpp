///////////////////////////////////////////////////////////////////////////////
//
// File: TestGmshReordering.cpp
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
// Description: Tests of the Gmsh to Nektar++ node reorderings.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Log.hpp>
#include <NekMesh/Module/InputModules/InputGmsh.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace Nektar::NekMeshGmshReorderingUnitTest
{
using namespace Nektar::NekMesh;

/**
 * @brief Every reordering has to be a permutation of the nodes it is given.
 *
 * A reordering says which of the file's nodes belongs at each position of the
 * element, so every node has to be used, and used once. A map that repeats one
 * node has necessarily dropped another, and the element is then built partly
 * out of copies of itself -- which is not something the mesh will report,
 * because the nodes are all still there and in plausible places. It shows up
 * much later as an element that is tangled for no visible reason.
 *
 * Only one of the three orderings known to have been wrong was of that kind: a
 * prism's interior was gathered with the wrong stride, so it was filled with
 * several copies of a fraction of its nodes, and the nodes left out are what
 * this check finds.
 *
 * The other two were permutations, and pass here. The interior of each face of
 * a hexahedron was mirrored about its diagonal, and a pyramid's interior was
 * left in the order the file gave it; both use every node exactly once, just
 * not in the right places. Those are caught by the Gmsh tests in
 * utilities/NekMesh/Tests/Gmsh, which read a mesh whose elements then have to
 * hold together -- reintroducing either fails several of them. That coverage
 * only exists because these nodes are now used for something; all three bugs
 * survived years of being read and never looked at.
 */
BOOST_AUTO_TEST_CASE(TestReorderingsArePermutations)
{
    Logger log;
    size_t checked = 0;

    for (auto &it : InputGmsh::GenElmMap())
    {
        std::vector<int> map = InputGmsh::CreateReordering(it.first, log);

        // An empty map means the file's ordering is already the one wanted.
        if (map.empty())
        {
            continue;
        }

        ++checked;

        std::vector<int> sorted = map, expected(map.size());
        std::sort(sorted.begin(), sorted.end());
        std::iota(expected.begin(), expected.end(), 0);

        BOOST_CHECK_MESSAGE(
            sorted == expected,
            "Reordering for gmsh element type "
                << it.first << " (" << LibUtilities::ShapeTypeMap[it.second.m_e]
                << ", order " << it.second.m_order << ", " << map.size()
                << " nodes) is not a permutation: it does not use each node "
                   "of the element exactly once.");
    }

    BOOST_CHECK_MESSAGE(checked > 0,
                        "No reorderings were checked, so this test proves "
                        "nothing; has GenElmMap stopped returning types?");
}

} // namespace Nektar::NekMeshGmshReorderingUnitTest
