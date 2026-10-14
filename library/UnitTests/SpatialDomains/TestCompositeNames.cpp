///////////////////////////////////////////////////////////////////////////////
//
// File: TestCompositeNames.cpp
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
// Description: Round-trip tests for the pyramid inverse mapping.
//
///////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/CompositeNames.h>

#include <boost/test/unit_test.hpp>

#include <map>
#include <string>
#include <vector>

namespace Nektar::CompositeNameTests
{

using SpatialDomains::ParseCompositeList;

static const std::map<int, std::string> labels = {
    {0, "fluid"}, {3, "inlet"}, {4, "outlet"},
    {5, "wall"},  {6, "wall"},  {7, "two words"}};

static std::vector<unsigned int> Parse(const std::string &str,
                                       bool expectResolved = true)
{
    std::vector<unsigned int> ids;
    BOOST_CHECK_EQUAL(ParseCompositeList(str, labels, ids), expectResolved);
    return ids;
}

BOOST_AUTO_TEST_CASE(TestNumericListsUnchanged)
{
    BOOST_CHECK(Parse("0") == std::vector<unsigned int>({0}));
    BOOST_CHECK(Parse("3,4") == std::vector<unsigned int>({3, 4}));
    BOOST_CHECK(Parse("1-4") == std::vector<unsigned int>({1, 2, 3, 4}));
    BOOST_CHECK(Parse("1-3,7") == std::vector<unsigned int>({1, 2, 3, 7}));
    BOOST_CHECK(Parse(" 1 , 2 ") == std::vector<unsigned int>({1, 2}));
}

BOOST_AUTO_TEST_CASE(TestNamesResolve)
{
    BOOST_CHECK(Parse("inlet") == std::vector<unsigned int>({3}));
    BOOST_CHECK(Parse("inlet,outlet") == std::vector<unsigned int>({3, 4}));
    BOOST_CHECK(Parse(" inlet , outlet ") == std::vector<unsigned int>({3, 4}));
}

BOOST_AUTO_TEST_CASE(TestNamesAndNumbersMix)
{
    BOOST_CHECK(Parse("0,inlet") == std::vector<unsigned int>({0, 3}));
    BOOST_CHECK(Parse("inlet,0") == std::vector<unsigned int>({3, 0}));
    BOOST_CHECK(Parse("1-2,outlet") == std::vector<unsigned int>({1, 2, 4}));
}

BOOST_AUTO_TEST_CASE(TestNameMayContainSpaces)
{
    // Only commas separate one entry from the next, so a label with a space
    // in it stays usable.
    BOOST_CHECK(Parse("two words") == std::vector<unsigned int>({7}));
}

BOOST_AUTO_TEST_CASE(TestSharedNameYieldsEveryComposite)
{
    // Two composites carrying one name is how a mesh says they are one patch.
    BOOST_CHECK(Parse("wall") == std::vector<unsigned int>({5, 6}));
}

BOOST_AUTO_TEST_CASE(TestNumbersAreNeverReadAsNames)
{
    // A composite named "3" must not shadow the composite whose ID is 3, so
    // that a list written before names existed cannot change meaning.
    std::map<int, std::string> awkward = {{9, "3"}};
    std::vector<unsigned int> ids;
    BOOST_CHECK(ParseCompositeList("3", awkward, ids));
    BOOST_CHECK(ids == std::vector<unsigned int>({3}));
}

BOOST_AUTO_TEST_CASE(TestMatchingIsCaseSensitive)
{
    Parse("INLET", false);
}

BOOST_AUTO_TEST_CASE(TestUnresolvedAreReported)
{
    std::vector<unsigned int> ids;
    std::vector<std::string> unresolved;

    BOOST_CHECK(
        !ParseCompositeList("inlet,nosuch,0", labels, ids, &unresolved));

    // What did resolve is still returned, so a caller that tolerates a miss
    // -- as one partition of a mesh must -- keeps the rest.
    BOOST_CHECK(ids == std::vector<unsigned int>({3, 0}));
    BOOST_CHECK(unresolved == std::vector<std::string>({"nosuch"}));
}

BOOST_AUTO_TEST_CASE(TestEmptyEntriesIgnored)
{
    BOOST_CHECK(Parse("inlet,,outlet") == std::vector<unsigned int>({3, 4}));
}

BOOST_AUTO_TEST_CASE(TestUnnamedCompositesAreNotMatchable)
{
    // An empty label is the absence of a name, not a name of its own.
    std::map<int, std::string> blank = {{2, ""}};
    std::vector<unsigned int> ids;
    BOOST_CHECK(!ParseCompositeList("", blank, ids));
    BOOST_CHECK(ids.empty());
}

} // namespace Nektar::CompositeNameTests
