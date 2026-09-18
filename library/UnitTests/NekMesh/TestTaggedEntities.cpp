///////////////////////////////////////////////////////////////////////////////
//
// File: TestTaggedEntities.cpp
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
// Description: Tests of the store that holds which entities carry which tag.
//
///////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Mesh.h>

#include <boost/test/unit_test.hpp>

#include <vector>

namespace Nektar::NekMeshTaggedEntitiesUnitTest
{
using namespace Nektar::NekMesh;

/**
 * @brief A mesh graph and @p n entities in it to use as keys.
 *
 * Points will do: the store keys on the geometry's address and never looks at
 * what it points to, so the cheapest geometry there is makes the clearest test.
 */
struct Entities
{
    Entities(int n) : graph(std::make_shared<SpatialDomains::MeshGraph>())
    {
        graph->SetMeshDimension(3);
        graph->SetSpaceDimension(3);
        geoms.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            geoms.push_back(graph->CreatePointGeom(3, i, (NekDouble)i, 0., 0.));
        }
    }

    SpatialDomains::Geometry *operator[](int i) const
    {
        return geoms[i];
    }

    SpatialDomains::MeshGraphSharedPtr graph;
    std::vector<SpatialDomains::PointGeom *> geoms;
};

/// The entities held, in the order they are iterated.
static std::vector<SpatialDomains::Geometry *> Order(const TaggedEntities &t)
{
    std::vector<SpatialDomains::Geometry *> out;
    for (auto &e : t)
    {
        out.push_back(e.first);
    }
    return out;
}

/**
 * @brief Tagging, reading back, and counting.
 */
BOOST_AUTO_TEST_CASE(TestTagAndLookup)
{
    Entities ents(4);
    TaggedEntities tags;

    BOOST_CHECK(tags.empty());
    BOOST_CHECK_EQUAL(tags.size(), 0u);

    tags[ents[0]] = 7;
    tags[ents[1]] = 9;

    BOOST_CHECK(!tags.empty());
    BOOST_CHECK_EQUAL(tags.size(), 2u);
    BOOST_CHECK_EQUAL(tags[ents[0]], 7);
    BOOST_CHECK_EQUAL(tags[ents[1]], 9);
    BOOST_CHECK_EQUAL(tags.count(ents[0]), 1u);
    BOOST_CHECK_EQUAL(tags.count(ents[2]), 0u);
    BOOST_CHECK(tags.find(ents[1]) != tags.end());
    BOOST_CHECK(tags.find(ents[2]) == tags.end());

    // Asking for an untagged entity tags it, as indexing the map this
    // replaces did.
    BOOST_CHECK_EQUAL(tags[ents[2]], 0);
    BOOST_CHECK_EQUAL(tags.size(), 3u);

    tags.clear();
    BOOST_CHECK(tags.empty());
    BOOST_CHECK_EQUAL(tags.size(), 0u);
    BOOST_CHECK(tags.begin() == tags.end());
}

/**
 * @brief Entities come back in the order they were tagged.
 *
 * This is the whole reason the store exists. Keying on the address alone, as
 * an unordered_map does, gives an order that depends on where the allocator
 * happened to put things, so the mesh that came out of NekMesh differed from
 * one run to the next.
 */
BOOST_AUTO_TEST_CASE(TestIterationFollowsTaggingOrder)
{
    Entities ents(6);
    TaggedEntities tags;

    // Deliberately not in address order.
    const int order[] = {4, 1, 5, 0, 3, 2};
    for (int i : order)
    {
        tags[ents[i]] = i;
    }

    std::vector<SpatialDomains::Geometry *> want;
    for (int i : order)
    {
        want.push_back(ents[i]);
    }

    BOOST_CHECK(Order(tags) == want);
}

/**
 * @brief Tagging something already tagged leaves it where and as it was.
 */
BOOST_AUTO_TEST_CASE(TestEmplaceKeepsTheFirstTag)
{
    Entities ents(3);
    TaggedEntities tags;

    auto first = tags.emplace(ents[0], 3);
    BOOST_CHECK(first.second);
    tags.emplace(ents[1], 4);

    auto again = tags.emplace(ents[0], 99);
    BOOST_CHECK(!again.second);
    BOOST_CHECK_EQUAL(tags[ents[0]], 3);
    BOOST_CHECK_EQUAL(tags.size(), 2u);
    BOOST_CHECK(Order(tags) ==
                (std::vector<SpatialDomains::Geometry *>{ents[0], ents[1]}));
}

/**
 * @brief Untagging hides the entity and leaves the rest in order.
 *
 * Removal leaves a hole rather than closing it up, so what is checked here is
 * that the hole is invisible: it is not iterated, not counted, and not found.
 */
BOOST_AUTO_TEST_CASE(TestEraseLeavesNoTraceButKeepsOrder)
{
    Entities ents(5);
    TaggedEntities tags;
    for (int i = 0; i < 5; ++i)
    {
        tags[ents[i]] = i;
    }

    BOOST_CHECK_EQUAL(tags.erase(ents[1]), 1u);
    BOOST_CHECK_EQUAL(tags.erase(ents[3]), 1u);
    // Untagging something that was not tagged removes nothing.
    BOOST_CHECK_EQUAL(tags.erase(ents[1]), 0u);

    BOOST_CHECK_EQUAL(tags.size(), 3u);
    BOOST_CHECK_EQUAL(tags.count(ents[1]), 0u);
    BOOST_CHECK(tags.find(ents[3]) == tags.end());
    BOOST_CHECK(Order(tags) == (std::vector<SpatialDomains::Geometry *>{
                                   ents[0], ents[2], ents[4]}));

    // Erasing through an iterator has to mean the same thing.
    tags.erase(tags.find(ents[2]));
    BOOST_CHECK_EQUAL(tags.size(), 2u);
    BOOST_CHECK(Order(tags) ==
                (std::vector<SpatialDomains::Geometry *>{ents[0], ents[4]}));
}

/**
 * @brief Tagging something again after untagging it puts it at the end.
 *
 * It is a fresh entry, not the old one brought back, so it belongs where
 * anything tagged now would go.
 */
BOOST_AUTO_TEST_CASE(TestRetaggingGoesToTheEnd)
{
    Entities ents(3);
    TaggedEntities tags;
    for (int i = 0; i < 3; ++i)
    {
        tags[ents[i]] = i;
    }

    tags.erase(ents[0]);
    tags[ents[0]] = 42;

    BOOST_CHECK_EQUAL(tags.size(), 3u);
    BOOST_CHECK_EQUAL(tags[ents[0]], 42);
    BOOST_CHECK(Order(tags) == (std::vector<SpatialDomains::Geometry *>{
                                   ents[1], ents[2], ents[0]}));
}

/**
 * @brief Collecting the holes changes nothing that can be seen.
 *
 * Enough entities are untagged here to take the store past the point where it
 * closes the holes up and renumbers, which is where an error in the rebuilt
 * index would show: tags read back against the wrong entity, or an entity that
 * can no longer be found at all.
 */
BOOST_AUTO_TEST_CASE(TestCompactionKeepsTagsWithTheirEntities)
{
    const int n = 400;
    Entities ents(n);
    TaggedEntities tags;
    for (int i = 0; i < n; ++i)
    {
        tags[ents[i]] = i * 10;
    }

    // Drop three out of every four, scattered rather than in a run.
    std::vector<SpatialDomains::Geometry *> want;
    for (int i = 0; i < n; ++i)
    {
        if (i % 4 == 0)
        {
            want.push_back(ents[i]);
        }
        else
        {
            tags.erase(ents[i]);
        }
    }

    BOOST_REQUIRE_EQUAL(tags.size(), want.size());
    BOOST_CHECK(Order(tags) == want);

    for (int i = 0; i < n; ++i)
    {
        if (i % 4 == 0)
        {
            BOOST_REQUIRE_MESSAGE(tags.find(ents[i]) != tags.end(),
                                  "entity " << i << " was lost by compaction");
            BOOST_CHECK_EQUAL(tags[ents[i]], i * 10);
        }
        else
        {
            BOOST_CHECK_EQUAL(tags.count(ents[i]), 0u);
        }
    }
}

/**
 * @brief Renumbering the entities does not disturb the store.
 *
 * This is what a std::map keyed on the global ID could not survive, and it is
 * not hypothetical: ProcessBL renumbers entities that are tagged at the time.
 * Ordering by a number that the owner is free to change puts the container's
 * invariant in someone else's hands, and once it is broken the entries are not
 * lost loudly -- they are simply not found, which came out as boundary
 * elements quietly losing their tags.
 */
BOOST_AUTO_TEST_CASE(TestRenumberingEntitiesKeepsTags)
{
    const int n = 200;
    Entities ents(n);
    TaggedEntities tags;
    for (int i = 0; i < n; ++i)
    {
        tags[ents[i]] = i * 10;
    }

    const std::vector<SpatialDomains::Geometry *> before = Order(tags);

    // Renumber, in an order that has nothing to do with the old one.
    for (int i = 0; i < n; ++i)
    {
        ents[i]->SetGlobalID((n - i) * 3 + 1);
    }

    BOOST_CHECK_EQUAL(tags.size(), (size_t)n);
    BOOST_CHECK(Order(tags) == before);
    for (int i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_MESSAGE(tags.find(ents[i]) != tags.end(),
                              "entity " << i
                                        << " could not be found after it "
                                           "was given a new global ID");
        BOOST_CHECK_EQUAL(tags[ents[i]], i * 10);
    }

    // And it still behaves afterwards.
    tags.erase(ents[0]);
    BOOST_CHECK_EQUAL(tags.size(), (size_t)n - 1);
    BOOST_CHECK_EQUAL(tags.count(ents[0]), 0u);
    BOOST_CHECK_EQUAL(tags[ents[1]], 10);
}

/**
 * @brief Dropping everything a condition holds for, in one pass.
 */
BOOST_AUTO_TEST_CASE(TestRemoveIf)
{
    const int n = 100;
    Entities ents(n);
    TaggedEntities tags;
    for (int i = 0; i < n; ++i)
    {
        tags[ents[i]] = i;
    }

    tags.RemoveIf(
        [](SpatialDomains::Geometry *, int tag) { return tag % 3 != 0; });

    std::vector<SpatialDomains::Geometry *> want;
    for (int i = 0; i < n; i += 3)
    {
        want.push_back(ents[i]);
    }

    BOOST_CHECK_EQUAL(tags.size(), want.size());
    BOOST_CHECK(Order(tags) == want);
    for (int i = 0; i < n; ++i)
    {
        BOOST_CHECK_EQUAL(tags.count(ents[i]), i % 3 == 0 ? 1u : 0u);
    }
}

} // namespace Nektar::NekMeshTaggedEntitiesUnitTest
