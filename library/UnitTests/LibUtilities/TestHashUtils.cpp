///////////////////////////////////////////////////////////////////////////////
//
// File: TestHashUtils.cpp
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
// Description: Collision checks on hash_combine for the small-integer pairs
// the data warehouse keys are built from.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/HashUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <set>
#include <string>

namespace Nektar::LibUtilities::HashUtilsUnitTest
{

/**
 * The data warehouse keys combine a block index with an interleave width
 * and are looked up by hash alone, so two such pairs must never hash
 * alike. The classic Boost combine step maps (b, 64) and (b + 1, 1) to the
 * same seed, which on a back-end with 64-wide interleaving handed one
 * block's geometric data to the next.
 */
BOOST_AUTO_TEST_CASE(TestBlockIndexWidthPairsDistinct)
{
    const unsigned int widths[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u};
    std::set<std::size_t> seen;
    for (unsigned int blk = 0; blk < 256; ++blk)
    {
        for (const unsigned int width : widths)
        {
            const std::size_t h = hash_combine(blk, width, "JacobianKey");
            BOOST_CHECK_MESSAGE(seen.insert(h).second,
                                "hash collision at block " << blk << " width "
                                                           << width);
        }
    }
    BOOST_CHECK_NE(hash_combine(3u, 64u), hash_combine(4u, 1u));
}

/// The combine is order dependent and seeded reproducibly.
BOOST_AUTO_TEST_CASE(TestOrderAndReproducibility)
{
    BOOST_CHECK_EQUAL(hash_combine(1u, 2u), hash_combine(1u, 2u));
    BOOST_CHECK_NE(hash_combine(1u, 2u), hash_combine(2u, 1u));
    std::size_t seed = 0;
    hash_combine(seed, 7u, std::string("key"));
    BOOST_CHECK_EQUAL(seed, hash_combine(7u, std::string("key")));
}

} // namespace Nektar::LibUtilities::HashUtilsUnitTest
