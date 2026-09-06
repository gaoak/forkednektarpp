///////////////////////////////////////////////////////////////////////////////
//
// File: TestCrystalRouter.cpp
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
// Description: Unit tests for the crystal-router routing decisions used by
// EntityResolver's CrystalRouterTransport.
//
///////////////////////////////////////////////////////////////////////////////

// Serial validation of the crystal-router *routing decisions* without MPI.
//
// It drives the shipped detail::CrystalPlanStep for every rank in lockstep,
// moving messages through an in-memory model that mirrors exactly what the MPI
// route() does (send to partner / one-way forward; receive is implied by the
// sender targeting). If routing is correct, the multiset of delivered messages
// equals the multiset of originally posted messages, and every message lands on
// its destination rank. This is the test that catches odd-split (non power of
// two) bugs.
//
// The resolver logic layered on top of the transport is tested separately in
// TestEntityResolver.cpp; the real MPI transports are covered by
// EntityResolverMPITest.cpp.

#include <LibUtilities/Communication/EntityResolver.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using Nektar::LibUtilities::RoutedMessage;
using Nektar::LibUtilities::detail::CrystalDestInMyHalf;
using Nektar::LibUtilities::detail::CrystalPlanStep;
using Nektar::LibUtilities::detail::CrystalStep;

namespace Nektar::UnitTests
{

namespace
{

/// Maximum number of routing rounds before we assume the route is not
/// converging; log2(P) rounds is expected, so this is very generous.
constexpr int MaxRounds = 1024;

// Per-rank state during the simulated route.
struct RankState
{
    int lo = 0;
    int hi = 0;
    std::vector<RoutedMessage> msgs;
    bool done() const
    {
        return lo >= hi;
    }
};

// Encode a message identity (src, dest, first payload byte) for multiset
// checks.
using MsgKey = std::tuple<int, int, int>;

MsgKey key(const RoutedMessage &m)
{
    int tag = m.bytes.empty() ? -1 : static_cast<int>(m.bytes[0]);
    return {m.src, m.dest, tag};
}

// Simulate the full crystal route for `np` ranks given each rank's initial
// outbound messages. Returns, per rank, the messages it ends up holding.
std::vector<std::vector<RoutedMessage>> SimulateRoute(
    int np, std::vector<std::vector<RoutedMessage>> initial)
{
    std::vector<RankState> state(np);
    for (int r = 0; r < np; ++r)
    {
        state[r].lo   = 0;
        state[r].hi   = np - 1;
        state[r].msgs = std::move(initial[r]);
    }

    // Iterate rounds until every rank has collapsed to a singleton group.
    // Different ranks may finish in different rounds (non power of two); a
    // finished rank simply stops participating, and -- as in the real
    // algorithm -- no active rank ever targets a finished one.
    bool anyActive = true;
    int guard      = 0;
    while (anyActive)
    {
        anyActive = false;
        BOOST_REQUIRE_MESSAGE(++guard <= MaxRounds,
                              "Round guard tripped (np=" << np << ")");

        // Phase 1: each active rank computes its keep pile and its single
        // outgoing (target, pile). We snapshot first, then deliver, so the
        // exchange is symmetric within the round.
        std::vector<std::vector<RoutedMessage>> keep(np);
        std::vector<std::vector<RoutedMessage>> incoming(np);
        std::vector<CrystalStep> steps(np);
        std::vector<bool> active(np, false);

        for (int r = 0; r < np; ++r)
        {
            if (state[r].done())
            {
                continue;
            }
            active[r]     = true;
            anyActive     = true;
            CrystalStep s = CrystalPlanStep(state[r].lo, state[r].hi, r);
            steps[r]      = s;

            std::vector<RoutedMessage> send;
            for (auto &m : state[r].msgs)
            {
                if (CrystalDestInMyHalf(s, m.dest))
                {
                    keep[r].push_back(std::move(m));
                }
                else
                {
                    send.push_back(std::move(m));
                }
            }
            state[r].msgs.clear();

            const int target = (s.partner >= 0) ? s.partner : s.forwardTo;
            for (auto &m : send)
            {
                incoming[target].push_back(std::move(m));
            }
        }

        // Phase 2: merge keep + delivered, then narrow each active rank's
        // group.
        for (int r = 0; r < np; ++r)
        {
            if (!active[r])
            {
                continue;
            }
            state[r].msgs = std::move(keep[r]);
            for (auto &m : incoming[r])
            {
                state[r].msgs.push_back(std::move(m));
            }
            if (steps[r].inLow)
            {
                state[r].hi = steps[r].mid;
            }
            else
            {
                state[r].lo = steps[r].mid + 1;
            }
        }
    }

    std::vector<std::vector<RoutedMessage>> out(np);
    for (int r = 0; r < np; ++r)
    {
        out[r] = std::move(state[r].msgs);
    }
    return out;
}

// One randomised trial for a given rank count: post `msgsPerRank` messages to
// random destinations on every rank and check they all arrive, exactly once.
void TrialForNp(int np, std::mt19937 &rng, int msgsPerRank)
{
    std::uniform_int_distribution<int> destDist(0, np - 1);
    std::uniform_int_distribution<int> tagDist(0, 120);

    std::vector<std::vector<RoutedMessage>> initial(np);
    std::multiset<MsgKey> posted;

    for (int r = 0; r < np; ++r)
    {
        for (int k = 0; k < msgsPerRank; ++k)
        {
            RoutedMessage m;
            m.src  = r;
            m.dest = destDist(rng);
            m.bytes.push_back(static_cast<std::byte>(tagDist(rng)));
            posted.insert(key(m));
            initial[r].push_back(std::move(m));
        }
    }

    auto result = SimulateRoute(np, std::move(initial));

    std::multiset<MsgKey> delivered;
    for (int r = 0; r < np; ++r)
    {
        for (auto &m : result[r])
        {
            BOOST_REQUIRE_MESSAGE(m.dest == r, "np=" << np << ": message for "
                                                     << m.dest
                                                     << " ended on rank " << r);
            delivered.insert(key(m));
        }
    }

    BOOST_REQUIRE_MESSAGE(
        delivered == posted,
        "np=" << np << ": delivered multiset != posted multiset ("
              << delivered.size() << " vs " << posted.size() << ")");
}

// Awkward counts around powers of two, plus a sweep, plus larger sizes.
const std::vector<int> npList = {
    1,  2,  3,  4,  5,  6,  7,  8,   9,   11,  13,  15,  16,  17,  23,
    31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256, 257};

} // namespace

/**
 * @brief Every posted message is delivered exactly once, to the right rank.
 *
 * Randomised traffic, including empty traffic, across rank counts that
 * deliberately straddle powers of two -- the odd-split case is where a
 * hypercube router most easily goes wrong.
 */
BOOST_AUTO_TEST_CASE(TestCrystalRouterDelivery)
{
    std::mt19937 rng(20260713);

    for (int np : npList)
    {
        BOOST_TEST_CONTEXT("np = " << np)
        {
            // Multiple trials per np, varying load including empty and heavy.
            for (int trial = 0; trial < 40; ++trial)
            {
                const int perRank = trial % 5; // 0..4, exercises empty traffic
                TrialForNp(np, rng, perRank);
            }

            // One heavier trial.
            TrialForNp(np, rng, 12);
        }
    }
}

/**
 * @brief All-to-all delivery: every rank sends one labelled message to every
 * other rank, and must receive exactly one message from every source.
 *
 * This is the dense counterpart to the randomised sparse traffic above.
 */
BOOST_AUTO_TEST_CASE(TestCrystalRouterAllToAll)
{
    for (int np : {1, 2, 3, 5, 6, 8, 13, 16, 17})
    {
        BOOST_TEST_CONTEXT("np = " << np)
        {
            std::vector<std::vector<RoutedMessage>> initial(np);
            for (int s = 0; s < np; ++s)
            {
                for (int d = 0; d < np; ++d)
                {
                    RoutedMessage m;
                    m.src  = s;
                    m.dest = d;
                    m.bytes.push_back(
                        static_cast<std::byte>(s)); // tag = source
                    initial[s].push_back(std::move(m));
                }
            }

            auto result = SimulateRoute(np, std::move(initial));

            std::vector<int> expected(np);
            for (int i = 0; i < np; ++i)
            {
                expected[i] = i;
            }

            for (int r = 0; r < np; ++r)
            {
                std::vector<int> from;
                for (auto &m : result[r])
                {
                    BOOST_REQUIRE_EQUAL(m.dest, r);
                    from.push_back(static_cast<int>(m.bytes[0]));
                }
                std::sort(from.begin(), from.end());

                BOOST_CHECK_EQUAL_COLLECTIONS(from.begin(), from.end(),
                                              expected.begin(), expected.end());
            }
        }
    }
}

} // namespace Nektar::UnitTests
