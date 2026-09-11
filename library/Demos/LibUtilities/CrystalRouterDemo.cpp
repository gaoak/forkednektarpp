///////////////////////////////////////////////////////////////////////////////
//
// File: CrystalRouterDemo.cpp
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
// Description: Serial check of the crystal router's routing decisions
// without MPI. It drives detail::CrystalPlanStep for every rank in
// lockstep through an in-memory model of what the MPI route does, and
// checks that the multiset of delivered messages equals the multiset
// posted and that every message lands on its destination rank. This is
// the check that catches odd (non power of two) rank-count splits.
//
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include <LibUtilities/Communication/EntityResolver.hpp>

using Nektar::LibUtilities::RoutedMessage;
using Nektar::LibUtilities::detail::CrystalDestInMyHalf;
using Nektar::LibUtilities::detail::CrystalPlanStep;
using Nektar::LibUtilities::detail::CrystalStep;

namespace
{

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
        if (++guard > 1024)
        {
            std::cerr << "Round guard tripped (np=" << np << ")\n";
            std::exit(2);
        }

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

// One randomised trial for a given rank count. Returns true on success.
bool TrialForNp(int np, std::mt19937 &rng, int msgsPerRank)
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
            if (m.dest != r)
            {
                std::cerr << "np=" << np << ": message for " << m.dest
                          << " ended on rank " << r << "\n";
                return false;
            }
            delivered.insert(key(m));
        }
    }

    if (delivered != posted)
    {
        std::cerr << "np=" << np << ": delivered multiset != posted multiset ("
                  << delivered.size() << " vs " << posted.size() << ")\n";
        return false;
    }
    return true;
}

// Verbose demonstration: every rank sends one labelled message to every other
// rank; after routing we print, per rank, which sources it received from. This
// makes the delivery visible rather than just asserting a multiset equality.
void DemoRoute(int np)
{
    std::vector<std::vector<RoutedMessage>> initial(np);
    for (int s = 0; s < np; ++s)
    {
        for (int d = 0; d < np; ++d)
        {
            RoutedMessage m;
            m.src  = s;
            m.dest = d;
            m.bytes.push_back(static_cast<std::byte>(s)); // tag = source
            initial[s].push_back(std::move(m));
        }
    }

    auto result = SimulateRoute(np, std::move(initial));

    std::cout << "\n=== Crystal-router delivery, np=" << np << " ===\n";
    std::cout << "each rank should receive exactly one message from every "
                 "source (0.."
              << (np - 1) << ")\n\n";
    bool allOk = true;
    for (int r = 0; r < np; ++r)
    {
        std::vector<int> from;
        bool destOk = true;
        for (auto &m : result[r])
        {
            if (m.dest != r)
            {
                destOk = false;
            }
            from.push_back(static_cast<int>(m.bytes[0]));
        }
        std::sort(from.begin(), from.end());

        std::vector<int> expected(np);
        for (int i = 0; i < np; ++i)
        {
            expected[i] = i;
        }
        const bool ok = destOk && (from == expected);
        allOk         = allOk && ok;

        std::cout << "rank " << std::setw(2) << r << " received from: [";
        for (size_t i = 0; i < from.size(); ++i)
        {
            std::cout << from[i] << (i + 1 < from.size() ? "," : "");
        }
        std::cout << "]  " << (ok ? "ok" : "MISMATCH") << "\n";
    }
    std::cout << (allOk ? "All ranks received the full set.\n"
                        : "DELIVERY MISMATCH.\n");
}

} // namespace

int main(int argc, char *argv[])
{
    std::mt19937 rng(20260713);

    // Awkward counts around powers of two, plus a sweep, plus larger sizes.
    std::vector<int> npList = {1,  2,  3,   4,   5,   6,   7,   8,   9,   11,
                               13, 15, 16,  17,  23,  31,  32,  33,  47,  63,
                               64, 65, 100, 127, 128, 129, 200, 255, 256, 257};

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
        {
            DemoRoute(6);
        }
        else
        {
            int n = std::atoi(a.c_str());
            if (n > 0)
            {
                DemoRoute(n);
            }
        }
    }

    int failures = 0;
    for (int np : npList)
    {
        bool ok = true;
        // Multiple trials per np, varying load including empty and heavy.
        for (int trial = 0; trial < 40 && ok; ++trial)
        {
            const int perRank = trial % 5; // 0..4, exercises empty traffic too
            ok                = TrialForNp(np, rng, perRank);
        }
        // One heavier trial.
        if (ok)
        {
            ok = TrialForNp(np, rng, 12);
        }

        std::cout << "np=" << np << (ok ? "  PASS" : "  FAIL") << "\n";
        if (!ok)
        {
            ++failures;
        }
    }

    if (failures)
    {
        std::cout << failures << " rank-count(s) FAILED\n";
        return 1;
    }
    std::cout << "All crystal-router routing trials passed.\n";
    return 0;
}
