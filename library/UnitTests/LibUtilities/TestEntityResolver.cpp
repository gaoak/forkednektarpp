///////////////////////////////////////////////////////////////////////////////
//
// File: TestEntityResolver.cpp
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
// Description: Unit tests for EntityResolver's rendezvous resolve logic,
// driven over an in-memory transport.
//
///////////////////////////////////////////////////////////////////////////////

// Validates the real EntityResolver::Resolve logic (hash bucketing, reduction,
// sharer discovery, reply) without MPI, by running P resolver instances on P
// threads that share an in-memory transport. The transport is barrier
// synchronised so each collective Exchange lines up across threads, exactly as
// MPI would. Results are checked against a brute-force oracle.
//
// Note: crystal-router *routing* is validated separately, in
// TestCrystalRouter.cpp. Since any correct transport delivers the same
// messages, a direct in-memory delivery here fully exercises the resolver logic
// on top. The real MPI transports are covered by EntityResolverMPITest.cpp.

#include <LibUtilities/Communication/EntityResolver.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Nektar::UnitTests
{

using namespace Nektar::LibUtilities;

namespace
{

// A reusable barrier for `n` participating threads.
class Barrier
{
public:
    explicit Barrier(int n) : m_n(n), m_count(0), m_gen(0)
    {
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        const int gen = m_gen;
        if (++m_count == m_n)
        {
            m_count = 0;
            ++m_gen;
            m_cv.notify_all();
        }
        else
        {
            m_cv.wait(lk, [&] { return gen != m_gen; });
        }
    }

private:
    std::mutex m_mtx;
    std::condition_variable m_cv;
    int m_n, m_count, m_gen;
};

// Shared switch backing all in-memory transports. Direct delivery: every
// outbound message is placed in its destination's inbox. Collective per round.
class InMemorySwitch
{
public:
    explicit InMemorySwitch(int np)
        : m_np(np), m_barrier(np), m_outbox(np), m_inbox(np), m_sorted(true)
    {
    }

    void Exchange(int rank, std::vector<RoutedMessage> &outbound,
                  std::vector<RoutedMessage> &inbound)
    {
        m_outbox[rank] = std::move(outbound);
        m_barrier.Wait();

        if (rank == 0)
        {
            for (auto &box : m_inbox)
            {
                box.clear();
            }
            for (int r = 0; r < m_np; ++r)
            {
                for (auto &m : m_outbox[r])
                {
                    RoutedMessage d;
                    d.src   = r;
                    d.dest  = m.dest;
                    d.bytes = std::move(m.bytes);
                    m_inbox[m.dest].push_back(std::move(d));
                }
            }
        }
        m_barrier.Wait();

        inbound = std::move(m_inbox[rank]);

        // ITransport requires inbound sorted by ascending src (stable within a
        // source). The delivery loop above already produces that order; record
        // a violation so this fake can't silently drift from the contract the
        // resolvers rely on for reproducible reductions. (Checked on the test
        // thread rather than asserted here: Boost.Test's assertion macros are
        // not thread safe.)
        if (!std::is_sorted(inbound.begin(), inbound.end(),
                            [](const RoutedMessage &a, const RoutedMessage &b) {
                                return a.src < b.src;
                            }))
        {
            m_sorted = false;
        }

        m_barrier.Wait(); // ensure all read before next round reuses buffers

        // Contract: Exchange() consumes outbound.
        outbound.clear();
    }

    int Np() const
    {
        return m_np;
    }

    /// False if any Exchange() delivered an inbox that was not sorted by src.
    bool InboundWasSorted() const
    {
        return m_sorted;
    }

private:
    int m_np;
    Barrier m_barrier;
    std::vector<std::vector<RoutedMessage>> m_outbox;
    std::vector<std::vector<RoutedMessage>> m_inbox;
    std::atomic<bool> m_sorted;
};

class InMemoryTransport : public ITransport
{
public:
    InMemoryTransport(InMemorySwitch &sw, int rank) : m_sw(sw), m_rank(rank)
    {
    }

    void Exchange(std::vector<RoutedMessage> &outbound,
                  std::vector<RoutedMessage> &inbound) override
    {
        m_sw.Exchange(m_rank, outbound, inbound);
    }
    int Rank() const override
    {
        return m_rank;
    }
    int Size() const override
    {
        return m_sw.Np();
    }
    CommSharedPtr Comm() const override
    {
        return CommSharedPtr();
    }

private:
    InMemorySwitch &m_sw;
    int m_rank;
};

// The registrations we hand to each rank, plus the oracle derived from them.
struct Workload
{
    // registrations[rank] = list of (id, order)
    std::vector<std::vector<std::pair<int64_t, int>>> registrations;
    // oracle: id -> (max order, sorted unique set of contributing ranks)
    std::map<int64_t, int> maxOrder;
    std::map<int64_t, std::vector<int>> sharers;
};

/**
 * @brief Build a workload that mirrors the adjacency patterns a real mesh
 * produces: ring edges, a multi-rank corner, a periodic pair, purely local
 * entities and a duplicate registration.
 */
Workload MakeWorkload(int np, std::mt19937 &rng)
{
    Workload w;
    w.registrations.resize(np);

    auto addReg = [&](int rank, int64_t id, int order) {
        w.registrations[rank].emplace_back(id, order);
        auto it = w.maxOrder.find(id);
        w.maxOrder[id] =
            (it == w.maxOrder.end()) ? order : std::max(it->second, order);
        w.sharers[id].push_back(rank);
    };

    std::uniform_int_distribution<int> orderDist(1, 15);

    // 1. Ring edges: rank r and rank (r+1)%np share edge id 1000 + r.
    //    Both endpoints register with independent orders.
    for (int r = 0; r < np; ++r)
    {
        const int64_t edge = 1000 + r;
        addReg(r, edge, orderDist(rng));
        addReg((r + 1) % np, edge, orderDist(rng));
    }

    // 2. A shared "corner" vertex touched by several ranks (multi-rank sharing,
    //    like a vertex-only connection in 3D). Up to min(np, 8) ranks.
    const int64_t corner = 500000;
    const int nCorner    = std::min(np, 8);
    for (int r = 0; r < nCorner; ++r)
    {
        addReg(r, corner, orderDist(rng));
    }

    // 3. A periodic pair: canonicalised to the same id, touched by rank 0 and
    //    rank np-1 (the "wrap" faces). Trivial when np == 1.
    const int64_t periodic = 900000;
    addReg(0, periodic, orderDist(rng));
    if (np > 1)
    {
        addReg(np - 1, periodic, orderDist(rng));
    }

    // 4. Purely local entities (single sharer) to check the degenerate case.
    for (int r = 0; r < np; ++r)
    {
        addReg(r, 2000000 + r, orderDist(rng));
    }

    // 5. A rank that registers the same id twice (duplicate contributions).
    addReg(0, 3000000, orderDist(rng));
    addReg(0, 3000000, orderDist(rng));

    for (auto &kv : w.sharers)
    {
        // GetSharers() returns ascending, duplicate-free ranks: a rank that
        // registers the same id twice (id 3000000 above, and edge 1000 when
        // np == 1) appears once. The oracle must match.
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()),
                        kv.second.end());
    }
    return w;
}

/// A single mismatch found while checking one rank's resolver output.
struct Failure
{
    bool failed = false;
    std::string message;
};

/**
 * @brief Compare rank @p rank's resolver output against the oracle.
 *
 * Runs on a worker thread, so it records failures into a per-rank slot rather
 * than calling Boost.Test macros (which are not thread safe); the caller
 * reports them once the threads have joined.
 */
void CheckRank(int rank, const Workload &w, EntityResolver<int> &res,
               Failure &out)
{
    auto fail = [&](const std::string &msg) {
        if (!out.failed)
        {
            out.failed  = true;
            out.message = "rank " + std::to_string(rank) + ": " + msg;
        }
    };

    // Every id this rank registered must be known and correct.
    std::map<int64_t, int> myIds; // id -> count registered locally
    for (auto &pr : w.registrations[rank])
    {
        myIds[pr.first]++;
    }

    for (auto &kv : myIds)
    {
        const int64_t id = kv.first;
        if (!res.Knows(id))
        {
            fail("missing id " + std::to_string(id));
            return;
        }
        if (res.GetResolved(id) != w.maxOrder.at(id))
        {
            std::ostringstream ss;
            ss << "id " << id << " resolved=" << res.GetResolved(id)
               << " expected=" << w.maxOrder.at(id);
            fail(ss.str());
            return;
        }
        // Do NOT sort `got`: ascending order and absence of duplicates are
        // part of the GetSharers() contract, so compare as returned.
        const std::vector<int> &got = res.GetSharers(id);
        if (!std::is_sorted(got.begin(), got.end()) ||
            std::adjacent_find(got.begin(), got.end()) != got.end())
        {
            fail("id " + std::to_string(id) + " sharers not sorted/unique");
            return;
        }
        if (got != w.sharers.at(id))
        {
            fail("id " + std::to_string(id) + " sharer mismatch");
            return;
        }
    }
}

/**
 * @brief Run one resolve over @p np simulated ranks and check every rank's
 * answer against the oracle.
 */
void TrialForNp(int np, std::mt19937 &rng)
{
    Workload w = MakeWorkload(np, rng);
    InMemorySwitch sw(np);

    std::vector<Failure> failures(np);

    auto runRank = [&](int rank) {
        auto transport = std::make_shared<InMemoryTransport>(sw, rank);
        EntityResolver<int> res(transport);
        for (auto &pr : w.registrations[rank])
        {
            res.Register(pr.first, pr.second);
        }
        res.Resolve([](int a, int b) { return std::max(a, b); });
        CheckRank(rank, w, res, failures[rank]);
    };

    std::vector<std::thread> threads;
    threads.reserve(np);
    for (int r = 0; r < np; ++r)
    {
        threads.emplace_back(runRank, r);
    }
    for (auto &t : threads)
    {
        t.join();
    }

    BOOST_REQUIRE_MESSAGE(sw.InboundWasSorted(),
                          "InMemorySwitch: inbound not sorted by src");

    for (auto &f : failures)
    {
        BOOST_REQUIRE_MESSAGE(!f.failed, f.message);
    }
}

} // namespace

/**
 * @brief EntityResolver agrees with a brute-force oracle on both the reduced
 * value and the sharer set, across a spread of rank counts.
 *
 * The rank counts deliberately straddle powers of two, since the rendezvous
 * bucketing and the transports beneath it are where odd splits bite.
 */
BOOST_AUTO_TEST_CASE(TestEntityResolverAgainstOracle)
{
    std::mt19937 rng(424242);

    const std::vector<int> npList = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                     13, 16, 17, 31, 32, 33, 47, 64, 65};

    for (int np : npList)
    {
        BOOST_TEST_CONTEXT("np = " << np)
        {
            for (int trial = 0; trial < 8; ++trial)
            {
                TrialForNp(np, rng);
            }
        }
    }
}

} // namespace Nektar::UnitTests
