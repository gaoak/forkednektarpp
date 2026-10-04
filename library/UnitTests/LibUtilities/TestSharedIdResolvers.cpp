///////////////////////////////////////////////////////////////////////////////
//
// File: TestSharedIdResolvers.cpp
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
// Description: Unit tests for the shared-id rendezvous and plan logic,
// driven over an in-memory transport.
//
///////////////////////////////////////////////////////////////////////////////

// Validates the real ConnectivityResolver, SharedPayloadResolver and
// SharedIdPlan logic (hash bucketing, sharer discovery, reply, and the
// persistent fold) without MPI, by running P instances on P threads that share
// an in-memory transport. The transport is barrier synchronised so each
// collective Exchange lines up across threads, exactly as MPI would. Results
// are checked against a brute-force oracle.
//
// Note: crystal-router *routing* is validated separately, in
// TestCrystalRouter.cpp. Since any correct transport delivers the same
// messages, a direct in-memory delivery here fully exercises the resolver logic
// on top. The real MPI transports are covered by
// SharedIdResolversMPITest.cpp.

#include <LibUtilities/Communication/SharedPayloadResolver.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
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

/// Oracle for the payload and plan tests, derived from the same Workload.
struct PlanOracle
{
    std::map<int64_t, double> sum;              // id -> sum of contributions
    std::map<int64_t, std::map<int, int>> mult; // id -> rank -> #registrations
    std::map<int64_t, bool> anyHigh;            // id -> any contribution > 7
};

PlanOracle MakePlanOracle(const Workload &w)
{
    PlanOracle o;
    for (size_t r = 0; r < w.registrations.size(); ++r)
    {
        for (auto &pr : w.registrations[r])
        {
            o.sum[pr.first] += pr.second;
            o.mult[pr.first][static_cast<int>(r)]++;
            o.anyHigh[pr.first] = o.anyHigh[pr.first] || (pr.second > 7);
        }
    }
    return o;
}

/// A payload that identifies its origin exactly.
struct Tag
{
    int64_t id;
    int rank;
    int slot;
};

/// The distinct ids rank @p rank registered, ascending -- what a caller hands
/// ConnectivityResolver, and the order it answers in.
std::vector<int64_t> DistinctIds(const Workload &w, int rank)
{
    std::vector<int64_t> ids;
    for (auto &pr : w.registrations[rank])
    {
        ids.push_back(pr.first);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

/**
 * @brief Check rank @p rank's one-shot resolvers against the oracle.
 *
 * Runs on a worker thread, so it records failures into a per-rank slot rather
 * than calling Boost.Test macros (which are not thread safe); the caller
 * reports them once the threads have joined. Every collective Resolve() is
 * issued before any check runs, so a mismatch on one rank returns without
 * stranding the others on the in-memory transport's barrier.
 */
void CheckOneShotRank(int rank, const Workload &w, const PlanOracle &o,
                      InMemorySwitch &sw, Failure &out)
{
    auto fail = [&](const std::string &msg) {
        if (!out.failed)
        {
            out.failed  = true;
            out.message = "rank " + std::to_string(rank) + ": " + msg;
        }
    };

    auto transport   = std::make_shared<InMemoryTransport>(sw, rank);
    const auto &regs = w.registrations[rank];
    const std::vector<int64_t> ids = DistinctIds(w, rank);

    // Discovery alone.
    ConnectivityResolver cr(transport);
    cr.Reserve(ids.size());
    cr.RegisterKeys(ids);
    cr.Resolve();

    // Every sharer's raw payload, keyed by id.
    SharedPayloadResolver<Tag> sp(transport);
    sp.Reserve(regs.size());
    for (size_t i = 0; i < regs.size(); ++i)
    {
        sp.Register(regs[i].first,
                    Tag{regs[i].first, rank, static_cast<int>(i)});
    }
    sp.Resolve();

    // bool payloads are held as char internally (std::vector<bool> has no
    // data()), so they get their own pass through the same machinery.
    SharedPayloadResolver<bool> sb(transport);
    sb.Reserve(regs.size());
    for (auto &pr : regs)
    {
        sb.Register(pr.first, pr.second > 7);
    }
    sb.Resolve();

    // ---- no more collectives past this point ----

    if (cr.NumResolved() != ids.size())
    {
        fail("ConnectivityResolver resolved " +
             std::to_string(cr.NumResolved()) + " ids, registered " +
             std::to_string(ids.size()));
        return;
    }
    for (size_t u = 0; u < cr.NumResolved(); ++u)
    {
        const int64_t id = cr.ResolvedId(u);
        if (id != ids[u])
        {
            fail("ConnectivityResolver: ids not returned ascending");
            return;
        }
        // Do NOT sort: ascending order and absence of duplicates are part of
        // the SharersAt() contract, so compare as returned.
        const ConnectivityResolver::SharerView sh = cr.SharersAt(u);
        const std::vector<int> got(sh.begin(), sh.end());
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
        if (cr.SharersOf(id).begin() != sh.begin())
        {
            fail("SharersOf disagrees with SharersAt for id " +
                 std::to_string(id));
            return;
        }
    }

    for (int64_t id : ids)
    {
        if (!sp.Knows(id) || !sb.Knows(id))
        {
            fail("missing id " + std::to_string(id));
            return;
        }

        // One entry per (sharer, registration), in unspecified order.
        std::map<int, int> count;
        bool anyHigh = false;
        for (auto &entry : sp.GetSharedPayloads(id))
        {
            if (entry.second.id != id || entry.second.rank != entry.first)
            {
                fail("misrouted payload for id " + std::to_string(id));
                return;
            }
            count[entry.first]++;
        }
        if (count != o.mult.at(id))
        {
            fail("wrong payload multiplicities for id " + std::to_string(id));
            return;
        }
        for (auto &entry : sb.GetSharedPayloads(id))
        {
            anyHigh = anyHigh || entry.second;
        }
        if (anyHigh != o.anyHigh.at(id))
        {
            fail("SharedPayloadResolver<bool> wrong for id " +
                 std::to_string(id));
            return;
        }

        std::vector<int> ranks = sp.GetSharers(id);
        std::sort(ranks.begin(), ranks.end());
        ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
        if (ranks != w.sharers.at(id))
        {
            fail("payload sharer mismatch for id " + std::to_string(id));
            return;
        }
    }
}

/**
 * @brief Run one round of one-shot resolves over @p np simulated ranks and
 * check every rank's answer against the oracle.
 */
void TrialForNp(int np, std::mt19937 &rng)
{
    const Workload w   = MakeWorkload(np, rng);
    const PlanOracle o = MakePlanOracle(w);
    InMemorySwitch sw(np);

    std::vector<Failure> failures(np);

    std::vector<std::thread> threads;
    threads.reserve(np);
    for (int r = 0; r < np; ++r)
    {
        threads.emplace_back(
            [&, r] { CheckOneShotRank(r, w, o, sw, failures[r]); });
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
 * @brief The one-shot resolvers agree with a brute-force oracle on the sharer
 * set and on every sharer's raw payload, across a spread of rank counts.
 *
 * The rank counts deliberately straddle powers of two, since the rendezvous
 * bucketing and the transports beneath it are where odd splits bite.
 */
BOOST_AUTO_TEST_CASE(TestOneShotResolversAgainstOracle)
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

// ---------------------------------------------------------------------------
// Persistent layer: SharedIdPlan, and SharedPayloadResolver's Setup() mode.
// Same threads-as-ranks harness, eTransport backend (the only one that runs
// without a real Comm).
// ---------------------------------------------------------------------------

namespace
{

/// The slot array a rank hands its plan: its registrations in order, with an
/// ignored slot after every third to exercise SharedIdPlan::kIgnore.
/// @p regOfSlot maps each slot back to its registration, or -1.
std::vector<int64_t> PlanSlots(const Workload &w, int rank,
                               std::vector<int> &regOfSlot)
{
    std::vector<int64_t> ids;
    regOfSlot.clear();
    const auto &regs = w.registrations[rank];
    for (size_t i = 0; i < regs.size(); ++i)
    {
        ids.push_back(regs[i].first);
        regOfSlot.push_back(static_cast<int>(i));
        if (i % 3 == 2)
        {
            ids.push_back(SharedIdPlan::kIgnore);
            regOfSlot.push_back(-1);
        }
    }
    return ids;
}

/// What one rank's plan delivered, for cross-rank checks after joining.
struct PlanRecord
{
    Failure failure;
    /// Whether this rank's plan took the one-slot-per-id fast path.
    bool fastPath = false;
    /// Ids whose UniqueMask() bit is set on this rank.
    std::vector<int64_t> maskedIds;
    /// Per id, the (rank, slot) list ExchangePayloads() delivered.
    std::map<int64_t, std::vector<std::pair<int, int>>> payloads;
};

void CheckPlanRank(int rank, const Workload &w, const PlanOracle &o,
                   InMemorySwitch &sw, PlanRecord &rec)
{
    auto fail = [&](const std::string &msg) {
        if (!rec.failure.failed)
        {
            rec.failure.failed  = true;
            rec.failure.message = "rank " + std::to_string(rank) + ": " + msg;
        }
    };

    auto transport   = std::make_shared<InMemoryTransport>(sw, rank);
    const auto &regs = w.registrations[rank];
    std::vector<int> regOfSlot;
    const std::vector<int64_t> ids = PlanSlots(w, rank, regOfSlot);

    SharedIdPlan plan(transport, ids, ExchangeBackend::eTransport);
    rec.fastPath = plan.UsesSingleSlotFastPath();

    // Gather(+), twice: every live slot holds the global sum; ignored slots
    // are untouched. The second pass proves the plan replays.
    for (int round = 0; round < 2; ++round)
    {
        std::vector<double> vals(ids.size(), -1.0);
        for (size_t s = 0; s < ids.size(); ++s)
        {
            if (regOfSlot[s] >= 0)
            {
                vals[s] = regs[regOfSlot[s]].second;
            }
        }
        plan.Gather(vals.data(), std::plus<double>{});
        for (size_t s = 0; s < ids.size(); ++s)
        {
            const double expect = regOfSlot[s] < 0 ? -1.0 : o.sum.at(ids[s]);
            if (vals[s] != expect)
            {
                fail("Gather: slot " + std::to_string(s) + " got " +
                     std::to_string(vals[s]) + ", expected " +
                     std::to_string(expect));
                return;
            }
        }
    }

    // Every slot past ActiveSlotBound() must be a single, unshared id: the
    // guarantee that lets a caller gather a shorter array.
    const std::vector<int> &s2u = plan.SlotToUnique();
    for (size_t s = plan.ActiveSlotBound(); s < ids.size(); ++s)
    {
        const int u = s2u[s];
        if (u >= 0 &&
            (plan.IsShared(u) || std::count(s2u.begin(), s2u.end(), u) > 1))
        {
            fail("slot " + std::to_string(s) +
                 " lies past ActiveSlotBound() but is shared or repeated");
            return;
        }
    }

    // Reduce(max) into per-unique results, leaving its input alone.
    std::vector<int> ord(ids.size(), 0);
    for (size_t s = 0; s < ids.size(); ++s)
    {
        if (regOfSlot[s] >= 0)
        {
            ord[s] = regs[regOfSlot[s]].second;
        }
    }
    const std::vector<int> ordBefore = ord;
    std::vector<int> uniq(plan.NumUnique());
    plan.Reduce(ord.data(), uniq.data(),
                [](int a, int b) { return std::max(a, b); });
    if (ord != ordBefore)
    {
        fail("Reduce modified its input");
        return;
    }
    for (size_t u = 0; u < plan.NumUnique(); ++u)
    {
        const int64_t id = plan.UniqueIds()[u];
        if (uniq[u] != w.maxOrder.at(id))
        {
            fail("Reduce: id " + std::to_string(id) + " wrong max");
            return;
        }
        if (plan.GetSharers(u) != w.sharers.at(id))
        {
            fail("plan sharers wrong for id " + std::to_string(id));
            return;
        }
    }

    // UniqueMask: recorded here, counted across ranks after the join.
    const std::vector<int> mask = plan.UniqueMask();
    for (size_t s = 0; s < ids.size(); ++s)
    {
        if (mask[s])
        {
            if (regOfSlot[s] < 0)
            {
                fail("UniqueMask set on an ignored slot");
                return;
            }
            rec.maskedIds.push_back(ids[s]);
        }
    }

    // Payload exchange: every sharer's every slot, ascending rank.
    plan.PreparePayloadExchange();
    std::vector<Tag> tags(ids.size());
    for (size_t s = 0; s < ids.size(); ++s)
    {
        tags[s] = Tag{ids[s], rank, static_cast<int>(s)};
    }
    std::vector<Tag> out;
    plan.ExchangePayloads(tags.data(), out);
    for (size_t u = 0; u < plan.NumUnique(); ++u)
    {
        const int64_t id = plan.UniqueIds()[u];
        std::map<int, int> count;
        int prev = -1;
        for (size_t e = plan.PayloadBegin(u); e < plan.PayloadEnd(u); ++e)
        {
            if (out[e].id != id || out[e].rank != plan.PayloadRank(e) ||
                plan.PayloadRank(e) < prev)
            {
                fail("ExchangePayloads: bad entry for id " +
                     std::to_string(id));
                return;
            }
            prev = plan.PayloadRank(e);
            count[out[e].rank]++;
            rec.payloads[id].emplace_back(out[e].rank, out[e].slot);
        }
        if (count != o.mult.at(id))
        {
            fail("ExchangePayloads: wrong multiplicities for id " +
                 std::to_string(id));
            return;
        }
    }
}

/// SharedPayloadResolver's Setup() mode, over several Resolve() rounds with
/// payloads that change between rounds -- a stale plan or a missed update
/// would show up as the previous round's answer.
void CheckPersistentResolver(int rank, const Workload &w, const PlanOracle &o,
                             InMemorySwitch &sw, Failure &f)
{
    auto fail = [&](const std::string &msg) {
        if (!f.failed)
        {
            f.failed  = true;
            f.message = "rank " + std::to_string(rank) + ": " + msg;
        }
    };

    auto transport   = std::make_shared<InMemoryTransport>(sw, rank);
    const auto &regs = w.registrations[rank];

    SharedPayloadResolver<Tag> sp(transport);
    for (size_t i = 0; i < regs.size(); ++i)
    {
        sp.Register(regs[i].first, Tag{regs[i].first, rank, 0});
    }
    sp.Setup(ExchangeBackend::eTransport);
    for (int round = 0; round < 2; ++round)
    {
        for (size_t i = 0; i < regs.size(); ++i)
        {
            sp.SetPayload(i, Tag{regs[i].first, rank, round});
        }
        sp.Resolve();
        for (auto &pr : regs)
        {
            std::map<int, int> count;
            for (auto &entry : sp.GetSharedPayloads(pr.first))
            {
                if (entry.second.id != pr.first ||
                    entry.second.rank != entry.first ||
                    entry.second.slot != round)
                {
                    fail("persistent SharedPayloadResolver: stale or "
                         "misrouted payload for id " +
                         std::to_string(pr.first));
                    return;
                }
                count[entry.first]++;
            }
            if (count != o.mult.at(pr.first))
            {
                fail("persistent SharedPayloadResolver: wrong "
                     "multiplicities for id " +
                     std::to_string(pr.first));
                return;
            }
        }
    }
}

void PlanTrialForNp(int np, std::mt19937 &rng)
{
    const Workload w   = MakeWorkload(np, rng);
    const PlanOracle o = MakePlanOracle(w);
    InMemorySwitch sw(np);
    std::vector<PlanRecord> recs(np);
    std::vector<Failure> persistent(np);

    auto runRank = [&](int rank) {
        CheckPlanRank(rank, w, o, sw, recs[rank]);
        CheckPersistentResolver(rank, w, o, sw, persistent[rank]);
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

    // Both Fold() paths must be exercised: rank 0 registers an id twice, so
    // it takes the general path, while every other rank is one slot per id
    // and takes the fast path. Without this the two could silently diverge.
    if (np >= 2)
    {
        int nFast = 0;
        for (int r = 0; r < np; ++r)
        {
            nFast += recs[r].fastPath ? 1 : 0;
        }
        BOOST_REQUIRE_MESSAGE(nFast > 0 && nFast < np,
                              "expected both SharedIdPlan fold paths to be "
                              "exercised, got " +
                                  std::to_string(nFast) + " fast of " +
                                  std::to_string(np));
    }

    for (int r = 0; r < np; ++r)
    {
        BOOST_REQUIRE_MESSAGE(!recs[r].failure.failed, recs[r].failure.message);
        BOOST_REQUIRE_MESSAGE(!persistent[r].failed, persistent[r].message);
    }

    // UniqueMask keeps exactly one slot per id, globally.
    std::map<int64_t, int> kept;
    for (auto &rec : recs)
    {
        for (int64_t id : rec.maskedIds)
        {
            kept[id]++;
        }
    }
    for (auto &kv : o.sum)
    {
        BOOST_REQUIRE_MESSAGE(kept[kv.first] == 1, "UniqueMask kept id "
                                                       << kv.first << " "
                                                       << kept[kv.first]
                                                       << " times");
    }

    // ExchangePayloads gives every sharer of an id the same list.
    for (auto &kv : w.sharers)
    {
        const auto &ref = recs[kv.second.front()].payloads.at(kv.first);
        for (int r : kv.second)
        {
            BOOST_REQUIRE_MESSAGE(recs[r].payloads.at(kv.first) == ref,
                                  "payload lists differ between sharers of id "
                                      << kv.first);
        }
    }
}

} // namespace

/**
 * @brief SharedIdPlan reproduces gs_gather / gs_unique semantics and the
 * payload exchange against the oracle, and SharedPayloadResolver's persistent
 * mode stays correct across repeated Resolve() calls with changing payloads.
 */
BOOST_AUTO_TEST_CASE(TestSharedIdPlanAgainstOracle)
{
    std::mt19937 rng(171717);

    const std::vector<int> npList = {1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 33};

    for (int np : npList)
    {
        BOOST_TEST_CONTEXT("np = " << np)
        {
            for (int trial = 0; trial < 4; ++trial)
            {
                PlanTrialForNp(np, rng);
            }
        }
    }
}

} // namespace Nektar::UnitTests
