///////////////////////////////////////////////////////////////////////////////
//
// File: EntityResolverDemo.cpp
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
// Description: Check of EntityResolver::Resolve (hash bucketing, reduction,
// sharer discovery, reply) without MPI: P resolver instances run on P
// threads over a barrier-synchronised in-memory transport, so each
// collective Exchange lines up across threads as it would under MPI, and
// the result is compared with a brute-force oracle. The crystal router's
// routing is checked separately by CrystalRouterDemo.
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <LibUtilities/Communication/EntityResolver.hpp>

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
        : m_np(np), m_barrier(np), m_outbox(np), m_inbox(np)
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
        m_barrier.Wait(); // ensure all read before next round reuses buffers
    }

    int Np() const
    {
        return m_np;
    }

private:
    int m_np;
    Barrier m_barrier;
    std::vector<std::vector<RoutedMessage>> m_outbox;
    std::vector<std::vector<RoutedMessage>> m_inbox;
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
    // oracle: id -> (max order, sorted multiset of contributing ranks)
    std::map<int64_t, int> maxOrder;
    std::map<int64_t, std::vector<int>> sharers; // sorted, with multiplicity
};

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
        std::sort(kv.second.begin(), kv.second.end());
    }
    return w;
}

// Returns true if rank `rank`'s resolver output matches the oracle.
bool CheckRank(int rank, const Workload &w, EntityResolver<int> &res)
{
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
            std::cerr << "rank " << rank << " missing id " << id << "\n";
            return false;
        }
        if (res.GetResolved(id) != w.maxOrder.at(id))
        {
            std::cerr << "rank " << rank << " id " << id
                      << " resolved=" << res.GetResolved(id)
                      << " expected=" << w.maxOrder.at(id) << "\n";
            return false;
        }
        std::vector<int> got = res.GetSharers(id);
        std::sort(got.begin(), got.end());
        if (got != w.sharers.at(id))
        {
            std::cerr << "rank " << rank << " id " << id
                      << " sharer mismatch\n";
            return false;
        }
    }
    return true;
}

bool TrialForNp(int np, std::mt19937 &rng)
{
    Workload w = MakeWorkload(np, rng);
    InMemorySwitch sw(np);

    std::vector<char> ok(np, 1); // char not bool: threads write distinct slots

    auto runRank = [&](int rank) {
        auto transport = std::make_shared<InMemoryTransport>(sw, rank);
        EntityResolver<int> res(transport);
        for (auto &pr : w.registrations[rank])
        {
            res.Register(pr.first, pr.second);
        }
        res.Resolve([](int a, int b) { return std::max(a, b); });
        ok[rank] = CheckRank(rank, w, res) ? 1 : 0;
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

    for (int r = 0; r < np; ++r)
    {
        if (!ok[r])
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    std::mt19937 rng(424242);

    std::vector<int> npList = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                               13, 16, 17, 31, 32, 33, 47, 64, 65};

    int failures = 0;
    for (int np : npList)
    {
        bool ok = true;
        for (int trial = 0; trial < 8 && ok; ++trial)
        {
            ok = TrialForNp(np, rng);
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
    std::cout << "All resolver trials passed.\n";
    return 0;
}
