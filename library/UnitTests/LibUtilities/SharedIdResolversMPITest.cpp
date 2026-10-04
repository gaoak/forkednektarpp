///////////////////////////////////////////////////////////////////////////////
//
// File: SharedIdResolversMPITest.cpp
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
// Description: Real-MPI integration test for the shared-id resolvers over
// both shipped transports.
//
///////////////////////////////////////////////////////////////////////////////

// Real-MPI integration test, driven by the Tester through
// Tests/SharedIdResolversMPI.tst at several rank counts. For the current
// communicator it:
//
//   1. Builds a workload with ring edges, a multi-rank corner vertex, a
//      periodic pair, purely local entities, and a duplicate registration.
//   2. Discovers it with BOTH transports and asserts they agree
//      (the cross-transport validation).
//   3. Asserts the reduced max-orders and sharer sets against an analytic
//      oracle computed the same way on every rank.
//   4. Replays each plan over every exchange backend, with values that change
//      between rounds, so a stale plan shows up as the previous answer.
//
// The resolver logic itself, and the crystal router's routing decisions, are
// unit tested serially in TestSharedIdResolvers.cpp and
// TestCrystalRouter.cpp;
// this test is what exercises them over real MPI.
//
// A single line "np=<n>  PASS" is printed on rank 0 and matched by the .tst
// regex metric; the exit code is 0 only if every rank passed.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <LibUtilities/Communication/SharedPayloadResolver.hpp>

using namespace Nektar::LibUtilities;

namespace
{

struct Workload
{
    std::vector<std::pair<int64_t, int>> myRegs; // this rank's (id, order)
    std::map<int64_t, int> oracleMax;            // id -> max order
    std::map<int64_t, long> oracleSum;           // id -> sum of orders
    std::map<int64_t, std::vector<int>>
        oracleShare; // id -> sorted, unique ranks
};

// Deterministic per-(rank,id) order so every rank can compute the oracle
// without communication. Mixes rank and id.
int OrderFor(int rank, int64_t id)
{
    uint64_t h = static_cast<uint64_t>(id) * 1315423911u + rank * 2654435761u;
    return 1 + static_cast<int>(h % 15u);
}

Workload BuildWorkload(int rank, int np)
{
    Workload w;

    // Helper records both "my registrations" and the global oracle. Because
    // OrderFor is a pure function of (rank, id), each rank can reconstruct the
    // full oracle for every id it cares about.
    auto contributor = [&](int64_t id, int who) {
        w.oracleMax[id] = w.oracleMax.count(id)
                              ? std::max(w.oracleMax[id], OrderFor(who, id))
                              : OrderFor(who, id);
        w.oracleShare[id].push_back(who);
        w.oracleSum[id] += OrderFor(who, id);
        if (who == rank)
        {
            w.myRegs.emplace_back(id, OrderFor(rank, id));
        }
    };

    // 1. Ring edges: edge (1000 + r) shared by r and (r+1)%np.
    for (int r = 0; r < np; ++r)
    {
        const int64_t edge = 1000 + r;
        contributor(edge, r);
        contributor(edge, (r + 1) % np);
    }

    // 2. Corner vertex shared by first min(np, 8) ranks.
    const int64_t corner = 500000;
    for (int r = 0; r < std::min(np, 8); ++r)
    {
        contributor(corner, r);
    }

    // 3. Periodic pair canonicalised to one id, on rank 0 and rank np-1.
    const int64_t periodic = 900000;
    contributor(periodic, 0);
    if (np > 1)
    {
        contributor(periodic, np - 1);
    }

    // 4. Purely local entity per rank.
    for (int r = 0; r < np; ++r)
    {
        contributor(2000000 + r, r);
    }

    // 5. Duplicate registration on rank 0.
    contributor(3000000, 0);
    contributor(3000000, 0); // second contribution, same rank

    for (auto &kv : w.oracleShare)
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

// This rank's slot ids and values, one entry per registration (so rank 0's
// duplicate id appears twice, as it would in a real assembly map).
void SlotArrays(const Workload &w, std::vector<int64_t> &ids,
                std::vector<int> &vals)
{
    ids.clear();
    vals.clear();
    for (auto &pr : w.myRegs)
    {
        ids.push_back(pr.first);
        vals.push_back(pr.second);
    }
}

// Discover the workload with a given transport; return id -> (value, sharers).
struct Resolved
{
    std::map<int64_t, int> value;
    std::map<int64_t, std::vector<int>> sharers;
};

Resolved RunWith(std::shared_ptr<ITransport> transport, const Workload &w)
{
    std::vector<int64_t> ids;
    std::vector<int> vals;
    SlotArrays(w, ids, vals);

    SharedIdPlan plan(transport, ids, ExchangeBackend::ePairwise);
    std::vector<int> uniq(plan.NumUnique());
    plan.Reduce(vals.data(), uniq.data(),
                [](int a, int b) { return std::max(a, b); });

    Resolved out;
    for (size_t u = 0; u < plan.NumUnique(); ++u)
    {
        const int64_t id = plan.UniqueIds()[u];
        out.value[id]    = uniq[u];
        out.sharers[id]  = plan.GetSharers(u);
    }
    return out;
}

// Persistent layer over real MPI: for every discovery transport and every
// exchange backend, a SharedIdPlan must reproduce the oracle and keep doing so
// over repeated calls with changing values -- which is the point.
void CheckPlans(const std::vector<std::shared_ptr<ITransport>> &transports,
                const Workload &w,
                const std::function<void(const std::string &)> &fail)
{
    const ExchangeBackend backends[] = {ExchangeBackend::ePairwise,
                                        ExchangeBackend::eNeighbourCollective,
                                        ExchangeBackend::eTransport};

    std::vector<int64_t> ids;
    std::vector<int> orders;
    SlotArrays(w, ids, orders);
    std::vector<long> vals(orders.begin(), orders.end());

    for (size_t t = 0; t < transports.size(); ++t)
    {
        for (ExchangeBackend b : backends)
        {
            const std::string where =
                "transport " + std::to_string(t) + ", backend " +
                std::to_string(static_cast<int>(b)) + ": ";

            SharedIdPlan plan(transports[t], ids, b);

            // Gather(+): every slot ends up holding the global sum. Values
            // change every round, so a plan that quietly kept the previous
            // round's data would show up here.
            for (int round = 0; round < 3; ++round)
            {
                // Scaling every contribution scales the sum, whatever an
                // id's multiplicity across ranks.
                std::vector<long> v(vals.size());
                for (size_t s = 0; s < vals.size(); ++s)
                {
                    v[s] = vals[s] * (round + 1);
                }
                plan.Gather(v.data(), std::plus<long>{});
                for (size_t s = 0; s < ids.size(); ++s)
                {
                    if (v[s] != w.oracleSum.at(ids[s]) * (round + 1))
                    {
                        fail(where + "Gather sum wrong for id " +
                             std::to_string(ids[s]));
                        break;
                    }
                }
            }

            // Reduce(max) into per-unique results, with the sharer sets the
            // plan discovered.
            for (int round = 0; round < 3; ++round)
            {
                std::vector<int> v(orders.size());
                for (size_t s = 0; s < orders.size(); ++s)
                {
                    v[s] = orders[s] + 100 * round;
                }
                std::vector<int> uniq(plan.NumUnique());
                plan.Reduce(v.data(), uniq.data(),
                            [](int x, int y) { return std::max(x, y); });
                for (size_t u = 0; u < plan.NumUnique(); ++u)
                {
                    const int64_t id = plan.UniqueIds()[u];
                    if (uniq[u] != w.oracleMax.at(id) + 100 * round ||
                        plan.GetSharers(u) != w.oracleShare.at(id))
                    {
                        fail(where + "persistent plan wrong for id " +
                             std::to_string(id));
                        break;
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    CommSharedPtr comm =
        GetCommFactory().CreateInstance("ParallelMPI", argc, argv);

    int rank = comm->GetRank(), np = comm->GetSize();

    Workload w = BuildWorkload(rank, np);

    auto alltoallv = std::make_shared<AlltoallvTransport>(comm);
    auto crystal   = std::make_shared<CrystalRouterTransport>(comm);

    Resolved rA = RunWith(alltoallv, w);
    Resolved rC = RunWith(crystal, w);

    int localOk = 1;
    auto fail   = [&](const std::string &msg) {
        std::cerr << "[rank " << rank << ", np " << np << "] " << msg << "\n";
        localOk = 0;
    };

    // Cross-transport agreement.
    if (rA.value != rC.value)
    {
        fail("transports disagree on resolved values");
    }
    if (rA.sharers != rC.sharers)
    {
        fail("transports disagree on sharer sets");
    }

    // Oracle agreement (check against rA; rC already shown equal).
    for (auto &pr : w.myRegs)
    {
        const int64_t id = pr.first;
        if (rA.value[id] != w.oracleMax.at(id))
        {
            fail("value mismatch for id " + std::to_string(id));
        }
        if (rA.sharers[id] != w.oracleShare.at(id))
        {
            fail("sharer mismatch for id " + std::to_string(id));
        }
    }

    CheckPlans({alltoallv, crystal}, w, fail);

    // Reduce pass/fail across all ranks.
    comm->AllReduce(localOk, ReduceMin);

    if (rank == 0)
    {
        std::cout << "np=" << np << (localOk ? "  PASS" : "  FAIL")
                  << std::endl;
    }

    comm->Finalise();

    return localOk ? 0 : 1;
}
