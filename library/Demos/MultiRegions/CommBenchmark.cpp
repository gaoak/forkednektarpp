///////////////////////////////////////////////////////////////////////////////
//
// File: CommBenchmark.cpp
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
// Description: Mesh-driven benchmark of the shared-id resolver family against
// gslib.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file CommBenchmark.cpp
 * @brief Mesh-driven strong/weak-scaling benchmark of SharedIdPlan,
 * ConnectivityResolver and SharedPayloadResolver against the equivalent gslib
 * operations, over both shipped transports.
 *
 * @details
 * ### What it measures
 * The universal-id array of a real partitioned mesh is used as the workload,
 * so the id distribution, duplicate rate and rank fan-out are those the solver
 * actually sees. Three families of test are run:
 *
 * - **assembly**: `Gs::Init` + `Gs::Gather(gs_add)` against SharedIdPlan's
 *   construction + `Gather` with a summing fold.
 * - **connectivity**: `Gs::Unique` against ConnectivityResolver plus the
 *   lowest-rank-owns tie-break that reproduces its result.
 * - **payload**: SharedPayloadResolver at several payload sizes. gslib has no
 *   equivalent (it can only fold, not exchange raw per-sharer values), so this
 *   family reports the two resolver transports against each other only.
 *
 * plus an optional **neighbour** test timing BuildNeighbourComm(). That one
 * needs MPI-3 for `MPI_Dist_graph_create_adjacent`, and is skipped on an
 * older library, as is the `neighbour` exchange backend.
 *
 * ### Comparing the numbers honestly
 * gslib splits into an expensive one-off `gs_setup` (which discovers the
 * communication graph and builds a persistent plan) and a cheap repeated
 * `gs_gather` that replays it. SharedIdPlan splits the same way, so
 * `plan-setup` / `plan-gather` are the like-for-like rows against
 * `gslib-setup` / `gslib-gather`, timed over every exchange backend
 * (`--bench-backends`) with discovery running over the row's transport. The
 * `<size>B/<backend>` SharedPayloadResolver rows are the id-keyed wrapper over
 * the same plan, and show what that convenience costs on top of it.
 *
 * The one-shot resolver rows re-run the full rendezvous on every call, so read
 * them against **gslib-setup + gslib-gather** (resolve once), not against
 * **gslib-gather** alone (resolve in a loop -- which is what `plan-gather` and
 * the persistent payload rows are for).
 *
 * Timings are per-iteration wall time, elementwise-max-reduced across ranks
 * (each iteration costed by its slowest rank, which is what a collective
 * actually costs), then summarised as min/median/mean/max over iterations. A
 * barrier precedes each iteration so a straggler from the previous one is not
 * charged to the next.
 *
 * ### Usage
 * @code{.sh}
 * mpirun -n 128 CommBenchmark session.xml \
 *     --bench-workload cg --bench-repeat 50 --bench-csv results.csv
 * @endcode
 *
 * Options (all optional):
 * - `--bench-workload cg|trace|entity` -- which id set to benchmark.
 *   `cg` uses the continuous-Galerkin global-to-universal map (needs boundary
 *   conditions in the session), `trace` the DG trace map, `entity` the raw
 *   mesh vertex/edge/face global ids with no field construction at all.
 * - `--bench-tests assembly,connectivity,payload,neighbour` -- subset to run.
 * - `--bench-transports a2a,crystal` -- subset of resolver transports.
 * - `--bench-repeat N`, `--bench-warmup N` -- iteration counts.
 * - `--bench-payload-bytes 8,64,512` -- SharedPayloadResolver payload sizes;
 *   must come from the compiled-in set {8,16,32,64,128,256,512,1024,4096}.
 * - `--bench-backends pairwise,neighbour,crystal,transport` -- exchange
 *   backends the persistent-plan rows are timed over (see ExchangeBackend).
 *   `transport` replays through the row's own discovery transport and is a
 *   diagnostic; `crystal` always routes through a crystal router.
 * - `--bench-csv FILE` -- write machine-readable results (rank 0 only).
 * - `--bench-no-validate` -- skip correctness cross-checks.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/Communication/GsLib.hpp>
#include <LibUtilities/Communication/SharedPayloadResolver.hpp>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace Nektar;
using LibUtilities::AlltoallvTransport;
using LibUtilities::CommSharedPtr;
using LibUtilities::ConnectivityResolver;
using LibUtilities::ConnectivityResolverTimings;
using LibUtilities::CrystalRouterTransport;
using LibUtilities::ExchangeBackend;
using LibUtilities::ITransport;
using LibUtilities::SharedIdPlan;
using LibUtilities::SharedPayloadResolver;
using LibUtilities::SharedPayloadResolverTimings;

namespace
{

// ---------------------------------------------------------------------------
// Command-line arguments. Registered before main() runs; SessionReader gives
// every one of these a std::string value, so they are parsed by hand below.
// ---------------------------------------------------------------------------
std::string cmdWorkload = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-workload", "",
    "Id set to benchmark: cg (default), trace, entity or synthetic (needs no "
    "mesh; see --bench-synth-*).");
std::string cmdTests = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-tests", "",
    "Comma-separated subset of assembly,connectivity,payload,neighbour.");
std::string cmdTransports =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-transports", "",
        "Comma-separated subset of a2a,crystal (default both).");
std::string cmdBackends = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-backends", "",
    "Comma-separated subset of pairwise,neighbour,crystal,transport: the "
    "exchange backends the persistent-plan rows are timed over (default "
    "pairwise,neighbour,crystal).");
std::string cmdSynthSlots =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-synth-slots", "",
        "Synthetic workload: slots per rank (default 19000).");
std::string cmdSynthActive =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-synth-active", "",
        "Synthetic workload: slots per rank carrying an id, the rest ignored "
        "(default 12000).");
std::string cmdSynthShared =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-synth-shared", "",
        "Synthetic workload: ids per rank shared with a neighbour (default "
        "6000).");
std::string cmdSynthNbrs = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-synth-neighbours", "",
    "Synthetic workload: neighbour ranks to share with (default 26, the "
    "face+edge+corner count of a structured 3D partition).");
std::string cmdSynthCorner =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-synth-corner-frac", "",
        "Synthetic workload: fraction of shared ids shared by a block of 8 "
        "ranks rather than a single neighbour, standing in for edge and "
        "corner DOF (default 0.1).");
std::string cmdRepeat = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-repeat", "", "Timed iterations per measurement (default 20).");
std::string cmdWarmup = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-warmup", "", "Untimed warm-up iterations (default 3).");
std::string cmdPayloadBytes =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "bench-payload-bytes", "",
        "Comma-separated SharedPayloadResolver payload sizes in bytes; must "
        "be drawn from {8,16,32,64,128,256,512,1024,4096}.");
std::string cmdCsv = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "bench-csv", "", "Write results as CSV to this file (rank 0 only).");
std::string cmdNoValidate = LibUtilities::SessionReader::RegisterCmdLineFlag(
    "bench-no-validate", "", "Skip the correctness cross-checks.");

using Clock = std::chrono::steady_clock;

double Now()
{
    return std::chrono::duration<double>(Clock::now().time_since_epoch())
        .count();
}

std::vector<std::string> Split(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        // Trim, so "a2a, crystal" works as well as "a2a,crystal".
        size_t b = item.find_first_not_of(" \t");
        size_t e = item.find_last_not_of(" \t");
        if (b != std::string::npos)
        {
            out.push_back(item.substr(b, e - b + 1));
        }
    }
    return out;
}

bool Contains(const std::vector<std::string> &v, const std::string &s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

/// The transport named by a --bench-transports entry.
std::shared_ptr<ITransport> MakeTransport(const CommSharedPtr &comm,
                                          const std::string &name)
{
    if (name == "crystal")
    {
        return std::make_shared<CrystalRouterTransport>(comm);
    }
    return std::make_shared<AlltoallvTransport>(comm);
}

/// The ExchangeBackend named by a --bench-backends entry.
ExchangeBackend ParseBackend(const std::string &name)
{
    if (name == "pairwise")
    {
        return ExchangeBackend::ePairwise;
    }
    if (name == "neighbour")
    {
        return ExchangeBackend::eNeighbourCollective;
    }
    if (name == "crystal")
    {
        return ExchangeBackend::eCrystalRouter;
    }
    return ExchangeBackend::eTransport;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/**
 * @brief One measured quantity: a family/implementation/variant label plus the
 * distribution of its per-iteration cost.
 *
 * Times are already max-reduced across ranks, so they describe the collective,
 * not any one rank.
 */
struct Result
{
    std::string test;      ///< assembly | connectivity | payload | neighbour
    std::string impl;      ///< gslib-setup, gslib-gather, plan-setup, ...
    std::string transport; ///< a2a | crystal | - (for gslib)
    std::string variant;   ///< exchange backend | payload size | -
    double tMin  = 0.0;
    double tMed  = 0.0;
    double tMean = 0.0;
    double tMax  = 0.0;
    /// Optional sub-phase breakdown, median over iterations. Empty name means
    /// the phase was not measured for this row.
    std::string p1Name, p2Name;
    double p1Med = 0.0, p2Med = 0.0;
};

std::vector<Result> g_results;

/// Cleared by any failed correctness check; drives the summary line and the
/// process exit code so the demo can double as a regression test.
int g_validateOk = 1;

/// Median of an already-populated sample vector (copies, so the caller's
/// ordering is preserved for the mean/min/max taken alongside it).
double Median(std::vector<double> v)
{
    if (v.empty())
    {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/// Signature of a benchmarked operation. The two out-parameters let an
/// operation report an internal phase split (e.g. request vs reply exchange);
/// leave them at zero if there is nothing to split.
using BenchFn = std::function<void(double &phase1, double &phase2)>;

/**
 * @brief Time @p fn over @p warmup + @p repeat iterations and append a Result.
 *
 * Each timed iteration is preceded by a barrier and the resulting per-iteration
 * times are elementwise max-reduced across ranks, so iteration @a i is costed
 * by whichever rank finished it last.
 */
void Measure(const CommSharedPtr &comm, const std::string &test,
             const std::string &impl, const std::string &transport,
             const std::string &variant, int warmup, int repeat,
             const BenchFn &fn, const std::string &p1Name = "",
             const std::string &p2Name = "")
{
    double d1 = 0.0, d2 = 0.0;
    bool output = comm->GetRank() == 0;
    if (output)
    {
        std::cout << "Running " << test << "/" << impl << "/" << transport
                  << "/" << variant << std::endl;
    }
    for (int i = 0; i < warmup; ++i)
    {
        if (output)
        {
            std::cout << "   warm up " << i << std::endl;
        }
        d1 = d2 = 0.0;
        fn(d1, d2);
    }

    Array<OneD, NekDouble> t(repeat, 0.0), a(repeat, 0.0), b(repeat, 0.0);
    for (int i = 0; i < repeat; ++i)
    {
        if (output)
        {
            std::cout << "   run " << i << std::endl;
        }
        d1 = d2 = 0.0;
        comm->Block();
        const double t0 = Now();
        fn(d1, d2);
        t[i] = Now() - t0;
        a[i] = d1;
        b[i] = d2;
    }

    comm->AllReduce(t, LibUtilities::ReduceMax);
    comm->AllReduce(a, LibUtilities::ReduceMax);
    comm->AllReduce(b, LibUtilities::ReduceMax);

    std::vector<double> tv(t.data(), t.data() + repeat);
    std::vector<double> av(a.data(), a.data() + repeat);
    std::vector<double> bv(b.data(), b.data() + repeat);

    Result r;
    r.test      = test;
    r.impl      = impl;
    r.transport = transport;
    r.variant   = variant;
    r.tMin      = *std::min_element(tv.begin(), tv.end());
    r.tMax      = *std::max_element(tv.begin(), tv.end());
    r.tMean     = std::accumulate(tv.begin(), tv.end(), 0.0) / repeat;
    r.tMed      = Median(tv);
    r.p1Name    = p1Name;
    r.p2Name    = p2Name;
    r.p1Med     = p1Name.empty() ? 0.0 : Median(av);
    r.p2Med     = p2Name.empty() ? 0.0 : Median(bv);
    g_results.push_back(std::move(r));
}

// ---------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------

/**
 * @brief The benchmark's id set, in exactly the shape gslib consumes it.
 *
 * @a gsIds is one entry per *local slot* (global DOF, trace DOF or
 * element-local mesh entity depending on the workload), holding that slot's
 * universal id, or 0 for a slot gslib should ignore. Everything else is
 * derived from it so that the resolvers and gslib see identical input.
 */
struct Workload
{
    std::string name;
    Array<OneD, long> gsIds;

    /// Distinct nonzero ids on this rank, ascending.
    std::vector<int64_t> uniqueIds;
    /// For each slot, index into uniqueIds, or -1 if the slot is ignored.
    std::vector<int> slotToUnique;
    /// Subset of uniqueIds that a topology pre-pass found on >1 rank.
    std::vector<int64_t> sharedIds;

    size_t nSlots  = 0; ///< gsIds.size()
    size_t nActive = 0; ///< slots with a nonzero id

    /// Topology summary from the pre-pass, the things per-gather cost scales
    /// with: ranks sharing at least one id with this one, the largest sharer
    /// count of any single id, and the number of (id, neighbour) links -- one
    /// value crossing the wire per link per gather.
    size_t nNeighbours  = 0;
    size_t maxFanout    = 0;
    size_t nLinkEntries = 0;
};

/// Build the derived index structures of a Workload from its gsIds array.
void FinaliseWorkload(Workload &w)
{
    w.nSlots = w.gsIds.size();

    std::vector<int64_t> ids;
    ids.reserve(w.nSlots);
    for (size_t i = 0; i < w.nSlots; ++i)
    {
        if (w.gsIds[i] != 0)
        {
            ids.push_back(static_cast<int64_t>(w.gsIds[i]));
        }
    }
    w.nActive = ids.size();

    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    w.uniqueIds = std::move(ids);

    w.slotToUnique.assign(w.nSlots, -1);
    for (size_t i = 0; i < w.nSlots; ++i)
    {
        if (w.gsIds[i] == 0)
        {
            continue;
        }
        const auto it = std::lower_bound(w.uniqueIds.begin(), w.uniqueIds.end(),
                                         static_cast<int64_t>(w.gsIds[i]));
        w.slotToUnique[i] = static_cast<int>(it - w.uniqueIds.begin());
    }
}

/**
 * @brief Topology pre-pass: fill Workload::sharedIds with the ids that live on
 * more than one rank.
 *
 * Untimed setup: the payload test's id set, and the input to the connectivity
 * test's correctness check. Uses the alltoallv transport unconditionally --
 * this is not a measured path, and both transports have already been shown to
 * agree in SharedIdResolversMPITest.
 */
void FindSharedIds(const CommSharedPtr &comm, Workload &w)
{
    ConnectivityResolver res(std::make_shared<AlltoallvTransport>(comm));
    res.Reserve(w.uniqueIds.size());
    res.RegisterKeys(w.uniqueIds);
    res.Resolve();

    // Topology summary, for attributing per-gather cost and its growth.
    const int rank = comm->GetRank();
    std::set<int> nbrs;
    w.maxFanout    = 0;
    w.nLinkEntries = 0;
    for (const auto &kv : res.AllSharers())
    {
        w.maxFanout = std::max(w.maxFanout, kv.second.size());
        for (int r : kv.second)
        {
            if (r != rank)
            {
                nbrs.insert(r);
                ++w.nLinkEntries;
            }
        }
    }
    w.nNeighbours = nbrs.size();

    w.sharedIds.clear();
    for (size_t i = 0; i < w.uniqueIds.size(); ++i)
    {
        const std::vector<int> *s = res.FindSharers(w.uniqueIds[i]);
        if (s && s->size() > 1)
        {
            w.sharedIds.push_back(w.uniqueIds[i]);
        }
    }
}

/// Universal ids of the continuous-Galerkin global DOF, exactly as
/// AssemblyMapCG hands them to Gs::Init: boundary DOF carry their universal
/// id, interior DOF are zeroed because they cannot be shared.
Workload BuildCgWorkload(const LibUtilities::SessionReaderSharedPtr &session,
                         const SpatialDomains::MeshGraphSharedPtr &graph)
{
    const std::string var = session->GetVariable(0);
    MultiRegions::ContFieldSharedPtr field =
        MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(session,
                                                                  graph, var);

    const MultiRegions::AssemblyMapCGSharedPtr &map =
        field->GetLocalToGlobalMap();
    const Array<OneD, const int> &g2u = map->GetGlobalToUniversalMap();

    Workload w;
    w.name  = "cg";
    w.gsIds = Array<OneD, long>(map->GetNumGlobalCoeffs(), 0L);
    for (int i = 0; i < map->GetNumGlobalBndCoeffs(); ++i)
    {
        w.gsIds[i] = g2u[i];
    }
    FinaliseWorkload(w);
    return w;
}

/// Universal ids of the DG trace DOF, as AssemblyMapDG hands them to Gs::Init.
Workload BuildTraceWorkload(const LibUtilities::SessionReaderSharedPtr &session,
                            const SpatialDomains::MeshGraphSharedPtr &graph)
{
    const std::string var = session->GetVariable(0);
    MultiRegions::DisContFieldSharedPtr field =
        MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
            session, graph, var);

    const MultiRegions::AssemblyMapDGSharedPtr &map = field->GetTraceMap();
    const Array<OneD, const int> &g2u = map->GetGlobalToUniversalBndMap();

    Workload w;
    w.name  = "trace";
    w.gsIds = Array<OneD, long>(g2u.size(), 0L);
    for (size_t i = 0; i < g2u.size(); ++i)
    {
        w.gsIds[i] = g2u[i];
    }
    FinaliseWorkload(w);
    return w;
}

/**
 * @brief Mesh vertex/edge/face global ids, one slot per (element, local
 * entity) pair.
 *
 * This is the resolvers' native workload -- "agree a value across every rank
 * touching this face" -- and needs no boundary conditions or field
 * construction, so it works on any mesh the graph can read. Ids are offset
 * into disjoint vertex/edge/face ranges using globally-reduced maxima, since
 * each family numbers from zero independently.
 */
Workload BuildEntityWorkload(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
{
    MultiRegions::ExpListSharedPtr exp =
        MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(session, graph);

    int maxVid = 0, maxEid = 0;
    for (int i = 0; i < exp->GetExpSize(); ++i)
    {
        SpatialDomains::Geometry *g = exp->GetExp(i)->GetGeom();
        for (int j = 0; j < g->GetNumVerts(); ++j)
        {
            maxVid = std::max(maxVid, g->GetVid(j));
        }
        for (int j = 0; j < g->GetNumEdges(); ++j)
        {
            maxEid = std::max(maxEid, g->GetEid(j));
        }
    }

    LibUtilities::CommSharedPtr comm = session->GetComm()->GetRowComm();
    comm->AllReduce(maxVid, LibUtilities::ReduceMax);
    comm->AllReduce(maxEid, LibUtilities::ReduceMax);

    // Disjoint id ranges, all strictly positive so that 0 keeps its "ignore
    // this slot" meaning for gslib. Only the two lower bases need a global
    // bound; faces occupy the open-ended top of the range.
    const int64_t vertBase = 1;
    const int64_t edgeBase = vertBase + maxVid + 1;
    const int64_t faceBase = edgeBase + maxEid + 1;

    std::vector<long> ids;
    for (int i = 0; i < exp->GetExpSize(); ++i)
    {
        SpatialDomains::Geometry *g = exp->GetExp(i)->GetGeom();
        for (int j = 0; j < g->GetNumVerts(); ++j)
        {
            ids.push_back(static_cast<long>(vertBase + g->GetVid(j)));
        }
        for (int j = 0; j < g->GetNumEdges(); ++j)
        {
            ids.push_back(static_cast<long>(edgeBase + g->GetEid(j)));
        }
        for (int j = 0; j < g->GetNumFaces(); ++j)
        {
            ids.push_back(static_cast<long>(faceBase + g->GetFid(j)));
        }
    }
    Workload w;
    w.name  = "entity";
    w.gsIds = Array<OneD, long>(ids.size(), ids.data());
    FinaliseWorkload(w);
    return w;
}

/**
 * @brief A mesh-free workload with the shape of a partitioned mesh: each rank
 * owns a fixed number of slots, some shared with a ring of neighbours, the
 * rest purely local, plus ignored slots standing in for interior DOF.
 *
 * Runs at any rank count with no mesh file, which makes it both a quick local
 * edit-measure loop for the per-rank work and a zero-setup scaling probe.
 * Defaults match the per-rank figures of the ARCHER2 CG runs (~19k slots, 12k
 * with an id, ~6k of them shared).
 *
 * Ids are built so sharing is symmetric without communication: neighbour
 * offsets come in +/- pairs, and a pair's ids are derived from the unordered
 * (lo, hi) rank pair, so both ends generate the same ids independently. A
 * @p cornerFrac slice instead goes to blocks of 8 consecutive ranks, standing
 * in for the edge and corner DOF a real mesh shares three ways or more --
 * without them the workload would be entirely two-sharer ids, which flatters
 * any implementation that special-cases that. Slots are then shuffled
 * deterministically, since a real global numbering does not hand the shared
 * ids to a contiguous block.
 */
Workload BuildSyntheticWorkload(const CommSharedPtr &comm, size_t nSlots,
                                size_t nActive, size_t nShared, size_t nNbrs,
                                double cornerFrac)
{
    const int rank = comm->GetRank();
    const int np   = comm->GetSize();

    ASSERTL0(nActive <= nSlots,
             "--bench-synth-active must not exceed --bench-synth-slots.");
    ASSERTL0(nShared <= nActive,
             "--bench-synth-shared must not exceed --bench-synth-active.");

    std::vector<int> nbrs;
    for (size_t k = 1; nbrs.size() < nNbrs && static_cast<int>(k) < np; ++k)
    {
        for (int sign : {1, -1})
        {
            const int n = static_cast<int>(
                ((rank + sign * static_cast<int>(k)) % np + np) % np);
            if (n != rank)
            {
                nbrs.push_back(n);
            }
        }
    }
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    if (nbrs.size() > nNbrs)
    {
        nbrs.resize(nNbrs);
    }

    // Disjoint id ranges: pairwise shared ids are keyed by rank pair, block
    // shared ids by block index, local ids by rank.
    const int64_t cornerBase = static_cast<int64_t>(1) << 45;
    const int64_t localBase  = static_cast<int64_t>(1) << 50;

    std::vector<long> ids;
    ids.reserve(nSlots);

    // Block-shared ids: every rank in a block of 8 generates the same ids.
    const size_t nCorner =
        std::min(nShared, static_cast<size_t>(cornerFrac * nShared));
    const int blockSize = std::min(8, np);
    const int64_t block = rank / blockSize;
    for (size_t j = 0; j < nCorner; ++j)
    {
        ids.push_back(static_cast<long>(
            cornerBase + block * static_cast<int64_t>(nCorner + 1) +
            static_cast<int64_t>(j)));
    }

    const size_t nPairwise = nShared - nCorner;
    const size_t perPair =
        nbrs.empty() ? 0 : std::max<size_t>(nPairwise / nbrs.size(), 1);
    for (int n : nbrs)
    {
        const int64_t lo      = std::min(rank, n);
        const int64_t hi      = std::max(rank, n);
        const int64_t pairKey = lo * static_cast<int64_t>(np) + hi;
        for (size_t j = 0; j < perPair && ids.size() < nShared; ++j)
        {
            ids.push_back(static_cast<long>(
                1 + pairKey * static_cast<int64_t>(perPair) + j));
        }
    }
    const size_t nLocal = nActive - ids.size();
    for (size_t i = 0; i < nLocal; ++i)
    {
        ids.push_back(static_cast<long>(localBase +
                                        static_cast<int64_t>(rank) *
                                            static_cast<int64_t>(nActive + 1) +
                                        static_cast<int64_t>(i)));
    }
    ids.resize(nSlots, 0); // the rest take no part, as interior DOF do

    std::mt19937 rng(20260914u + static_cast<unsigned>(rank));
    std::shuffle(ids.begin(), ids.end(), rng);

    Workload w;
    w.name  = "synthetic";
    w.gsIds = Array<OneD, long>(ids.size(), ids.data());
    FinaliseWorkload(w);
    return w;
}

// ---------------------------------------------------------------------------
// Assembly: SharedIdPlan::Gather vs Gs::Gather(gs_add)
// ---------------------------------------------------------------------------

/// Per-slot contribution. Kept integral so that a sum is exact in double and
/// gslib and the plan can be compared bit-for-bit despite folding in
/// different orders.
double SlotValue(int rank, size_t slot)
{
    return static_cast<double>((rank + 1) * 1000 + static_cast<int>(slot % 97));
}

/// The workload's slot ids in SharedIdPlan's convention: gslib's "ignore this
/// slot" id 0 becomes SharedIdPlan::kIgnore.
std::vector<int64_t> PlanIds(const Workload &w)
{
    std::vector<int64_t> ids(w.nSlots);
    for (size_t i = 0; i < w.nSlots; ++i)
    {
        ids[i] = w.gsIds[i] != 0 ? static_cast<int64_t>(w.gsIds[i])
                                 : SharedIdPlan::kIgnore;
    }
    return ids;
}

void RunAssembly(const CommSharedPtr &comm, const Workload &w,
                 const std::vector<std::string> &transports,
                 const std::vector<std::string> &backends, int warmup,
                 int repeat, bool validate)
{
    const int rank = comm->GetRank();

    // Reference values every implementation must reproduce, plus the gslib
    // plan itself.
    Array<OneD, long> gsIdCopy(w.nSlots, w.gsIds.data());
    Gs::gs_data *gsh =
        comm->IsSerial() ? nullptr : Gs::Init(gsIdCopy, comm, false);

    // gslib setup, timed on its own: the like-for-like partner of the
    // plan-setup rows below.
    if (!comm->IsSerial())
    {
        Measure(comm, "assembly", "gslib-setup", "-", "-", warmup, repeat,
                [&](double &, double &) {
                    Array<OneD, long> tmp(w.nSlots, w.gsIds.data());
                    Gs::gs_data *h = Gs::Init(tmp, comm, false);
                    Gs::Free(h);
                });
    }

    // gslib gather, replaying the persistent plan.
    Array<OneD, NekDouble> u(w.nSlots, 0.0);
    auto gather = [&](double &, double &) {
        for (size_t i = 0; i < w.nSlots; ++i)
        {
            u[i] = SlotValue(rank, i);
        }
        Gs::Gather(u, Gs::gs_add, gsh);
    };
    if (comm->IsSerial())
    {
        // Gs::Gather is a no-op without MPI, so timing it would report a
        // meaningless row -- but still run it once to populate `u` with the
        // (purely local) reference values below.
        double d1 = 0.0, d2 = 0.0;
        gather(d1, d2);
    }
    else
    {
        Measure(comm, "assembly", "gslib-gather", "-", "-", warmup, repeat,
                gather);
    }

    // Reference: gathered value per distinct id, from the last gslib gather.
    std::vector<double> gsRef(w.uniqueIds.size(), 0.0);
    for (size_t i = 0; i < w.nSlots; ++i)
    {
        if (w.slotToUnique[i] >= 0)
        {
            gsRef[w.slotToUnique[i]] = u[i];
        }
    }

    auto plus = [](double a, double b) { return a + b; };

    bool foldReported = false;

    const std::vector<int64_t> planIds = PlanIds(w);

    for (const std::string &tname : transports)
    {
        std::shared_ptr<ITransport> transport = MakeTransport(comm, tname);

        // Persistent plan: discovery once (plan-setup, over this transport),
        // then one neighbour exchange per call over each backend
        // (plan-gather). The like-for-like rows against gslib.
        for (const std::string &bname : backends)
        {
            const ExchangeBackend backend = ParseBackend(bname);

            Measure(
                comm, "assembly", "plan-setup", tname, bname, warmup, repeat,
                [&](double &p1, double &p2) {
                    ConnectivityResolverTimings tm;
                    SharedIdPlan plan(transport, planIds, backend, &tm);
                    p1 = tm.requestExchangeSeconds;
                    p2 = tm.replyExchangeSeconds;
                },
                "request-exchange", "reply-exchange");

            LibUtilities::SharedIdPlanTimings planBreak;
            SharedIdPlan plan(transport, planIds, backend, nullptr, &planBreak);

            // Report once what the fold actually has to do: whether the
            // one-slot-per-id fast path applies, and how much of the traffic
            // comes from ids shared by more than two ranks -- the part that
            // costs O(k) per rank rather than O(1).
            if (!foldReported)
            {
                foldReported = true;
                long fast    = plan.UsesSingleSlotFastPath() ? 1 : 0;
                long idK2 = 0, idHi = 0, linkK2 = 0, linkHi = 0;
                for (size_t u = 0; u < plan.NumUnique(); ++u)
                {
                    const size_t k = plan.NumSharers(u);
                    if (k < 2)
                    {
                        continue;
                    }
                    if (k == 2)
                    {
                        ++idK2;
                        ++linkK2;
                    }
                    else
                    {
                        ++idHi;
                        linkHi += static_cast<long>(k - 1);
                    }
                }
                // What an owner-based fold would cost instead: for an id
                // shared by k ranks we currently put k-1 values on the wire
                // per rank (k(k-1) in total); letting the lowest sharer
                // collect, fold and reply makes that 2(k-1) in total.
                long sendNow = 0, sendOwner = 0;
                for (size_t u = 0; u < plan.NumUnique(); ++u)
                {
                    const size_t k = plan.NumSharers(u);
                    if (k < 2)
                    {
                        continue;
                    }
                    sendNow += static_cast<long>(k - 1);
                    sendOwner += (plan.GetSharers(u).front() == comm->GetRank())
                                     ? static_cast<long>(k - 1)
                                     : 1;
                }
                long sendNowMax = sendNow, sendOwnerMax = sendOwner;
                comm->AllReduce(sendNow, LibUtilities::ReduceSum);
                comm->AllReduce(sendOwner, LibUtilities::ReduceSum);
                comm->AllReduce(sendNowMax, LibUtilities::ReduceMax);
                comm->AllReduce(sendOwnerMax, LibUtilities::ReduceMax);

                double bGroup = planBreak.groupSeconds;
                double bDisc  = planBreak.discoverySeconds;
                double bLay   = planBreak.layoutSeconds;
                double bBack  = planBreak.backendSeconds;
                comm->AllReduce(bGroup, LibUtilities::ReduceMax);
                comm->AllReduce(bDisc, LibUtilities::ReduceMax);
                comm->AllReduce(bLay, LibUtilities::ReduceMax);
                comm->AllReduce(bBack, LibUtilities::ReduceMax);

                comm->AllReduce(fast, LibUtilities::ReduceMin);
                comm->AllReduce(idK2, LibUtilities::ReduceSum);
                comm->AllReduce(idHi, LibUtilities::ReduceSum);
                comm->AllReduce(linkK2, LibUtilities::ReduceSum);
                comm->AllReduce(linkHi, LibUtilities::ReduceSum);
                if (comm->GetRank() == 0)
                {
                    const long links = linkK2 + linkHi;
                    std::cout
                        << "  plan fold      : fast path "
                        << (fast ? "on" : "OFF on at least one rank")
                        << "; shared ids " << (idK2 + idHi) << " (" << idK2
                        << " two-sharer, " << idHi
                        << " higher fan-out); link entries " << links << " ("
                        << (links ? 100.0 * linkHi / links : 0.0)
                        << "% from higher fan-out)\n"
                        << "  plan setup     : group/sort " << bGroup
                        << " s, discovery " << bDisc << " s, layout " << bLay
                        << " s, backend " << bBack << " s (max over ranks)\n"
                        << "  owner-based    : would move " << sendOwner
                        << " values instead of " << sendNow << " ("
                        << (sendNow ? 100.0 * (sendNow - sendOwner) / sendNow
                                    : 0.0)
                        << "% less), max/rank " << sendOwnerMax
                        << " instead of " << sendNowMax << "\n"
                        << std::endl;
                }
            }

            std::vector<double> pu(w.nSlots, 0.0);
            Measure(
                comm, "assembly", "plan-gather", tname, bname, warmup, repeat,
                [&](double &p1, double &) {
                    for (size_t i = 0; i < w.nSlots; ++i)
                    {
                        pu[i] = SlotValue(rank, i);
                    }
                    plan.Gather(pu.data(), plus, &p1);
                },
                "exchange");

            if (validate && !comm->IsSerial())
            {
                int ok = 1;
                for (size_t i = 0; i < w.nSlots && ok; ++i)
                {
                    if (w.slotToUnique[i] >= 0 &&
                        pu[i] != gsRef[w.slotToUnique[i]])
                    {
                        ok = 0;
                    }
                }
                comm->AllReduce(ok, LibUtilities::ReduceMin);
                g_validateOk &= ok;
                if (comm->GetRank() == 0)
                {
                    std::cout
                        << "  validate assembly/" << tname << "/" << bname
                        << " (plan): " << (ok ? "OK" : "MISMATCH vs gslib")
                        << std::endl;
                }
            }
        }
    }

    Gs::Free(gsh);
}

// ---------------------------------------------------------------------------
// Connectivity: ConnectivityResolver vs Gs::Unique
// ---------------------------------------------------------------------------

/**
 * @brief Compare ConnectivityResolver against gslib's `gs_unique`.
 *
 * `gs_unique` negates every reference to a universal id except one chosen
 * globally, so that a distributed dot product counts each DOF once. The
 * resolver equivalent is: discover the sharer set, let the lowest-numbered
 * sharer own the id, and keep its first local slot positive.
 *
 * The two need not pick the *same* owner, so validation checks the property
 * that matters instead of the choice: summing the keep-flags over every rank
 * touching an id must give exactly 1.
 */
void RunConnectivity(const CommSharedPtr &comm, const Workload &w,
                     const std::vector<std::string> &transports, int warmup,
                     int repeat, bool validate)
{
    const int rank = comm->GetRank();

    const std::vector<int64_t> planIds = PlanIds(w);

    // gslib. gs_unique does its own setup internally, so this is the whole
    // operation and is directly comparable to a ConnectivityResolver Resolve().
    std::vector<int> gsKeep(w.nSlots, 0);
    if (!comm->IsSerial())
    {
        Measure(comm, "connectivity", "gslib-unique", "-", "-", warmup, repeat,
                [&](double &, double &) {
                    Array<OneD, long> tmp(w.nSlots, w.gsIds.data());
                    Gs::Unique(tmp, comm);
                    for (size_t i = 0; i < w.nSlots; ++i)
                    {
                        gsKeep[i] = (tmp[i] >= 0 && w.gsIds[i] != 0) ? 1 : 0;
                    }
                });
    }

    for (const std::string &tname : transports)
    {
        std::shared_ptr<ITransport> transport = MakeTransport(comm, tname);

        std::vector<int> resKeep(w.nSlots, 0);
        Measure(
            comm, "connectivity", "connectivityresolver", tname, "-", warmup,
            repeat,
            [&](double &p1, double &p2) {
                ConnectivityResolver res(transport);
                res.Reserve(w.uniqueIds.size());
                res.RegisterKeys(w.uniqueIds);

                ConnectivityResolverTimings tm;
                res.Resolve(&tm);
                p1 = tm.requestExchangeSeconds;
                p2 = tm.replyExchangeSeconds;

                // Lowest sharer owns; that owner keeps its first local slot.
                std::vector<int> ownUnique(w.uniqueIds.size(), 0);
                for (size_t i = 0; i < w.uniqueIds.size(); ++i)
                {
                    const std::vector<int> *s = res.FindSharers(w.uniqueIds[i]);
                    ownUnique[i] = (s && !s->empty() && s->front() == rank);
                }
                std::vector<char> taken(w.uniqueIds.size(), 0);
                std::fill(resKeep.begin(), resKeep.end(), 0);
                for (size_t i = 0; i < w.nSlots; ++i)
                {
                    const int u = w.slotToUnique[i];
                    if (u >= 0 && ownUnique[u] && !taken[u])
                    {
                        taken[u]   = 1;
                        resKeep[i] = 1;
                    }
                }
            },
            "request-exchange", "reply-exchange");

        // The plan equivalent of gs_unique: build a plan and read its mask.
        // A plan built for Gather() anyway gives the mask for free, so this
        // is an upper bound on its cost.
        std::vector<int> planKeep;
        Measure(
            comm, "connectivity", "plan-unique", tname, "-", warmup, repeat,
            [&](double &p1, double &p2) {
                ConnectivityResolverTimings tm;
                SharedIdPlan plan(transport, planIds,
                                  ExchangeBackend::ePairwise, &tm);
                planKeep = plan.UniqueMask();
                p1       = tm.requestExchangeSeconds;
                p2       = tm.replyExchangeSeconds;
            },
            "request-exchange", "reply-exchange");

        if (validate && !comm->IsSerial())
        {
            // Each id must be kept exactly once globally. Check by gathering
            // the keep-flags with gslib itself: every active slot must read
            // back 1.
            auto checkOnce = [&](const std::vector<int> &keep) {
                Array<OneD, long> idTmp(w.nSlots, w.gsIds.data());
                Gs::gs_data *h = Gs::Init(idTmp, comm, false);
                Array<OneD, NekDouble> f(w.nSlots, 0.0);
                for (size_t i = 0; i < w.nSlots; ++i)
                {
                    f[i] = keep[i];
                }
                Gs::Gather(f, Gs::gs_add, h);
                Gs::Free(h);
                int ok = 1;
                for (size_t i = 0; i < w.nSlots; ++i)
                {
                    if (w.gsIds[i] != 0 && f[i] != 1.0)
                    {
                        ok = 0;
                        break;
                    }
                }
                comm->AllReduce(ok, LibUtilities::ReduceMin);
                return ok;
            };

            const int okGs   = checkOnce(gsKeep);
            const int okRes  = checkOnce(resKeep);
            const int okPlan = checkOnce(planKeep);
            g_validateOk &= (okGs & okRes & okPlan);
            if (comm->GetRank() == 0)
            {
                std::cout << "  validate connectivity/" << tname
                          << ": gslib=" << (okGs ? "OK" : "BAD")
                          << " resolver=" << (okRes ? "OK" : "BAD")
                          << " plan=" << (okPlan ? "OK" : "BAD") << std::endl;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SharedPayloadResolver
// ---------------------------------------------------------------------------

/// Fixed-size trivially-copyable payload. The first 8 bytes carry a hash of
/// (id, source rank) so the receiver can check it got what was sent.
template <size_t N> struct Blob
{
    std::byte data[N];
};

uint64_t PayloadHash(int64_t id, int src)
{
    return static_cast<uint64_t>(id) * 1315423911ull +
           static_cast<uint64_t>(src) * 2654435761ull + 0x9e3779b97f4a7c15ull;
}

template <size_t N> Blob<N> MakeBlob(int64_t id, int src)
{
    static_assert(N >= 8, "payload must be at least 8 bytes to carry a hash");
    Blob<N> b;
    const uint64_t h = PayloadHash(id, src);
    std::memcpy(b.data, &h, sizeof(h));
    std::memset(b.data + sizeof(h), static_cast<int>(h & 0xff), N - sizeof(h));
    return b;
}

template <size_t N>
void RunPayloadImpl(const CommSharedPtr &comm, const Workload &w,
                    const std::string &tname,
                    const std::vector<std::string> &backends, int warmup,
                    int repeat, bool validate)
{
    const int rank = comm->GetRank();

    std::shared_ptr<ITransport> transport = MakeTransport(comm, tname);

    // Only shared ids are worth exchanging payloads for: an id nobody else
    // touches would produce a one-entry list containing this rank's own value.
    const std::vector<int64_t> &ids = w.sharedIds;

    // Entries whose payload hash does not match its (id, source rank).
    auto countBad = [&](const auto &res) {
        int bad = 0;
        for (int64_t id : ids)
        {
            const auto *pl = res.FindSharedPayloads(id);
            if (!pl)
            {
                ++bad;
                continue;
            }
            for (const auto &pr : *pl)
            {
                uint64_t h = 0;
                std::memcpy(&h, pr.second.data, sizeof(h));
                if (h != PayloadHash(id, pr.first))
                {
                    ++bad;
                }
            }
        }
        return bad;
    };
    auto report = [&](const std::string &label, int bad) {
        comm->AllReduce(bad, LibUtilities::ReduceSum);
        g_validateOk &= (bad == 0);
        if (comm->GetRank() == 0)
        {
            std::cout << "  validate payload/" << tname << "/" << label << ": "
                      << (bad == 0 ? "OK"
                                   : "CORRUPT (" + std::to_string(bad) + ")")
                      << std::endl;
        }
    };

    int badPairs = 0;
    Measure(
        comm, "payload", "sharedpayloadresolver", tname,
        std::to_string(N) + "B", warmup, repeat,
        [&](double &p1, double &p2) {
            SharedPayloadResolver<Blob<N>> res(transport);
            res.Reserve(ids.size());
            for (int64_t id : ids)
            {
                res.Register(id, MakeBlob<N>(id, rank));
            }

            SharedPayloadResolverTimings tm;
            res.Resolve(&tm);
            p1 = tm.discoverySeconds;
            p2 = tm.payloadExchangeSeconds;

            if (validate)
            {
                badPairs += countBad(res);
            }
        },
        "discovery", "payload-exchange");

    if (validate)
    {
        report(std::to_string(N) + "B", badPairs);
    }

    // Persistent mode: discovery once in Setup(), then each Resolve() is the
    // payload exchange alone. Payloads are rewritten every iteration so the
    // row pays the same packing work as the one-shot row's Register() loop.
    auto registerAll = [&](SharedPayloadResolver<Blob<N>> &res) {
        res.Reserve(ids.size());
        for (int64_t id : ids)
        {
            res.Register(id, MakeBlob<N>(id, rank));
        }
    };
    for (const std::string &bname : backends)
    {
        const ExchangeBackend backend = ParseBackend(bname);
        const std::string label       = std::to_string(N) + "B/" + bname;

        Measure(
            comm, "payload", "sharedpayload-setup", tname, label, warmup,
            repeat,
            [&](double &p1, double &p2) {
                SharedPayloadResolver<Blob<N>> res(transport);
                registerAll(res);
                ConnectivityResolverTimings tm;
                res.Setup(backend, &tm);
                p1 = tm.requestExchangeSeconds;
                p2 = tm.replyExchangeSeconds;
            },
            "request-exchange", "reply-exchange");

        SharedPayloadResolver<Blob<N>> res(transport);
        registerAll(res);
        res.Setup(backend);
        Measure(
            comm, "payload", "sharedpayloadresolver", tname, label, warmup,
            repeat,
            [&](double &p1, double &) {
                Blob<N> *pl = res.Payloads();
                for (size_t k = 0; k < ids.size(); ++k)
                {
                    pl[k] = MakeBlob<N>(ids[k], rank);
                }
                SharedPayloadResolverTimings tm;
                res.Resolve(&tm);
                p1 = tm.payloadExchangeSeconds;
            },
            "payload-exchange");

        if (validate)
        {
            report(label, countBad(res));
        }
    }
}

/// Runtime-to-compile-time dispatch over the compiled-in payload sizes.
bool RunPayloadSize(size_t nbytes, const CommSharedPtr &comm, const Workload &w,
                    const std::string &tname,
                    const std::vector<std::string> &backends, int warmup,
                    int repeat, bool validate)
{
    switch (nbytes)
    {
        case 8:
            RunPayloadImpl<8>(comm, w, tname, backends, warmup, repeat,
                              validate);
            return true;
        case 16:
            RunPayloadImpl<16>(comm, w, tname, backends, warmup, repeat,
                               validate);
            return true;
        case 32:
            RunPayloadImpl<32>(comm, w, tname, backends, warmup, repeat,
                               validate);
            return true;
        case 64:
            RunPayloadImpl<64>(comm, w, tname, backends, warmup, repeat,
                               validate);
            return true;
        case 128:
            RunPayloadImpl<128>(comm, w, tname, backends, warmup, repeat,
                                validate);
            return true;
        case 256:
            RunPayloadImpl<256>(comm, w, tname, backends, warmup, repeat,
                                validate);
            return true;
        case 512:
            RunPayloadImpl<512>(comm, w, tname, backends, warmup, repeat,
                                validate);
            return true;
        case 1024:
            RunPayloadImpl<1024>(comm, w, tname, backends, warmup, repeat,
                                 validate);
            return true;
        case 4096:
            RunPayloadImpl<4096>(comm, w, tname, backends, warmup, repeat,
                                 validate);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Neighbour communicator construction
// ---------------------------------------------------------------------------

void RunNeighbour(const CommSharedPtr &comm, const Workload &w,
                  const std::vector<std::string> &transports, int warmup,
                  int repeat)
{
    for (const std::string &tname : transports)
    {
        std::shared_ptr<ITransport> transport = MakeTransport(comm, tname);

        // Discovery is done once, outside the timing: what is being measured
        // here is the MPI_Dist_graph_create_adjacent that turns an already
        // known sharer set into a reusable neighbourhood communicator.
        ConnectivityResolver res(transport);
        res.Reserve(w.uniqueIds.size());
        res.RegisterKeys(w.uniqueIds);
        res.Resolve();

        Measure(comm, "neighbour", "buildneighbourcomm", tname, "-", warmup,
                repeat, [&](double &, double &) {
                    CommSharedPtr nbr = res.BuildNeighbourComm(comm);
                    (void)nbr;
                });
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void PrintTable(std::ostream &os)
{
    os << "\n"
       << std::left << std::setw(14) << "test" << std::setw(24) << "impl"
       << std::setw(9) << "transp" << std::setw(19) << "variant" << std::right
       << std::setw(12) << "min(s)" << std::setw(12) << "median(s)"
       << std::setw(12) << "mean(s)" << std::setw(12) << "max(s)" << "   "
       << std::left << "phase breakdown (median)" << "\n";
    os << std::string(140, '-') << "\n";

    os << std::scientific << std::setprecision(3);
    for (const Result &r : g_results)
    {
        os << std::left << std::setw(14) << r.test << std::setw(24) << r.impl
           << std::setw(9) << r.transport << std::setw(19) << r.variant
           << std::right << std::setw(12) << r.tMin << std::setw(12) << r.tMed
           << std::setw(12) << r.tMean << std::setw(12) << r.tMax << "   ";
        if (!r.p1Name.empty())
        {
            os << std::left << r.p1Name << "=" << r.p1Med;
        }
        if (!r.p2Name.empty())
        {
            os << "  " << r.p2Name << "=" << r.p2Med;
        }
        os << "\n";
    }
    os << std::defaultfloat << std::endl;
}

void WriteCsv(const std::string &path, const std::string &workload, int nRanks,
              int repeat, long gSlots, long gUnique, long gShared, long nbrMax,
              long fanMax, long linkMax)
{
    std::ofstream f(path);
    if (!f)
    {
        std::cerr << "CommBenchmark: cannot open '" << path
                  << "' for writing; CSV skipped." << std::endl;
        return;
    }
    f << "test,impl,transport,variant,ranks,workload,global_slots,"
         "global_unique_ids,global_shared_ids,max_neighbours,max_fanout,"
         "max_link_entries,repeat,t_min,t_median,t_mean,"
         "t_max,phase1,phase1_median,phase2,phase2_median\n";
    f << std::scientific << std::setprecision(9);
    for (const Result &r : g_results)
    {
        f << r.test << ',' << r.impl << ',' << r.transport << ',' << r.variant
          << ',' << nRanks << ',' << workload << ',' << gSlots << ',' << gUnique
          << ',' << gShared << ',' << nbrMax << ',' << fanMax << ',' << linkMax
          << ',' << repeat << ',' << r.tMin << ',' << r.tMed << ',' << r.tMean
          << ',' << r.tMax << ',' << (r.p1Name.empty() ? "-" : r.p1Name) << ','
          << r.p1Med << ',' << (r.p2Name.empty() ? "-" : r.p2Name) << ','
          << r.p2Med << '\n';
    }
}

/// Read a registered string option, or return @p dflt if it was not given.
std::string Opt(const LibUtilities::SessionReaderSharedPtr &s,
                const std::string &name, const std::string &dflt)
{
    return s->DefinesCmdLineArgument(name)
               ? s->GetCmdLineArgument<std::string>(name)
               : dflt;
}

} // namespace

int main(int argc, char *argv[])
{
    LibUtilities::SessionReaderSharedPtr session;

    try
    {
        session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    }
    catch (const std::runtime_error &)
    {
        return 1;
    }

    try
    {
        CommSharedPtr comm = session->GetComm()->GetRowComm();

        const std::string workloadName = Opt(session, "bench-workload", "cg");
        const std::vector<std::string> tests = Split(Opt(
            session, "bench-tests", "assembly,connectivity,payload,neighbour"));
        std::vector<std::string> transports =
            Split(Opt(session, "bench-transports", "a2a,crystal"));
        const std::vector<std::string> backendsRequested =
            Split(Opt(session, "bench-backends", "pairwise,neighbour,crystal"));
        const int repeat = std::stoi(Opt(session, "bench-repeat", "20"));
        const int warmup = std::stoi(Opt(session, "bench-warmup", "3"));
        const std::vector<std::string> payloadBytes =
            Split(Opt(session, "bench-payload-bytes", "8,64,512"));
        const std::string csvPath = Opt(session, "bench-csv", "");
        const bool validate =
            !session->DefinesCmdLineArgument("bench-no-validate");

        ASSERTL0(repeat > 0, "--bench-repeat must be positive.");
        ASSERTL0(warmup >= 0, "--bench-warmup must be non-negative.");
        for (const std::string &t : transports)
        {
            ASSERTL0(t == "a2a" || t == "crystal",
                     "--bench-transports accepts only 'a2a' and 'crystal'.");
        }
        std::vector<std::string> backends;
        for (const std::string &b : backendsRequested)
        {
            ASSERTL0(b == "pairwise" || b == "neighbour" || b == "crystal" ||
                         b == "transport",
                     "--bench-backends accepts only 'pairwise', 'neighbour', "
                     "'crystal' and 'transport'.");
            // The neighbourhood collective needs MPI-3. Rather than report a
            // row that SharedIdPlan has quietly turned into the pairwise
            // backend, leave it out and say why.
            if (b == "neighbour" && std::get<0>(comm->GetVersion()) < 3)
            {
                if (comm->GetRank() == 0)
                {
                    std::cout << "  note: skipping the 'neighbour' backend, "
                                 "which needs MPI-3"
                              << std::endl;
                }
                continue;
            }
            backends.push_back(b);
        }

        // The synthetic workload needs no mesh, so the session file is only
        // read for MPI and command-line setup.
        SpatialDomains::MeshGraphSharedPtr graph;
        if (workloadName != "synthetic")
        {
            graph = SpatialDomains::MeshGraphIO::Read(session);
        }

        double tSetup0 = Now();
        Workload w;
        if (workloadName == "cg")
        {
            w = BuildCgWorkload(session, graph);
        }
        else if (workloadName == "trace")
        {
            w = BuildTraceWorkload(session, graph);
        }
        else if (workloadName == "entity")
        {
            w = BuildEntityWorkload(session, graph);
        }
        else if (workloadName == "synthetic")
        {
            w = BuildSyntheticWorkload(
                comm, std::stoul(Opt(session, "bench-synth-slots", "19000")),
                std::stoul(Opt(session, "bench-synth-active", "12000")),
                std::stoul(Opt(session, "bench-synth-shared", "6000")),
                std::stoul(Opt(session, "bench-synth-neighbours", "26")),
                std::stod(Opt(session, "bench-synth-corner-frac", "0.1")));
        }
        else
        {
            NEKERROR(ErrorUtil::efatal, "--bench-workload must be one of cg, "
                                        "trace, entity or synthetic.");
        }
        const double tWorkload = Now() - tSetup0;

        FindSharedIds(comm, w);

        long gSlots  = static_cast<long>(w.nSlots);
        long gActive = static_cast<long>(w.nActive);
        long gUnique = static_cast<long>(w.uniqueIds.size());
        long gShared = static_cast<long>(w.sharedIds.size());
        comm->AllReduce(gSlots, LibUtilities::ReduceSum);
        comm->AllReduce(gActive, LibUtilities::ReduceSum);
        comm->AllReduce(gUnique, LibUtilities::ReduceSum);
        comm->AllReduce(gShared, LibUtilities::ReduceSum);

        long maxUnique = static_cast<long>(w.uniqueIds.size());
        long maxShared = static_cast<long>(w.sharedIds.size());
        comm->AllReduce(maxUnique, LibUtilities::ReduceMax);
        comm->AllReduce(maxShared, LibUtilities::ReduceMax);

        long nbrMin = static_cast<long>(w.nNeighbours);
        long nbrMax = nbrMin, nbrSum = nbrMin;
        comm->AllReduce(nbrMin, LibUtilities::ReduceMin);
        comm->AllReduce(nbrMax, LibUtilities::ReduceMax);
        comm->AllReduce(nbrSum, LibUtilities::ReduceSum);
        long fanMax  = static_cast<long>(w.maxFanout);
        long linkMax = static_cast<long>(w.nLinkEntries);
        long linkSum = linkMax;
        comm->AllReduce(fanMax, LibUtilities::ReduceMax);
        comm->AllReduce(linkMax, LibUtilities::ReduceMax);
        comm->AllReduce(linkSum, LibUtilities::ReduceSum);

        if (comm->GetRank() == 0)
        {
            std::cout << "SharedIdPlan / gslib communication benchmark\n"
                      << "  session        : " << session->GetSessionName()
                      << "\n"
                      << "  ranks          : " << comm->GetSize() << "\n"
                      << "  workload       : " << w.name << "\n"
                      << "  local slots    : " << gSlots << " total ("
                      << gActive << " with a nonzero id)\n"
                      << "  distinct ids   : " << gUnique
                      << " summed over ranks, max " << maxUnique
                      << " on a rank\n"
                      << "  shared ids     : " << gShared
                      << " summed over ranks, max " << maxShared
                      << " on a rank\n"
                      << "  neighbours     : min " << nbrMin << ", mean "
                      << (static_cast<double>(nbrSum) / comm->GetSize())
                      << ", max " << nbrMax << " per rank\n"
                      << "  max id fan-out : " << fanMax << " ranks\n"
                      << "  gather traffic : " << linkMax
                      << " values/rank max (" << (linkMax * sizeof(NekDouble))
                      << " B), " << linkSum << " summed\n"
                      << "  repeat/warmup  : " << repeat << "/" << warmup
                      << "\n"
                      << "  workload build : " << tWorkload << " s\n"
                      << std::endl;
            if (comm->IsSerial())
            {
                std::cout << "  NOTE: serial communicator -- gslib is a no-op "
                             "here, so the gslib rows and the assembly "
                             "validation are skipped.\n"
                          << std::endl;
            }
        }

        if (Contains(tests, "assembly"))
        {
            RunAssembly(comm, w, transports, backends, warmup, repeat,
                        validate);
        }
        if (Contains(tests, "connectivity"))
        {
            RunConnectivity(comm, w, transports, warmup, repeat, validate);
        }
        if (Contains(tests, "payload"))
        {
            for (const std::string &tname : transports)
            {
                for (const std::string &bs : payloadBytes)
                {
                    const size_t nb = static_cast<size_t>(std::stoul(bs));
                    if (!RunPayloadSize(nb, comm, w, tname, backends, warmup,
                                        repeat, validate) &&
                        comm->GetRank() == 0)
                    {
                        std::cerr << "CommBenchmark: payload size " << nb
                                  << " is not compiled in; skipping. Allowed: "
                                     "8,16,32,64,128,256,512,1024,4096."
                                  << std::endl;
                    }
                }
            }
        }
        if (Contains(tests, "neighbour"))
        {
            // BuildNeighbourComm() needs MPI_Dist_graph_create_adjacent,
            // which is MPI-3. Unlike the neighbourhood exchange backend
            // there is no fallback -- a neighbourhood communicator is the
            // whole point of the test -- so it simply cannot run here.
            if (std::get<0>(comm->GetVersion()) < 3)
            {
                if (comm->GetRank() == 0)
                {
                    std::cout << "  note: skipping the 'neighbour' test, "
                                 "which needs MPI-3"
                              << std::endl;
                }
            }
            else
            {
                RunNeighbour(comm, w, transports, warmup, std::min(repeat, 10));
            }
        }

        if (comm->GetRank() == 0)
        {
            PrintTable(std::cout);
            if (!csvPath.empty())
            {
                WriteCsv(csvPath, w.name, comm->GetSize(), repeat, gSlots,
                         gUnique, gShared, nbrMax, fanMax, linkMax);
                std::cout << "CSV written to " << csvPath << std::endl;
            }
        }

        // Single machine-readable summary line, so the demo can be driven as a
        // regression test as well as a benchmark. Serial runs have nothing to
        // check gslib against, so they report SKIPPED rather than a pass they
        // did not earn.
        const bool checked = validate && !comm->IsSerial();
        if (comm->GetRank() == 0)
        {
            std::cout << "CommBenchmark: np=" << comm->GetSize()
                      << " workload=" << w.name << " validation="
                      << (!checked ? "SKIPPED"
                                   : (g_validateOk ? "PASS" : "FAIL"))
                      << std::endl;
        }

        session->Finalise();

        if (checked && !g_validateOk)
        {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "CommBenchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
