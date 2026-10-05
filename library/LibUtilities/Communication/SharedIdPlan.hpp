///////////////////////////////////////////////////////////////////////////////
//
// File: SharedIdPlan.hpp
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
// Description: Persistent gather-scatter plan over shared global ids.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file SharedIdPlan.hpp
 * @brief Persistent gather-scatter plan over shared global ids: the
 * replacement for gslib's gs_setup / gs_gather / gs_unique.
 *
 * @details
 * Discovery is paid once (via #ConnectivityResolver); afterwards each call is
 * one fixed-pattern neighbour exchange carrying values only -- no ids, no
 * counts, no rendezvous -- over the backend chosen by #ExchangeBackend.
 * Callers working on flat arrays, such as the assembly maps, use this
 * directly; the id-keyed SharedPayloadResolver sits on top of it.
 *
 * See Transport.hpp for the module overview.
 */

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_SHAREDIDPLAN_HPP
#define NEKTAR_LIBUTILITIES_COMMUNICATION_SHAREDIDPLAN_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/Communication/ConnectivityResolver.hpp>

namespace Nektar::LibUtilities
{

// ===========================================================================
// Persistent (setup-once, execute-many) layer
// ===========================================================================

/**
 * @brief How a #SharedIdPlan moves data on each execute call.
 *
 * Discovery (building the plan) always runs over the ITransport the plan was
 * constructed with; this only selects the backend for the fixed-pattern
 * exchange the plan then replays on every SharedIdPlan::Gather(),
 * SharedIdPlan::Reduce() and SharedIdPlan::ExchangePayloads(). Because the
 * neighbour set and every message size are fixed once the plan is built, none
 * of these need a count exchange or any O(P) array.
 */
enum class ExchangeBackend
{
    /// One `MPI_Isend` / `MPI_Irecv` per neighbour -- the analogue of gslib's
    /// pairwise method, and the default.
    ePairwise,
    /// A single `MPI_Neighbor_alltoallv` over a distributed-graph
    /// communicator created once when the plan is built. Needs MPI-3; on an
    /// older library the plan silently uses #ePairwise instead, which moves
    /// the same data (see detail::ResolveBackend()). Backend() reports which
    /// of the two is actually in use.
    eNeighbourCollective,
    /// Route every execute through a #CrystalRouterTransport of its own,
    /// whatever transport discovery used: log P rounds carrying the whole
    /// neighbour exchange, with no array sized by the rank count and no
    /// per-neighbour message count. The analogue of gslib's crystal-router
    /// method, and the one to reach for when a rank has far more neighbours
    /// than log P.
    eCrystalRouter,
    /// Replay through the plan's own ITransport::Exchange(). Pays that
    /// transport's per-call overhead on every execute (AlltoallvTransport's
    /// O(P) count exchange), but works with any transport, including ones
    /// with no underlying Comm such as the in-memory fake in the serial unit
    /// tests. A diagnostic, not a production choice: prefer
    /// #eCrystalRouter, whose cost does not depend on what discovery used.
    eTransport
};

namespace detail
{

/// Read element @p i of a trivially-copyable @p T from a byte buffer.
template <typename T>
inline T LoadAt(const std::vector<std::byte> &buf, size_t i)
{
    T v;
    std::memcpy(static_cast<void *>(&v), buf.data() + i * sizeof(T), sizeof(T));
    return v;
}

/// Write element @p i of a trivially-copyable @p T into a byte buffer.
template <typename T>
inline void StoreAt(std::vector<std::byte> &buf, size_t i, const T &v)
{
    std::memcpy(buf.data() + i * sizeof(T), &v, sizeof(T));
}

/**
 * @brief A fixed-pattern exchange over a known neighbour set.
 *
 * Block @a i of the send buffer goes to neighbour @a i and block @a i of the
 * receive buffer arrives from it. Counts and displacements are in bytes, one
 * entry per neighbour, in the ascending-rank neighbour order the exchange was
 * constructed with. The caller pads both buffers to at least one byte, so an
 * empty vector never reaches Nektar's Comm wrappers (whose GetPointer()
 * asserts on one).
 */
class IFixedExchange
{
public:
    virtual ~IFixedExchange() = default;

    /// Collective over the plan's communicator: every rank calls it, ranks
    /// with no neighbours included.
    virtual void Execute(const std::vector<std::byte> &send,
                         const std::vector<int> &sendCounts,
                         const std::vector<int> &sendDispls,
                         std::vector<std::byte> &recv,
                         const std::vector<int> &recvCounts,
                         const std::vector<int> &recvDispls) = 0;
};

/**
 * @brief ExchangeBackend::ePairwise: one non-blocking send and receive per
 * neighbour.
 *
 * Uses the communicator's default tag. That is safe because every execute
 * completes (WaitAll) before returning and all ranks issue the plan's
 * collectives in the same order, so MPI's non-overtaking rule matches each
 * message to the receive posted for it -- the same argument gslib's pairwise
 * method and AssemblyCommDG's Pairwise exchange rely on.
 */
class PairwiseFixedExchange final : public IFixedExchange
{
public:
    PairwiseFixedExchange(CommSharedPtr comm, std::vector<int> nbrs)
        : m_comm(std::move(comm)), m_nbrs(std::move(nbrs)),
          m_sendReq(m_comm->CreateRequest(static_cast<int>(m_nbrs.size()))),
          m_recvReq(m_comm->CreateRequest(static_cast<int>(m_nbrs.size())))
    {
    }

    void Execute(const std::vector<std::byte> &send,
                 const std::vector<int> &sendCounts,
                 const std::vector<int> &sendDispls,
                 std::vector<std::byte> &recv,
                 const std::vector<int> &recvCounts,
                 const std::vector<int> &recvDispls) override
    {
        const int n = static_cast<int>(m_nbrs.size());

        // Post receives first so every matching send finds a buffer. A
        // zero-count slot is not posted; its request is still
        // MPI_REQUEST_NULL (initially, and after every WaitAll), which
        // WaitAll accepts.
        for (int i = 0; i < n; ++i)
        {
            if (recvCounts[i] > 0)
            {
                m_comm->Irecv(m_nbrs[i], recv[recvDispls[i]], recvCounts[i],
                              m_recvReq, i);
            }
        }
        for (int i = 0; i < n; ++i)
        {
            if (sendCounts[i] > 0)
            {
                // Comm::Isend() only takes a non-const reference; MPI never
                // writes through a send buffer.
                m_comm->Isend(m_nbrs[i],
                              const_cast<std::byte &>(send[sendDispls[i]]),
                              sendCounts[i], m_sendReq, i);
            }
        }
        m_comm->WaitAll(m_recvReq);
        m_comm->WaitAll(m_sendReq);
    }

private:
    CommSharedPtr m_comm;
    std::vector<int> m_nbrs;
    CommRequestSharedPtr m_sendReq;
    CommRequestSharedPtr m_recvReq;
};

/**
 * @brief ExchangeBackend::eNeighbourCollective: one `MPI_Neighbor_alltoallv`
 * over a distributed-graph communicator built at construction.
 */
class NeighbourCollectiveFixedExchange final : public IFixedExchange
{
public:
    NeighbourCollectiveFixedExchange(const CommSharedPtr &comm,
                                     std::vector<int> nbrs)
        : m_nbrs(std::move(nbrs))
    {
        // Sharing is symmetric, so sources and destinations are the same
        // list. reorder = 0 keeps block i meaning neighbour i.
        std::vector<int> weights;
        m_graph = comm->DistGraphCreateAdjacent(m_nbrs, weights, 0);
    }

    void Execute(const std::vector<std::byte> &send,
                 const std::vector<int> &sendCounts,
                 const std::vector<int> &sendDispls,
                 std::vector<std::byte> &recv,
                 const std::vector<int> &recvCounts,
                 const std::vector<int> &recvDispls) override
    {
        if (m_nbrs.empty())
        {
            // Still collective over the graph communicator, so a rank with no
            // neighbours must take part. Comm's wrappers assert on empty
            // vectors; hand it one-element dummies, none of which MPI reads
            // at in/outdegree 0.
            std::vector<int> zero(1, 0);
            std::vector<std::byte> s(1), r(1);
            m_graph->NeighborAlltoAllv(s, zero, zero, r, zero, zero);
            return;
        }

        // The Comm wrapper takes non-const references throughout; nothing
        // but the receive buffer is written.
        m_graph->NeighborAlltoAllv(const_cast<std::vector<std::byte> &>(send),
                                   const_cast<std::vector<int> &>(sendCounts),
                                   const_cast<std::vector<int> &>(sendDispls),
                                   recv,
                                   const_cast<std::vector<int> &>(recvCounts),
                                   const_cast<std::vector<int> &>(recvDispls));
    }

private:
    std::vector<int> m_nbrs;
    CommSharedPtr m_graph;
};

/**
 * @brief ExchangeBackend::eTransport: replay the fixed pattern through an
 * arbitrary ITransport, one RoutedMessage per neighbour.
 */
class TransportFixedExchange final : public IFixedExchange
{
public:
    TransportFixedExchange(std::shared_ptr<ITransport> transport,
                           std::vector<int> nbrs)
        : m_transport(std::move(transport)), m_nbrs(std::move(nbrs))
    {
    }

    void Execute(const std::vector<std::byte> &send,
                 const std::vector<int> &sendCounts,
                 const std::vector<int> &sendDispls,
                 std::vector<std::byte> &recv,
                 const std::vector<int> &recvCounts,
                 const std::vector<int> &recvDispls) override
    {
        const size_t n = m_nbrs.size();
        const int me   = m_transport->Rank();

        std::vector<RoutedMessage> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            if (sendCounts[i] == 0)
            {
                continue;
            }
            RoutedMessage m;
            m.dest = m_nbrs[i];
            m.src  = me;
            m.bytes.assign(send.begin() + sendDispls[i],
                           send.begin() + sendDispls[i] + sendCounts[i]);
            out.push_back(std::move(m));
        }

        std::vector<RoutedMessage> in;
        m_transport->Exchange(out, in);

        // ITransport may split one source's traffic over several messages
        // (see ITransport::Exchange), so append at a per-source cursor.
        std::vector<int> filled(n, 0);
        for (const RoutedMessage &m : in)
        {
            auto it = std::lower_bound(m_nbrs.begin(), m_nbrs.end(), m.src);
            ASSERTL0(it != m_nbrs.end() && *it == m.src,
                     "TransportFixedExchange: message from a rank that is "
                     "not a plan neighbour");
            const size_t i   = static_cast<size_t>(it - m_nbrs.begin());
            const int nBytes = static_cast<int>(m.bytes.size());
            ASSERTL0(filled[i] + nBytes <= recvCounts[i],
                     "TransportFixedExchange: neighbour sent more than the "
                     "plan expects");
            std::copy(m.bytes.begin(), m.bytes.end(),
                      recv.begin() + recvDispls[i] + filled[i]);
            filled[i] += nBytes;
        }
        for (size_t i = 0; i < n; ++i)
        {
            ASSERTL0(filled[i] == recvCounts[i],
                     "TransportFixedExchange: neighbour sent less than the "
                     "plan expects");
        }
    }

private:
    std::shared_ptr<ITransport> m_transport;
    std::vector<int> m_nbrs;
};

/**
 * @brief The backend a plan can actually use, given what @p comm supports.
 *
 * ExchangeBackend::eNeighbourCollective needs
 * `MPI_Dist_graph_create_adjacent`, which arrived in MPI-3. Some MPI
 * libraries still in service predate it -- Microsoft MPI among them -- so
 * there is nothing to build the neighbourhood communicator with. Fall back
 * to ExchangeBackend::ePairwise, which moves exactly the same data with one
 * send and receive per neighbour; results are identical either way, only the
 * MPI calls differ. MultiRegions::AssemblyCommDG drops its own neighbourhood
 * method on the same test.
 *
 * @param backend Backend the caller asked for.
 * @param comm    Communicator the plan will run over; null for a transport
 *                with no underlying Comm, where nothing is downgraded.
 * @return @p backend, or ePairwise if it cannot be honoured.
 */
inline ExchangeBackend ResolveBackend(ExchangeBackend backend,
                                      const CommSharedPtr &comm)
{
    if (backend == ExchangeBackend::eNeighbourCollective && comm &&
        std::get<0>(comm->GetVersion()) < 3)
    {
        return ExchangeBackend::ePairwise;
    }
    return backend;
}

/// Construct the IFixedExchange for @p backend over @p nbrs.
inline std::unique_ptr<IFixedExchange> MakeFixedExchange(
    ExchangeBackend backend, const std::shared_ptr<ITransport> &transport,
    const std::vector<int> &nbrs)
{
    if (backend == ExchangeBackend::eTransport)
    {
        return std::make_unique<TransportFixedExchange>(transport, nbrs);
    }

    CommSharedPtr comm = transport->Comm();
    ASSERTL0(comm, "Every ExchangeBackend but eTransport needs a transport "
                   "with an underlying Comm; use ExchangeBackend::eTransport "
                   "otherwise.");
    if (backend == ExchangeBackend::eCrystalRouter)
    {
        // Its own router, so that the cost of an execute never depends on
        // which transport discovery happened to run over.
        return std::make_unique<TransportFixedExchange>(
            std::make_shared<CrystalRouterTransport>(comm), nbrs);
    }
    if (backend == ExchangeBackend::ePairwise)
    {
        return std::make_unique<PairwiseFixedExchange>(comm, nbrs);
    }
    return std::make_unique<NeighbourCollectiveFixedExchange>(comm, nbrs);
}

} // namespace detail

/**
 * @brief Optional breakdown of where SharedIdPlan's construction time goes,
 * for deciding which stage is worth optimising. Pass one to the constructor;
 * leave it null for no measurement overhead. Seconds; the stages are
 * disjoint and sum to roughly the whole build.
 */
struct SharedIdPlanTimings
{
    /// Grouping slots by id: the sort over (id, slot) pairs.
    double groupSeconds = 0.0;
    /// The whole ConnectivityResolver::Resolve() pass, exchanges included.
    double discoverySeconds = 0.0;
    /// Sharer, neighbour, link and fold index construction.
    double layoutSeconds = 0.0;
    /// Building the exchange backend (for eNeighbourCollective, this creates
    /// the distributed-graph communicator).
    double backendSeconds = 0.0;
};

/**
 * @class SharedIdPlan
 * @brief A persistent gather-scatter plan over shared global ids: discovery
 * is paid once, then a fixed neighbour exchange is replayed as often as
 * needed. The replacement for gslib's `gs_setup` / `gs_gather` / `gs_unique`,
 * and the engine behind #SharedPayloadResolver's persistent mode.
 *
 * @details
 * ### Why a plan
 * The rendezvous resolvers answer a question once: every Resolve()
 * rediscovers who shares what. That suits one-off agreements at mesh-setup
 * time, but assembly-style operations repeat the *same* exchange over the
 * *same* ids with new values many times. A plan does the discovery once and
 * afterwards moves only values.
 *
 * ### Model
 * The plan is built over an array of per-*slot* ids, exactly like the id
 * array gslib takes: a slot is any local storage location (a global DOF, a
 * trace point, an element's copy of a face, ...). Several slots may carry
 * the same id; slots whose id is #kIgnore take no part. Construction:
 *
 * -# groups slots by id into this rank's *unique* ids, in ascending order;
 * -# runs one #ConnectivityResolver pass over the plan's transport to find
 *    the sharer set of every unique id -- the plan's only rendezvous;
 * -# lists, for each neighbour rank, the ids shared with it in ascending-id
 *    order. Sharer sets are identical on every sharer, so that list is the
 *    same at both ends of every link: message layouts line up without ever
 *    exchanging an index, and no id crosses the wire again;
 * -# sets up the fixed-pattern exchange selected by #ExchangeBackend.
 *
 * Every execute call is then a local fold of slots into per-id partials, one
 * neighbour exchange of those partials (sizeof(T) bytes per shared id per
 * link -- no ids, no counts), a fold across sharers, and a scatter back.
 *
 * ### Reproducibility
 * Each id's value is folded across its sharers in ascending-rank order, each
 * rank contributing one partial that it folded over its own slots in slot
 * order. Every sharer therefore applies `op` to identical operands in an
 * identical order and ends with a bitwise-identical result, whatever the
 * backend or message arrival order. Note that a rank's repeated slots for
 * one id are pre-folded locally before the exchange, so for a
 * non-associative `op` (floating-point addition) the result can differ in
 * the last bits from a scheme that folds every contribution individually --
 * gslib's `gs_gather` included.
 *
 * ### Cost
 * Construction is one ConnectivityResolver::Resolve() (two transport
 * exchanges) plus O(local slots) sorting, and is collective. Every execute
 * call is collective over the plan's communicator; its traffic is
 * point-to-point between sharers only (O(k^2) links for an id shared by k
 * ranks, as for gslib's pairwise method), and memory is O(local slots +
 * neighbour links). Nothing is sized by P or by N_global.
 *
 * Per-call work is proportional to the shared interface, not to the slot
 * array: ids that are neither shared nor repeated locally are never touched.
 * When every id owns exactly one slot -- the CG and DG trace case -- values
 * are packed straight from the slot array, with no intermediate per-id
 * buffer (see UsesSingleSlotFastPath()).
 *
 * ### Example (CG assembly, in place of gslib)
 * @code{.cpp}
 * SharedIdPlan plan(transport, universalIds);            // gs_setup
 * std::vector<int> owned = plan.UniqueMask();             // gs_unique
 * for (...)
 *     plan.Gather(values.data(), std::plus<double>{});    // gs_gather(gs_add)
 * @endcode
 */
class SharedIdPlan
{
public:
    /// Slot id meaning "take no part", the role id 0 plays for gslib. Mesh
    /// entity ids use 0 legitimately, so a distinct sentinel is used; a
    /// caller porting from gslib maps its zeros to this.
    static constexpr int64_t kIgnore = std::numeric_limits<int64_t>::min();

    /**
     * @brief Build the plan. Collective over @p transport's ranks.
     *
     * @tparam IdContainer Any iterable of integer ids.
     * @param transport Transport for the one-off discovery; also the
     *                  communicator the execute calls run over.
     * @param slotIds   One id per local slot, #kIgnore for slots to skip.
     * @param backend   Exchange used by every execute call.
     * @param timings   Optional discovery-exchange timing breakdown.
     */
    template <typename IdContainer>
    SharedIdPlan(std::shared_ptr<ITransport> transport,
                 const IdContainer &slotIds,
                 ExchangeBackend backend = ExchangeBackend::ePairwise,
                 ConnectivityResolverTimings *timings = nullptr,
                 SharedIdPlanTimings *breakdown       = nullptr)
        : m_transport(std::move(transport)), m_rank(m_transport->Rank()),
          m_nproc(m_transport->Size()),
          m_backend(detail::ResolveBackend(backend, m_transport->Comm()))
    {
        std::vector<int64_t> ids;
        for (auto id : slotIds)
        {
            ids.push_back(static_cast<int64_t>(id));
        }
        Build(ids, timings, breakdown);
    }

    SharedIdPlan(const SharedIdPlan &)            = delete;
    SharedIdPlan &operator=(const SharedIdPlan &) = delete;

    /**
     * @brief In-place gather-scatter over the slot array: the `gs_gather`
     * equivalent. Collective.
     *
     * On return every non-ignored slot holds the fold, under @p op, of every
     * slot on every rank carrying the same id; ignored slots are untouched.
     * A slot whose id is neither shared nor repeated locally is not accessed
     * at all (its fold is itself), so @p slots need only extend to
     * ActiveSlotBound(). gslib behaves the same way, and AssemblyMap's
     * multi-level static condensation relies on it.
     *
     * @param slots           NumSlots() values, updated in place.
     * @param op              Associative, commutative `(T, T) -> T`.
     * @param exchangeSeconds Optional: wall time of the exchange alone.
     */
    template <typename T, typename Op>
    void Gather(T *slots, Op &&op, double *exchangeSeconds = nullptr)
    {
        if (m_allSingleSlot)
        {
            // One slot per id: write the result straight back, skipping the
            // per-id slot loop and its two bound loads.
            Fold<T>(
                slots, op,
                [&](size_t u, const T &v) { slots[m_ownSlot[u]] = v; },
                exchangeSeconds);
            return;
        }

        Fold<T>(
            slots, op,
            [&](size_t u, const T &v) {
                for (int p = m_uSlotPtr[u]; p < m_uSlotPtr[u + 1]; ++p)
                {
                    slots[m_uSlots[p]] = v;
                }
            },
            exchangeSeconds);
    }

    /**
     * @brief As Gather(), but leaves @p slots untouched and writes one result
     * per unique id (in UniqueIds() order) to @p uniqueOut. Collective.
     */
    template <typename T, typename Op>
    void Reduce(const T *slots, T *uniqueOut, Op &&op,
                double *exchangeSeconds = nullptr)
    {
        Fold<T>(
            slots, op, [&](size_t u, const T &v) { uniqueOut[u] = v; },
            exchangeSeconds);

        // Ids Fold() skips are a single, unshared slot: the slot itself is
        // the result.
        for (const int u : m_inactiveUnique)
        {
            uniqueOut[u] = slots[m_uSlots[m_uSlotPtr[u]]];
        }
    }

    /**
     * @brief One-off, collective setup for ExchangePayloads(): tells each
     * neighbour how many slots this rank holds per shared id. Idempotent.
     *
     * Kept out of the constructor so pure Gather()/Reduce() users do not pay
     * for it.
     */
    void PreparePayloadExchange()
    {
        if (m_payloadReady)
        {
            return;
        }

        const size_t nU = m_uniqueIds.size();
        const size_t nE = m_nbrUnique.size();
        const size_t n  = m_nbrs.size();

        // Exchange per-link slot multiplicities over the plan's own backend.
        m_sendBuf.resize(std::max<size_t>(nE * sizeof(int32_t), 1));
        m_recvBuf.resize(std::max<size_t>(nE * sizeof(int32_t), 1));
        for (size_t e = 0; e < nE; ++e)
        {
            const int u = m_nbrUnique[e];
            detail::StoreAt<int32_t>(m_sendBuf, e,
                                     m_uSlotPtr[u + 1] - m_uSlotPtr[u]);
        }
        RunExchange(sizeof(int32_t), m_nbrPtr, m_nbrPtr, nullptr);
        std::vector<int> remoteMult(nE);
        for (size_t e = 0; e < nE; ++e)
        {
            remoteMult[e] = detail::LoadAt<int32_t>(m_recvBuf, e);
        }

        // Send layout: per link, each shared id's local slots in slot order.
        m_paySendPtr.assign(n + 1, 0);
        m_paySendSlots.clear();
        for (size_t i = 0; i < n; ++i)
        {
            for (int e = m_nbrPtr[i]; e < m_nbrPtr[i + 1]; ++e)
            {
                const int u = m_nbrUnique[e];
                for (int p = m_uSlotPtr[u]; p < m_uSlotPtr[u + 1]; ++p)
                {
                    m_paySendSlots.push_back(m_uSlots[p]);
                }
            }
            m_paySendPtr[i + 1] = static_cast<int>(m_paySendSlots.size());
        }

        // Receive layout: the mirror image, using the neighbour's counts.
        m_payRecvPtr.assign(n + 1, 0);
        std::vector<int> entryOff(nE, 0);
        int off = 0;
        for (size_t i = 0; i < n; ++i)
        {
            for (int e = m_nbrPtr[i]; e < m_nbrPtr[i + 1]; ++e)
            {
                entryOff[e] = off;
                off += remoteMult[e];
            }
            m_payRecvPtr[i + 1] = off;
        }

        // Per-id list of (rank, source), in the fold's ascending-rank order.
        // A source below NumSlots() is a local slot; anything else is
        // NumSlots() + an element offset into the receive buffer.
        const int nSlots = static_cast<int>(m_nSlots);
        m_payPtr.assign(1, 0);
        m_paySrc.clear();
        m_payRank.clear();
        for (size_t u = 0; u < nU; ++u)
        {
            for (int f = m_foldPtr[u]; f < m_foldPtr[u + 1]; ++f)
            {
                const int s = m_foldSrc[f];
                if (s < static_cast<int>(nU))
                {
                    for (int p = m_uSlotPtr[u]; p < m_uSlotPtr[u + 1]; ++p)
                    {
                        m_paySrc.push_back(m_uSlots[p]);
                        m_payRank.push_back(m_rank);
                    }
                }
                else
                {
                    const int e   = s - static_cast<int>(nU);
                    const int nbr = m_nbrs[std::upper_bound(m_nbrPtr.begin(),
                                                            m_nbrPtr.end(), e) -
                                           m_nbrPtr.begin() - 1];
                    for (int k = 0; k < remoteMult[e]; ++k)
                    {
                        m_paySrc.push_back(nSlots + entryOff[e] + k);
                        m_payRank.push_back(nbr);
                    }
                }
            }
            m_payPtr.push_back(static_cast<int>(m_paySrc.size()));
        }

        m_payloadReady = true;
    }

    /**
     * @brief Deliver every slot's value to every rank sharing its id, with no
     * reduction. Collective; requires PreparePayloadExchange().
     *
     * On return, entries [PayloadBegin(u), PayloadEnd(u)) of @p out are the
     * values of every slot carrying unique id @a u on every sharer, this
     * rank's included, ordered by ascending rank (PayloadRank()) and by slot
     * order within a rank. Every sharer sees the same list.
     */
    template <typename T>
    void ExchangePayloads(const T *slots, std::vector<T> &out,
                          double *exchangeSeconds = nullptr)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "SharedIdPlan payloads must be trivially copyable");
        ASSERTL0(m_payloadReady, "SharedIdPlan::ExchangePayloads: call "
                                 "PreparePayloadExchange() first.");

        const size_t nSend = m_paySendSlots.size();
        const size_t nRecv = static_cast<size_t>(m_payRecvPtr.back());
        m_sendBuf.resize(std::max<size_t>(nSend * sizeof(T), 1));
        m_recvBuf.resize(std::max<size_t>(nRecv * sizeof(T), 1));
        for (size_t k = 0; k < nSend; ++k)
        {
            detail::StoreAt<T>(m_sendBuf, k, slots[m_paySendSlots[k]]);
        }

        RunExchange(sizeof(T), m_paySendPtr, m_payRecvPtr, exchangeSeconds);

        const int nSlots = static_cast<int>(m_nSlots);
        out.resize(m_paySrc.size());
        for (size_t e = 0; e < m_paySrc.size(); ++e)
        {
            const int s = m_paySrc[e];
            out[e]      = s < nSlots ? slots[s]
                                     : detail::LoadAt<T>(m_recvBuf, s - nSlots);
        }
    }

    /**
     * @brief The `gs_unique` equivalent: 1 on exactly one slot per id
     * globally (the lowest-numbered slot on the lowest-numbered sharer), 0
     * everywhere else, ignored slots included. Local; no communication.
     */
    std::vector<int> UniqueMask() const
    {
        std::vector<int> mask(m_nSlots, 0);
        for (size_t u = 0; u < m_uniqueIds.size(); ++u)
        {
            if (m_shrRanks[m_shrPtr[u]] == m_rank)
            {
                mask[m_uSlots[m_uSlotPtr[u]]] = 1;
            }
        }
        return mask;
    }

    /// @return Number of slots the plan was built over.
    size_t NumSlots() const
    {
        return m_nSlots;
    }

    /// @return Number of distinct non-ignored ids on this rank.
    size_t NumUnique() const
    {
        return m_uniqueIds.size();
    }

    /// @return This rank's distinct ids, ascending; index = unique index.
    const std::vector<int64_t> &UniqueIds() const
    {
        return m_uniqueIds;
    }

    /// @return Per slot, its unique index, or -1 for an ignored slot.
    const std::vector<int> &SlotToUnique() const
    {
        return m_slotToUnique;
    }

    /// @return Unique index of @p id, or -1 if this rank does not hold it.
    int FindUnique(int64_t id) const
    {
        auto it = std::lower_bound(m_uniqueIds.begin(), m_uniqueIds.end(), id);
        return (it != m_uniqueIds.end() && *it == id)
                   ? static_cast<int>(it - m_uniqueIds.begin())
                   : -1;
    }

    /// @return Sharer ranks of unique id @p u, ascending, this rank included.
    std::vector<int> GetSharers(size_t u) const
    {
        return std::vector<int>(m_shrRanks.begin() + m_shrPtr[u],
                                m_shrRanks.begin() + m_shrPtr[u + 1]);
    }

    /// @return Number of ranks sharing unique id @p u, this rank included.
    size_t NumSharers(size_t u) const
    {
        return static_cast<size_t>(m_shrPtr[u + 1] - m_shrPtr[u]);
    }

    /// @return Whether unique id @p u lives on any rank but this one.
    bool IsShared(size_t u) const
    {
        return m_shrPtr[u + 1] - m_shrPtr[u] > 1;
    }

    /// @return One past the highest slot Gather() or Reduce() reads or
    /// writes. Every slot beyond it holds an id that is neither shared nor
    /// repeated locally, so an array handed to Gather() may stop there.
    size_t ActiveSlotBound() const
    {
        return m_activeSlotBound;
    }

    /// @return Every rank sharing at least one id with this one, ascending.
    const std::vector<int> &Neighbours() const
    {
        return m_nbrs;
    }

    /// @return First index in ExchangePayloads() output for unique id @p u.
    size_t PayloadBegin(size_t u) const
    {
        return static_cast<size_t>(m_payPtr[u]);
    }

    /// @return One past the last ExchangePayloads() index for unique id @p u.
    size_t PayloadEnd(size_t u) const
    {
        return static_cast<size_t>(m_payPtr[u + 1]);
    }

    /// @return The rank that owns entry @p e of ExchangePayloads() output.
    int PayloadRank(size_t e) const
    {
        return m_payRank[e];
    }

    /// @return Whether the one-slot-per-id fast path is in use. True for the
    /// CG and DG trace id arrays; false once any id owns several slots.
    bool UsesSingleSlotFastPath() const
    {
        return m_allSingleSlot;
    }

    /// @return The backend every execute call uses. Not necessarily the one
    /// asked for: see detail::ResolveBackend().
    ExchangeBackend Backend() const
    {
        return m_backend;
    }

    /**
     * @brief Build a distributed-graph communicator over Neighbours(), for
     * callers who want to drive their own neighbourhood collectives.
     * Collective over @p base.
     */
    CommSharedPtr BuildNeighbourComm(const CommSharedPtr &base) const
    {
        std::vector<int> ranks = m_nbrs, weights;
        return base->DistGraphCreateAdjacent(ranks, weights, 0);
    }

private:
    /// Everything the constructor does beyond collecting ids.
    void Build(const std::vector<int64_t> &ids,
               ConnectivityResolverTimings *timings,
               SharedIdPlanTimings *breakdown = nullptr)
    {
        m_nSlots = ids.size();
        ASSERTL0(m_nSlots <=
                     static_cast<size_t>(std::numeric_limits<int>::max()),
                 "SharedIdPlan: more than INT_MAX slots on one rank.");

        // Group active slots by id. Sorting (id, slot) pairs leaves each
        // id's slots in ascending slot order, which fixes the local fold
        // order and the "first slot" UniqueMask() keeps.
        std::vector<std::pair<int64_t, int>> order;
        order.reserve(m_nSlots);
        for (size_t s = 0; s < m_nSlots; ++s)
        {
            if (ids[s] != kIgnore)
            {
                order.emplace_back(ids[s], static_cast<int>(s));
            }
        }
        const auto tGroup0 = std::chrono::steady_clock::now();
        std::sort(order.begin(), order.end());

        m_slotToUnique.assign(m_nSlots, -1);
        m_uSlotPtr.assign(1, 0);
        m_uSlots.reserve(order.size());
        for (size_t i = 0; i < order.size(); ++i)
        {
            if (i == 0 || order[i].first != order[i - 1].first)
            {
                if (i > 0)
                {
                    m_uSlotPtr.push_back(static_cast<int>(i));
                }
                m_uniqueIds.push_back(order[i].first);
            }
            m_slotToUnique[order[i].second] =
                static_cast<int>(m_uniqueIds.size() - 1);
            m_uSlots.push_back(order[i].second);
        }
        if (!order.empty())
        {
            m_uSlotPtr.push_back(static_cast<int>(order.size()));
        }
        const size_t nU = m_uniqueIds.size();

        // Discovery: the plan's one and only rendezvous.
        if (breakdown)
        {
            breakdown->groupSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              tGroup0)
                    .count();
        }

        const auto tDisc0 = std::chrono::steady_clock::now();
        ConnectivityResolver disc(m_transport);
        disc.Reserve(nU);
        disc.RegisterKeys(m_uniqueIds);
        disc.Resolve(timings);
        const auto tLayout0 = std::chrono::steady_clock::now();
        if (breakdown)
        {
            breakdown->discoverySeconds =
                std::chrono::duration<double>(tLayout0 - tDisc0).count();
        }

        // Sharer lists (ascending, self included) and the neighbour set.
        m_shrPtr.assign(1, 0);
        m_shrPtr.reserve(nU + 1);
        // We registered exactly m_uniqueIds, ascending and deduped, and the
        // resolver returns its answers in the same order -- so walk the two
        // in step rather than looking up every id, which at ~200k ids a rank
        // was a measurable slice of the build.
        ASSERTL1(disc.NumResolved() == nU,
                 "SharedIdPlan: discovery returned a different id count");
        for (size_t u = 0; u < nU; ++u)
        {
            ASSERTL1(disc.ResolvedId(u) == m_uniqueIds[u],
                     "SharedIdPlan: discovery reordered the ids");
            const ConnectivityResolver::SharerView sh = disc.SharersAt(u);
            ASSERTL1(std::binary_search(sh.begin(), sh.end(), m_rank),
                     "SharedIdPlan: discovery lost this rank's own id");
            m_shrRanks.insert(m_shrRanks.end(), sh.begin(), sh.end());
            m_shrPtr.push_back(static_cast<int>(m_shrRanks.size()));
            for (int r : sh)
            {
                if (r != m_rank)
                {
                    m_nbrs.push_back(r);
                }
            }
        }
        std::sort(m_nbrs.begin(), m_nbrs.end());
        m_nbrs.erase(std::unique(m_nbrs.begin(), m_nbrs.end()), m_nbrs.end());

        auto nbrIndex = [&](int r) {
            return static_cast<size_t>(
                std::lower_bound(m_nbrs.begin(), m_nbrs.end(), r) -
                m_nbrs.begin());
        };

        // Per-link id lists and the fold order, in one pass. Visiting ids in
        // ascending order keeps every link's list ascending by id, which is
        // what makes both ends of a link agree on its layout. Each fold
        // source is either this rank's own partial (index u) or a receive
        // entry (NumUnique() + its position in the receive buffer, which
        // mirrors the send layout exactly).
        m_nbrPtr.assign(m_nbrs.size() + 1, 0);
        for (size_t u = 0; u < nU; ++u)
        {
            for (int k = m_shrPtr[u]; k < m_shrPtr[u + 1]; ++k)
            {
                if (m_shrRanks[k] != m_rank)
                {
                    ++m_nbrPtr[nbrIndex(m_shrRanks[k]) + 1];
                }
            }
        }
        for (size_t i = 0; i < m_nbrs.size(); ++i)
        {
            m_nbrPtr[i + 1] += m_nbrPtr[i];
        }

        m_nbrUnique.assign(m_nbrPtr.back(), -1);
        std::vector<int> cursor(m_nbrPtr.begin(), m_nbrPtr.end() - 1);
        m_foldPtr.assign(1, 0);
        m_foldPtr.reserve(nU + 1);
        m_foldSrc.reserve(nU + m_nbrUnique.size());
        for (size_t u = 0; u < nU; ++u)
        {
            for (int k = m_shrPtr[u]; k < m_shrPtr[u + 1]; ++k)
            {
                const int r = m_shrRanks[k];
                if (r == m_rank)
                {
                    m_foldSrc.push_back(static_cast<int>(u));
                }
                else
                {
                    const int pos    = cursor[nbrIndex(r)]++;
                    m_nbrUnique[pos] = static_cast<int>(u);
                    m_foldSrc.push_back(static_cast<int>(nU) + pos);
                }
            }
            m_foldPtr.push_back(static_cast<int>(m_foldSrc.size()));
        }

        // Only ids shared with another rank or held in more than one slot
        // need work on each call; any other id is a single, unshared slot
        // whose fold is itself. Skipping those matches gslib, which never
        // touches them either -- a contract AssemblyMap relies on when a
        // multi-level static condensation level gathers through its parent
        // level's handle with an array that stops short of the parent's
        // condensed-away slots -- and keeps per-call work proportional to
        // the shared interface rather than the whole slot array.
        m_activeUnique.clear();
        m_inactiveUnique.clear();
        m_activeSlotBound = 0;
        for (size_t u = 0; u < nU; ++u)
        {
            if (IsShared(u) || m_uSlotPtr[u + 1] - m_uSlotPtr[u] > 1)
            {
                m_activeUnique.push_back(static_cast<int>(u));
                m_activeSlotBound = std::max(
                    m_activeSlotBound,
                    static_cast<size_t>(m_uSlots[m_uSlotPtr[u + 1] - 1]) + 1);
            }
            else
            {
                m_inactiveUnique.push_back(static_cast<int>(u));
            }
        }

        // When every id occupies exactly one slot -- the CG and DG trace id
        // arrays, where a global dof appears once per rank -- the local fold
        // is just a copy. Fold() can then pack the send buffer straight from
        // the slot array and read own contributions from it too, dropping the
        // partial buffer and a whole pass over the interface in each
        // direction. Precompute the two index maps that needs.
        m_allSingleSlot = (m_uSlots.size() == nU);
        m_ownSlot.clear();
        m_sendSlot.clear();
        m_k2OwnFirst.clear();
        m_k2OwnLast.clear();
        m_foldGeneral.clear();
        if (m_allSingleSlot)
        {
            m_ownSlot.resize(nU);
            for (size_t u = 0; u < nU; ++u)
            {
                m_ownSlot[u] = m_uSlots[m_uSlotPtr[u]];
            }
            m_sendSlot.resize(m_nbrUnique.size());
            for (size_t e = 0; e < m_nbrUnique.size(); ++e)
            {
                m_sendSlot[e] = m_ownSlot[m_nbrUnique[e]];
            }

            // An id shared with exactly one other rank -- the bulk of a CG
            // interface -- folds two values whose origins are known at build
            // time. Splitting those by rank order lets Fold() run them in two
            // branch-free loops that still apply op in the canonical
            // ascending-rank order. Anything else keeps the general loop.
            const int nUi = static_cast<int>(nU);
            for (const int u : m_activeUnique)
            {
                const int f0 = m_foldPtr[u], f1 = m_foldPtr[u + 1];
                if (f1 - f0 != 2)
                {
                    m_foldGeneral.push_back(u);
                    continue;
                }
                const int s0 = m_foldSrc[f0], s1 = m_foldSrc[f0 + 1];
                const bool ownFirst = (s0 < nUi);
                K2Fold k;
                k.unique = u;
                k.slot   = m_ownSlot[u];
                k.recv   = (ownFirst ? s1 : s0) - nUi;
                (ownFirst ? m_k2OwnFirst : m_k2OwnLast).push_back(k);
            }
        }

        const auto tBackend0 = std::chrono::steady_clock::now();
        if (breakdown)
        {
            breakdown->layoutSeconds =
                std::chrono::duration<double>(tBackend0 - tLayout0).count();
        }

        m_exchange = detail::MakeFixedExchange(m_backend, m_transport, m_nbrs);

        if (breakdown)
        {
            breakdown->backendSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              tBackend0)
                    .count();
        }
    }

    /**
     * @brief Run the backend over the current send/receive buffers.
     *
     * @param elemBytes Size of one element.
     * @param sendPtr   Per-link CSR offsets into the send buffer, elements.
     * @param recvPtr   Likewise for the receive buffer.
     * @param seconds   Optional: filled with the exchange's wall time.
     */
    void RunExchange(size_t elemBytes, const std::vector<int> &sendPtr,
                     const std::vector<int> &recvPtr, double *seconds)
    {
        const size_t n = m_nbrs.size();
        ASSERTL0(
            static_cast<int64_t>(std::max(sendPtr.back(), recvPtr.back())) *
                    static_cast<int64_t>(elemBytes) <=
                std::numeric_limits<int>::max(),
            "SharedIdPlan: exchange exceeds the MPI int count limit "
            "(2^31-1 bytes) on one rank.");

        const int sz = static_cast<int>(elemBytes);
        m_sc.resize(n);
        m_sd.resize(n);
        m_rc.resize(n);
        m_rd.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            m_sd[i] = sendPtr[i] * sz;
            m_sc[i] = (sendPtr[i + 1] - sendPtr[i]) * sz;
            m_rd[i] = recvPtr[i] * sz;
            m_rc[i] = (recvPtr[i + 1] - recvPtr[i]) * sz;
        }

        const auto t0 = std::chrono::steady_clock::now();
        // A single rank has no neighbours and nothing to exchange; skipping
        // keeps serial runs off backends that need a parallel Comm.
        if (m_nproc > 1)
        {
            m_exchange->Execute(m_sendBuf, m_sc, m_sd, m_recvBuf, m_rc, m_rd);
        }
        if (seconds)
        {
            *seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        }
    }

    /**
     * @brief Shared body of Gather() and Reduce(): local fold, exchange,
     * cross-rank fold, then hand each unique id's result to @p sink.
     *
     * All local partials are computed before @p sink is first called, so a
     * sink that writes back into @p slots (Gather()) is safe.
     */
    template <typename T, typename Op, typename Sink>
    void Fold(const T *slots, Op &op, Sink &&sink, double *exchangeSeconds)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "SharedIdPlan values must be trivially copyable");

        if (m_allSingleSlot)
        {
            FoldSingleSlot<T>(slots, op, sink, exchangeSeconds);
            return;
        }

        const size_t nU = m_uniqueIds.size();
        const size_t nE = m_nbrUnique.size();

        // Only active ids are visited (see Build()); partials of the others
        // are never read, since only shared ids go on the wire.
        m_partial.resize(std::max<size_t>(nU * sizeof(T), 1));
        for (const int u : m_activeUnique)
        {
            T acc = slots[m_uSlots[m_uSlotPtr[u]]];
            for (int p = m_uSlotPtr[u] + 1; p < m_uSlotPtr[u + 1]; ++p)
            {
                acc = op(acc, slots[m_uSlots[p]]);
            }
            detail::StoreAt<T>(m_partial, u, acc);
        }

        m_sendBuf.resize(std::max<size_t>(nE * sizeof(T), 1));
        m_recvBuf.resize(std::max<size_t>(nE * sizeof(T), 1));
        for (size_t e = 0; e < nE; ++e)
        {
            std::memcpy(m_sendBuf.data() + e * sizeof(T),
                        m_partial.data() + m_nbrUnique[e] * sizeof(T),
                        sizeof(T));
        }

        RunExchange(sizeof(T), m_nbrPtr, m_nbrPtr, exchangeSeconds);

        const int nUi = static_cast<int>(nU);
        auto source   = [&](int i) {
            return i < nUi ? detail::LoadAt<T>(m_partial, i)
                             : detail::LoadAt<T>(m_recvBuf, i - nUi);
        };
        for (const int u : m_activeUnique)
        {
            T acc = source(m_foldSrc[m_foldPtr[u]]);
            for (int f = m_foldPtr[u] + 1; f < m_foldPtr[u + 1]; ++f)
            {
                acc = op(acc, source(m_foldSrc[f]));
            }
            sink(u, acc);
        }
    }

    /**
     * @brief Fold() for the one-slot-per-id case.
     *
     * Skips the partial buffer entirely: the send buffer is packed straight
     * from @p slots, and an id's own contribution is read from there too.
     * Visits fold sources in exactly the order the general path does, so
     * results are bitwise identical; only the data movement differs.
     *
     * Safe against a sink that writes back into @p slots (Gather()): the send
     * buffer is packed before any sink runs, and distinct ids own distinct
     * slots, so no id's own value can be overwritten before it is read.
     */
    template <typename T, typename Op, typename Sink>
    void FoldSingleSlot(const T *slots, Op &op, Sink &&sink,
                        double *exchangeSeconds)
    {
        const size_t nE = m_nbrUnique.size();

        m_sendBuf.resize(std::max<size_t>(nE * sizeof(T), 1));
        m_recvBuf.resize(std::max<size_t>(nE * sizeof(T), 1));
        for (size_t e = 0; e < nE; ++e)
        {
            detail::StoreAt<T>(m_sendBuf, e, slots[m_sendSlot[e]]);
        }

        RunExchange(sizeof(T), m_nbrPtr, m_nbrPtr, exchangeSeconds);

        // Two-sharer ids: no per-entry branch and no inner loop. The two
        // loops differ only in which operand comes first, which is what keeps
        // the canonical ascending-rank fold order.
        for (const K2Fold &k : m_k2OwnFirst)
        {
            sink(k.unique,
                 op(slots[k.slot], detail::LoadAt<T>(m_recvBuf, k.recv)));
        }
        for (const K2Fold &k : m_k2OwnLast)
        {
            sink(k.unique,
                 op(detail::LoadAt<T>(m_recvBuf, k.recv), slots[k.slot]));
        }

        const int nUi = static_cast<int>(m_uniqueIds.size());
        auto source   = [&](int i) {
            return i < nUi ? slots[m_ownSlot[i]]
                             : detail::LoadAt<T>(m_recvBuf, i - nUi);
        };
        for (const int u : m_foldGeneral)
        {
            const int f0 = m_foldPtr[u], f1 = m_foldPtr[u + 1];
            T acc = source(m_foldSrc[f0]);
            for (int f = f0 + 1; f < f1; ++f)
            {
                acc = op(acc, source(m_foldSrc[f]));
            }
            sink(u, acc);
        }
    }

    /// Discovery transport, and the eTransport backend's exchange.
    std::shared_ptr<ITransport> m_transport;
    /// This rank's index.
    int m_rank = 0;
    /// Total rank count.
    int m_nproc = 0;
    /// Backend chosen at construction.
    ExchangeBackend m_backend;
    /// Number of slots, ignored ones included.
    size_t m_nSlots = 0;

    /// Distinct ids on this rank, ascending.
    std::vector<int64_t> m_uniqueIds;
    /// Slot -> unique index, -1 if ignored.
    std::vector<int> m_slotToUnique;
    /// CSR unique id -> its slots, ascending slot order.
    std::vector<int> m_uSlotPtr, m_uSlots;
    /// CSR unique id -> its sharer ranks, ascending, self included.
    std::vector<int> m_shrPtr, m_shrRanks;

    /// Neighbour ranks, ascending.
    std::vector<int> m_nbrs;
    /// CSR neighbour -> unique ids shared with it, ascending by id. Entry
    /// positions double as send and receive buffer element offsets.
    std::vector<int> m_nbrPtr, m_nbrUnique;
    /// CSR unique id -> fold sources in ascending sharer-rank order: own
    /// partial (index < NumUnique()) or receive entry (NumUnique() + pos).
    std::vector<int> m_foldPtr, m_foldSrc;
    /// Unique ids that are shared or held in several slots -- the only ones
    /// Fold() visits -- and the rest.
    std::vector<int> m_activeUnique, m_inactiveUnique;
    /// One past the highest slot of any active unique id.
    size_t m_activeSlotBound = 0;
    /// Whether every id owns exactly one slot, enabling FoldSingleSlot().
    bool m_allSingleSlot = false;
    /// Fast path only: each id's single slot, and the slot each link entry
    /// sends from. Empty otherwise.
    std::vector<int> m_ownSlot, m_sendSlot;

    /// One two-sharer id on the fast path: where this rank's value lives,
    /// where the other rank's landed, and which id to hand the result to.
    struct K2Fold
    {
        int unique;
        int slot;
        int recv;
    };
    /// Fast-path two-sharer ids, split by whether this rank sorts first, plus
    /// the active ids still needing the general fold. Empty off the fast path.
    std::vector<K2Fold> m_k2OwnFirst, m_k2OwnLast;
    std::vector<int> m_foldGeneral;

    /// Whether PreparePayloadExchange() has run.
    bool m_payloadReady = false;
    /// Payload send layout: CSR offsets per link, and the slots to pack.
    std::vector<int> m_paySendPtr, m_paySendSlots;
    /// Payload receive layout: CSR offsets per link.
    std::vector<int> m_payRecvPtr;
    /// CSR unique id -> ExchangePayloads() entries, their sources and ranks.
    std::vector<int> m_payPtr, m_paySrc, m_payRank;

    /// The fixed-pattern exchange every execute call replays.
    std::unique_ptr<detail::IFixedExchange> m_exchange;
    /// Scratch reused across calls, so execute calls do not allocate once
    /// warm.
    std::vector<std::byte> m_partial, m_sendBuf, m_recvBuf;
    /// Per-link byte counts and displacements for the current call.
    std::vector<int> m_sc, m_sd, m_rc, m_rd;
};

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIBUTILITIES_COMMUNICATION_SHAREDIDPLAN_HPP
