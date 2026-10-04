///////////////////////////////////////////////////////////////////////////////
//
// File: SharedPayloadResolver.hpp
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
// Description: Id-keyed exchange of every sharer's raw payload.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file SharedPayloadResolver.hpp
 * @brief Id-keyed exchange of every sharer's raw payload for a shared entity.
 *
 * @details
 * The top layer of the shared-id module: discovery is delegated to a composed
 * #ConnectivityResolver, then each rank ships its payload straight to the
 * sharers that discovery found, peer to peer. Use it where there is no
 * sensible way to fold contributions into one value -- periodic vertex
 * coordinates, say, where every sharer's own copy is wanted rather than an
 * "agreed" coordinate. Where a fold *is* what you want, use #SharedIdPlan
 * instead.
 *
 * This header also includes the rest of the module, so including it alone is
 * enough to reach #ITransport, #ConnectivityResolver and #SharedIdPlan.
 *
 * See Transport.hpp for the module overview.
 */

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_SHAREDPAYLOADRESOLVER_HPP
#define NEKTAR_LIBUTILITIES_COMMUNICATION_SHAREDPAYLOADRESOLVER_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/Communication/SharedIdPlan.hpp>

namespace Nektar::LibUtilities
{

/**
 * @brief Optional breakdown of wall-clock time spent in
 * SharedPayloadResolver::Resolve()'s two kinds of round trip, for isolating
 * the fixed per-Exchange() synchronisation cost from payload-volume cost.
 * Not templated on PayloadT -- it only measures time, so one type serves
 * every SharedPayloadResolver<T> instantiation.
 *
 * Pass a pointer to Resolve() to have it filled in; leave null (the
 * default) for zero measurement overhead. Both fields are seconds, timed
 * strictly around Exchange() call(s) -- local packing/unpacking and the
 * id-grouping sort in Resolve() are excluded from both. `discoverySeconds`
 * is the sum of the composed ConnectivityResolver's own
 * ConnectivityResolverTimings::requestExchangeSeconds and
 * ::replyExchangeSeconds (see that struct for the finer per-round split),
 * so -- unlike an earlier revision of this struct, which wrapped the whole
 * ConnectivityResolver::Resolve() call including its local gather/pack/
 * unpack work -- the two fields here are now genuinely both exchange-only
 * and directly comparable to each other.
 *
 * Placed here, before SharedPayloadResolver's own class-level `@class` doc
 * block, rather than between it and the class definition: an intervening
 * documented entity there previously left the class doc orphaned from its
 * class in practice, relying entirely on the explicit `@class` command to
 * reattach it -- fragile against a future edit that drops that command.
 */
struct SharedPayloadResolverTimings
{
    /// Wall time in the composed ConnectivityResolver's two Exchange()
    /// calls (ids out, sharer ranks back), summed. See
    /// ConnectivityResolverTimings for the individual request/reply split.
    double discoverySeconds = 0.0;

    /// Wall time in this class's own single direct payload Exchange() call
    /// (this rank's payloads addressed straight to the sharer ranks discovery
    /// found).
    double payloadExchangeSeconds = 0.0;
};

/**
 * @class SharedPayloadResolver
 * @brief Discovery-then-direct-exchange resolver: for every registered
 * (entity id, payload) pair, discovers every rank sharing that entity and
 * returns each of their raw payloads -- no reduction, and PayloadT never
 * transits a rendezvous rank.
 *
 * @tparam PayloadT Trivially-copyable value type shipped alongside each
 * entity id (e.g. a small POD struct of vertex coordinates). Must satisfy
 * `std::is_trivially_copyable_v`, since payloads are shipped as raw bytes
 * with no custom MPI datatype.
 *
 * @details
 * Use this instead of #SharedIdPlan when there is no sensible way to fold
 * multiple ranks' contributions into one value -- e.g. periodic vertex
 * coordinates, where every sharer's own copy is needed rather than an
 * "agreed" single coordinate.
 *
 * The algorithm proceeds in two phases:
 * -# **Discovery**, delegated entirely to a composed #ConnectivityResolver:
 *    every id this rank registered is sent, once each, to its rendezvous
 *    rank, which gathers the global set of ranks touching that id and ships
 *    the (small, payload-free) rank list back to every one of them. Same
 *    O(local) two round trips as before, but the rendezvous rank only ever
 *    sees ids and rank numbers -- never PayloadT.
 * -# **Payload exchange**: armed with the sharer set for every id, this
 *    rank builds one outbound message per sharer rank containing this
 *    rank's payload(s) for every id shared with them, and calls
 *    ITransport::Exchange() directly, addressed peer-to-peer at the real
 *    destination rank rather than hashed to a rendezvous host. Discovery
 *    gives every sharer of an id an identical view of who else shares it,
 *    so this step is inherently symmetric: if A sends to B because B is in
 *    A's sharer set for id X, B independently sends to A for the same
 *    reason -- no separate reply round trip is needed here, unlike the
 *    request/reply pair a rendezvous-mediated reduction needs.
 *
 * Net effect: PayloadT crosses the network exactly once per (contributor,
 * sharer) pair, on a direct hop, instead of twice via a third-party rank. This
 * can be important in cases where PayloadT is relatively large (typically 100s
 * of bytes).
 *
 * ### Persistent mode
 * When the same ids are exchanged repeatedly with new payloads, call Setup()
 * once after registering: discovery then happens once, and every Resolve()
 * is just the payload exchange over a #SharedIdPlan. Update payloads between
 * calls with SetPayload() or Payloads().
 *
 * ### Example (periodic vertex coordinates)
 * @code{.cpp}
 * struct Point3D { double x, y, z; };
 * SharedPayloadResolver<Point3D> res(transport);
 * res.Register(periodicVertexId, myCoord);
 * res.Resolve();
 * for (auto &[rank, coord] : res.GetSharedPayloads(periodicVertexId))
 *     if (rank != myRank)
 *         std::cout << "rank " << rank << " has " << coord.x << "\n";
 * @endcode
 */
template <typename PayloadT> class SharedPayloadResolver
{
    static_assert(std::is_trivially_copyable_v<PayloadT> &&
                      std::is_default_constructible_v<PayloadT>,
                  "PayloadT must be trivially copyable and "
                  "default-constructible (UnpackPod default-constructs a "
                  "scratch value internally before the memcpy). For "
                  "variable-length payloads, add a pack/unpack step.");

public:
    /**
     * @brief Construct a resolver bound to @p transport.
     * @param transport Exchange mechanism to use in Resolve(); typically an
     * AlltoallvTransport or CrystalRouterTransport. The same transport
     * drives both the composed ConnectivityResolver's discovery round trips
     * and this class's own direct payload exchange.
     */
    explicit SharedPayloadResolver(std::shared_ptr<ITransport> transport)
        : m_transport(transport), m_connResolver(std::move(transport)),
          m_rank(m_transport->Rank())
    {
    }

    /**
     * @brief Register an entity this rank touches, along with this rank's
     * own payload for it (e.g. its copy of a periodic vertex's
     * coordinates).
     * @param globalId Global entity id.
     * @param value This rank's payload contribution for @p globalId.
     *
     * Unlike the rendezvous-shunted implementation this replaces, nothing
     * is shipped anywhere at registration time: the payload is simply
     * appended to a local, per-id list, and only the bare id (never the
     * payload) is later handed to the composed ConnectivityResolver inside
     * Resolve(). May be called more than once for the same id; each call is
     * a separate contribution and will appear as a separate entry in
     * GetSharedPayloads().
     */
    void Register(int64_t globalId, const PayloadT &value)
    {
        // A flat push_back, not a hash-map insert keyed by globalId. The
        // earlier revision of this method inserted into an
        // unordered_map<int64_t, vector<PayloadT>> keyed by the raw entity
        // id -- for callers with O(local mesh entities) unique ids (e.g. one
        // key per facet), that map can be orders of magnitude larger than
        // the old rendezvous-bucketed store it replaced (which had at most
        // `nproc` keys), and paid for it in both scattered-hash insert cost
        // here *and* two more cache-hostile full traversals inside
        // Resolve(). Grouping by id is deferred to Resolve(), where it's
        // done with a sort over this flat array instead.
        ASSERTL0(!m_plan, "SharedPayloadResolver::Register: ids are fixed "
                          "once Setup() has run; call Clear() to start over.");
        m_localRegs.emplace_back(globalId, value);
    }

    /**
     * @brief Bulk registration of many ids sharing one payload.
     * @tparam IdContainer Any iterable of integer ids (e.g.
     * `std::vector<int>`).
     * @param ids Ids to register, each contributing @p commonValue.
     * @param commonValue Value registered against every id in @p ids.
     */
    template <typename IdContainer>
    void RegisterKeys(const IdContainer &ids, const PayloadT &commonValue)
    {
        for (auto id : ids)
        {
            Register(static_cast<int64_t>(id), commonValue);
        }
    }

    /**
     * @brief Hint that @p n more Register() calls are coming, to reduce
     * reallocations on large local meshes.
     * @param n Number of additional (id, value) pairs about to be
     * registered.
     *
     * Unlike ConnectivityResolver::Reserve(), this is a real, exact
     * pre-size (not a best-effort hint bounded by
     * `nproc`): m_localRegs is a flat array, not a map keyed by
     * destination, so there's no hashing to worry about -- it's a plain
     * `std::vector::reserve(n)`.
     */
    void Reserve(size_t n)
    {
        m_localRegs.reserve(n);
    }

    /**
     * @brief Discover, for every registered id, the full set of (rank,
     * payload) pairs from every sharer.
     *
     * After this call, GetSharedPayloads() and GetSharers() are valid for
     * every id this rank registered. This runs the composed
     * ConnectivityResolver's two payload-free discovery round trips, then
     * one further Exchange() to ship payloads directly to the sharer ranks
     * discovery found -- three Exchange() calls in total, but only the
     * third ever carries a PayloadT, and it does so contributor-to-sharer
     * directly rather than via a rendezvous host.
     *
     * Does not consume registrations: calling Resolve() again without an
     * intervening Register() re-resolves the same registrations and is safe
     * (contrast with ConnectivityResolver::Resolve(), which consumes
     * its own). Before Setup() each
     * call repeats discovery; after Setup() (persistent mode) a call is just
     * the payload exchange of the current payloads (see SetPayload()) over
     * the plan built there, and `timings->discoverySeconds` is zero.
     *
     * @param timings Optional out-parameter; if non-null, filled in with a
     * discovery-vs-payload-round wall-time split (see
     * SharedPayloadResolverTimings). Leave null for zero measurement
     * overhead -- this is a diagnostic hook, not something to leave wired
     * up in a hot production call site once you've got the number you need.
     */
    void Resolve(SharedPayloadResolverTimings *timings = nullptr)
    {
        if (m_plan)
        {
            ResolvePersistent(timings);
            return;
        }

        m_sharedPayloads.clear();
        // Every key m_sharedPayloads ever gets populated with comes from
        // this rank's own registrations (see the symmetry argument in the
        // phase-2 comment below), so m_localRegs.size() is an exact upper
        // bound on the final key count (exact, not just an estimate, unless
        // some id was registered more than once locally).
        m_sharedPayloads.reserve(m_localRegs.size());

        // Group local registrations by id via a sort over compact (id,
        // index) pairs, not an index array whose comparator needs a
        // random-access load into (potentially large) m_localRegs on every
        // comparison. Keeping the sort keys themselves contiguous is
        // meaningfully cheaper at the scale this class targets, and -- like
        // the index-array approach it replaces -- still never moves a
        // PayloadT: only the compact (id, index) pairs get reordered here,
        // and m_localRegs[idx].second is looked up by index afterwards.
        // uint32_t is enough for any realistic per-rank local entity count.
        std::vector<std::pair<int64_t, uint32_t>> order;
        order.reserve(m_localRegs.size());
        for (size_t i = 0; i < m_localRegs.size(); ++i)
        {
            order.emplace_back(m_localRegs[i].first, static_cast<uint32_t>(i));
        }
        std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });

        // Phase 1: discover, per id, *who* shares it -- ids only, no payload
        // ever reaches the rendezvous rank. One
        // ConnectivityResolver::Register() per unique id (not per Register()
        // call on *this* class -- each contiguous run in `order` collapses to a
        // single call), so the discovered sharer set is a plain rank set even
        // if we hold several local contributions for the same id.
        m_connResolver.Clear();
        for (auto it = order.begin(); it != order.end();)
        {
            const int64_t id = it->first;
            m_connResolver.Register(id);
            it = std::find_if(it, order.end(),
                              [&](const auto &kv) { return kv.first != id; });
        }

        // Forward a ConnectivityResolverTimings through to the composed
        // resolver so discoverySeconds is genuinely exchange-only (summed
        // from its two Exchange()-only timings) rather than wrapping the
        // whole ConnectivityResolver::Resolve() call, local gather/pack/
        // unpack work included -- see SharedPayloadResolverTimings' doc.
        ConnectivityResolverTimings connTimings;
        m_connResolver.Resolve(timings ? &connTimings : nullptr);
        if (timings)
        {
            timings->discoverySeconds = connTimings.requestExchangeSeconds +
                                        connTimings.replyExchangeSeconds;
        }

        // Phase 2: every id's sharer set is now known -- and, by the symmetry
        // of rendezvous discovery, known identically by every sharer -- so ship
        // this rank's payloads straight to each one, addressed by real rank
        // rather than rendezvous hash. Walked as contiguous runs over the same
        // sorted `order`, so this is a second linear scan, not a second
        // hash-map traversal.
        std::unordered_map<int, std::vector<std::byte>> outBuckets;
        for (auto it = order.begin(); it != order.end();)
        {
            const int64_t id = it->first;
            auto runEnd = std::find_if(it, order.end(), [&](const auto &kv) {
                return kv.first != id;
            });

            for (int r : m_connResolver.SharersOf(id))
            {
                if (r == m_rank)
                {
                    continue; // self contribution added directly below
                }
                std::vector<std::byte> &buf = outBuckets[r];
                detail::PackPod<int64_t>(buf, id);
                detail::PackPod<int32_t>(buf,
                                         static_cast<int32_t>(runEnd - it));
                for (auto rit = it; rit != runEnd; ++rit)
                {
                    detail::PackPod<PayloadT>(buf,
                                              m_localRegs[rit->second].second);
                }
            }
            it = runEnd;
        }

        std::vector<RoutedMessage> outbound =
            detail::DrainBucketsToMessages(outBuckets, m_rank);

        std::vector<RoutedMessage> inbound;
        const auto exchangeStart = std::chrono::steady_clock::now();
        m_transport->Exchange(outbound, inbound);
        if (timings)
        {
            timings->payloadExchangeSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              exchangeStart)
                    .count();
        }

        // Store what every other sharer sent us
        for (const auto &m : inbound)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id = detail::UnpackPod<int64_t>(cur);
                const int32_t n  = detail::UnpackPod<int32_t>(cur);

                std::vector<std::pair<int, PayloadT>> &entries =
                    m_sharedPayloads[id];
                entries.reserve(entries.size() + n);
                for (int32_t i = 0; i < n; ++i)
                {
                    entries.emplace_back(m.src,
                                         detail::UnpackPod<PayloadT>(cur));
                }
            }
            ASSERTL1(cur == end,
                     "SharedPayloadResolver::Resolve: malformed payload "
                     "buffer (record lengths didn't sum to the message "
                     "size)");
        }

        // Add this rank's own contributions; these were never shipped over the
        // wire (see the `r == m_rank` skip in phase 2). Plain linear scan over
        // the flat store, no grouping needed here.
        for (const auto &reg : m_localRegs)
        {
            m_sharedPayloads[reg.first].emplace_back(m_rank, reg.second);
        }
    }

    /**
     * @brief Test whether Resolve() returned a result for @p globalId.
     * @param globalId Entity id to query.
     * @return `true` iff @p globalId was registered by this rank prior to
     * the most recent Resolve() call.
     */
    bool Knows(int64_t globalId) const
    {
        return m_sharedPayloads.find(globalId) != m_sharedPayloads.end();
    }

    /**
     * @brief Switch to persistent mode: freeze the ids registered so far
     * into a #SharedIdPlan, so every later Resolve() is just the payload
     * exchange. Collective.
     *
     * Discovery, and the one-off multiplicity exchange the payload layout
     * needs, run here, once. The Register() call order becomes the index
     * SetPayload() takes, and Register() is refused from here on (Clear()
     * starts over). GetSharedPayloads() and friends work unchanged; each
     * id's list is ordered by ascending rank, then by registration order
     * within a rank.
     *
     * @param backend Exchange replayed by every subsequent Resolve().
     * @param timings Optional discovery-exchange timing breakdown.
     */
    void Setup(ExchangeBackend backend = ExchangeBackend::ePairwise,
               ConnectivityResolverTimings *timings = nullptr)
    {
        ASSERTL0(!m_plan, "SharedPayloadResolver::Setup: already set up; "
                          "call Clear() to start over.");

        std::vector<int64_t> ids(m_localRegs.size());
        m_regPayloads.resize(m_localRegs.size());
        for (size_t i = 0; i < m_localRegs.size(); ++i)
        {
            ids[i]           = m_localRegs[i].first;
            m_regPayloads[i] = m_localRegs[i].second;
        }
        // m_regPayloads is the plan's slot array from here on; drop the
        // paired copy rather than hold every payload twice.
        std::vector<std::pair<int64_t, PayloadT>>().swap(m_localRegs);

        m_plan =
            std::make_unique<SharedIdPlan>(m_transport, ids, backend, timings);
        m_plan->PreparePayloadExchange();

        // Lay out m_sharedPayloads once, ranks filled in, and record where
        // each exchanged entry lands. Nothing is inserted or resized after
        // this, so those addresses stay valid and Resolve() only overwrites
        // values.
        const std::vector<int64_t> &uids = m_plan->UniqueIds();
        m_sharedPayloads.clear();
        m_sharedPayloads.reserve(uids.size());
        m_payloadDest.assign(
            uids.empty() ? 0 : m_plan->PayloadEnd(uids.size() - 1), nullptr);
        for (size_t u = 0; u < uids.size(); ++u)
        {
            const size_t b = m_plan->PayloadBegin(u);
            const size_t e = m_plan->PayloadEnd(u);
            std::vector<std::pair<int, PayloadT>> &entries =
                m_sharedPayloads[uids[u]];
            entries.resize(e - b);
            for (size_t k = 0; k < e - b; ++k)
            {
                entries[k].first     = m_plan->PayloadRank(b + k);
                m_payloadDest[b + k] = &entries[k].second;
            }
        }
    }

    /// @return Whether Setup() has run (persistent mode).
    bool IsSetup() const
    {
        return m_plan != nullptr;
    }

    /// @return Number of registrations held: the index range of
    /// SetPayload().
    size_t NumRegistrations() const
    {
        return m_plan ? m_regPayloads.size() : m_localRegs.size();
    }

    /**
     * @brief Replace the payload of registration @p regIndex (Register()
     * call order) for the next Resolve() -- the persistent-mode way to feed
     * new payloads over fixed ids.
     */
    void SetPayload(size_t regIndex, const PayloadT &value)
    {
        ASSERTL1(regIndex < NumRegistrations(),
                 "SharedPayloadResolver::SetPayload: index out of range");
        if (m_plan)
        {
            m_regPayloads[regIndex] = value;
        }
        else
        {
            m_localRegs[regIndex].second = value;
        }
    }

    /// @return Mutable view of every registration's payload in Register()
    /// order, for bulk updates between Resolve() calls. Persistent mode only.
    PayloadT *Payloads()
    {
        static_assert(!std::is_same_v<PayloadT, bool>,
                      "SharedPayloadResolver<bool>::Payloads() is unavailable "
                      "(bool payloads are stored as char); use SetPayload() "
                      "instead.");
        ASSERTL0(m_plan, "SharedPayloadResolver::Payloads: call Setup() "
                         "first.");
        return m_regPayloads.data();
    }

    /// @return The plan built by Setup(); persistent mode only.
    const SharedIdPlan &Plan() const
    {
        ASSERTL0(m_plan, "SharedPayloadResolver::Plan: call Setup() first.");
        return *m_plan;
    }

    /**
     * @brief Single-lookup alternative to `Knows(id)` then
     * `GetSharedPayloads(id)`.
     * @param globalId Entity id to query.
     * @return Pointer to the (rank, payload) list if @p globalId was
     * registered by this rank prior to the most recent Resolve() call, else
     * `nullptr`.
     */
    const std::vector<std::pair<int, PayloadT>> *FindSharedPayloads(
        int64_t globalId) const
    {
        auto it = m_sharedPayloads.find(globalId);
        return it == m_sharedPayloads.end() ? nullptr : &it->second;
    }

    /**
     * @brief Retrieve every sharer's raw payload for a resolved entity.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return (rank, payload) pairs from every rank that registered this
     * entity, including this rank. Order is unspecified; contains a
     * duplicate entry if a rank registered the id more than once.
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     */
    const std::vector<std::pair<int, PayloadT>> &GetSharedPayloads(
        int64_t globalId) const
    {
        return m_sharedPayloads.at(globalId);
    }

    /**
     * @brief Direct read-only access to every resolved id's (rank, payload)
     * list, for callers that want to iterate everything this rank knows
     * about rather than querying id-by-id.
     * @return The full map from resolved entity id (i.e. every id this rank
     * registered, regardless of whether any other rank shares it) to its
     * (rank, payload) list. This is the underlying storage, not a filtered
     * view: unlike a hypothetical "only genuinely shared" accessor, ids
     * with exactly one contributor (this rank) are present too, each with
     * a single-entry list. Iteration order is unspecified.
     */
    const std::unordered_map<int64_t, std::vector<std::pair<int, PayloadT>>> &
    AllSharedPayloads() const
    {
        return m_sharedPayloads;
    }

    /**
     * @brief Convenience accessor returning just the sharer ranks, for
     * callers that only need connectivity from a resolver they are already
     * using for payload data.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return Ranks that touch this entity, one per entry in
     * GetSharedPayloads() (so may repeat under the same conditions).
     * Returned by value, unlike ConnectivityResolver::SharersOf(), since
     * no ranks-only list is stored separately.
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     */
    std::vector<int> GetSharers(int64_t globalId) const
    {
        const auto &entries = m_sharedPayloads.at(globalId);
        std::vector<int> ranks;
        ranks.reserve(entries.size());
        for (const auto &entry : entries)
        {
            ranks.push_back(entry.first);
        }
        return ranks;
    }

    /**
     * @brief Build a persistent distributed-graph communicator over the
     * discovered topological neighbours (self excluded).
     * @param base Base communicator to derive the graph communicator from.
     * @return A neighbourhood communicator; use this once after Resolve(),
     * then drive downstream halo exchanges with `MPI_Neighbor_alltoallv`
     * etc. rather than rebuilding it per exchange.
     */
    CommSharedPtr BuildNeighbourComm(CommSharedPtr base) const
    {
        // The plan (persistent mode) or the composed ConnectivityResolver
        // (one-shot mode) already holds exactly this neighbour set, so
        // delegate rather than re-deriving it from m_sharedPayloads.
        return m_plan ? m_plan->BuildNeighbourComm(base)
                      : m_connResolver.BuildNeighbourComm(base);
    }

    /// @brief Clear registrations and results, so the resolver can be
    /// reused for a fresh Register()/Resolve() cycle.
    void Clear()
    {
        m_localRegs.clear();
        m_connResolver.Clear();
        m_sharedPayloads.clear();
        m_plan.reset();
        m_regPayloads.clear();
        m_payloadDest.clear();
        m_payloadScratch.clear();
    }

private:
    /// Storage for values handed to the plan: PayloadT itself, except that
    /// bool is held as char, since std::vector<bool> packs bits and has no
    /// data() to pass on.
    using Slot =
        std::conditional_t<std::is_same_v<PayloadT, bool>, char, PayloadT>;

    /// Persistent-mode Resolve(): one payload exchange over the plan, values
    /// written straight into the layout Setup() built.
    void ResolvePersistent(SharedPayloadResolverTimings *timings)
    {
        double seconds = 0.0;
        m_plan->ExchangePayloads(m_regPayloads.data(), m_payloadScratch,
                                 timings ? &seconds : nullptr);
        for (size_t e = 0; e < m_payloadScratch.size(); ++e)
        {
            *m_payloadDest[e] = m_payloadScratch[e];
        }
        if (timings)
        {
            timings->discoverySeconds       = 0.0;
            timings->payloadExchangeSeconds = seconds;
        }
    }

    /// Exchange mechanism used by Resolve().
    std::shared_ptr<ITransport> m_transport;

    /// Discovery-only helper: figures out, per registered id, the sharer
    /// rank set (ids and ranks cross the wire, never PayloadT). Driven
    /// entirely from within this class's Resolve()/Clear() -- its own
    /// Register()/Resolve()/Clear() cycle is not exposed to callers.
    ConnectivityResolver m_connResolver;

    /// This rank's index, cached from m_transport->Rank().
    int m_rank = 0;

    /// This rank's own registrations, in Register()-call order (i.e.
    /// unsorted, possibly with repeated ids). Deliberately a flat array
    /// rather than a map keyed by id: Register() is called once per local
    /// mesh entity touch, which can be orders of magnitude more numerous
    /// than `nproc`, and a hash map at that scale costs real time both to
    /// insert into and, worse, to iterate (scattered-bucket traversal,
    /// visited twice per Resolve() call). Resolve() groups this by id with
    /// a sort instead. Sized O(local registrations).
    std::vector<std::pair<int64_t, PayloadT>> m_localRegs;

    /// (rank, payload) pairs per resolved entity id, populated by
    /// Resolve(). Contains only entities this rank registered -- O(local),
    /// not O(N_global).
    std::unordered_map<int64_t, std::vector<std::pair<int, PayloadT>>>
        m_sharedPayloads;

    /// Persistent plan built by Setup(); null in one-shot mode.
    std::unique_ptr<SharedIdPlan> m_plan;
    /// Persistent mode: payloads in Register() order -- the plan's slot
    /// array, taking over from m_localRegs, which Setup() releases.
    std::vector<Slot> m_regPayloads;
    /// Persistent mode: where each exchanged entry lands in
    /// m_sharedPayloads.
    std::vector<PayloadT *> m_payloadDest;
    /// Persistent mode: the plan's output for the latest Resolve().
    std::vector<Slot> m_payloadScratch;
};

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIBUTILITIES_COMMUNICATION_ENTITYRESOLVER_HPP
