///////////////////////////////////////////////////////////////////////////////
//
// File: ConnectivityResolver.hpp
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
// Description: Rendezvous discovery of shared-id sharer sets.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file ConnectivityResolver.hpp
 * @brief Rendezvous discovery of which ranks share each global id.
 *
 * @details
 * The payload-free discovery pass: every id goes to a rendezvous rank chosen
 * by hashing it, which sees every contribution for that id and ships the
 * resulting sharer list back to each contributor. Two exchanges, and memory
 * that is O(local ids) rather than O(P) or O(N_global).
 *
 * This is the discovery step underneath #SharedIdPlan, and is also usable on
 * its own where only topology is wanted (AssemblyCommDG does exactly that).
 *
 * See Transport.hpp for the module overview.
 */

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_CONNECTIVITYRESOLVER_HPP
#define NEKTAR_LIBUTILITIES_COMMUNICATION_CONNECTIVITYRESOLVER_HPP

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
#include <LibUtilities/Communication/Transport.hpp>

namespace Nektar::LibUtilities
{

/**
 * @brief Optional breakdown of wall-clock time spent in
 * ConnectivityResolver::Resolve()'s two Exchange() calls, for isolating
 * network/collective time from the local packing, gathering, and unpacking
 * work around them. Pass a pointer to Resolve() to have it filled in; leave
 * null (the default) for zero measurement overhead.
 *
 * Placed here, before ConnectivityResolver's own class-level `@class` doc
 * block, rather than between it and the class definition: an intervening
 * documented entity there previously left the class doc orphaned from its
 * class in practice, relying entirely on the explicit `@class` command to
 * reattach it -- fragile against a future edit that drops that command.
 * SharedPayloadResolverTimings is ordered the same way below, for the same
 * reason.
 */
struct ConnectivityResolverTimings
{
    /// Wall time in the request-phase Exchange() (ids -> rendezvous ranks).
    double requestExchangeSeconds = 0.0;
    /// Wall time in the reply-phase Exchange() (sharer lists -> contributors).
    double replyExchangeSeconds = 0.0;
};

/**
 * @class ConnectivityResolver
 * @brief Rendezvous discovery on its own, for the common case where you
 * only need to find *which ranks share each entity* and not a reduced
 * value.
 *
 * @details
 * It ships only the entity id on the wire (no payload, no reduction op), so
 * both the request and reply are small, and the API is just
 * RegisterKeys()/Resolve()/SharersOf(). This is the discovery step
 * #SharedIdPlan and #SharedPayloadResolver are both built on -- see the
 * @ref Transport.hpp "file-level documentation" for the rendezvous
 * algorithm and the O(local) memory argument.
 *
 * ### Example (partition-interface discovery)
 * @code{.cpp}
 * ConnectivityResolver res(transport);
 * res.RegisterKeys(uniqueEdgeIds);          // a vector<int> you already have
 * res.Resolve();
 * for (int e : uniqueEdgeIds)
 *     for (int r : res.GetSharers(e))
 *         if (r != myRank) sharedWith[r].push_back(e);
 * @endcode
 */
class ConnectivityResolver
{
public:
    /**
     * @brief Construct a resolver bound to @p transport.
     * @param transport Exchange mechanism to use in Resolve(); typically an
     * AlltoallvTransport or CrystalRouterTransport.
     */
    explicit ConnectivityResolver(std::shared_ptr<ITransport> transport)
        : m_transport(std::move(transport)), m_rank(m_transport->Rank()),
          m_nproc(m_transport->Size())
    {
    }

    /**
     * @brief Register an entity id this rank touches.
     * @param globalId Global entity id (vertex/edge/face, etc.).
     *
     * Packed straight into the per-rendezvous send buffer; no payload is
     * stored or shipped.
     */
    void Register(int64_t globalId)
    {
        std::vector<std::byte> &buf =
            m_reqBuckets[detail::RendezvousRank(globalId, m_nproc)];
        detail::PackPod<int64_t>(buf, globalId);
    }

    /**
     * @brief Bulk registration from any iterable of ids.
     * @tparam IdContainer Any iterable of integer ids (e.g.
     * `std::vector<int>`).
     * @param ids Ids to register.
     */
    template <typename IdContainer> void RegisterKeys(const IdContainer &ids)
    {
        for (auto id : ids)
        {
            Register(static_cast<int64_t>(id));
        }
    }

    /**
     * @brief Best-effort hint that @p n more Register() calls are coming.
     * @param n Number of additional ids about to be registered.
     * Only the outer per-rendezvous-rank bucket map (at most `nproc` keys)
     * can usefully be pre-sized; the byte buffer within each bucket can't
     * be, since which bucket an id lands in isn't known until Register()
     * hashes it.
     */
    void Reserve(size_t n)
    {
        m_reqBuckets.reserve(std::min(n, static_cast<size_t>(m_nproc)));
    }

    /**
     * @brief Discover, for every registered id, the full set of ranks that
     * touch it.
     *
     * After this call, GetSharers() is valid for every id this rank
     * registered. Two Exchange() round trips are performed: one to gather
     * registrations at each entity's rendezvous rank, one to ship the
     * sharer list back to every contributor.
     *
     * Consumes registrations: this drains m_reqBuckets, so a second
     * Resolve() call without an intervening Register() will find nothing to
     * do and return empty results. Register() again before every Resolve()
     * if you need to run it more than once (contrast with
     * SharedPayloadResolver::Resolve(), which keeps its registrations until
     * Clear() and so is safe to call repeatedly).
     *
     * @param timings Optional out-parameter; if non-null, filled in with a
     * request-vs-reply-round wall-time split (see
     * ConnectivityResolverTimings), timed strictly around the two
     * Exchange() calls with the local gather/pack/unpack work excluded from
     * both -- unlike SharedPayloadResolverTimings::discoverySeconds prior to
     * this being added, which wrapped this entire function including that
     * local work. Leave null for zero measurement overhead.
     */
    void Resolve(ConnectivityResolverTimings *timings = nullptr)
    {
        m_ids.clear();
        m_shrOff.clear();
        m_shrCnt.clear();
        m_shrFlat.clear();
        m_sharers.clear();
        m_mapBuilt = false;

        // Every id this rank hears back about is one it registered, so the
        // ids sitting in m_reqBuckets bound the reply count -- counted here,
        // before DrainBucketsToMessages() empties them below.
        size_t localEntries = 0;
        for (const auto &kv : m_reqBuckets)
        {
            localEntries += kv.second.size() / sizeof(int64_t);
        }

        // Request phase: ship ids to their rendezvous ranks
        std::vector<RoutedMessage> outbound =
            detail::DrainBucketsToMessages(m_reqBuckets, m_rank);

        std::vector<RoutedMessage> inbound;
        {
            const auto t0 = std::chrono::steady_clock::now();
            m_transport->Exchange(outbound, inbound);
            if (timings)
            {
                timings->requestExchangeSeconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
            }
        }

        // At the rendezvous, group by id -- by sorting (id, source) pairs in
        // one flat array, not by hashing each id into a map of per-id
        // vectors. Both cost O(n), but at the ~200k ids a rank hosts on a
        // large mesh the map version spends most of its time on scattered
        // hashing and one heap allocation per id, which measured as the
        // dominant cost of the whole plan build. Sorting also leaves each
        // id's sources ascending and adjacent, so the dedupe below is a
        // comparison against the previous entry rather than a sort per id.
        size_t totalEntries = 0;
        for (const auto &m : inbound)
        {
            totalEntries += m.bytes.size() / sizeof(int64_t);
        }

        struct Contribution
        {
            int64_t id;
            int32_t src;
        };
        std::vector<Contribution> recv;
        recv.reserve(totalEntries);

        for (const auto &m : inbound)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                recv.push_back({detail::UnpackPod<int64_t>(cur),
                                static_cast<int32_t>(m.src)});
            }
            ASSERTL1(cur == end,
                     "ConnectivityResolver::Resolve: malformed request "
                     "buffer (record lengths didn't sum to the message "
                     "size)");
        }

        std::sort(recv.begin(), recv.end(),
                  [](const Contribution &a, const Contribution &b) {
                      return a.id != b.id ? a.id < b.id : a.src < b.src;
                  });

        // Reply phase: send (id, sharers) back to each sharer. A rank that
        // registered the same id twice appears once: rank identity is all a
        // sharer list conveys, and deduping keeps the reply smaller and
        // spares every caller doing it defensively.
        std::unordered_map<int, std::vector<std::byte>> repBuckets;
        std::vector<std::byte> rec;
        std::vector<int> shs;
        for (size_t i = 0; i < recv.size();)
        {
            size_t j = i;
            shs.clear();
            while (j < recv.size() && recv[j].id == recv[i].id)
            {
                if (shs.empty() || shs.back() != recv[j].src)
                {
                    shs.push_back(recv[j].src);
                }
                ++j;
            }

            rec.clear();
            detail::PackPod<int64_t>(rec, recv[i].id);
            detail::PackPod<int32_t>(rec, static_cast<int32_t>(shs.size()));
            for (int other : shs)
            {
                detail::PackPod<int32_t>(rec, other);
            }
            for (int other : shs)
            {
                std::vector<std::byte> &buf = repBuckets[other];
                buf.insert(buf.end(), rec.begin(), rec.end());
            }

            i = j;
        }

        std::vector<RoutedMessage> replyOut =
            detail::DrainBucketsToMessages(repBuckets, m_rank);

        std::vector<RoutedMessage> replyIn;
        {
            const auto t0 = std::chrono::steady_clock::now();
            m_transport->Exchange(replyOut, replyIn);
            if (timings)
            {
                timings->replyExchangeSeconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
            }
        }

        // Store the answers as flat arrays indexed by ascending id: the
        // ranks land in one contiguous block and each id keeps an offset and
        // a count into it, so there is again no per-id allocation. The
        // id-keyed map the public accessors hand out is built from this only
        // if something asks for it (see EnsureMap()).
        struct Entry
        {
            int64_t id;
            uint32_t off;
            uint32_t n;
        };
        std::vector<Entry> entries;
        entries.reserve(localEntries);
        m_shrFlat.reserve(localEntries * 2);

        for (const auto &m : replyIn)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id    = detail::UnpackPod<int64_t>(cur);
                const int32_t nShar = detail::UnpackPod<int32_t>(cur);
                entries.push_back({id, static_cast<uint32_t>(m_shrFlat.size()),
                                   static_cast<uint32_t>(nShar)});
                for (int32_t i = 0; i < nShar; ++i)
                {
                    m_shrFlat.push_back(detail::UnpackPod<int32_t>(cur));
                }
            }
            ASSERTL1(cur == end,
                     "ConnectivityResolver::Resolve: malformed reply buffer "
                     "(record lengths didn't sum to the message size)");
        }

        std::sort(entries.begin(), entries.end(),
                  [](const Entry &a, const Entry &b) { return a.id < b.id; });
        // A locally-repeated Register() produces one reply per registration;
        // they are identical, so keep the first.
        entries.erase(std::unique(entries.begin(), entries.end(),
                                  [](const Entry &a, const Entry &b) {
                                      return a.id == b.id;
                                  }),
                      entries.end());

        m_ids.resize(entries.size());
        m_shrOff.resize(entries.size());
        m_shrCnt.resize(entries.size());
        for (size_t i = 0; i < entries.size(); ++i)
        {
            m_ids[i]    = entries[i].id;
            m_shrOff[i] = entries[i].off;
            m_shrCnt[i] = entries[i].n;
        }
    }

    /**
     * @brief A resolved id's sharer ranks, as a range over the resolver's own
     * storage -- ascending, deduped, and including this rank. Valid until the
     * next Resolve()/Clear().
     */
    struct SharerView
    {
        const int *first = nullptr;
        const int *last  = nullptr;

        const int *begin() const
        {
            return first;
        }
        const int *end() const
        {
            return last;
        }
        size_t size() const
        {
            return static_cast<size_t>(last - first);
        }
        bool empty() const
        {
            return first == last;
        }
        int front() const
        {
            return *first;
        }
    };

    /**
     * @brief Index of @p globalId among the resolved ids, or -1 if this rank
     * did not register it. Binary search over ascending ids.
     */
    int FindIndex(int64_t globalId) const
    {
        auto it = std::lower_bound(m_ids.begin(), m_ids.end(), globalId);
        return (it != m_ids.end() && *it == globalId)
                   ? static_cast<int>(it - m_ids.begin())
                   : -1;
    }

    /// @return Number of ids resolved for this rank, ascending by id.
    size_t NumResolved() const
    {
        return m_ids.size();
    }

    /// @return The @p i th resolved id, ascending.
    int64_t ResolvedId(size_t i) const
    {
        return m_ids[i];
    }

    /// @return Sharers of the @p i th resolved id, without a lookup.
    SharerView SharersAt(size_t i) const
    {
        const int *base = m_shrFlat.data() + m_shrOff[i];
        return SharerView{base, base + m_shrCnt[i]};
    }

    /**
     * @brief Sharers of @p globalId, or an empty view if this rank did not
     * register it. The allocation-free alternative to GetSharers().
     */
    SharerView SharersOf(int64_t globalId) const
    {
        const int i = FindIndex(globalId);
        return i < 0 ? SharerView{} : SharersAt(static_cast<size_t>(i));
    }

    /**
     * @brief Test whether Resolve() returned a result for @p globalId.
     * @param globalId Entity id to query.
     * @return `true` iff @p globalId was registered by this rank prior to
     * the most recent Resolve() call.
     */
    bool Knows(int64_t globalId) const
    {
        return FindIndex(globalId) >= 0;
    }

    /**
     * @brief Single-lookup alternative to `Knows(id)` then `GetSharers(id)`.
     * @param globalId Entity id to query.
     * @return Pointer to the sharer list if @p globalId was registered by
     * this rank prior to the most recent Resolve() call, else `nullptr`.
     *
     * Materialises the id-keyed map on first use; SharersOf() answers the
     * same question straight from the flat storage.
     */
    const std::vector<int> *FindSharers(int64_t globalId) const
    {
        EnsureMap();
        auto it = m_sharers.find(globalId);
        return it == m_sharers.end() ? nullptr : &it->second;
    }

    /**
     * @brief Direct read-only access to every resolved id's sharer list.
     * @return The full map from resolved entity id to its sharer-rank list.
     * Iteration order is unspecified. Materialises the map on first use.
     */
    const std::unordered_map<int64_t, std::vector<int>> &AllSharers() const
    {
        EnsureMap();
        return m_sharers;
    }

    /**
     * @brief Retrieve the full sharer set for a resolved entity.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return Ranks that touch this entity, including this rank. Deduped: a
     * rank appears once even if it registered the same id more than once.
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     *
     * Materialises the id-keyed map on first use; prefer SharersOf() in a
     * loop over many ids.
     */
    const std::vector<int> &GetSharers(int64_t globalId) const
    {
        EnsureMap();
        return m_sharers.at(globalId);
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
        // Straight from the flat sharer array, so this does not drag the
        // id-keyed map into existence.
        std::set<int> nbrs;
        for (int r : m_shrFlat)
        {
            if (r != m_rank)
            {
                nbrs.insert(r);
            }
        }
        std::vector<int> ranks(nbrs.begin(), nbrs.end()), weights;
        return base->DistGraphCreateAdjacent(ranks, weights, 0);
    }

    /// @brief Clear registrations and results, so the resolver can be
    /// reused for a fresh Register()/Resolve() cycle.
    void Clear()
    {
        m_reqBuckets.clear();
        m_ids.clear();
        m_shrOff.clear();
        m_shrCnt.clear();
        m_shrFlat.clear();
        m_sharers.clear();
        m_mapBuilt = false;
    }

private:
    /// Build the id-keyed map the legacy accessors return, from the flat
    /// arrays Resolve() fills. Done once, and only if asked for.
    void EnsureMap() const
    {
        if (m_mapBuilt)
        {
            return;
        }
        m_sharers.clear();
        m_sharers.reserve(m_ids.size());
        for (size_t i = 0; i < m_ids.size(); ++i)
        {
            const int *base = m_shrFlat.data() + m_shrOff[i];
            m_sharers.emplace(m_ids[i],
                              std::vector<int>(base, base + m_shrCnt[i]));
        }
        m_mapBuilt = true;
    }

    /// Exchange mechanism used by Resolve().
    std::shared_ptr<ITransport> m_transport;
    /// This rank's index, cached from m_transport->Rank().
    int m_rank = 0;
    /// Total rank count, cached from m_transport->Size().
    int m_nproc = 0;

    /// Per-rendezvous-rank request buffers, filled incrementally by
    /// Register(). Sparse: only ranks this rank actually sends to are present,
    /// so this is O(local registrations), never O(nproc).
    std::unordered_map<int, std::vector<std::byte>> m_reqBuckets;

    /// Resolved ids, ascending: this rank's registrations, deduped.
    std::vector<int64_t> m_ids;
    /// Per resolved id, where its sharers start in m_shrFlat and how many.
    std::vector<uint32_t> m_shrOff, m_shrCnt;
    /// Every resolved id's sharer ranks, one contiguous block.
    std::vector<int> m_shrFlat;

    /// The id-keyed view the legacy accessors hand out, built on demand from
    /// the arrays above rather than during Resolve().
    mutable std::unordered_map<int64_t, std::vector<int>> m_sharers;
    mutable bool m_mapBuilt = false;
};

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIBUTILITIES_COMMUNICATION_CONNECTIVITYRESOLVER_HPP
