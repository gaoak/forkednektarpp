///////////////////////////////////////////////////////////////////////////////
//
// File: Transport.hpp
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
// Description: Personalised all-to-all transports for shared-id exchange.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file Transport.hpp
 * @brief Personalised all-to-all exchange of opaque byte blobs, and the
 * overview of the shared-id module built on it.
 *
 * @details
 * ### Where things live
 * This header is the bottom layer, holding #ITransport, its two
 * implementations and the `detail` helpers the layers above share. Every
 * other header in the module includes it, so the module overview lives here:
 * - ConnectivityResolver.hpp -- rendezvous discovery of sharer sets;
 * - SharedIdPlan.hpp -- the persistent gather-scatter plan;
 * - SharedPayloadResolver.hpp -- the id-keyed per-sharer payload exchange.
 *
 * ### The problem being solved
 * Several places in the code need to answer "for every global entity ID I
 * touch (vertex / edge / face, possibly periodically identified), find every
 * other rank that also touches it and agree on a canonical value (e.g. the
 * highest polynomial order)". The naive implementation AllReduces arrays
 * sized by the *global* entity count onto *every* rank: O(N_global) memory
 * per rank and O(N_global * P) total bandwidth. This does not scale to large
 * rank counts.
 *
 * ### Classes in the module
 * There are three main classes:
 *
 * - #SharedIdPlan -- the core: rendezvous discovery once, then a fixed
 *   neighbour exchange replayed on every call, folding each id's
 *   contributions to one agreed value (e.g. the highest polynomial order
 *   across a shared edge). The replacement for gslib's `gs_setup` /
 *   `gs_gather` / `gs_unique`, and what SharedPayloadResolver::Setup()
 *   builds. Code working on flat arrays (assembly maps) uses it directly,
 *   usually through LibUtilities::GatherScatter.
 * - #ConnectivityResolver -- the discovery half on its own: for each entity
 *   it finds the set of ranks sharing it, and nothing else. Use this when
 *   you only need topology, not a value. SharedIdPlan builds on it.
 * - #SharedPayloadResolver -- discovery (delegated to a composed
 *   ConnectivityResolver) followed by a direct, peer-to-peer payload
 *   exchange. Use this when there's no sensible way to fold contributions into
 *   one value (e.g. periodic vertex coordinates).
 *
 * There are also some additional utility classes:
 *
 * - #SharedPayloadResolverTimings -- optional diagnostic breakdown of
 *   SharedPayloadResolver::Resolve()'s discovery-round vs payload-round
 *   wall time; zero-cost unless a caller explicitly asks for it.
 * - #ConnectivityResolverTimings -- the finer-grained request-vs-reply
 *   Exchange() split that feeds SharedPayloadResolverTimings::
 *   discoverySeconds; also usable directly against a bare
 *   ConnectivityResolver.
 * - #ITransport -- abstract personalised all-to-all exchange of opaque byte
 *   blobs. Every resolver below is built on this; supply whichever
 *   implementation suits your rank count.
 *   - #AlltoallvTransport -- `MPI_Alltoall` of counts + `MPI_Alltoallv` of
 *     data. Simple and robust; keeps O(P) count arrays, so suitable below a
 *     few thousand ranks.
 *   - #CrystalRouterTransport -- hypercube personalised all-to-all (Fox /
 *     gslib crystal router). O(log P) rounds, no O(P) arrays; the variant
 *     for very large rank counts with sparse traffic.
 *
 * ### The approach (rendezvous / distributed directory)
 * -# Each rank hashes each of its entity IDs to a "rendezvous" rank and ships
 *    (id, value) there. All contributions for a given ID meet on one rank.
 * -# The rendezvous rank reduces the values and -- for free -- learns the
 *    complete set of ranks that touch each ID (the true topological
 *    neighbour set, including vertex-only and periodic adjacency).
 * -# Results (and the sharer list) are shipped back to each contributor.
 *
 * Memory per rank is O(local entities); nothing is ever sized by P or by the
 * global entity count. Every map that persists past a call --
 * #Nektar::LibUtilities::SharedPayloadResolver::m_sharedPayloads and the
 * ConnectivityResolver equivalent -- is keyed only by entities this rank
 * actually registered or was a rendezvous host for, so its size is bounded by
 * local entity count plus, in expectation, N_global / P at the rendezvous
 * step. No structure is ever sized by P or by N_global directly. The sharer
 * lists let you build a persistent neighbourhood communicator
 * (BuildNeighbourComm()) so downstream halo-style exchanges become cheap
 * neighbourhood collectives.
 *
 * This O(local) claim is about the *number of entities*, not the *fan-out*
 * of any one of them: ConnectivityResolver's reply phase sends the full
 * k-entry sharer list to each of an entity's k sharers,
 * so wire volume for a single high-fan-out entity is O(k^2). Fine at the
 * k=2-8 typical of facet/edge adjacency; worth watching for entities that
 * can be shared by dozens of ranks (e.g. a partition corner with
 * periodicity) or, at very large rank counts, a genuinely global entity.
 *
 * Payloads are shipped as raw bytes (no custom MPI datatype), which assumes
 * a homogeneous cluster: all ranks share the same endianness and struct
 * layout for PayloadT. True of essentially every HPC deployment, but not
 * checked or enforced here.
 *
 * ### One-shot versus persistent
 * Out of the box a resolver answers its question once: every Resolve()
 * rediscovers the sharers. That is optimal for a one-off agreement but
 * wasteful when the same ids are resolved again and again with new values --
 * the gslib use case. For that, #SharedIdPlan pays discovery once in its
 * constructor, and every Gather() / Reduce() thereafter is a single neighbour
 * exchange carrying only values -- no ids, no counts, no rendezvous -- over
 * the backend chosen by #ExchangeBackend. SharedPayloadResolver offers the
 * same split behind its id-keyed interface: Register() the ids once, call
 * Setup() once, then update payloads with SetPayload() and Resolve() as
 * often as needed.
 *
 * @code{.cpp}
 * SharedIdPlan plan(transport, ids);        // discovery, once
 * for (int step = 0; step < nSteps; ++step)
 * {
 *     plan.Gather(u.data(), std::plus<double>{});  // one neighbour exchange
 * }
 * @endcode
 *
 * ### Transport is pluggable
 * - #AlltoallvTransport: simple, good up to O(10^4) ranks. Still keeps O(P)
 *   count arrays internally (an MPI_Alltoall of counts), so not the endgame
 *   at 1M ranks.
 * - #CrystalRouterTransport: hypercube personalised all-to-all (Fox / gslib).
 *   O(log P) rounds, no O(P) arrays, follows the actual sparse traffic.
 *   Prefer this at scale.
 *
 * ### Example
 * @code{.cpp}
 * auto transport = std::make_shared<CrystalRouterTransport>(comm);
 * SharedIdPlan plan(transport, ids);     // ids[i] is slot i's entity id
 * std::vector<int> order = myOrders;     // one per slot
 * plan.Gather(order.data(), [](int a, int b){ return std::max(a, b); });
 * // order[i] is now the highest order across every rank sharing ids[i].
 * CommSharedPtr nbr = plan.BuildNeighbourComm(comm); // for halo exchange
 * @endcode
 */

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_TRANSPORT_HPP
#define NEKTAR_LIBUTILITIES_COMMUNICATION_TRANSPORT_HPP

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

namespace Nektar::LibUtilities
{

namespace detail
{

/**
 * @brief Append a trivially-copyable POD value to a byte buffer.
 *
 * @tparam T     Trivially-copyable value type.
 * @param  buf   Buffer to append to; grown by `sizeof(T)` bytes.
 * @param  v     Value to serialise.
 */
template <typename T>
inline void PackPod(std::vector<std::byte> &buf, const T &v)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "PackPod requires a trivially-copyable type");
    const auto *p = reinterpret_cast<const std::byte *>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}

/**
 * @brief Read a trivially-copyable POD value from a cursor and advance it.
 *
 * @tparam T     Trivially-copyable value type.
 * @param  cur   Cursor into a byte buffer, positioned at the start of a value
 *               of type `T`; advanced by `sizeof(T)` bytes on return.
 *
 * @return The unpacked value.
 */
template <typename T> inline T UnpackPod(const std::byte *&cur)
{
    static_assert(std::is_trivially_copyable_v<T> &&
                      std::is_default_constructible_v<T>,
                  "UnpackPod requires a trivially-copyable, "
                  "default-constructible type (T v; default-constructs "
                  "before the memcpy below)");
    T v;
    std::memcpy(static_cast<void *>(&v), cur, sizeof(T));
    cur += sizeof(T);
    return v;
}

/**
 * @brief Crystal-router routing decision for one bisection step.
 *
 * This is the single source of truth for who this rank should communicate with
 * in the recursive-bisection hypercube route. Group [lo, hi] splits into low =
 * [lo, mid] and high = [mid+1, hi]. Equal halves pair directly; an odd split
 * leaves one unpaired rank in the larger half that forwards its cross-group
 * messages one-way (#forwardTo) to the first rank of the other half. The mirror
 * side of that forward is #recvForwardFrom.
 */
struct CrystalStep
{
    /// Lower bound (inclusive) of the current group.
    int lo;
    /// Upper bound (inclusive) of the current group.
    int hi;
    /// Split point: low half is [lo, mid].
    int mid;
    /// Number of ranks in the low half, `mid - lo + 1`.
    int lowSize;
    /// Number of ranks in the high half, `hi - mid`.
    int highSize;
    /// Whether the querying rank is in the low half.
    bool inLow;
    /// Rank to exchange with, or -1 if unpaired extra.
    int partner;
    /// One-way forward target (valid iff partner == -1).
    int forwardTo;
    /// Rank we must receive a one-way forward from, or -1.
    int recvForwardFrom;
};

/**
 * @brief Compute this rank's routing decision for one bisection step of the
 * crystal router.
 *
 * @param  lo   Lower bound (inclusive) of the current group of ranks.
 * @param  hi   Upper bound (inclusive) of the current group of ranks.
 * @param  rank The rank computing its own routing decision (normally the
 *              calling rank's own rank).
 *
 * @return A #CrystalStep describing which half `rank` falls in and who it
 * should exchange (or one-way forward) messages with this round.
 */
inline CrystalStep CrystalPlanStep(int lo, int hi, int rank)
{
    CrystalStep s;

    // Compute step parameters according to comments in CrystalStep struct.
    s.lo              = lo;
    s.hi              = hi;
    s.mid             = lo + (hi - lo) / 2;
    s.lowSize         = s.mid - lo + 1;
    s.highSize        = hi - s.mid;
    s.inLow           = (rank <= s.mid);
    s.partner         = -1;
    s.forwardTo       = -1;
    s.recvForwardFrom = -1;

    // Invariant of `mid = lo + (hi - lo) / 2` (floor division): the low
    // half is never smaller than the high half -- lowSize is always either
    // highSize or highSize + 1, never less. Consequently the "unpaired
    // extra in the larger *high* half" branch below, and the
    // `highSize == lowSize + 1` mirror case, can never trigger for this
    // split rule; they exist only so behaviour stays correct (rather than
    // silently wrong) if the split rule above is ever changed to something
    // that can produce a larger high half. Asserted here, once, rather than
    // deleting the "dead" branches, so that change would fail loudly here
    // instead of routing messages incorrectly somewhere in route().
    ASSERTL1(s.lowSize == s.highSize || s.lowSize == s.highSize + 1,
             "CrystalPlanStep: split invariant violated (lowSize must be "
             "highSize or highSize + 1) -- if the split rule above changed, "
             "the branches guarded by this invariant need revisiting too");

    if (s.inLow)
    {
        const int idx = rank - lo;
        if (idx < s.highSize)
        {
            // pair with matching rank in high half
            s.partner = s.mid + 1 + idx;
        }
        else
        {
            // unpaired extra in the larger low half
            s.forwardTo = s.mid + 1;
        }
    }
    else
    {
        const int idx = rank - (s.mid + 1);
        if (idx < s.lowSize)
        {
            // pair with matching rank in low half
            s.partner = lo + idx;
        }
        else
        {
            // unpaired extra in the larger high half
            s.forwardTo = lo;
        }
    }

    // Mirror of another rank's one-way forward: exactly the first rank of the
    // smaller half receives it. Derived from group sizes so both sides agree.
    if (s.lowSize == s.highSize + 1 && rank == s.mid + 1)
    {
        s.recvForwardFrom = s.mid;
    }
    else if (s.highSize == s.lowSize + 1 && rank == lo)
    {
        s.recvForwardFrom = s.hi;
    }

    return s;
}

/**
 * @brief Test whether a message destined for @p dest belongs in the
 * querying rank's half of the current bisection.
 *
 * @param s     Routing decision for the current step, from CrystalPlanStep().
 * @param dest  Destination rank of the message being tested.
 *
 * @return `true` if `dest` is in the same half as the querying rank (so the
 * message should be kept for a later round), `false` if it must be sent
 * across to the other half this round.
 */
inline bool CrystalDestInMyHalf(const CrystalStep &s, int dest)
{
    const bool destLow = (dest <= s.mid);
    return destLow == s.inLow;
}

} // namespace detail

/**
 * @brief An opaque byte payload plus routing metadata.
 *
 * The transport layer moves these between ranks without ever interpreting
 * `bytes`; only #ConnectivityResolver, #SharedIdPlan and
 * #SharedPayloadResolver know how to pack and unpack the payload.
 */
struct RoutedMessage
{
    int dest = -1;                ///< destination rank
    int src  = -1;                ///< originating rank
    std::vector<std::byte> bytes; ///< opaque payload
};

/**
 * @brief Abstract interface for a personalised all-to-all exchange of opaque
 * byte blobs.
 *
 * The idea of this class is that a resolver produces at most one outbound
 * message per destination rank. On return, `inbound` holds one message per
 * originating rank that sent to us, with `src` set and `dest == this rank`.
 *
 * Concrete implementations are #AlltoallvTransport and
 * #CrystalRouterTransport; either can be handed to #ConnectivityResolver,
 * #SharedIdPlan or #SharedPayloadResolver interchangeably.
 */
class ITransport
{
public:
    virtual ~ITransport() = default;

    /**
     * @brief Perform a personalised all-to-all exchange of the outbound
     * messages.
     *
     * @param[in,out] outbound Messages to send, at most one per destination
     *                         rank; always left empty on return (every
     *                         implementation consumes it, whether by move or
     *                         by explicit clear -- callers should not rely
     *                         on `outbound` still holding anything
     *                         afterwards).
     * @param[out]    inbound  Filled with the messages received from other
     *                         ranks. May hold one message per originating
     *                         rank (as AlltoallvTransport delivers, coalesced)
     *                         or several (as CrystalRouterTransport delivers,
     *                         one per original outbound message that reached
     *                         this rank) -- do not assume a fixed count per
     *                         source; concatenate/iterate instead.
     */
    virtual void Exchange(std::vector<RoutedMessage> &outbound,
                          std::vector<RoutedMessage> &inbound) = 0;

    /// @return This rank's index within Comm().
    virtual int Rank() const = 0;

    /// @return The number of ranks in Comm().
    virtual int Size() const = 0;

    /// @return The underlying communicator this transport operates over.
    virtual CommSharedPtr Comm() const = 0;
};

/**
 * @brief ITransport implementation built on `MPI_Alltoall` (of counts)
 * followed by `MPI_Alltoallv` (of data).
 *
 * Simple and robust, but the count exchange is \f$ O(P) \f$ memory per rank
 * (see the note in Exchange()), so this transport mechanism is suitable for
 * below a few thousand ranks, and can be used as the correctness reference for
 * #CrystalRouterTransport.
 */
class AlltoallvTransport : public ITransport
{
public:
    /**
     * @brief Construct a transport bound to @p comm.
     *
     * @param comm Communicator whose ranks participate in each Exchange().
     */
    explicit AlltoallvTransport(CommSharedPtr comm)
        : m_comm(std::move(comm)), m_nproc(m_comm->GetSize()),
          m_rank(m_comm->GetRank())
    {
    }

    int Rank() const override
    {
        return m_rank;
    }
    int Size() const override
    {
        return m_nproc;
    }
    CommSharedPtr Comm() const override
    {
        return m_comm;
    }

    /**
     * @brief Personalised all-to-all via `MPI_Alltoall` and `MPI_Alltoallv`.
     * @param[in] outbound Messages to send, at most one per destination
     * rank; multiple messages to the same destination are concatenated.
     * Left empty on return (see ITransport::Exchange).
     * @param[out] inbound One message per rank that sent to this rank.
     *
     * @note The `sendCounts`/`recvCounts`/`sdispls`/`rdispls` arrays below
     * are sized O(P) -- this is the one place in the module where memory
     * scales with rank count rather than local entity count. It is bounded
     * by the `MPI_Alltoall` of counts, not by the (much larger) data
     * exchange, and is the documented trade-off of this transport; use
     * CrystalRouterTransport instead once P becomes large.
     */
    void Exchange(std::vector<RoutedMessage> &outbound,
                  std::vector<RoutedMessage> &inbound) override
    {
        const int np = m_nproc;
        std::vector<int> sendCounts(np, 0), recvCounts(np, 0);
        for (auto &m : outbound)
        {
            ASSERTL1(m.dest >= 0 && m.dest < np,
                     "AlltoallvTransport::Exchange: message has an "
                     "out-of-range dest rank");
            sendCounts[m.dest] += static_cast<int>(m.bytes.size());
        }

        m_comm->AlltoAll(sendCounts, recvCounts);

        std::vector<int> sdispls(np, 0), rdispls(np, 0);
        for (int p = 1; p < np; ++p)
        {
            sdispls[p] = sdispls[p - 1] + sendCounts[p - 1];
            rdispls[p] = rdispls[p - 1] + recvCounts[p - 1];
        }
        const int totSend = sdispls[np - 1] + sendCounts[np - 1];
        const int totRecv = rdispls[np - 1] + recvCounts[np - 1];

        // sendCounts / sdispls / totSend are `int` because MPI's count
        // arguments are `int`; a rank whose aggregate outbound volume
        // exceeds INT_MAX silently wraps in the accumulation above rather
        // than failing loudly, corrupting data instead of crashing.
        // ASSERTL0 (checked in release too) since this is exactly the kind
        // of wall people hit only once meshes get large, and the whole
        // point is to fail loudly rather than ship a wrapped count.
        {
            int64_t check = 0;
            for (auto &m : outbound)
            {
                check += static_cast<int64_t>(m.bytes.size());
            }
            ASSERTL0(check == static_cast<int64_t>(totSend),
                     "AlltoallvTransport::Exchange: outbound byte volume "
                     "overflowed the MPI int count limit (2^31-1 bytes); "
                     "switch to CrystalRouterTransport or shard the "
                     "exchange.");
        }

        // Pad to one element: a rank with no traffic at all in a round is
        // routine (the rendezvous buckets are sparse), but Nektar's
        // CommDataTypeTraits::GetPointer asserts on an empty vector, so a
        // zero-length buffer would trip an ASSERTL1 in debug and take the
        // address of element 0 of an empty vector in release. The counts stay
        // zero either way, so nothing is actually sent or received.
        std::vector<std::byte> sbuf(std::max(totSend, 1)),
            rbuf(std::max(totRecv, 1));

        // Copy each message into its destination's slot at an
        // independently-advancing cursor, not always at sdispls[dest]: two
        // messages to the same destination previously both landed at the
        // start of that destination's slot, so the second silently
        // overwrote the first and the unwritten tail of the slot shipped as
        // zero bytes -- wrong data with no exception, since
        // DrainBucketsToMessages() (the only caller today) happens to key
        // by dest and so never triggers this. CrystalRouterTransport was
        // never affected, since it serialises whole messages rather than
        // concatenating by destination.
        std::vector<int> cursor = sdispls;
        for (auto &m : outbound)
        {
            std::copy(m.bytes.begin(), m.bytes.end(),
                      sbuf.begin() + cursor[m.dest]);
            cursor[m.dest] += static_cast<int>(m.bytes.size());
        }

        m_comm->AlltoAllv(sbuf, sendCounts, sdispls, rbuf, recvCounts, rdispls);

        // Unpack received buffer back into per-source messages.
        inbound.clear();
        for (int p = 0; p < np; ++p)
        {
            if (recvCounts[p] == 0)
            {
                continue;
            }
            RoutedMessage m;
            m.src  = p;
            m.dest = m_rank;
            m.bytes.assign(rbuf.begin() + rdispls[p],
                           rbuf.begin() + rdispls[p] + recvCounts[p]);
            inbound.push_back(std::move(m));
        }

        // Guarantee the same postcondition CrystalRouterTransport gives
        // (outbound left empty), rather than leaving it fully intact as
        // before -- see ITransport::Exchange.
        outbound.clear();
    }

private:
    CommSharedPtr m_comm;
    int m_nproc = 0;
    int m_rank  = 0;
};

/**
 * @brief ITransport implementation performing a personalised all-to-all on a
 * (bisected) hypercube (a Fox/gslib-style crystal router).
 *
 * @details
 * O(log P) communication rounds; each message is forwarded at most
 * `log2(P)` hops, with no O(P) arrays. This is the variant to use at very large
 * rank counts.
 *
 * The recursive-bisection formulation in Route() handles non-power-of-two
 * rank counts directly: at each step the current group [lo, hi] is split
 * into [lo, mid] and [mid+1, hi]. Equal halves pair up directly; an odd
 * split leaves one unpaired rank in the larger half, which forwards its
 * cross-group messages one-way to a designated rank in the other half.
 */
class CrystalRouterTransport : public ITransport
{
public:
    /// @brief Construct a transport bound to @p comm.
    /// @param comm Communicator whose ranks participate in each Exchange().
    explicit CrystalRouterTransport(CommSharedPtr comm)
        : m_comm(std::move(comm)), m_nproc(m_comm->GetSize()),
          m_rank(m_comm->GetRank())
    {
    }

    int Rank() const override
    {
        return m_rank;
    }
    int Size() const override
    {
        return m_nproc;
    }
    CommSharedPtr Comm() const override
    {
        return m_comm;
    }

    /**
     * @brief Personalised all-to-all via recursive-bisection hypercube
     * routing.
     * @param[in,out] outbound Messages to send, at most one per destination
     * rank; moved from on entry.
     * @param[out] inbound Filled with every message addressed to this rank
     * once routing completes (i.e. once `dest == Rank()` for all of them).
     */
    void Exchange(std::vector<RoutedMessage> &outbound,
                  std::vector<RoutedMessage> &inbound) override
    {
        Flatten(outbound);
        // Standard guarantees a moved-from vector is left in a valid but
        // unspecified state, not necessarily empty -- clear explicitly so
        // the "outbound left empty" postcondition (see ITransport::Exchange)
        // doesn't depend on stdlib implementation details.
        outbound.clear();
        Route();
        // After routing every record has dest == m_rank.
        Rebuild(inbound);
    }

private:
    /**
     * @brief One message in flight, as an index into #m_pool.
     *
     * Routing moves 16-byte records and contiguous byte ranges, never
     * per-message containers: the whole point of the flat representation is
     * that a hop costs one pass over two arrays rather than an allocation
     * per message. Cf. gslib's crystal router, which likewise partitions a
     * flat array of fixed-size records at every hop.
     */
    struct Record
    {
        int32_t dest; ///< destination rank
        int32_t src;  ///< originating rank
        uint32_t off; ///< offset of the payload in the pool
        uint32_t len; ///< payload length in bytes
    };

    /// Bytes of a record's wire header: dest, src, len.
    static constexpr size_t kHeaderBytes = 3 * sizeof(int32_t);

    /// Slack below which reclaiming departed payloads is not worth a pass.
    static constexpr size_t kCompactSlack = 64 * 1024;

    /**
     * @brief Copy @p outbound into the flat (#m_recs, #m_pool) form routing
     * works on. The only place a caller's per-message buffer is read.
     */
    void Flatten(const std::vector<RoutedMessage> &outbound)
    {
        size_t total = 0;
        for (const auto &m : outbound)
        {
            total += m.bytes.size();
        }

        m_recs.clear();
        m_recs.reserve(outbound.size());
        m_pool.clear();
        m_pool.resize(total);

        size_t off = 0;
        for (const auto &m : outbound)
        {
            const size_t n = m.bytes.size();
            if (n > 0)
            {
                std::memcpy(m_pool.data() + off, m.bytes.data(), n);
            }
            m_recs.push_back(Record{
                static_cast<int32_t>(m.dest), static_cast<int32_t>(m.src),
                static_cast<uint32_t>(off), static_cast<uint32_t>(n)});
            off += n;
        }
    }

    /**
     * @brief Hand the routed records back as RoutedMessage, the one
     * allocation per message the interface obliges us to make.
     */
    void Rebuild(std::vector<RoutedMessage> &inbound)
    {
        inbound.clear();
        inbound.reserve(m_recs.size());
        for (const Record &r : m_recs)
        {
            RoutedMessage m;
            m.dest = r.dest;
            m.src  = r.src;
            m.bytes.assign(m_pool.begin() + r.off,
                           m_pool.begin() + r.off + r.len);
            inbound.push_back(std::move(m));
        }
    }

    /**
     * @brief Write the records selected by @p send into #m_sendBuf, in wire
     * format: a record count, then per record its header and payload.
     *
     * The count is always written, so the buffer is never empty and never
     * reaches Comm's GetPointer(), which asserts on an empty container.
     */
    void PackSend(const std::vector<Record> &send)
    {
        size_t bytes = sizeof(int32_t);
        for (const Record &r : send)
        {
            bytes += kHeaderBytes + r.len;
        }
        m_sendBuf.resize(bytes);

        std::byte *cur      = m_sendBuf.data();
        const int32_t nSend = static_cast<int32_t>(send.size());
        std::memcpy(cur, &nSend, sizeof(int32_t));
        cur += sizeof(int32_t);

        for (const Record &r : send)
        {
            const int32_t hdr[3] = {r.dest, r.src, static_cast<int32_t>(r.len)};
            std::memcpy(cur, hdr, kHeaderBytes);
            cur += kHeaderBytes;
            if (r.len > 0)
            {
                std::memcpy(cur, m_pool.data() + r.off, r.len);
                cur += r.len;
            }
        }
    }

    /**
     * @brief Append the records in @p buf (wire format, from PackSend()) to
     * #m_recs, their payloads going on the end of #m_pool.
     */
    void UnpackInto(const std::vector<std::byte> &buf, size_t &liveBytes)
    {
        if (buf.empty())
        {
            return;
        }

        const std::byte *cur                  = buf.data();
        [[maybe_unused]] const std::byte *end = buf.data() + buf.size();

        int32_t n = 0;
        std::memcpy(&n, cur, sizeof(int32_t));
        cur += sizeof(int32_t);

        m_recs.reserve(m_recs.size() + static_cast<size_t>(n));
        // No reserve() on the pool: an exact-size reserve on every round
        // would defeat the containers geometric growth and reallocate each
        // time. insert() already grows it amortised-constant.
        for (int32_t i = 0; i < n; ++i)
        {
            int32_t hdr[3];
            std::memcpy(hdr, cur, kHeaderBytes);
            cur += kHeaderBytes;

            const auto len = static_cast<uint32_t>(hdr[2]);
            const auto off = static_cast<uint32_t>(m_pool.size());
            m_pool.insert(m_pool.end(), cur, cur + len);
            cur += len;

            m_recs.push_back(Record{hdr[0], hdr[1], off, len});
            liveBytes += len;
        }

        ASSERTL1(cur == end,
                 "CrystalRouterTransport: malformed buffer (record lengths "
                 "didn't sum to the buffer size)");
    }

    /**
     * @brief Symmetric size-then-data exchange of #m_sendBuf with @p partner,
     * leaving the reply in #m_recvBuf.
     */
    void ExchangeWith(int partner)
    {
        int64_t sn = static_cast<int64_t>(m_sendBuf.size());
        int64_t rn = 0;

        m_comm->SendRecv(partner, sn, partner, rn);

        m_recvBuf.resize(static_cast<size_t>(rn));
        m_comm->SendRecv(partner, m_sendBuf, partner, m_recvBuf);
    }

    /**
     * @brief One-way forward of #m_sendBuf from an unpaired rank in the
     * larger half to @p partner, which has no direct partner this round.
     */
    void SendOneWay(int partner)
    {
        int64_t sn = static_cast<int64_t>(m_sendBuf.size());
        m_comm->Send(partner, sn);
        m_comm->Send(partner, m_sendBuf);
    }

    /// @brief Receive the mirror side of a SendOneWay() into #m_recvBuf.
    void RecvOneWay(int partner)
    {
        int64_t rn = 0;
        m_comm->Recv(partner, rn);

        m_recvBuf.resize(static_cast<size_t>(rn));
        m_comm->Recv(partner, m_recvBuf);
    }

    /**
     * @brief Drive the recursive-bisection hypercube routing loop until
     * every record in #m_recs has reached its destination rank.
     *
     * Each iteration narrows the active group [lo, hi] by half (via
     * #detail::CrystalPlanStep) and moves any record whose destination is no
     * longer in this rank's half across to a partner (or forwards it one-way
     * if unpaired), so after `ceil(log2(nproc))` rounds every record has
     * arrived. Records that stay are compacted into #m_next alongside the
     * ones that arrive, so the pool never accumulates the payloads of
     * records that have left.
     */
    void Route()
    {
        int lo = 0;
        int hi = m_nproc - 1;

        while (lo < hi)
        {
            const detail::CrystalStep s =
                detail::CrystalPlanStep(lo, hi, m_rank);

            // Split: records whose destination has left this half are sent,
            // the rest stay exactly where they are. Keeping them in place is
            // what makes a hop cheap -- the pool only ever grows within a
            // call, so a kept record's offset stays valid and no payload is
            // copied to hold on to it.
            m_keep.clear();
            m_send.clear();
            size_t liveBytes = 0;
            for (const Record &r : m_recs)
            {
                if (detail::CrystalDestInMyHalf(s, r.dest))
                {
                    m_keep.push_back(r);
                    liveBytes += r.len;
                }
                else
                {
                    m_send.push_back(r);
                }
            }

            // Reads m_pool, so it must precede the appends below.
            PackSend(m_send);
            m_recs.swap(m_keep);

            if (s.partner >= 0)
            {
                ExchangeWith(s.partner);
                UnpackInto(m_recvBuf, liveBytes);
            }
            else
            {
                // Unpaired rank in the larger half: forward one-way into the
                // other half.
                SendOneWay(s.forwardTo);
            }

            // If we are the designated receiver of a one-way forward, post
            // it.
            if (s.recvForwardFrom >= 0)
            {
                RecvOneWay(s.recvForwardFrom);
                UnpackInto(m_recvBuf, liveBytes);
            }

            // The payloads of records that have left are still sitting in
            // the pool. Reclaim them only once they dominate it, so the
            // common hop pays no copy at all and the pool stays inside a
            // constant factor of what is live.
            if (m_pool.size() > 2 * liveBytes + kCompactSlack)
            {
                Compact();
            }

            if (s.inLow)
            {
                hi = s.mid;
            }
            else
            {
                lo = s.mid + 1;
            }
        }
    }

    /// Rebuild the pool from the live records, dropping departed payloads.
    void Compact()
    {
        m_next.clear();
        for (Record &r : m_recs)
        {
            const auto off = static_cast<uint32_t>(m_next.size());
            m_next.insert(m_next.end(), m_pool.begin() + r.off,
                          m_pool.begin() + r.off + r.len);
            r.off = off;
        }
        m_pool.swap(m_next);
    }

    /// Records in flight, and the contiguous pool their payloads live in.
    std::vector<Record> m_recs;
    std::vector<std::byte> m_pool;
    /// Scratch for one round, kept as members so the buffers are grown once
    /// and then reused for the life of the transport.
    std::vector<Record> m_keep, m_send;
    std::vector<std::byte> m_next, m_sendBuf, m_recvBuf;

    CommSharedPtr m_comm;
    int m_nproc = 0;
    int m_rank  = 0;
};

namespace detail
{

/**
 * @brief Bit-mixing hash (splitmix64 finaliser) mapping an entity id to its
 * rendezvous rank, so sequential IDs spread across rendezvous ranks and
 * don't create hot spots.
 * @param id Global entity id.
 * @param nproc Number of ranks to hash into, i.e. `Comm()->GetSize()`.
 * @return The rendezvous rank for @p id, in `[0, nproc)`.
 */
inline int RendezvousRank(int64_t id, int nproc)
{
    uint64_t h = static_cast<uint64_t>(id);
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return static_cast<int>(h % static_cast<uint64_t>(nproc));
}

/**
 * @brief Move sparse per-destination byte buckets into transport messages,
 * clearing the buckets.
 * @param[in,out] buckets Per-destination-rank byte buffers, keyed only by
 * ranks this rank actually has data for (i.e. sparse, not O(nproc));
 * cleared on return.
 * @param selfRank This rank's index, stamped into each produced message's
 * `src` field.
 * @return One RoutedMessage per non-empty bucket. Only non-empty
 * destinations are present, so this is O(local).
 */
inline std::vector<RoutedMessage> DrainBucketsToMessages(
    std::unordered_map<int, std::vector<std::byte>> &buckets, int selfRank)
{
    std::vector<RoutedMessage> msgs;
    msgs.reserve(buckets.size());
    for (auto &kv : buckets)
    {
        msgs.push_back(RoutedMessage{kv.first, selfRank, std::move(kv.second)});
    }
    buckets.clear();
    return msgs;
}

/**
 * @brief Build a persistent distributed-graph communicator over the
 * discovered topological neighbours (self excluded).
 * @param base Base communicator to derive the graph communicator from.
 * @param selfRank This rank's index; excluded from its own neighbour list.
 * @param sharers Resolved sharer lists, keyed by entity id (as populated by
 * ConnectivityResolver::Resolve()).
 * @return A distributed-graph communicator (via `DistGraphCreateAdjacent`)
 * whose neighbours are the union of all ranks appearing in @p sharers,
 * excluding @p selfRank.
 */
inline CommSharedPtr BuildNeighbourComm(
    const CommSharedPtr &base, int selfRank,
    const std::unordered_map<int64_t, std::vector<int>> &sharers)
{
    std::set<int> nbrs;
    for (const auto &kv : sharers)
    {
        for (int r : kv.second)
        {
            if (r != selfRank)
            {
                nbrs.insert(r);
            }
        }
    }
    std::vector<int> ranks(nbrs.begin(), nbrs.end());
    std::vector<int> weights;

    return base->DistGraphCreateAdjacent(ranks, weights, 0);
}

} // namespace detail

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIBUTILITIES_COMMUNICATION_TRANSPORT_HPP
