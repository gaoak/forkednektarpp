///////////////////////////////////////////////////////////////////////////////
//
// File: EntityResolver.hpp
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
// Description: Shared ID discovery and exchange classes.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file EntityResolver.hpp
 * @brief Scalable passes for shared-id discovery, reductions and payload
 * synchronisation.
 *
 * @details
 * ### The problem being solved
 * Several places in the code need to answer "for every global entity ID I
 * touch (vertex / edge / face, possibly periodically identified), find every
 * other rank that also touches it and agree on a canonical value (e.g. the
 * highest polynomial order)". The naive implementation AllReduces arrays
 * sized by the *global* entity count onto *every* rank: O(N_global) memory
 * per rank and O(N_global * P) total bandwidth. This does not scale to large
 * rank counts.
 *
 * ### Classes in this file
 * There are three main classes this file provides:
 *
 * - #EntityResolver -- rendezvous *and* reduction: discovers every rank
 *   sharing an entity and folds their contributions to one agreed value
 *   (e.g. the highest polynomial order across a shared edge). This functions
 *   similarly to gslib.
 * - #ConnectivityResolver -- the payload-free sibling of EntityResolver:
 *   discovers just the sharer set for each entity, nothing else. Use this
 *   when you only need topology, not a value.
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
 * global entity count. Every unordered_map that persists past a call --
 * #Nektar::LibUtilities::EntityResolver::m_reqBuckets,
 * #Nektar::LibUtilities::EntityResolver::m_resolved and
 * #Nektar::LibUtilities::EntityResolver::m_sharers, and their
 * ConnectivityResolver equivalents -- is keyed only by entities this rank
 * actually registered or was a rendezvous host for, so its size is bounded by
 * local entity count plus, in expectation, N_global / P at the rendezvous
 * step. No structure is ever sized by P or by N_global directly. The sharer
 * lists let you build a persistent neighbourhood communicator
 * (BuildNeighbourComm()) so downstream halo-style exchanges become cheap
 * neighbourhood collectives.
 *
 * This O(local) claim is about the *number of entities*, not the *fan-out*
 * of any one of them: EntityResolver's and ConnectivityResolver's reply
 * phase sends the full k-entry sharer list to each of an entity's k sharers,
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
 * EntityResolver<int> res(transport);          // payload = polynomial order
 * for (auto &e : myTraces)
 *     res.Register(e.globalId, e.order);
 * res.Resolve([](int a, int b){ return std::max(a, b); });
 * int agreed = res.GetResolved(someId);        // highest order across ranks
 * MPI_Comm nbr = res.BuildNeighbourComm(comm); // reuse for halo exchange
 * @endcode
 */

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_ENTITYRESOLVER_HPP
#define NEKTAR_LIBUTILITIES_COMMUNICATION_ENTITYRESOLVER_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
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
    std::memcpy(&v, cur, sizeof(T));
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
 * `bytes`; only #EntityResolver and #ConnectivityResolver know how to pack and
 * unpack the payload.
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
 * The idea of this class is that #EntityResolver produces at most one outbound
 * message per destination rank. On return, `inbound` holds one message per
 * originating rank that sent to us, with `src` set and `dest == this rank`.
 *
 * Concrete implementations are #AlltoallvTransport and
 * #CrystalRouterTransport; either can be handed to #EntityResolver or
 * #ConnectivityResolver interchangeably.
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
        std::vector<RoutedMessage> msgs = std::move(outbound);
        // Standard guarantees a moved-from vector is left in a valid but
        // unspecified state, not necessarily empty -- clear explicitly so
        // the "outbound left empty" postcondition (see ITransport::Exchange)
        // doesn't depend on stdlib implementation details.
        outbound.clear();
        Route(msgs);
        // After routing every message has dest == m_rank.
        inbound = std::move(msgs);
    }

private:
    /**
     * @brief Flatten a list of RoutedMessage into a single wire-format byte
     * buffer (count, then per-message dest/src/length/bytes).
     * @param v Messages to serialise.
     * @return The packed byte buffer.
     */
    static std::vector<std::byte> Serialize(const std::vector<RoutedMessage> &v)
    {
        std::vector<std::byte> buf;
        detail::PackPod<int32_t>(buf, static_cast<int32_t>(v.size()));
        for (auto &m : v)
        {
            detail::PackPod<int32_t>(buf, static_cast<int32_t>(m.dest));
            detail::PackPod<int32_t>(buf, static_cast<int32_t>(m.src));
            detail::PackPod<int32_t>(buf, static_cast<int32_t>(m.bytes.size()));
            buf.insert(buf.end(), m.bytes.begin(), m.bytes.end());
        }
        return buf;
    }

    /**
     * @brief Inverse of Serialize(): unpack a wire-format byte buffer back
     * into a list of RoutedMessage, appending to @p out.
     * @param buf Buffer previously produced by Serialize(). An empty buffer
     * is treated as "no messages" and is a no-op.
     * @param[out] out Messages are appended here (not cleared first).
     */
    static void Deserialize(const std::vector<std::byte> &buf,
                            std::vector<RoutedMessage> &out)
    {
        if (buf.empty())
        {
            return;
        }
        const std::byte *cur = buf.data();
        // Only read by the ASSERTL1 below, which compiles out in release.
        [[maybe_unused]] const std::byte *end = buf.data() + buf.size();
        const int32_t n                       = detail::UnpackPod<int32_t>(cur);
        out.reserve(out.size() + n);
        for (int32_t i = 0; i < n; ++i)
        {
            RoutedMessage m;
            m.dest            = detail::UnpackPod<int32_t>(cur);
            m.src             = detail::UnpackPod<int32_t>(cur);
            const int32_t len = detail::UnpackPod<int32_t>(cur);
            m.bytes.assign(cur, cur + len);
            cur += len;
            out.push_back(std::move(m));
        }
        ASSERTL1(cur == end,
                 "CrystalRouterTransport::Deserialize: malformed buffer "
                 "(record lengths didn't sum to the buffer size)");
    }

    /**
     * @brief Symmetric size-then-data exchange with a paired partner rank.
     * @param send Messages to send to @p partner.
     * @param partner Rank to exchange with (this round's
     * #detail::CrystalStep::partner).
     * @return The messages received from @p partner.
     */
    std::vector<RoutedMessage> ExchangeWith(
        const std::vector<RoutedMessage> &send, int partner)
    {
        std::vector<std::byte> sbuf = Serialize(send);
        int64_t sn                  = static_cast<int64_t>(sbuf.size());
        int64_t rn                  = 0;

        m_comm->SendRecv(partner, sn, partner, rn);

        std::vector<std::byte> rbuf(static_cast<size_t>(rn));
        m_comm->SendRecv(partner, sbuf, partner, rbuf);

        std::vector<RoutedMessage> recv;
        Deserialize(rbuf, recv);
        return recv;
    }

    /**
     * @brief One-way forward from an unpaired rank in the larger half; sent
     * by an "extra" rank that has no direct partner this round.
     * @param send Messages to forward.
     * @param partner Destination rank (this round's
     * #detail::CrystalStep::forwardTo).
     */
    void SendOneWay(const std::vector<RoutedMessage> &send, int partner)
    {
        std::vector<std::byte> sbuf = Serialize(send);
        int64_t sn                  = static_cast<int64_t>(sbuf.size());

        m_comm->Send(partner, sn);
        m_comm->Send(partner, sbuf);
    }

    /**
     * @brief Receive the mirror side of a SendOneWay() forward.
     * @param partner Rank the forward is expected from (this round's
     * #detail::CrystalStep::recvForwardFrom).
     * @return The forwarded messages.
     */
    std::vector<RoutedMessage> RecvOneWay(int partner)
    {
        int64_t rn = 0;
        m_comm->Recv(partner, rn);

        std::vector<std::byte> rbuf(static_cast<size_t>(rn));
        m_comm->Recv(partner, rbuf);

        std::vector<RoutedMessage> recv;
        Deserialize(rbuf, recv);
        return recv;
    }

    /**
     * @brief Drive the recursive-bisection hypercube routing loop until
     * every message has reached its destination rank.
     * @param[in,out] msgs Messages to route, addressed by `dest`; on return,
     * every message satisfies `dest == m_rank` (i.e. this holds the inbound
     * set for this rank).
     *
     * Each iteration narrows the active group [lo, hi] by half (via
     * #detail::CrystalPlanStep) and moves any message whose destination is
     * no longer in this rank's half across to a partner (or forwards it
     * one-way if unpaired), so after `ceil(log2(nproc))` rounds every
     * message has arrived.
     */
    void Route(std::vector<RoutedMessage> &msgs)
    {
        int lo = 0;
        int hi = m_nproc - 1;

        while (lo < hi)
        {
            const detail::CrystalStep s =
                detail::CrystalPlanStep(lo, hi, m_rank);

            // Split: messages for the other half are sent, the rest kept.
            std::vector<RoutedMessage> keep, send;
            keep.reserve(msgs.size());
            for (auto &m : msgs)
            {
                if (detail::CrystalDestInMyHalf(s, m.dest))
                {
                    keep.push_back(std::move(m));
                }
                else
                {
                    send.push_back(std::move(m));
                }
            }
            msgs.clear();

            std::vector<RoutedMessage> recv;
            if (s.partner >= 0)
            {
                recv = ExchangeWith(send, s.partner);
            }
            else
            {
                // Unpaired rank in the larger half: forward one-way into the
                // other half.
                SendOneWay(send, s.forwardTo);
            }

            // If we are the designated receiver of a one-way forward, post it.
            if (s.recvForwardFrom >= 0)
            {
                std::vector<RoutedMessage> extra =
                    RecvOneWay(s.recvForwardFrom);
                for (auto &m : extra)
                {
                    recv.push_back(std::move(m));
                }
            }

            // Merge kept + received, then narrow to my half.
            msgs = std::move(keep);
            for (auto &m : recv)
            {
                msgs.push_back(std::move(m));
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
 * EntityResolver::Resolve() or ConnectivityResolver::Resolve()).
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

/**
 * @class EntityResolver
 * @brief Distributed-directory (rendezvous) resolver: for every registered
 * (entity id, value) pair, discovers all ranks sharing that entity and
 * reduces their values to a single agreed answer.
 *
 * @tparam PayloadT Trivially-copyable value type shipped alongside each
 * entity id (e.g. an `int` polynomial order, or a small POD struct). Must
 * satisfy `std::is_trivially_copyable_v`, since payloads are shipped as raw
 * bytes with no custom MPI datatype.
 *
 * @details
 * See the @ref EntityResolver.hpp "file-level documentation" for the
 * rendezvous algorithm and the O(local) memory argument. Reduction is
 * supplied as a callable `(a, b) -> reduced` (see ReduceOp).
 *
 * If you only need to discover *who shares each entity* and not a reduced
 * value, use #ConnectivityResolver instead -- it ships no payload at all. If
 * you need every sharer's own payload rather than a single reduced value
 * (e.g. periodic vertex coordinates, where there's no sensible "agreed"
 * coordinate), use #SharedPayloadResolver instead.
 */
template <typename PayloadT> class EntityResolver
{
    static_assert(std::is_trivially_copyable_v<PayloadT> &&
                      std::is_default_constructible_v<PayloadT>,
                  "PayloadT must be trivially copyable and "
                  "default-constructible (UnpackPod default-constructs a "
                  "scratch value internally before the memcpy). For "
                  "variable-length payloads, add a pack/unpack step.");

public:
    /// Callable reducing two payloads sharing the same entity id to one.
    /// Kept as a named type for callers who want to store or pass one
    /// around; Resolve() itself is templated on the op (see below) so a
    /// lambda isn't paid for via std::function's indirect call in the
    /// per-contribution fold -- a std::function<...> value still binds fine
    /// to the template parameter if that's what you have.
    using ReduceOp =
        std::function<PayloadT(const PayloadT &, const PayloadT &)>;

    /**
     * @brief Construct a resolver bound to @p transport.
     * @param transport Exchange mechanism to use in Resolve(); typically an
     * AlltoallvTransport or CrystalRouterTransport.
     */
    explicit EntityResolver(std::shared_ptr<ITransport> transport)
        : m_transport(std::move(transport)), m_rank(m_transport->Rank()),
          m_nproc(m_transport->Size())
    {
    }

    /**
     * @brief Register an entity this rank touches.
     * @param globalId Global entity id (vertex/edge/face, etc.).
     * @param value This rank's contribution for @p globalId.
     *
     * The (id, value) is packed straight into the per-rendezvous send
     * buffer, so nothing is buffered twice and no dense O(nproc) array is
     * ever allocated -- only the rendezvous ranks this rank actually
     * communicates with appear as keys. May be called more than once for
     * the same id (each call is a separate contribution).
     */
    void Register(int64_t globalId, const PayloadT &value)
    {
        std::vector<std::byte> &buf = m_reqBuckets[RendezvousRank(globalId)];
        detail::PackPod<int64_t>(buf, globalId);
        detail::PackPod<PayloadT>(buf, value);
    }

    /**
     * @brief Bulk registration of many ids sharing one payload -- the
     * common case for pure connectivity discovery, where the payload is
     * unused.
     * @tparam IdContainer Any iterable of integer ids (e.g.
     * `std::vector<int>`).
     * @param ids Ids to register, each contributing @p commonValue.
     * @param commonValue Value registered against every id in @p ids.
     *
     * Takes any iterable of ids you already have, so callers don't
     * hand-roll a per-element Register() loop.
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
     * @brief Best-effort hint that @p n more Register() calls are coming,
     * to reduce reallocations on large local meshes.
     * @param n Number of additional (id, value) pairs about to be
     * registered.
     *
     * Only the outer per-rendezvous-rank bucket map can usefully be
     * pre-sized here (bounded by `nproc` distinct keys) -- the byte buffer
     * *within* each bucket can't be, since which bucket a given id lands in
     * isn't known until Register() hashes it. Safe to skip: this only
     * avoids some outer-map rehashing, it never changes behaviour.
     */
    void Reserve(size_t n)
    {
        m_reqBuckets.reserve(std::min(n, static_cast<size_t>(m_nproc)));
    }

    /**
     * @brief Discover sharers and reduce values across all ranks for every
     * entity registered since construction or the last Clear().
     *
     * @param op Associative, commutative reduction folding two
     * contributions for the same entity id into one; applied over the
     * sharers' contributions in ascending-rank order (see the @note below),
     * so it should not depend on argument order for correctness but *can*
     * rely on that order being deterministic.
     *
     * After this call, GetResolved() and GetSharers() are valid for every
     * id this rank registered. Two Exchange() round trips are performed:
     * one to gather contributions at each entity's rendezvous rank, one to
     * ship the reduced value and sharer list back to every contributor.
     *
     * Consumes registrations: this drains m_reqBuckets, so a second
     * Resolve() call without an intervening Register() will find nothing to
     * do and return empty results. Register() again before every Resolve()
     * if you need to run it more than once (contrast with
     * SharedPayloadResolver::Resolve(), which keeps its registrations until
     * Clear() and so is safe to call repeatedly).
     */
    template <typename Op> void Resolve(Op &&op)
    {
        m_resolved.clear();
        m_sharers.clear();

        // Every key m_resolved/m_sharers ends up with comes from an id
        // *this* rank registered, so the number of entries currently in
        // m_reqBuckets is an exact upper bound on the final key count --
        // computed here, before DrainBucketsToMessages() empties
        // m_reqBuckets below (mirrors ConnectivityResolver::Resolve(),
        // which needs the same up-front scan for the same reason).
        {
            size_t localEntries = 0;
            for (const auto &kv : m_reqBuckets)
            {
                localEntries +=
                    kv.second.size() / (sizeof(int64_t) + sizeof(PayloadT));
            }
            m_resolved.reserve(localEntries);
            m_sharers.reserve(localEntries);
        }

        // Request phase: buckets were already filled during Register, so we
        // just hand them to the transport.
        std::vector<RoutedMessage> outbound =
            detail::DrainBucketsToMessages(m_reqBuckets, m_rank);

        std::vector<RoutedMessage> inbound;
        m_transport->Exchange(outbound, inbound);

        // Gather every contribution per id, unfolded -- the fold itself
        // happens below, after sorting, so it doesn't depend on the order
        // messages happened to arrive in (see the @note on Resolve()).
        struct Group
        {
            /// Every (source rank, contribution) pair seen for this id, in
            /// receive order (not yet sorted or folded).
            std::vector<std::pair<int, PayloadT>> contributions;
        };
        std::unordered_map<int64_t, Group> groups;

        // Pre-size `groups` from the inbound byte volume, exactly as
        // ConnectivityResolver::Resolve() does for its own gather map -- see
        // the comment there for why this matters at scale.
        {
            size_t totalEntries = 0;
            for (const auto &m : inbound)
            {
                totalEntries +=
                    m.bytes.size() / (sizeof(int64_t) + sizeof(PayloadT));
            }
            groups.reserve(totalEntries);
        }

        for (const auto &m : inbound)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id   = detail::UnpackPod<int64_t>(cur);
                const PayloadT val = detail::UnpackPod<PayloadT>(cur);
                groups[id].contributions.emplace_back(m.src, val);
            }
            ASSERTL1(cur == end,
                     "EntityResolver::Resolve: malformed request buffer "
                     "(record lengths didn't sum to the message size)");
        }

        // Reply phase: send (id, reduced, sharers) back to each sharer.
        // Sparse buckets keyed by sharer rank -- no dense O(nproc) array.
        std::unordered_map<int, std::vector<std::byte>> repBuckets;
        std::vector<std::byte> rec; // hoisted scratch buffer, reused per id
        for (auto &kv : groups)
        {
            const int64_t id = kv.first;
            std::vector<std::pair<int, PayloadT>> &contributions =
                kv.second.contributions;

            // Stable-sort by source rank before folding, so `op` always
            // combines a given set of contributions in the same relative
            // order regardless of which transport delivered inbound
            // messages in which order (AlltoallvTransport: ascending source
            // rank; CrystalRouterTransport: routing order) -- makes
            // Resolve() bitwise-reproducible for the same registrations
            // across transports, which matters for any floating-point fold.
            // Two contributions from the same rank (a locally-repeated
            // Register() for this id) keep their original relative order,
            // since stable_sort only reorders across distinct keys and that
            // relative order is itself already transport-independent (it's
            // fixed by the order Register() packed them, unpacked back in
            // the same order every time).
            std::stable_sort(
                contributions.begin(), contributions.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

            PayloadT reduced = contributions.front().second;
            for (size_t i = 1; i < contributions.size(); ++i)
            {
                reduced = op(reduced, contributions[i].second);
            }

            // Sharer list is rank *identity* for reply routing, not a fold
            // input, so -- unlike `contributions` above -- it's fine (and
            // cheaper on the wire, and avoids O(k^2) blowup when a rank
            // registers the same id many times) to dedupe: a rank that
            // registered twice still only needs telling once.
            std::vector<int> sharers;
            sharers.reserve(contributions.size());
            for (const auto &c : contributions)
            {
                sharers.push_back(c.first);
            }
            std::sort(sharers.begin(), sharers.end());
            sharers.erase(std::unique(sharers.begin(), sharers.end()),
                          sharers.end());

            rec.clear();
            detail::PackPod<int64_t>(rec, id);
            detail::PackPod<PayloadT>(rec, reduced);
            detail::PackPod<int32_t>(rec, static_cast<int32_t>(sharers.size()));
            for (int s : sharers)
            {
                detail::PackPod<int32_t>(rec, s);
            }

            for (int s : sharers)
            {
                std::vector<std::byte> &buf = repBuckets[s];
                buf.insert(buf.end(), rec.begin(), rec.end());
            }
        }

        std::vector<RoutedMessage> replyOut =
            detail::DrainBucketsToMessages(repBuckets, m_rank);

        std::vector<RoutedMessage> replyIn;
        m_transport->Exchange(replyOut, replyIn);

        // Finally, store resolved values and sharer lists.
        for (const auto &m : replyIn)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id    = detail::UnpackPod<int64_t>(cur);
                const PayloadT val  = detail::UnpackPod<PayloadT>(cur);
                const int32_t nShar = detail::UnpackPod<int32_t>(cur);

                std::vector<int> sharers;
                sharers.reserve(nShar);
                for (int32_t i = 0; i < nShar; ++i)
                {
                    sharers.push_back(detail::UnpackPod<int32_t>(cur));
                }
                m_resolved[id] = val;
                m_sharers[id]  = std::move(sharers);
            }
            ASSERTL1(cur == end,
                     "EntityResolver::Resolve: malformed reply buffer "
                     "(record lengths didn't sum to the message size)");
        }
    }

    /**
     * @brief Test whether Resolve() returned a result for @p globalId.
     * @param globalId Entity id to query.
     * @return `true` iff *this* rank registered @p globalId prior to the
     * most recent Resolve() call. Replies only ever reach contributors, so
     * this is never true for an id only some *other* rank registered.
     */
    bool Knows(int64_t globalId) const
    {
        return m_resolved.find(globalId) != m_resolved.end();
    }

    /**
     * @brief Single-lookup alternative to `Knows(id)` then `GetResolved(id)`,
     * which hashes @p globalId twice.
     * @param globalId Entity id to query.
     * @return Pointer to the resolved value if @p globalId was registered by
     * this rank prior to the most recent Resolve() call, else `nullptr`.
     * Invalidated by the next Resolve()/Clear() call, same as any reference
     * into the underlying map.
     */
    const PayloadT *FindResolved(int64_t globalId) const
    {
        auto it = m_resolved.find(globalId);
        return it == m_resolved.end() ? nullptr : &it->second;
    }

    /**
     * @brief Retrieve the reduced value for a resolved entity.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return The value obtained by folding every registered contribution
     * for @p globalId (across all sharers) through the `op` passed to
     * Resolve().
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     */
    const PayloadT &GetResolved(int64_t globalId) const
    {
        return m_resolved.at(globalId);
    }

    /**
     * @brief Retrieve the full sharer set for a resolved entity.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return Ranks that touch this entity, including this rank. Deduped:
     * a rank appears once even if it registered the same id more than once.
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     */
    const std::vector<int> &GetSharers(int64_t globalId) const
    {
        return m_sharers.at(globalId);
    }

    /**
     * @brief Single-lookup alternative to `Knows(id)` then `GetSharers(id)`.
     * @param globalId Entity id to query.
     * @return Pointer to the sharer list if @p globalId was registered by
     * this rank prior to the most recent Resolve() call, else `nullptr`.
     */
    const std::vector<int> *FindSharers(int64_t globalId) const
    {
        auto it = m_sharers.find(globalId);
        return it == m_sharers.end() ? nullptr : &it->second;
    }

    /**
     * @brief Direct read-only access to every resolved id's reduced value,
     * for callers that want to iterate everything this rank knows about
     * rather than querying id-by-id. Mirrors
     * SharedPayloadResolver::AllSharedPayloads().
     * @return The full map from resolved entity id to its reduced value.
     * Iteration order is unspecified.
     */
    const std::unordered_map<int64_t, PayloadT> &AllResolved() const
    {
        return m_resolved;
    }

    /**
     * @brief Direct read-only access to every resolved id's sharer list.
     * @return The full map from resolved entity id to its sharer-rank list.
     * Iteration order is unspecified.
     */
    const std::unordered_map<int64_t, std::vector<int>> &AllSharers() const
    {
        return m_sharers;
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
        return detail::BuildNeighbourComm(base, m_rank, m_sharers);
    }

    /// @brief Clear registrations and results, so the resolver can be reused
    /// for a fresh Register()/Resolve() cycle.
    void Clear()
    {
        m_reqBuckets.clear();
        m_resolved.clear();
        m_sharers.clear();
    }

private:
    /**
     * @brief Hash an entity id to its rendezvous rank.
     * @param id Global entity id.
     * @return The rendezvous rank for @p id (see #detail::RendezvousRank
     * for the bit-mixing splitmix64 finaliser used).
     */
    int RendezvousRank(int64_t id) const
    {
        return detail::RendezvousRank(id, m_nproc);
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

    /// Reduced value per resolved entity id, populated by Resolve().  Contains
    /// only entities this rank registered -- O(local), not O(N_global).
    std::unordered_map<int64_t, PayloadT> m_resolved;

    /// Sharer ranks per resolved entity id, populated by Resolve().  Contains
    /// only entities this rank registered -- O(local), not O(N_global).
    std::unordered_map<int64_t, std::vector<int>> m_sharers;
};

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
 * @brief The payload-free sibling of #EntityResolver, for the common case
 * where you only need to discover *which ranks share each entity* and not a
 * reduced value.
 *
 * @details
 * It ships only the entity id on the wire (no payload, no reduction op), so
 * both the request and reply are smaller, and the API is just
 * Register()/Resolve()/GetSharers(). Uses the same rendezvous algorithm and
 * O(local) memory guarantees as EntityResolver -- see the
 * @ref EntityResolver.hpp "file-level documentation".
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
     * See EntityResolver::Reserve() for the same caveat about only the
     * outer bucket map being pre-sizable here.
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
        m_sharers.clear();

        // Every key m_sharers ends up with comes from an id *this* rank
        // registered, so the number of ids currently in m_reqBuckets is an
        // exact upper bound (exact unless some id was Register()'d more than
        // once locally) on the final key count -- computed here, before
        // DrainBucketsToMessages() empties m_reqBuckets below.
        {
            size_t localEntries = 0;
            for (const auto &kv : m_reqBuckets)
            {
                localEntries += kv.second.size() / sizeof(int64_t);
            }
            m_sharers.reserve(localEntries);
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

        // At the rendezvous, group by id and collect sharers
        std::unordered_map<int64_t, std::vector<int>> groups;

        // Pre-size `groups` from the inbound byte volume rather than letting it
        // grow one insert at a time. Every request entry here is exactly one
        // int64_t id (no payload), so the entry count is exact, not an
        // estimate. This rank may be the rendezvous for entries aggregated from
        // every other rank registering ids that hash here, so without this,
        // `groups` can end up considerably larger than any single rank's own
        // local registrations and pay for repeated rehashing accordingly.
        size_t totalEntries = 0;
        for (const auto &m : inbound)
        {
            totalEntries += m.bytes.size() / sizeof(int64_t);
        }
        groups.reserve(totalEntries);

        for (const auto &m : inbound)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id = detail::UnpackPod<int64_t>(cur);
                groups[id].push_back(m.src);
            }
            ASSERTL1(cur == end,
                     "ConnectivityResolver::Resolve: malformed request "
                     "buffer (record lengths didn't sum to the message "
                     "size)");
        }

        // Dedupe each id's sharer list before replying: a rank that
        // registered the same id more than once currently appears once per
        // registration, which both inflates reply wire volume unnecessarily
        // and means every caller has to defensively dedupe GetSharers()
        // themselves. Rank *identity* is all a sharer list needs to convey.
        for (auto &kv : groups)
        {
            std::vector<int> &shs = kv.second;
            std::sort(shs.begin(), shs.end());
            shs.erase(std::unique(shs.begin(), shs.end()), shs.end());
        }

        // Reply phase: send (id, sharers) back to each sharer. Packed once
        // per id into a hoisted scratch buffer and memcpy'd into every
        // recipient's bucket, rather than re-running PackPod() once per
        // (id, sharer) pair.
        std::unordered_map<int, std::vector<std::byte>> repBuckets;
        std::vector<std::byte> rec;
        for (const auto &kv : groups)
        {
            const int64_t id            = kv.first;
            const std::vector<int> &shs = kv.second;

            rec.clear();
            detail::PackPod<int64_t>(rec, id);
            detail::PackPod<int32_t>(rec, static_cast<int32_t>(shs.size()));
            for (int other : shs)
            {
                detail::PackPod<int32_t>(rec, other);
            }

            for (int s : shs)
            {
                std::vector<std::byte> &buf = repBuckets[s];
                buf.insert(buf.end(), rec.begin(), rec.end());
            }
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

        // Store sharer lists
        for (const auto &m : replyIn)
        {
            const std::byte *cur = m.bytes.data();
            const std::byte *end = cur + m.bytes.size();
            while (cur < end)
            {
                const int64_t id    = detail::UnpackPod<int64_t>(cur);
                const int32_t nShar = detail::UnpackPod<int32_t>(cur);
                std::vector<int> sharers;
                sharers.reserve(nShar);
                for (int32_t i = 0; i < nShar; ++i)
                {
                    sharers.push_back(detail::UnpackPod<int32_t>(cur));
                }
                m_sharers[id] = std::move(sharers);
            }
            ASSERTL1(cur == end,
                     "ConnectivityResolver::Resolve: malformed reply buffer "
                     "(record lengths didn't sum to the message size)");
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
        return m_sharers.find(globalId) != m_sharers.end();
    }

    /**
     * @brief Single-lookup alternative to `Knows(id)` then `GetSharers(id)`.
     * @param globalId Entity id to query.
     * @return Pointer to the sharer list if @p globalId was registered by
     * this rank prior to the most recent Resolve() call, else `nullptr`.
     */
    const std::vector<int> *FindSharers(int64_t globalId) const
    {
        auto it = m_sharers.find(globalId);
        return it == m_sharers.end() ? nullptr : &it->second;
    }

    /**
     * @brief Direct read-only access to every resolved id's sharer list.
     * @return The full map from resolved entity id to its sharer-rank list.
     * Iteration order is unspecified.
     */
    const std::unordered_map<int64_t, std::vector<int>> &AllSharers() const
    {
        return m_sharers;
    }

    /**
     * @brief Retrieve the full sharer set for a resolved entity.
     * @param globalId Entity id to query; must satisfy `Knows(globalId)`.
     * @return Ranks that touch this entity, including this rank. Deduped: a
     * rank appears once even if it registered the same id more than once.
     * @throws std::out_of_range if @p globalId was not registered/resolved.
     */
    const std::vector<int> &GetSharers(int64_t globalId) const
    {
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
        return detail::BuildNeighbourComm(base, m_rank, m_sharers);
    }

    /// @brief Clear registrations and results, so the resolver can be
    /// reused for a fresh Register()/Resolve() cycle.
    void Clear()
    {
        m_reqBuckets.clear();
        m_sharers.clear();
    }

private:
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

    /// Sharer ranks per resolved entity id, populated by Resolve().  Contains
    /// only entities this rank registered -- O(local), not O(N_global).
    std::unordered_map<int64_t, std::vector<int>> m_sharers;
};

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
 * `std::is_trivially_copyable_v`, the same constraint as #EntityResolver.
 *
 * @details
 * Use this instead of #EntityResolver when there is no sensible way to fold
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
 *    request/reply pair EntityResolver needs for its rendezvous-mediated
 *    reduction.
 *
 * Net effect: PayloadT crosses the network exactly once per (contributor,
 * sharer) pair, on a direct hop, instead of twice via a third-party rank. This
 * can be important in cases where PayloadT is relatively large (typically 100s
 * of bytes).
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
     * Unlike EntityResolver::Reserve()/ConnectivityResolver::Reserve(),
     * this is a real, exact pre-size (not a best-effort hint bounded by
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
     * Does not consume registrations: m_localRegs is left intact, so
     * calling Resolve() again without an intervening Register() simply
     * re-resolves the same registrations and is safe (contrast with
     * EntityResolver::Resolve()/ConnectivityResolver::Resolve(), which
     * drain their request buckets and so return nothing on a second call
     * without re-registering).
     *
     * @param timings Optional out-parameter; if non-null, filled in with a
     * discovery-vs-payload-round wall-time split (see
     * SharedPayloadResolverTimings). Leave null for zero measurement
     * overhead -- this is a diagnostic hook, not something to leave wired
     * up in a hot production call site once you've got the number you need.
     */
    void Resolve(SharedPayloadResolverTimings *timings = nullptr)
    {
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

            for (int r : m_connResolver.GetSharers(id))
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
     * Returned by value, unlike EntityResolver::GetSharers(), since no
     * ranks-only list is stored separately.
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
        // The composed ConnectivityResolver already holds exactly this
        // neighbour set as a ranks-only view, so delegate rather than
        // re-deriving it from m_sharedPayloads.
        return m_connResolver.BuildNeighbourComm(base);
    }

    /// @brief Clear registrations and results, so the resolver can be
    /// reused for a fresh Register()/Resolve() cycle.
    void Clear()
    {
        m_localRegs.clear();
        m_connResolver.Clear();
        m_sharedPayloads.clear();
    }

private:
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
};

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIBUTILITIES_COMMUNICATION_ENTITYRESOLVER_HPP
