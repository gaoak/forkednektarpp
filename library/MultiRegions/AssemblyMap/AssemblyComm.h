///////////////////////////////////////////////////////////////////////////////
//
// File: AssemblyComm.h
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
// Description: C0-continuous pairwise communication
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MULTIREGIONS_ASSEMBLYMAP_ASSEMBLYCOMM
#define NEKTAR_MULTIREGIONS_ASSEMBLYMAP_ASSEMBLYCOMM

#include <map>
#include <numeric>
#include <vector>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <LibUtilities/Communication/Comm.h>

namespace Nektar::MultiRegions
{

/**
 * @brief Persistent point-to-point exchange of shared data with neighbouring
 * ranks.
 *
 * Was AssemblyCommCG, and continuous assembly is still what the first
 * constructor serves: given a local-index-to-universal-id map it discovers
 * which ranks share each id, groups the shared entries by rank, and opens a
 * persistent send/receive request per neighbour so that a step reduces to
 * BeginComm() / EndComm() around whatever local work can overlap it.
 *
 * None of that machinery is specific to a continuous field, which is why the
 * name no longer says CG. A discontinuous trace flux shares the same shape -
 * a set of entries, each owned jointly with exactly one other rank, exchanged
 * every step - and reaches it through the second constructor, which takes
 * connectivity already discovered rather than rediscovering it here.
 *
 * ### Fixed and variable payloads
 *
 * The assembly path exchanges one value per entry, or @c numComp of them, and
 * uses the InitSendRecvComms() overloads taking a component count. A trace
 * carries a whole quadrature line or face per entry, and under variable
 * polynomial order the two sides of a trace need not even carry the same
 * number of points, so the payload is neither one value nor a fixed multiple
 * of one. That case uses InitSendRecvCommsVar(), which takes an explicit
 * element count per entry for each direction and lays the buffers out as
 * their running sums; GetSendOffsets() and GetRecvOffsets() then say where
 * each entry's payload begins.
 */
template <typename TData> class AssemblyComm
{
public:
    AssemblyComm(LibUtilities::CommSharedPtr comm,
                 const Array<OneD, long> &gids_to_uids)
        : m_comm(comm), m_gids_to_uids(gids_to_uids)
    {
        m_rank = comm->GetRank();
        m_size = comm->GetSize();

        DiscoverSharedIDsRing();
        BuildSendRecvMaps();
    }

    /**
     * @brief Construct from connectivity discovered elsewhere.
     *
     * The other constructor finds its neighbours by passing every local id
     * round a ring of all P ranks. A caller that already knows who its
     * neighbours are - having resolved them through
     * LibUtilities::SharedPayloadResolver, say, which answers in two
     * rendezvous exchanges rather than P - has no reason to pay for that
     * again, and hands the answer in here instead.
     *
     * @param comm          Communicator to exchange over.
     * @param sharedEntries Local entry indices to exchange with each
     *                      neighbouring rank. Both ends of every pair must
     *                      list their entries in a matching order, since it
     *                      is position within the list, and nothing else,
     *                      that pairs a sent entry with a received one. Order
     *                      by an id both ranks agree on to get that.
     */
    AssemblyComm(LibUtilities::CommSharedPtr comm,
                 const std::map<int, std::vector<size_t>> &sharedEntries)
        : m_comm(comm)
    {
        m_rank = comm->GetRank();
        m_size = comm->GetSize();

        size_t offset = 0;
        for (const auto &[rank, entries] : sharedEntries)
        {
            m_sr_blocks.push_back({rank, offset, entries.size()});

            for (auto idx : entries)
            {
                m_sr_entries.push_back(idx);
                m_fromRank.push_back(rank);
            }

            offset += entries.size();
        }
    }

    void InitSendRecvComms()
    {
        m_send_storage.resize(m_sr_entries.size());
        m_recv_storage.resize(m_sr_entries.size());
        InitSendRecvComms(m_sr_entries.size(), m_send_storage.data(),
                          m_recv_storage.data());
    }

    void InitSendRecvComms(const unsigned nstorage, TData *send_storage_ptr,
                           TData *recv_storage_ptr, const unsigned numComp = 1)
    {
        ASSERTL0(nstorage >= m_sr_blocks.size() * numComp,
                 "Pointer size is not sufficent for send recieve storage");

        // Create send/recv requests
        m_send_reqs = m_comm->CreateRequest(m_sr_blocks.size());
        m_recv_reqs = m_comm->CreateRequest(m_sr_blocks.size());

        int reqCnt = 0;
        for (const auto &block : m_sr_blocks)
        {
            CommPtrView<TData> send_ptr(send_storage_ptr +
                                            block.offset * numComp,
                                        block.count * numComp);
            CommPtrView<TData> recv_ptr(recv_storage_ptr +
                                            block.offset * numComp,
                                        block.count * numComp);
            m_comm->SendInit(block.rank, send_ptr, block.count * numComp,
                             m_send_reqs, reqCnt);
            m_comm->RecvInit(block.rank, recv_ptr, block.count * numComp,
                             m_recv_reqs, reqCnt++);
        }
    }

    /**
     * @brief Open persistent requests for payloads that differ in size from
     * entry to entry.
     *
     * @param send_storage_ptr Send buffer, laid out as the running sum of
     *                         @p sendCounts.
     * @param sendCounts       Elements this rank sends for each entry, in
     *                         GetSREntries() order.
     * @param recv_storage_ptr Receive buffer, laid out as the running sum of
     *                         @p recvCounts.
     * @param recvCounts       Elements this rank receives for each entry, in
     *                         the same order. Equal to the neighbour's send
     *                         count for the matching entry, which need not be
     *                         this rank's - the two sides of a trace may
     *                         carry different numbers of points.
     */
    void InitSendRecvCommsVar(TData *send_storage_ptr,
                              const std::vector<size_t> &sendCounts,
                              TData *recv_storage_ptr,
                              const std::vector<size_t> &recvCounts)
    {
        ASSERTL0(sendCounts.size() == m_sr_entries.size() &&
                     recvCounts.size() == m_sr_entries.size(),
                 "One send and one receive count is needed per shared entry");

        BuildOffsets(sendCounts, m_send_offsets);
        BuildOffsets(recvCounts, m_recv_offsets);

        m_send_reqs = m_comm->CreateRequest(m_sr_blocks.size());
        m_recv_reqs = m_comm->CreateRequest(m_sr_blocks.size());

        int reqCnt = 0;
        for (const auto &block : m_sr_blocks)
        {
            // A block's entries are contiguous in GetSREntries(), so its
            // payload is contiguous in the buffer too, and one message
            // carries the whole block however its entries are sized.
            const size_t sendBeg = m_send_offsets[block.offset];
            const size_t sendEnd = m_send_offsets[block.offset + block.count];
            const size_t recvBeg = m_recv_offsets[block.offset];
            const size_t recvEnd = m_recv_offsets[block.offset + block.count];

            CommPtrView<TData> send_ptr(send_storage_ptr + sendBeg,
                                        sendEnd - sendBeg);
            CommPtrView<TData> recv_ptr(recv_storage_ptr + recvBeg,
                                        recvEnd - recvBeg);

            m_comm->SendInit(block.rank, send_ptr, sendEnd - sendBeg,
                             m_send_reqs, reqCnt);
            m_comm->RecvInit(block.rank, recv_ptr, recvEnd - recvBeg,
                             m_recv_reqs, reqCnt++);
        }
    }

    /// Start of each entry's payload in the send buffer, with a final entry
    /// holding the total. Valid after InitSendRecvCommsVar().
    const std::vector<size_t> &GetSendOffsets() const
    {
        return m_send_offsets;
    }

    /// Start of each entry's payload in the receive buffer; see
    /// GetSendOffsets().
    const std::vector<size_t> &GetRecvOffsets() const
    {
        return m_recv_offsets;
    }

    const std::vector<unsigned> &GetFromRank() const
    {
        return m_fromRank;
    }

    const std::vector<size_t> &GetSREntries() const
    {
        return m_sr_entries;
    }

    inline void BeginComm()
    {
        ASSERTL1(
            m_recv_reqs,
            "InitSendRecvComm() must be called before BeginComm can be used");
        // Post receives
        m_comm->StartAll(m_recv_reqs);

        // Post sends
        m_comm->StartAll(m_send_reqs);
    }

    inline void EndComm()
    {
        ASSERTL1(
            m_send_reqs,
            "InitSendRecvComm() must be called before BeginComm can be used");

        // Wait for send-recv to finish
        m_comm->WaitAll(m_send_reqs);
        m_comm->WaitAll(m_recv_reqs);
    }

    void EndAssemble(Array<OneD, TData> &values)
    {
        EndComm();

        // Accumulate
        for (size_t i = 0; i < m_sr_entries.size(); ++i)
        {
            values[m_sr_entries[i]] += m_recv_storage[i];
        }
    }

    void BeginAssemble(const Array<OneD, TData> &values)
    {
        // Fill send buffer
        for (size_t i = 0; i < m_sr_entries.size(); ++i)
        {
            m_send_storage[i] = values[m_sr_entries[i]];
        }

        BeginComm();
    }

    void FillUniqueMap(Array<OneD, long> &unique)
    {
        ASSERTL0(unique.size() == m_gids_to_uids.size(), "wut");

        int nShared = 0;

        // Mask out entries we do not own
        for (const auto &[uid, ranks] : m_uid_to_ranks)
        {
            // Disregard any uids we solely own
            if (ranks.size() == 1)
            {
                continue;
            }

            int owner = *std::min_element(ranks.begin(), ranks.end());
            if (owner == m_rank)
            {
                nShared++;
            }
            else
            {
                auto idx    = m_uid_to_index[uid];
                unique[idx] = -unique[idx];
            }
        }
        m_comm->AllReduce(nShared, LibUtilities::ReduceSum);
    }

private:
    struct SendRecvBlock
    {
        int rank;
        size_t offset;
        size_t count;
    };

    /// List of entries that map local indices to send/recv storage
    std::vector<size_t> m_sr_entries;
    /// rank where assembled point has been communciated from
    std::vector<unsigned> m_fromRank;
    std::vector<SendRecvBlock> m_sr_blocks;

    /// Send storage if required
    std::vector<TData> m_send_storage;
    /// Recv storage if required
    std::vector<TData> m_recv_storage;

    /// Running sums of the per-entry payload sizes, when those vary; see
    /// InitSendRecvCommsVar(). Both hold one more element than there are
    /// entries, the last being the buffer total.
    std::vector<size_t> m_send_offsets;
    std::vector<size_t> m_recv_offsets;

    /// Turn per-entry counts into the running sum the buffers are laid out by.
    static void BuildOffsets(const std::vector<size_t> &counts,
                             std::vector<size_t> &offsets)
    {
        offsets.resize(counts.size() + 1);
        offsets[0] = 0;
        std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    }

    /// Send requests
    LibUtilities::CommRequestSharedPtr m_send_reqs = nullptr;
    /// Recv requests
    LibUtilities::CommRequestSharedPtr m_recv_reqs = nullptr;

    LibUtilities::CommSharedPtr m_comm;
    unsigned m_rank;
    unsigned m_size;

    /// List of local IDs, where each entry denotes a potential shared dof. 0
    /// entry denotes local to this process, no communication required.
    Array<OneD, long> m_gids_to_uids;
    /// Map each ID to a vector of ranks that own this ID
    std::unordered_map<long, std::vector<int>> m_uid_to_ranks;
    /// Map each ID to the index inside m_local_ids where it lives
    std::unordered_map<long, size_t> m_uid_to_index;

    inline void DiscoverSharedIDsRing()
    {
        // Create a set of IDs that lie on this process for fast ID lookups
        std::unordered_set<long> gid_to_uid_set(m_gids_to_uids.begin(),
                                                m_gids_to_uids.end());

        int nNonZero = 0;

        for (auto uid : m_gids_to_uids)
        {
            // Zero IDs are masked out from communication
            if (uid == 0)
            {
                continue;
            }

            // always shared with self
            m_uid_to_ranks[uid].push_back(m_rank);

            // count number of non-zero entries in m_local_ids
            nNonZero++;
        }

        Array<OneD, long> send_uids(nNonZero), recv_uids;

        // Isolate all non-zero ids
        for (size_t cnt = 0, i = 0; i < m_gids_to_uids.size(); ++i)
        {
            if (m_gids_to_uids[i] == 0)
            {
                continue;
            }

            send_uids[cnt++] = m_gids_to_uids[i];
        }

        // Ring communication: send IDs in an all-to-all but without O(n)
        // storage cost.
        for (int k = 1; k < m_size; ++k)
        {
            int send_rank = (m_rank + k) % m_size;
            int recv_rank = (m_rank - k + m_size) % m_size;

            // How many dofs do we need to send/receive?
            int send_count = send_uids.size();
            int recv_count = 0;

            // Communicate number of entries to send/receive
            m_comm->SendRecv(send_rank, send_count, recv_rank, recv_count);
            recv_uids = Array<OneD, long>(recv_count);

            // Do the actual send/receive
            m_comm->SendRecv(send_rank, send_uids, recv_rank, recv_uids);

            for (long uid : recv_uids)
            {
                if (gid_to_uid_set.find(uid) != gid_to_uid_set.end())
                {
                    m_uid_to_ranks[uid].push_back(recv_rank);
                }
            }
        }

        // Deduplicate, so that rank list is unique for each ID
        for (auto &[uid, ranks] : m_uid_to_ranks)
        {
            std::sort(ranks.begin(), ranks.end());
            ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
        }

#if 0
    // debug print
    for (auto &[uid, ranks] : m_uid_to_ranks)
    {
        if (ranks.size() == 1)
        {
            continue;
        }
        std::cout << "[RANK " << m_rank << "]: uid " << uid << " ->";
        for (auto &rank : ranks)
        {
            std::cout << " " << rank;
        }
        std::cout << std::endl;
    }
#endif
    }

    inline void BuildSendRecvMaps()
    {
        // Set up map of local IDs -> location inside m_local_ids
        for (size_t i = 0; i < m_gids_to_uids.size(); ++i)
        {
            if (m_gids_to_uids[i] == 0)
            {
                continue;
            }

            m_uid_to_index[m_gids_to_uids[i]] = i;
        }

        // Map that takes ranks to entries to send/receive. Basically the
        // inverse of m_uid_to_ranks.
        std::unordered_map<int, std::vector<size_t>> sr_map;

        for (const auto &[uid, ranks] : m_uid_to_ranks)
        {
            // Loop over ranks, and sort out list of IDs to send/receive
            for (int r : ranks)
            {
                if (r == m_rank)
                {
                    continue;
                }

                sr_map[r].push_back(uid);
            }
        }

        // Flatten send/recv blocks and create entries
        size_t offset = 0;
        for (auto &[rank, indices] : sr_map)
        {
            // Create a block that will eventually be sent to each rank from
            // within the send/receive storage.
            m_sr_blocks.push_back({rank, offset, indices.size()});

            // Sort indices to ensure same send/recv order for IDs, otherwise we
            // will mismatch.
            std::sort(indices.begin(), indices.end());

            // Note this is by reference to change the ultimate entry inside
            // indices
            for (auto &idx : indices)
            {
                idx = m_uid_to_index[idx];
                m_sr_entries.push_back(idx);
                m_fromRank.push_back(rank);
            }

            offset += indices.size();
        }
    }
};

} // namespace Nektar::MultiRegions

#endif
