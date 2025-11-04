///////////////////////////////////////////////////////////////////////////////
//
// File: AssemblyCommCG.cpp
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

#include <unordered_set>

#include <MultiRegions/AssemblyMap/AssemblyCommCG.h>

namespace Nektar::MultiRegions
{

void AssemblyCommCG::BeginAssemble(const Array<OneD, double> &values)
{
    // Fill send buffer
    for (size_t i = 0; i < m_sr_entries.size(); ++i)
    {
        m_send_storage[i] = values[m_sr_entries[i]];
    }

    BeginComm();
}

// Wait for comms and accumulate into values vector
void AssemblyCommCG::EndAssemble(Array<OneD, double> &values)
{
    EndComm();

    // Accumulate
    for (size_t i = 0; i < m_sr_entries.size(); ++i)
    {
        values[m_sr_entries[i]] += m_recv_storage[i];
    }
}

void AssemblyCommCG::DiscoverSharedIDsRing()
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

    // Ring communication: send IDs in an all-to-all but without O(n) storage
    // cost.
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

void AssemblyCommCG::FillUniqueMap(Array<OneD, long> &unique)
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

void AssemblyCommCG::BuildSendRecvMaps()
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

    // Map that takes ranks to entries to send/receive. Basically the inverse of
    // m_uid_to_ranks.
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
        // Create a block that will eventually be sent to each rank from within
        // the send/receive storage.
        m_sr_blocks.push_back({rank, offset, indices.size()});

        // Sort indices to ensure same send/recv order for IDs, otherwise we
        // will mismatch.
        std::sort(indices.begin(), indices.end());

        // Note this is by reference to change the ultimate entry inside indices
        for (auto &idx : indices)
        {
            idx = m_uid_to_index[idx];
            m_sr_entries.push_back(idx);
            m_fromRank.push_back(rank);
        }

        offset += indices.size();
    }
}

void AssemblyCommCG::InitSendRecvComms(const unsigned nstorage,
                                       double *send_storage_ptr,
                                       double *recv_storage_ptr,
                                       const unsigned numComp)
{
    ASSERTL0(nstorage >= m_sr_blocks.size() * numComp,
             "Pointer size is not sufficent for send recieve storage");

    // Create send/recv requests
    m_send_reqs = m_comm->CreateRequest(m_sr_blocks.size());
    m_recv_reqs = m_comm->CreateRequest(m_sr_blocks.size());

    int reqCnt = 0;
    for (const auto &block : m_sr_blocks)
    {
        CommPtrView<double> send_ptr(send_storage_ptr + block.offset * numComp,
                                     block.count * numComp);
        CommPtrView<double> recv_ptr(recv_storage_ptr + block.offset * numComp,
                                     block.count * numComp);
        m_comm->SendInit(block.rank, send_ptr, block.count * numComp,
                         m_send_reqs, reqCnt);
        m_comm->RecvInit(block.rank, recv_ptr, block.count * numComp,
                         m_recv_reqs, reqCnt++);
    }
}

} // namespace Nektar::MultiRegions
