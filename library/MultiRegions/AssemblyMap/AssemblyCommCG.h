///////////////////////////////////////////////////////////////////////////////
//
// File: AssemblyCommCG.h
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

#ifndef NEKTAR_MULTIREGIONS_ASSEMBLYMAP_ASSEMBLYCOMMCG
#define NEKTAR_MULTIREGIONS_ASSEMBLYMAP_ASSEMBLYCOMMCG

#include <vector>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <LibUtilities/Communication/Comm.h>

namespace Nektar::MultiRegions
{
class AssemblyCommCG;
typedef std::shared_ptr<AssemblyCommCG> AssemblyCommCGSharedPtr;

class AssemblyCommCG
{
public:
    AssemblyCommCG(LibUtilities::CommSharedPtr comm,
                   const Array<OneD, long> &gids_to_uids)
        : m_comm(comm), m_gids_to_uids(gids_to_uids)
    {
        m_rank = comm->GetRank();
        m_size = comm->GetSize();

        DiscoverSharedIDsRing();
        BuildSendRecvMaps();
    }

    void InitSendRecvComms()
    {
        m_send_storage.resize(m_sr_entries.size());
        m_recv_storage.resize(m_sr_entries.size());
        InitSendRecvComms(m_sr_entries.size(), m_send_storage.data(),
                          m_recv_storage.data());
    }

    void InitSendRecvComms(const unsigned nstorage, double *send_storage_ptr,
                           double *recv_storage_ptr, const unsigned ncom = 1);

    const std::vector<unsigned> &GetFromRank() const
    {
        return m_fromRank;
    }

    const std::vector<size_t> &GetSREntries() const
    {
        return m_sr_entries;
    }

    void BeginComm()
    {
        ASSERTL1(
            m_recv_reqs,
            "InitSendRecvComm() must be called before BeginComm can be used");
        // Post receives
        m_comm->StartAll(m_recv_reqs);

        // Post sends
        m_comm->StartAll(m_send_reqs);
    }

    void EndComm()
    {
        ASSERTL1(
            m_send_reqs,
            "InitSendRecvComm() must be called before BeginComm can be used");

        // Wait for send-recv to finish
        m_comm->WaitAll(m_send_reqs);
        m_comm->WaitAll(m_recv_reqs);
    }

    void BeginAssemble(const Array<OneD, double> &values);
    void EndAssemble(Array<OneD, double> &values);
    void FillUniqueMap(Array<OneD, long> &unique);

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
    std::vector<double> m_send_storage;
    /// Recv storage if required
    std::vector<double> m_recv_storage;

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

    void DiscoverSharedIDsRing();
    void BuildSendRecvMaps();
};

} // namespace Nektar::MultiRegions

#endif
