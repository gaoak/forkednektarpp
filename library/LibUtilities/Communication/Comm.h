///////////////////////////////////////////////////////////////////////////////
//
// File: Comm.h
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
// Description: Base communication class
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_COMM_H
#define NEKTAR_LIB_UTILITIES_COMM_H

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/LibUtilitiesDeclspec.h>

#if defined(NEKTAR_BUILD_REDESIGN)
#include <Operators/Common/Spaces.hpp>
#include <Operators/Field/MemoryRegion.hpp>
#endif
#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Communication/CommDataType.h>

namespace Nektar::LibUtilities
{

// Forward declarations
class Comm;

/// Pointer to a Communicator object.
typedef std::shared_ptr<Comm> CommSharedPtr;

/// Datatype of the NekFactory used to instantiate classes derived from
/// the EquationSystem class.
typedef LibUtilities::NekFactory<std::string, Comm, int, char **> CommFactory;

LIB_UTILITIES_EXPORT CommFactory &GetCommFactory();

/// Type of operation to perform in AllReduce.
enum ReduceOperator
{
    ReduceSum,
    ReduceMax,
    ReduceMin,
    SIZE_ReduceOperator
};

const char *const ReduceOperatorMap[] = {"ReduceSum", "ReduceMax", "ReduceMin"};

/// Class for communicator request type
class CommRequest
{
public:
    /// Default constructor
    CommRequest() = default;
    /// Default destructor
    virtual ~CommRequest() = default;
};

typedef std::shared_ptr<CommRequest> CommRequestSharedPtr;

/// Base communications class
class Comm : public std::enable_shared_from_this<Comm>
{
public:
    LIB_UTILITIES_EXPORT Comm(int narg, char *arg[]);
    LIB_UTILITIES_EXPORT virtual ~Comm();

    LIB_UTILITIES_EXPORT inline void Finalise();

    LIB_UTILITIES_EXPORT inline bool IsGPUAware(void)
    {
        return m_gpu_aware;
    }

    /// Returns number of processes
    LIB_UTILITIES_EXPORT inline int GetSize() const;
    LIB_UTILITIES_EXPORT inline int GetRank();
    LIB_UTILITIES_EXPORT inline const std::string &GetType() const;

    LIB_UTILITIES_EXPORT inline bool TreatAsRankZero();
    LIB_UTILITIES_EXPORT inline bool IsSerial();
    LIB_UTILITIES_EXPORT inline bool IsParallelInTime();
    LIB_UTILITIES_EXPORT inline std::tuple<int, int, int> GetVersion();

    /// Block execution until all processes reach this point
    LIB_UTILITIES_EXPORT inline void Block();

    /// Return the time in seconds
    LIB_UTILITIES_EXPORT inline NekDouble Wtime();

    template <class T> void Send(int pProc, T &pData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void Send(int pProc, Operators::MemoryRegion<T> &pData);
#endif

    template <class T> void Recv(int pProc, T &pData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void Recv(int pProc, Operators::MemoryRegion<T> &pData);
#endif

    template <class T>
    void SendRecv(int pSendProc, T &pSendData, int pRecvProc, T &pRecvData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void SendRecv(int pSendProc, Operators::MemoryRegion<T> &pSendData,
                  int pRecvProc, Operators::MemoryRegion<T> &pRecvData);
#endif

    template <class T> void AllReduce(T &pData, enum ReduceOperator pOp);
    template <class T>
    void AllReduceBegin(T &pData, enum ReduceOperator pOp,
                        CommRequestSharedPtr request);
    template <class T>
    void AllReduceEnd(T &pData, CommRequestSharedPtr request);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AllReduce(Operators::MemoryRegion<T> &pData, enum ReduceOperator pOp);
    template <class MemSpace, class T>
    void AllReduceBegin(Operators::MemoryRegion<T> &pData,
                        enum ReduceOperator pOp, CommRequestSharedPtr request);
    template <class MemSpace, class T>
    void AllReduceEnd(Operators::MemoryRegion<T> &pData,
                      CommRequestSharedPtr request);
#endif

    template <class T> void AlltoAll(T &pSendData, T &pRecvData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AlltoAll(Operators::MemoryRegion<T> &pSendData,
                  Operators::MemoryRegion<T> &pRecvData);
#endif
    template <class T1, class T2>
    void AlltoAllv(T1 &pSendData, T2 &pSendDataSizeMap, T2 &pSendDataOffsetMap,
                   T1 &pRecvData, T2 &pRecvDataSizeMap, T2 &pRecvDataOffsetMap);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AlltoAllv(Operators::MemoryRegion<T> &pSendData,
                   Operators::MemoryRegion<int> &pSendDataSizeMap,
                   Operators::MemoryRegion<int> &pSendDataOffsetMap,
                   Operators::MemoryRegion<T> &pRecvData,
                   Operators::MemoryRegion<int> &pRecvDataSizeMap,
                   Operators::MemoryRegion<int> &pRecvDataOffsetMap);
#endif

    template <class T> void AllGather(T &pSendData, T &pRecvData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AllGather(Operators::MemoryRegion<T> &pSendData,
                   Operators::MemoryRegion<T> &pRecvData);
#endif
    template <class T>
    void AllGatherv(T &pSendData, T &pRecvData,
                    Array<OneD, int> &pRecvDataSizeMap,
                    Array<OneD, int> &pRecvDataOffsetMap);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AllGatherv(Operators::MemoryRegion<T> &pSendData,
                    Operators::MemoryRegion<T> &pRecvData,
                    Operators::MemoryRegion<int> &pRecvDataSizeMap,
                    Operators::MemoryRegion<int> &pRecvDataOffsetMap);
#endif
    template <class T>
    void AllGatherv(T &pRecvData, Array<OneD, int> &pRecvDataSizeMap,
                    Array<OneD, int> &pRecvDataOffsetMap);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void AllGatherv(Operators::MemoryRegion<T> &pRecvData,
                    Operators::MemoryRegion<int> &pRecvDataSizeMap,
                    Operators::MemoryRegion<int> &pRecvDataOffsetMap);
#endif

    template <class T> void Bcast(T &pData, int pRoot);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void Bcast(Operators::MemoryRegion<T> &pData, int pRoot);
#endif

    template <class T> T Gather(int rootProc, T &val);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    Operators::MemoryRegion<T> Gather(int rootProc,
                                      Operators::MemoryRegion<T> &val);
#endif

    template <class T> T Scatter(int rootProc, T &pData);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    Operators::MemoryRegion<T> Scatter(int rootProc,
                                       Operators::MemoryRegion<T> &pData);
#endif

    template <class T>
    void DistGraphCreateAdjacent(T &sources, T &sourceweights, int reorder);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void DistGraphCreateAdjacent(Operators::MemoryRegion<T> &sources,
                                 Operators::MemoryRegion<T> &sourceweights,
                                 int reorder);
#endif

    template <class T1, class T2>
    void NeighborAlltoAllv(T1 &pSendData, T2 &pSendDataSizeMap,
                           T2 &pSendDataOffsetMap, T1 &pRecvData,
                           T2 &pRecvDataSizeMap, T2 &pRecvDataOffsetMap);
#if defined(NEKTAR_BUILD_REDESIGN)
    template <class MemSpace, class T>
    void NeighborAlltoAllv(Operators::MemoryRegion<T> &pSendData,
                           Operators::MemoryRegion<int> &pSendDataSizeMap,
                           Operators::MemoryRegion<int> &pSendDataOffsetMap,
                           Operators::MemoryRegion<T> &pRecvData,
                           Operators::MemoryRegion<int> &pRecvDataSizeMap,
                           Operators::MemoryRegion<int> &pRecvDataOffsetMap);
#endif

    template <class T>
    void Irsend(int pProc, T &pData, int count,
                const CommRequestSharedPtr &request, int loc);
    template <class T>
    void Isend(int pProc, T &pData, int count,
               const CommRequestSharedPtr &request, int loc);
    template <class T>
    void SendInit(int pProc, T &pData, int count,
                  const CommRequestSharedPtr &request, int loc);
    template <class T>
    void Irecv(int pProc, T &pData, int count,
               const CommRequestSharedPtr &request, int loc);
    template <class T>
    void RecvInit(int pProc, T &pData, int count,
                  const CommRequestSharedPtr &request, int loc);

    inline void StartAll(const CommRequestSharedPtr &request);
    inline void WaitAll(const CommRequestSharedPtr &request);

    inline CommRequestSharedPtr CreateRequest(int num);
    LIB_UTILITIES_EXPORT inline CommSharedPtr CommCreateIf(int flag);
    LIB_UTILITIES_EXPORT inline void SplitComm(int pRows, int pColumns,
                                               int pTime = 0);
    LIB_UTILITIES_EXPORT inline CommSharedPtr GetRowComm();
    LIB_UTILITIES_EXPORT inline CommSharedPtr GetColumnComm();
    LIB_UTILITIES_EXPORT inline CommSharedPtr GetTimeComm();
    LIB_UTILITIES_EXPORT inline CommSharedPtr GetSpaceComm();
    LIB_UTILITIES_EXPORT inline bool RemoveExistingFiles();
    LIB_UTILITIES_EXPORT inline std::pair<CommSharedPtr, CommSharedPtr>
    SplitCommNode();

protected:
    bool m_gpu_aware = false;   ///< Flag for GPU-aware MPI
    int m_size;                 ///< Number of processes
    std::string m_type;         ///< Type of communication
    CommSharedPtr m_commRow;    ///< Row communicator
    CommSharedPtr m_commColumn; ///< Column communicator
    CommSharedPtr m_commTime;
    CommSharedPtr m_commSpace;

    Comm();

    virtual void v_Finalise()                                               = 0;
    virtual int v_GetRank()                                                 = 0;
    virtual bool v_TreatAsRankZero()                                        = 0;
    virtual bool v_IsSerial()                                               = 0;
    virtual std::tuple<int, int, int> v_GetVersion()                        = 0;
    virtual void v_Block()                                                  = 0;
    virtual NekDouble v_Wtime()                                             = 0;
    virtual void v_Send(const void *buf, int count, CommDataType dt,
                        int dest)                                           = 0;
    virtual void v_Recv(void *buf, int count, CommDataType dt, int source)  = 0;
    virtual void v_SendRecv(const void *sendbuf, int sendcount,
                            CommDataType sendtype, int dest, void *recvbuf,
                            int recvcount, CommDataType recvtype,
                            int source)                                     = 0;
    virtual void v_AllReduce(void *buf, int count, CommDataType dt,
                             enum ReduceOperator pOp)                       = 0;
    virtual void v_AllReduceBegin(void *buf, int count, CommDataType dt,
                                  enum ReduceOperator pOp,
                                  CommRequestSharedPtr request)             = 0;
    virtual void v_AllReduceEnd(CommRequestSharedPtr request)               = 0;
    virtual void v_AlltoAll(const void *sendbuf, int sendcount,
                            CommDataType sendtype, void *recvbuf, int recvcount,
                            CommDataType recvtype)                          = 0;
    virtual void v_AlltoAllv(const void *sendbuf, const int *sendcounts,
                             const int *senddispls, CommDataType sendtype,
                             void *recvbuf, const int *recvcounts,
                             const int *recvdispls, CommDataType recvtype)  = 0;
    virtual void v_AllGather(const void *sendbuf, int sendcount,
                             CommDataType sendtype, void *recvbuf,
                             int recvcount, CommDataType recvtype)          = 0;
    virtual void v_AllGatherv(const void *sendbuf, int sendcount,
                              CommDataType sendtype, void *recvbuf,
                              const int *recvcounts, const int *recvdispls,
                              CommDataType recvtype)                        = 0;
    virtual void v_AllGatherv(void *recvbuf, const int *recvcounts,
                              const int *recvdispls, CommDataType recvtype) = 0;
    virtual void v_Bcast(void *buffer, int count, CommDataType dt,
                         int root)                                          = 0;
    virtual void v_Gather(const void *sendbuf, int sendcount,
                          CommDataType sendtype, void *recvbuf, int recvcount,
                          CommDataType recvtype, int root)                  = 0;
    virtual void v_Scatter(const void *sendbuf, int sendcount,
                           CommDataType sendtype, void *recvbuf, int recvcount,
                           CommDataType recvtype, int root)                 = 0;

    virtual void v_DistGraphCreateAdjacent(int indegree, const int *sources,
                                           const int *sourceweights,
                                           int reorder) = 0;

    virtual void v_NeighborAlltoAllv(const void *sendbuf, const int *sendcounts,
                                     const int *senddispls,
                                     CommDataType sendtype, void *recvbuf,
                                     const int *recvcounts,
                                     const int *recvdispls,
                                     CommDataType recvtype) = 0;

    virtual void v_Irsend(const void *buf, int count, CommDataType dt, int dest,
                          CommRequestSharedPtr request, int loc)   = 0;
    virtual void v_Isend(const void *buf, int count, CommDataType dt, int dest,
                         CommRequestSharedPtr request, int loc)    = 0;
    virtual void v_SendInit(const void *buf, int count, CommDataType dt,
                            int dest, CommRequestSharedPtr request,
                            int loc)                               = 0;
    virtual void v_Irecv(void *buf, int count, CommDataType dt, int source,
                         CommRequestSharedPtr request, int loc)    = 0;
    virtual void v_RecvInit(void *buf, int count, CommDataType dt, int source,
                            CommRequestSharedPtr request, int loc) = 0;
    virtual void v_StartAll(CommRequestSharedPtr request)          = 0;
    virtual void v_WaitAll(CommRequestSharedPtr request)           = 0;
    virtual CommRequestSharedPtr v_CreateRequest(int num)          = 0;

    virtual void v_SplitComm(int pRows, int pColumns, int pTime) = 0;

    virtual CommSharedPtr v_CommCreateIf(int flag) = 0;
    LIB_UTILITIES_EXPORT virtual std::pair<CommSharedPtr, CommSharedPtr>
    v_SplitCommNode();
};

/**
 *
 */
inline void Comm::Finalise()
{
    v_Finalise();
}

/**
 *
 */
inline int Comm::GetSize() const
{
    return m_size;
}

/**
 *
 */
inline int Comm::GetRank()
{
    return v_GetRank();
}

/**
 *
 */
inline const std::string &Comm::GetType() const
{
    return m_type;
}

/**
 *
 */
inline bool Comm::TreatAsRankZero()
{
    return v_TreatAsRankZero();
}

/**
 *
 */
inline bool Comm::IsSerial()
{
    return v_IsSerial();
}

/**
 *
 */
inline bool Comm::IsParallelInTime()
{
    return m_commTime.get();
}

/**
 * @return tuple of {major, minor, patch} version numbers
 */
inline std::tuple<int, int, int> Comm::GetVersion()
{
    return v_GetVersion();
}

/**
 *
 */
inline void Comm::Block()
{
    v_Block();
}

/**
 *
 */
inline double Comm::Wtime()
{
    return v_Wtime();
}

/**
 *
 */
template <class T> void Comm::Send(int pProc, T &pData)
{
    v_Send(CommDataTypeTraits<T>::GetPointer(pData),
           CommDataTypeTraits<T>::GetCount(pData),
           CommDataTypeTraits<T>::GetDataType(), pProc);
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::Send(int pProc, Operators::MemoryRegion<T> &pData)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_Send(pData.template GetPtr<MemSpace, ReadOnly>(), pData.size(),
               CommDataTypeTraits<T>::GetDataType(), pProc);
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_Send(pData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
               pData.size(), CommDataTypeTraits<T>::GetDataType(), pProc);
    }
}
#endif

/**
 *
 */
template <class T> void Comm::Recv(int pProc, T &pData)
{
    v_Recv(CommDataTypeTraits<T>::GetPointer(pData),
           CommDataTypeTraits<T>::GetCount(pData),
           CommDataTypeTraits<T>::GetDataType(), pProc);
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::Recv(int pProc, Operators::MemoryRegion<T> &pData)
{
    if (m_gpu_aware)
    {
        v_Recv(pData.template GetPtr<MemSpace, WriteOnly>(), pData.size(),
               CommDataTypeTraits<T>::GetDataType(), pProc);
    }
    else
    {
        // MPI is NOT GPU-aware, data is received from the host.
        v_Recv(pData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
               pData.size(), CommDataTypeTraits<T>::GetDataType(), pProc);
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T>
void Comm::SendRecv(int pSendProc, T &pSendData, int pRecvProc, T &pRecvData)
{
    v_SendRecv(CommDataTypeTraits<T>::GetPointer(pSendData),
               CommDataTypeTraits<T>::GetCount(pSendData),
               CommDataTypeTraits<T>::GetDataType(), pSendProc,
               CommDataTypeTraits<T>::GetPointer(pRecvData),
               CommDataTypeTraits<T>::GetCount(pRecvData),
               CommDataTypeTraits<T>::GetDataType(), pRecvProc);
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::SendRecv(int pSendProc, Operators::MemoryRegion<T> &pSendData,
                    int pRecvProc, Operators::MemoryRegion<T> &pRecvData)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_SendRecv(pSendData.template GetPtr<MemSpace, ReadOnly>(),
                   pSendData.size(), CommDataTypeTraits<T>::GetDataType(),
                   pSendProc, pRecvData.template GetPtr<MemSpace, WriteOnly>(),
                   pRecvData.size(), CommDataTypeTraits<T>::GetDataType(),
                   pRecvProc);
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_SendRecv(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pSendData.size(), CommDataTypeTraits<T>::GetDataType(), pSendProc,
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            pRecvData.size(), CommDataTypeTraits<T>::GetDataType(), pRecvProc);
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T> void Comm::AllReduce(T &pData, enum ReduceOperator pOp)
{
    v_AllReduce(CommDataTypeTraits<T>::GetPointer(pData),
                CommDataTypeTraits<T>::GetCount(pData),
                CommDataTypeTraits<T>::GetDataType(), pOp);
}

/**
 *
 */
template <class T>
void Comm::AllReduceBegin(T &pData, enum ReduceOperator pOp,
                          CommRequestSharedPtr request)
{
    v_AllReduceBegin(CommDataTypeTraits<T>::GetPointer(pData),
                     CommDataTypeTraits<T>::GetCount(pData),
                     CommDataTypeTraits<T>::GetDataType(), pOp, request);
}

/**
 *
 */
template <class T>
void Comm::AllReduceEnd([[maybe_unused]] T &pData, CommRequestSharedPtr request)
{
    v_AllReduceEnd(request);
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AllReduce(Operators::MemoryRegion<T> &pData, enum ReduceOperator pOp)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AllReduce(pData.template GetPtr<MemSpace, ReadWrite>(), pData.size(),
                    CommDataTypeTraits<T>::GetDataType(), pOp);
    }
    else
    {
        v_AllReduce(pData.template GetPtr<NektarSpaces::HostSpace, ReadWrite>(),
                    pData.size(), CommDataTypeTraits<T>::GetDataType(), pOp);
    }
}

/**
 *
 */
template <class MemSpace, class T>
void Comm::AllReduceBegin(Operators::MemoryRegion<T> &pData,
                          enum ReduceOperator pOp, CommRequestSharedPtr request)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AllReduceBegin(pData.template GetPtr<MemSpace, ReadWrite>(),
                         pData.size(), CommDataTypeTraits<T>::GetDataType(),
                         pOp, request);
    }
    else
    {
        v_AllReduceBegin(
            pData.template GetPtr<NektarSpaces::HostSpace, ReadWrite>(),
            pData.size(), CommDataTypeTraits<T>::GetDataType(), pOp, request);
    }
}

/**
 *
 */
template <class MemSpace, class T>
void Comm::AllReduceEnd([[maybe_unused]] Operators::MemoryRegion<T> &pData,
                        CommRequestSharedPtr request)
{
    if (m_gpu_aware)
    {
        v_AllReduceEnd(request);
    }
    else
    {
        v_AllReduceEnd(request);
    }
}
#endif

/**
 *
 */
template <class T> void Comm::AlltoAll(T &pSendData, T &pRecvData)
{
    static_assert(CommDataTypeTraits<T>::IsVector,
                  "AlltoAll only valid with Array or vector arguments.");
    int sendSize = CommDataTypeTraits<T>::GetCount(pSendData);
    int recvSize = CommDataTypeTraits<T>::GetCount(pRecvData);
    ASSERTL0(sendSize == recvSize,
             "Send and Recv arrays have incompatible sizes in AlltoAll");

    int count = sendSize / GetSize();
    ASSERTL0(count * GetSize() == sendSize,
             "Array size incompatible with size of communicator");

    v_AlltoAll(CommDataTypeTraits<T>::GetPointer(pSendData), count,
               CommDataTypeTraits<T>::GetDataType(),
               CommDataTypeTraits<T>::GetPointer(pRecvData), count,
               CommDataTypeTraits<T>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AlltoAll(Operators::MemoryRegion<T> &pSendData,
                    Operators::MemoryRegion<T> &pRecvData)
{
    int sendSize = pSendData.size();
    int recvSize = pRecvData.size();
    ASSERTL0(sendSize == recvSize,
             "Send and Recv arrays have incompatible sizes in AlltoAll");

    int count = sendSize / GetSize();
    ASSERTL0(count * GetSize() == sendSize,
             "Array size incompatible with size of communicator");

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AlltoAll(pSendData.template GetPtr<MemSpace, ReadOnly>(), count,
                   CommDataTypeTraits<T>::GetDataType(),
                   pRecvData.template GetPtr<MemSpace, WriteOnly>(), count,
                   CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_AlltoAll(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            count, CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            count, CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T1, class T2>
void Comm::AlltoAllv(T1 &pSendData, T2 &pSendDataSizeMap,
                     T2 &pSendDataOffsetMap, T1 &pRecvData,
                     T2 &pRecvDataSizeMap, T2 &pRecvDataOffsetMap)
{
    static_assert(CommDataTypeTraits<T1>::IsVector,
                  "AlltoAllv only valid with Array or vector arguments.");
    static_assert(std::is_same_v<T2, std::vector<int>> ||
                      std::is_same_v<T2, Array<OneD, int>>,
                  "Alltoallv size and offset maps should be integer vectors.");
    v_AlltoAllv(CommDataTypeTraits<T1>::GetPointer(pSendData),
                (int *)CommDataTypeTraits<T2>::GetPointer(pSendDataSizeMap),
                (int *)CommDataTypeTraits<T2>::GetPointer(pSendDataOffsetMap),
                CommDataTypeTraits<T1>::GetDataType(),
                CommDataTypeTraits<T1>::GetPointer(pRecvData),
                (int *)CommDataTypeTraits<T2>::GetPointer(pRecvDataSizeMap),
                (int *)CommDataTypeTraits<T2>::GetPointer(pRecvDataOffsetMap),
                CommDataTypeTraits<T1>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AlltoAllv(Operators::MemoryRegion<T> &pSendData,
                     Operators::MemoryRegion<int> &pSendDataSizeMap,
                     Operators::MemoryRegion<int> &pSendDataOffsetMap,
                     Operators::MemoryRegion<T> &pRecvData,
                     Operators::MemoryRegion<int> &pRecvDataSizeMap,
                     Operators::MemoryRegion<int> &pRecvDataOffsetMap)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AlltoAllv(pSendData.template GetPtr<MemSpace, ReadOnly>(),
                    pSendDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
                    pSendDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
                    CommDataTypeTraits<T>::GetDataType(),
                    pRecvData.template GetPtr<MemSpace, WriteOnly>(),
                    pRecvDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
                    pRecvDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
                    CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_AlltoAllv(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pSendDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pSendDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            pRecvDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pRecvDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T> void Comm::AllGather(T &pSendData, T &pRecvData)
{
    BOOST_STATIC_ASSERT_MSG(
        CommDataTypeTraits<T>::IsVector,
        "AllGather only valid with Array or vector arguments.");

    int sendSize = CommDataTypeTraits<T>::GetCount(pSendData);
    int recvSize = sendSize;

    pRecvData = T(recvSize * GetSize());

    v_AllGather(CommDataTypeTraits<T>::GetPointer(pSendData), sendSize,
                CommDataTypeTraits<T>::GetDataType(),
                CommDataTypeTraits<T>::GetPointer(pRecvData), recvSize,
                CommDataTypeTraits<T>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AllGather(Operators::MemoryRegion<T> &pSendData,
                     Operators::MemoryRegion<T> &pRecvData)
{
    int sendSize = pSendData.size();
    int recvSize = sendSize;

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AllGather(pSendData.template GetPtr<MemSpace, ReadOnly>(), sendSize,
                    CommDataTypeTraits<T>::GetDataType(),
                    pRecvData.template GetPtr<MemSpace, WriteOnly>(), recvSize,
                    CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_AllGather(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            sendSize, CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            recvSize, CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T>
void Comm::AllGatherv(T &pSendData, T &pRecvData,
                      Array<OneD, int> &pRecvDataSizeMap,
                      Array<OneD, int> &pRecvDataOffsetMap)
{
    BOOST_STATIC_ASSERT_MSG(
        CommDataTypeTraits<T>::IsVector,
        "AllGatherv only valid with Array or vector arguments.");

    int sendSize = CommDataTypeTraits<T>::GetCount(pSendData);

    v_AllGatherv(CommDataTypeTraits<T>::GetPointer(pSendData), sendSize,
                 CommDataTypeTraits<T>::GetDataType(),
                 CommDataTypeTraits<T>::GetPointer(pRecvData),
                 pRecvDataSizeMap.data(), pRecvDataOffsetMap.data(),
                 CommDataTypeTraits<T>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AllGatherv(Operators::MemoryRegion<T> &pSendData,
                      Operators::MemoryRegion<T> &pRecvData,
                      Operators::MemoryRegion<int> &pRecvDataSizeMap,
                      Operators::MemoryRegion<int> &pRecvDataOffsetMap)
{
    int sendSize = pSendData.size();

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AllGatherv(pSendData.template GetPtr<MemSpace, ReadOnly>(), sendSize,
                     CommDataTypeTraits<T>::GetDataType(),
                     pRecvData.template GetPtr<MemSpace, WriteOnly>(),
                     pRecvDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
                     pRecvDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
                     CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_AllGatherv(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            sendSize, CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            pRecvDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pRecvDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T>
void Comm::AllGatherv(T &pRecvData, Array<OneD, int> &pRecvDataSizeMap,
                      Array<OneD, int> &pRecvDataOffsetMap)
{
    BOOST_STATIC_ASSERT_MSG(
        CommDataTypeTraits<T>::IsVector,
        "AllGatherv only valid with Array or vector arguments.");

    v_AllGatherv(CommDataTypeTraits<T>::GetPointer(pRecvData),
                 pRecvDataSizeMap.data(), pRecvDataOffsetMap.data(),
                 CommDataTypeTraits<T>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::AllGatherv(Operators::MemoryRegion<T> &pRecvData,
                      Operators::MemoryRegion<int> &pRecvDataSizeMap,
                      Operators::MemoryRegion<int> &pRecvDataOffsetMap)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_AllGatherv(pRecvData.template GetPtr<MemSpace, ReadWrite>(),
                     pRecvDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
                     pRecvDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
                     CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_AllGatherv(
            pRecvData.template GetPtr<NektarSpaces::HostSpace, ReadWrite>(),
            pRecvDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pRecvDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 *
 */
template <class T> void Comm::Bcast(T &pData, int pRoot)
{
    v_Bcast(CommDataTypeTraits<T>::GetPointer(pData),
            CommDataTypeTraits<T>::GetCount(pData),
            CommDataTypeTraits<T>::GetDataType(), pRoot);
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 *
 */
template <class MemSpace, class T>
void Comm::Bcast(Operators::MemoryRegion<T> &pData, int pRoot)
{
    if (m_gpu_aware)
    {
        // Synchronize memory on the root process.
        if (GetRank() == pRoot)
        {
            nekStreamSynchronize(nullptr);
            pData.template GetPtr<MemSpace, ReadOnly>();
        }

        v_Bcast(pData.template GetPtr<MemSpace, WriteOnly>(), pData.size(),
                CommDataTypeTraits<T>::GetDataType(), pRoot);
    }
    else
    {
        // Synchronize memory on the root process.
        if (GetRank() == pRoot)
        {
            pData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }
        v_Bcast(pData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
                pData.size(), CommDataTypeTraits<T>::GetDataType(), pRoot);
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        if (GetRank() != pRoot)
        {
            pData.template GetPtr<MemSpace, ReadOnly>();
        }
    }
}
#endif

/**
 * Concatenate all the input arrays, in rank order, onto the process with rank
 * == rootProc
 */
template <class T> T Comm::Gather(const int rootProc, T &val)
{
    static_assert(CommDataTypeTraits<T>::IsVector,
                  "Gather only valid with Array or vector arguments.");
    bool amRoot  = (GetRank() == rootProc);
    unsigned nEl = CommDataTypeTraits<T>::GetCount(val);

    unsigned nOut = amRoot ? GetSize() * nEl : 0;
    T ans(nOut);
    void *recvbuf = amRoot ? CommDataTypeTraits<T>::GetPointer(ans) : nullptr;

    v_Gather(CommDataTypeTraits<T>::GetPointer(val), nEl,
             CommDataTypeTraits<T>::GetDataType(), recvbuf, nEl,
             CommDataTypeTraits<T>::GetDataType(), rootProc);
    return ans;
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 * Concatenate all the input arrays, in rank order, onto the process with rank
 * == rootProc
 */
template <class MemSpace, class T>
Operators::MemoryRegion<T> Comm::Gather(const int rootProc,
                                        Operators::MemoryRegion<T> &val)
{
    bool amRoot  = (GetRank() == rootProc);
    unsigned nEl = val.size();

    unsigned nOut = amRoot ? GetSize() * nEl : 0;
    Operators::MemoryRegion<T> ans =
        Operators::MemoryRegion<T>(nOut, eHostPinned, val.GetAlignment());

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        if (!amRoot)
        {
            nekStreamSynchronize(nullptr);
        }

        void *recvbuf =
            amRoot ? ans.template GetPtr<MemSpace, WriteOnly>() : nullptr;
        v_Gather(val.template GetPtr<MemSpace, ReadOnly>(), nEl,
                 CommDataTypeTraits<T>::GetDataType(), recvbuf, nEl,
                 CommDataTypeTraits<T>::GetDataType(), rootProc);
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        void *recvbuf =
            amRoot ? ans.template GetPtr<NektarSpaces::HostSpace, WriteOnly>()
                   : nullptr;
        v_Gather(val.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(), nEl,
                 CommDataTypeTraits<T>::GetDataType(), recvbuf, nEl,
                 CommDataTypeTraits<T>::GetDataType(), rootProc);
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        if (amRoot)
        {
            ans.template GetPtr<MemSpace, ReadOnly>();
        }
    }
    return ans;
}
#endif

/**
 * Scatter pData across ranks in chunks of len(pData)/num_ranks
 */
template <class T> T Comm::Scatter(const int rootProc, T &pData)
{
    static_assert(CommDataTypeTraits<T>::IsVector,
                  "Scatter only valid with Array or vector arguments.");

    bool amRoot  = (GetRank() == rootProc);
    unsigned nEl = CommDataTypeTraits<T>::GetCount(pData) / GetSize();

    const void *sendbuf =
        amRoot ? CommDataTypeTraits<T>::GetPointer(pData) : nullptr;
    T ans(nEl);

    v_Scatter(sendbuf, nEl, CommDataTypeTraits<T>::GetDataType(),
              CommDataTypeTraits<T>::GetPointer(ans), nEl,
              CommDataTypeTraits<T>::GetDataType(), rootProc);
    return ans;
}

#if defined(NEKTAR_BUILD_REDESIGN)
/**
 * Scatter pData across ranks in chunks of len(pData)/num_ranks
 */
template <class MemSpace, class T>
Operators::MemoryRegion<T> Comm::Scatter(const int rootProc,
                                         Operators::MemoryRegion<T> &pData)
{
    bool amRoot  = (GetRank() == rootProc);
    unsigned nEl = pData.size() / GetSize();

    Operators::MemoryRegion<T> ans =
        Operators::MemoryRegion<T>(nEl, eHostPinned, pData.GetAlignment());

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        if (amRoot)
        {
            nekStreamSynchronize(nullptr);
        }

        const void *sendbuf =
            amRoot ? pData.template GetPtr<MemSpace, ReadOnly>() : nullptr;
        v_Scatter(sendbuf, nEl, CommDataTypeTraits<T>::GetDataType(),
                  ans.template GetPtr<MemSpace, WriteOnly>(), nEl,
                  CommDataTypeTraits<T>::GetDataType(), rootProc);
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        const void *sendbuf =
            amRoot ? pData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>()
                   : nullptr;
        v_Scatter(sendbuf, nEl, CommDataTypeTraits<T>::GetDataType(),
                  ans.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
                  nEl, CommDataTypeTraits<T>::GetDataType(), rootProc);
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        ans.template GetPtr<MemSpace, ReadOnly>();
    }
    return ans;
}
#endif

/**
 * This replaces the current MPI communicator with a new one that also holds
 * the distributed graph topology information. If reordering is enabled using
 * this might break code where process/rank numbers are assumed to remain
 * constant. This also assumes that the graph is bi-directional, so all
 * sources are also destinations with equal weighting.
 *
 * @param sources       Ranks of processes for which the calling process is the
 *                      destination/source
 * @param sourceweights Weights of the corresponding edges into the calling
 *                      process
 * @param reorder       Ranks may be reordered (true) or not (false)
 */
template <class T>
void Comm::DistGraphCreateAdjacent(T &sources, T &sourceweights, int reorder)
{
    static_assert(
        CommDataTypeTraits<T>::IsVector,
        "DistGraphCreateAdjacent only valid with Array or vector arguments.");

    ASSERTL0(CommDataTypeTraits<T>::GetCount(sources) ==
                 CommDataTypeTraits<T>::GetCount(sourceweights),
             "Sources and weights array sizes don't match");

    int indegree = CommDataTypeTraits<T>::GetCount(sources);

    v_DistGraphCreateAdjacent(
        indegree, (const int *)CommDataTypeTraits<T>::GetPointer(sources),
        (const int *)CommDataTypeTraits<T>::GetPointer(sourceweights), reorder);
}

#if defined(NEKTAR_BUILD_REDESIGN)
template <class MemSpace, class T>
void Comm::DistGraphCreateAdjacent(Operators::MemoryRegion<T> &sources,
                                   Operators::MemoryRegion<T> &sourceweights,
                                   int reorder)
{
    ASSERTL0(sources.size() == sourceweights.size(),
             "Sources and weights array sizes don't match");

    int indegree = CommDataTypeTraits<T>::GetCount(sources);

    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_DistGraphCreateAdjacent(
            indegree, sources.template GetPtr<MemSpace, ReadOnly>(),
            sourceweights.template GetPtr<MemSpace, ReadOnly>(), reorder);
    }
    else
    {
        v_DistGraphCreateAdjacent(
            indegree,
            sources.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            sourceweights.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            reorder);
    }
}
#endif

/**
 * Sends data to neighboring processes in a virtual topology communicator. All
 * processes send different amounts of data to, and receive different amounts
 * of data from, all neighbors
 *
 * @param pSendData          Array/vector to send to neighbors
 * @param pSendDataSizeMap   Array/vector where entry i specifies the number
 *                           of elements to send to neighbor i
 * @param pSendDataOffsetMap Array/vector where entry i specifies the
 *                           displacement (offset from pSendData) from which to
 *                           send data to neighbor i
 * @param pRecvData          Array/vector to place incoming data in to
 * @param pRecvDataSizeMap   Array/vector where entry i specifies the number
 *                           of elements to receive from neighbor i
 * @param pRecvDataOffsetMap Array/vector where entry i specifies the
 *                           displacement (offset from pRecvData) from which to
 *                           receive data from neighbor i
 */
template <class T1, class T2>
void Comm::NeighborAlltoAllv(T1 &pSendData, T2 &pSendDataSizeMap,
                             T2 &pSendDataOffsetMap, T1 &pRecvData,
                             T2 &pRecvDataSizeMap, T2 &pRecvDataOffsetMap)
{
    static_assert(
        CommDataTypeTraits<T1>::IsVector,
        "NeighbourAlltoAllv only valid with Array or vector arguments.");
    static_assert(
        std::is_same_v<T2, std::vector<int>> ||
            std::is_same_v<T2, Array<OneD, int>>,
        "NeighborAllToAllv size and offset maps should be integer vectors.");
    v_NeighborAlltoAllv(
        CommDataTypeTraits<T1>::GetPointer(pSendData),
        (int *)CommDataTypeTraits<T2>::GetPointer(pSendDataSizeMap),
        (int *)CommDataTypeTraits<T2>::GetPointer(pSendDataOffsetMap),
        CommDataTypeTraits<T1>::GetDataType(),
        CommDataTypeTraits<T1>::GetPointer(pRecvData),
        (int *)CommDataTypeTraits<T2>::GetPointer(pRecvDataSizeMap),
        (int *)CommDataTypeTraits<T2>::GetPointer(pRecvDataOffsetMap),
        CommDataTypeTraits<T1>::GetDataType());
}

#if defined(NEKTAR_BUILD_REDESIGN)
template <class MemSpace, class T>
void Comm::NeighborAlltoAllv(Operators::MemoryRegion<T> &pSendData,
                             Operators::MemoryRegion<int> &pSendDataSizeMap,
                             Operators::MemoryRegion<int> &pSendDataOffsetMap,
                             Operators::MemoryRegion<T> &pRecvData,
                             Operators::MemoryRegion<int> &pRecvDataSizeMap,
                             Operators::MemoryRegion<int> &pRecvDataOffsetMap)
{
    if (m_gpu_aware)
    {
        // Synchronize stream before communication.
        nekStreamSynchronize(nullptr);

        v_NeighborAlltoAllv(
            pSendData.template GetPtr<MemSpace, ReadOnly>(),
            pSendDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
            pSendDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<MemSpace, WriteOnly>(),
            pRecvDataSizeMap.template GetPtr<MemSpace, ReadOnly>(),
            pRecvDataOffsetMap.template GetPtr<MemSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType());
    }
    else
    {
        // MPI is NOT GPU-aware, data must be copied to the host before data
        // transfer.
        v_NeighborAlltoAllv(
            pSendData.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pSendDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pSendDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType(),
            pRecvData.template GetPtr<NektarSpaces::HostSpace, WriteOnly>(),
            pRecvDataSizeMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            pRecvDataOffsetMap
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
            CommDataTypeTraits<T>::GetDataType());
        // MPI is NOT GPU-aware, data must be copied back to MemSpace after data
        // transfer.
        pRecvData.template GetPtr<MemSpace, ReadOnly>();
    }
}
#endif

/**
 * Starts a ready-mode nonblocking send
 *
 * @param pProc   Rank of destination
 * @param pData   Array/vector to send
 * @param count   Number of elements to send in pData
 * @param request Communication request object
 * @param loc     Location in request to use
 */
template <class T>
void Comm::Irsend(int pProc, T &pData, int count,
                  const CommRequestSharedPtr &request, int loc)
{
    v_Irsend(CommDataTypeTraits<T>::GetPointer(pData), count,
             CommDataTypeTraits<T>::GetDataType(), pProc, request, loc);
}

/**
 * Starts a nonblocking send
 *
 * @param pProc   Rank of destination
 * @param pData   Array/vector to send
 * @param count   Number of elements to send in pData
 * @param request Communication request object
 * @param loc     Location in request to use
 */
template <class T>
void Comm::Isend(int pProc, T &pData, int count,
                 const CommRequestSharedPtr &request, int loc)
{
    v_Isend(CommDataTypeTraits<T>::GetPointer(pData), count,
            CommDataTypeTraits<T>::GetDataType(), pProc, request, loc);
}

/**
 * Creates a persistent request for a send
 *
 * @param pProc   Rank of destination
 * @param pData   Array/vector to send
 * @param count   Number of elements to send in pData
 * @param request Communication request object
 * @param loc     Location in request to use
 */
template <class T>
void Comm::SendInit(int pProc, T &pData, int count,
                    const CommRequestSharedPtr &request, int loc)
{
    v_SendInit(CommDataTypeTraits<T>::GetPointer(pData), count,
               CommDataTypeTraits<T>::GetDataType(), pProc, request, loc);
}

/**
 * Begins a nonblocking receive
 *
 * @param pProc   Rank of source
 * @param pData   Array/vector to place incoming data in to
 * @param count   Number of elements to receive in to pData
 * @param request Communication request object
 * @param loc     Location in request to use
 */
template <class T>
void Comm::Irecv(int pProc, T &pData, int count,
                 const CommRequestSharedPtr &request, int loc)
{
    v_Irecv(CommDataTypeTraits<T>::GetPointer(pData), count,
            CommDataTypeTraits<T>::GetDataType(), pProc, request, loc);
}

/**
 * Create a persistent request for a receive
 *
 * @param pProc   Rank of source
 * @param pData   Array/vector to place incoming data in to
 * @param count   Number of elements to receive in to pData
 * @param request Communication request object
 * @param loc     Location in request to use
 */
template <class T>
void Comm::RecvInit(int pProc, T &pData, int count,
                    const CommRequestSharedPtr &request, int loc)
{
    v_RecvInit(CommDataTypeTraits<T>::GetPointer(pData), count,
               CommDataTypeTraits<T>::GetDataType(), pProc, request, loc);
}

/**
 * Starts a collection of persistent requests
 *
 * @param request Communication request object
 */
inline void Comm::StartAll(const CommRequestSharedPtr &request)
{
    v_StartAll(request);
}

/**
 * Waits for all CommRequests in the request object to complete.
 *
 * @param request Communication request object
 */
inline void Comm::WaitAll(const CommRequestSharedPtr &request)
{
    v_WaitAll(request);
}

/**
 * Creates a number of CommRequests.
 *
 * @param num Number of requests to generate in the communication request object
 *
 * @return Communication request object
 */
inline CommRequestSharedPtr Comm::CreateRequest(int num)
{
    return v_CreateRequest(num);
}

/**
 * @brief If the flag is non-zero create a new communicator.
 */
inline CommSharedPtr Comm::CommCreateIf(int flag)
{
    return v_CommCreateIf(flag);
}

/**
 * @brief Splits this communicator into a grid of size pRows*pColumns
 * and creates row and column communicators. By default the communicator
 * is a single row.
 */
inline void Comm::SplitComm(int pRows, int pColumns, int pTime)
{
    v_SplitComm(pRows, pColumns, pTime);
}

/**
 * @brief Retrieve the row communicator to which this process belongs.
 */
inline CommSharedPtr Comm::GetRowComm()
{
    if (!m_commRow.get())
    {
        return shared_from_this();
    }
    else
    {
        return m_commRow;
    }
}

/**
 * @brief Retrieve the column communicator to which this process
 * belongs.
 */
inline CommSharedPtr Comm::GetColumnComm()
{
    if (!m_commColumn.get())
    {
        return shared_from_this();
    }
    else
    {
        return m_commColumn;
    }
}

/**
 * @brief Retrieve the time communicator to which this process
 * belongs.
 */
inline CommSharedPtr Comm::GetTimeComm()
{
    if (!m_commTime.get())
    {
        return shared_from_this();
    }
    else
    {
        return m_commTime;
    }
}

/**
 * @brief Retrieve the space communicator to which this process
 * belongs.
 */
inline CommSharedPtr Comm::GetSpaceComm()
{
    if (!m_commSpace.get())
    {
        return shared_from_this();
    }
    else
    {
        return m_commSpace;
    }
}

/**
 *
 */
inline bool Comm::RemoveExistingFiles()
{
    return true;
}

/**
 *
 */
std::pair<CommSharedPtr, CommSharedPtr> Comm::SplitCommNode()
{
    return v_SplitCommNode();
}

} // namespace Nektar::LibUtilities

#endif
