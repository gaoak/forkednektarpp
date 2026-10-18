///////////////////////////////////////////////////////////////////////////////
//
// File: CommMpi.cpp
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
// Description: MPI communication implementation
//
///////////////////////////////////////////////////////////////////////////////

#include <cstdlib>

#ifdef NEKTAR_USING_PETSC
#include "petscsys.h"
#endif

#include <LibUtilities/Communication/CommMpi.h>

#if defined(NEKTAR_USE_MPIX_STREAM)
#include <mutex>
#if defined(SYCL_ENABLE_CUDA)
#include <LibUtilities/Backends/CUDAStream.hpp>
#elif defined(SYCL_ENABLE_HIP)
#include <LibUtilities/Backends/HIPStream.hpp>
#endif
#endif

#if __has_include(<mpi-ext.h>)
/* Needed for MPIX_Query_*_support() */
#include <mpi-ext.h> // OpenMPI
#endif

namespace Nektar::LibUtilities
{

/**
 * Return the MPI operation of @p pOp.
 */
static MPI_Op GetMpiOp(enum ReduceOperator pOp)
{
    switch (pOp)
    {
        case ReduceMax:
            return MPI_MAX;
        case ReduceMin:
            return MPI_MIN;
        case ReduceSum:
        default:
            return MPI_SUM;
    }
}

/**
 * Return whether the MPI implementation is GPU-aware. The SYCL back-ends
 * offloading to CUDA or HIP devices are detected as the native back-ends.
 */
static bool IsMpiGPUAware()
{
    bool gpuAware = false;

#if (defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)) &&              \
    defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    // CUDA-aware OpenMPI/MVAPICH2
    // https://docs.open-mpi.org/en/main/tuning-apps/networking/cuda.html
    gpuAware = (bool)MPIX_Query_cuda_support();
#elif (defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)) &&              \
    defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
    // ROCM-aware OpenMPI
    // https://docs.open-mpi.org/en/main/tuning-apps/networking/rocm.html
    // https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
    gpuAware = (bool)MPIX_Query_rocm_support();
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP)
    // CUDA/ROCM-aware CRAY MPICH
    // https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/
    // Note: Environment variable must bet set as follow:
    //       export MPICH_GPU_SUPPORT_ENABLED=1
    // Note: It is the user responsibilities to compile Nektar++ with Cray MPICH
    // when using the MPICH_GPU_SUPPORT_ENABLED option.
    gpuAware = (bool)std::getenv("MPICH_GPU_SUPPORT_ENABLED");
#elif defined(SYCL_ENABLE_INTEL)
    // GPU-aware Intel MPI
    // https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/current/gpu-support.html
    // Note: Environment variable must be set as follow:
    //       export I_MPI_OFFLOAD=1
    gpuAware = (bool)std::getenv("I_MPI_OFFLOAD");
#endif

#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||               \
    defined(NEKTAR_ENABLE_SYCL)
    // Can't automatically determine if MPI is GPU aware or not. Manually set
    // GPU-aware flag. Note: Environment variable must bet set as follow:
    //       export MPI_GPU_AWARE=1
    // Note: It is the user responsibilities to compile Nektar++ with GPU-aware
    // implementation when using the MPI_GPU_AWARE option.
    gpuAware = gpuAware || (bool)std::getenv("MPI_GPU_AWARE");
#endif

    return gpuAware;
}

#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
/**
 * Return the node-local rank the MPI launcher sets in the environment, or -1
 * when it sets none.
 */
static int GetLauncherLocalRank()
{
    static const char *const names[] = {
        "MPI_LOCALRANKID",            // MPICH (Hydra)
        "OMPI_COMM_WORLD_LOCAL_RANK", // Open MPI
        "MV2_COMM_WORLD_LOCAL_RANK",  // MVAPICH2
        "PALS_LOCAL_RANKID",          // Cray PALS
        "SLURM_LOCALID"};             // Slurm srun

    for (const char *name : names)
    {
        if (const char *value = std::getenv(name))
        {
            char *end;
            const long rank = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && rank >= 0)
            {
                return static_cast<int>(rank);
            }
        }
    }

    return -1;
}
#endif

#if defined(NEKTAR_USE_MPIX_STREAM)
#if defined(NEKTAR_ENABLE_SYCL) && defined(__DPCPP_COMPILER) &&                \
    !defined(SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND)
#error "NEKTAR_USE_MPIX_STREAM with DPC++ requires the \
sycl_ext_oneapi_enqueue_native_command extension."
#endif

/**
 * Run @p command with the native stream the work on @p streamID is queued
 * on. Under SYCL, that is the stream behind the queue, reached from a native
 * command submitted to it, which the runtime may run on a thread of its own.
 */
template <typename Command>
static void RunOnStream(const unsigned int streamID, const Command &command)
{
#if defined(NEKTAR_ENABLE_CUDA)
    command(CUDAStream::GetInstance(streamID));
#elif defined(NEKTAR_ENABLE_HIP)
    command(HIPStream::GetInstance(streamID));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([&](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        auto native = [command](sycl::interop_handle ih) {
            command(
                static_cast<nekStream>(ih.get_native_queue<sycl_backend>()));
        };
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(native);
#elif defined(__DPCPP_COMPILER)
        cgh.ext_oneapi_enqueue_native_command(native);
#endif
    });
    SYCLQueue::SetEvent(streamID, e);
#endif
}

/**
 * Stream communicator of the MPIX_Stream extension over the ranks of
 * #m_comm, with the device stream it enqueues on and the events that order
 * that stream against the caller's.
 */
struct CommMpi::MpixState
{
    /// Communicator bound to #commStream.
    MPI_Comm streamComm    = MPI_COMM_NULL;
    MPIX_Stream mpixStream = MPIX_STREAM_NULL;
    /// Non-blocking stream the enqueued communication runs on.
    nekStream commStream = nullptr;
    /// Recorded on the caller's stream before the communication.
    nekEvent readyEvent = nullptr;
    /// Recorded on #commStream after the communication.
    nekEvent doneEvent = nullptr;
    /// Serialises the enqueueing, which SYCL runs on its own threads.
    std::mutex mutex;

    /**
     * Create #commStream, non-blocking so that it does not synchronise with the
     * default stream, its events, and the stream communicator over @p mpiComm
     * bound to it. Collective over @p mpiComm.
     */
    void Init(MPI_Comm mpiComm)
    {
        // Under SYCL, created in a native command so that it shares the device
        // and context of the queues.
        RunOnStream(0, [this](nekStream) {
            CHECK_HIPCUDA_ERROR(
                nekStreamCreateWithFlags(&commStream, nekStreamNonBlocking));
            CHECK_HIPCUDA_ERROR(
                nekEventCreateWithFlags(&readyEvent, nekEventDisableTiming));
            CHECK_HIPCUDA_ERROR(
                nekEventCreateWithFlags(&doneEvent, nekEventDisableTiming));
        });
#if defined(NEKTAR_ENABLE_SYCL)
        SYCLQueue::GetInstance(0).wait();
#endif

#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
        const char *streamType = "cudaStream_t";
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
        const char *streamType = "hipStream_t";
#endif

        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "type", streamType);
        MPIX_Info_set_hex(info, "value", &commStream, sizeof(commStream));
        int retval = MPIX_Stream_create(info, &mpixStream);
        MPI_Info_free(&info);

        ASSERTL0(retval == MPI_SUCCESS, "MPI error creating MPIX stream.");

        retval = MPIX_Stream_comm_create(mpiComm, mpixStream, &streamComm);

        ASSERTL0(retval == MPI_SUCCESS,
                 "MPI error creating MPIX stream communicator.");
    }

    /**
     * Free the stream communicator and the device resources behind it. The MPI
     * objects are left to MPI_Finalize() once it has run.
     */
    void Free()
    {
#if defined(NEKTAR_ENABLE_SYCL)
        // Native commands still queued may use the resources freed below.
        if (commStream != nullptr)
        {
            nekDeviceSynchronize();
        }
#endif

        int flag;
        MPI_Finalized(&flag);
        if (!flag)
        {
            if (streamComm != MPI_COMM_NULL)
            {
                MPI_Comm_free(&streamComm);
            }
            if (mpixStream != MPIX_STREAM_NULL)
            {
                MPIX_Stream_free(&mpixStream);
            }
        }
        streamComm = MPI_COMM_NULL;
        mpixStream = MPIX_STREAM_NULL;

        // Errors are ignored: this may run after the device runtime has
        // unloaded.
        if (doneEvent != nullptr)
        {
            (void)nekEventDestroy(doneEvent);
        }
        if (readyEvent != nullptr)
        {
            (void)nekEventDestroy(readyEvent);
        }
        if (commStream != nullptr)
        {
            (void)nekStreamDestroy(commStream);
        }
        doneEvent  = nullptr;
        readyEvent = nullptr;
        commStream = nullptr;
    }

    /**
     * Enqueue the communication @p enqueue, given #streamComm, on #commStream,
     * ordered after the work queued so far on @p streamID. With @p join, the
     * work queued next on @p streamID is ordered after the communication;
     * otherwise JoinCommDone() orders it.
     */
    template <typename Enqueue>
    void EnqueueOnStream(const unsigned int streamID, const Enqueue &enqueue,
                         const bool join = true)
    {
        RunOnStream(streamID, [this, enqueue, join](nekStream stream) {
            std::lock_guard<std::mutex> lock(mutex);

            CHECK_HIPCUDA_ERROR(nekEventRecord(readyEvent, stream));
            CHECK_HIPCUDA_ERROR(nekStreamWaitEvent(commStream, readyEvent));

            enqueue(streamComm);

            CHECK_HIPCUDA_ERROR(nekEventRecord(doneEvent, commStream));
            if (join)
            {
                CHECK_HIPCUDA_ERROR(nekStreamWaitEvent(stream, doneEvent));
            }
        });
    }

    /**
     * Order the work queued next on @p streamID after the communication last
     * enqueued on #commStream.
     */
    void JoinCommDone(const unsigned int streamID)
    {
        RunOnStream(streamID, [this](nekStream stream) {
            std::lock_guard<std::mutex> lock(mutex);

            CHECK_HIPCUDA_ERROR(nekStreamWaitEvent(stream, doneEvent));
        });
    }
};
#endif

std::string CommMpi::className = GetCommFactory().RegisterCreatorFunction(
    "ParallelMPI", CommMpi::create, "Parallel communication using MPI.");

/**
 *
 */
CommMpi::CommMpi(int narg, char *arg[]) : Comm(narg, arg)
{
    int init = 0;
    MPI_Initialized(&init);

    if (!init)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        // Bind local MPI rank to GPU. Skipped entirely when no device is
        // visible to this process, as is the case for a build with device
        // support enabled running on a machine without a GPU. A GPU-aware MPI
        // sets up its device support on the current device in MPI_Init, so
        // the binding comes first wherever the launcher gives the local rank.
        const auto num_device   = nekGetNumDevice();
        const int launcher_rank = num_device > 0 ? GetLauncherLocalRank() : -1;
        if (launcher_rank >= 0)
        {
            nekSetDevice(launcher_rank % num_device);
        }
#endif

        int thread_support = 0;
        if (MPI_Init_thread(&narg, &arg, MPI_THREAD_MULTIPLE,
                            &thread_support) != MPI_SUCCESS)
        {
            NEKERROR(
                ErrorUtil::ewarning,
                "Initializing MPI using MPI_Init, if scotch version > 6 and is "
                "compiled with multi-threading, it might cause deadlocks.")
            ASSERTL0(MPI_Init(&narg, &arg) == MPI_SUCCESS,
                     "Failed to initialise MPI");
        }
        // store bool to indicate that Nektar++ is in charge of finalizing MPI.
        m_controls_mpi = true;

#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        // Otherwise bind it from its rank on the node, once MPI is up.
        if (num_device > 0 && launcher_rank < 0)
        {
            MPI_Comm local_comm;
            int local_rank, local_size;

            // Split MPI_COMM_WORLD based on shared memory access (effectively
            // per node).
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &local_comm);

            // Get the rank and size in the new local communicator.
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_size(local_comm, &local_size);

            // Use round-Robin distribution.
            auto device_rank = local_rank % num_device;
            nekSetDevice(device_rank);

            // Free communicator.
            MPI_Comm_free(&local_comm);
        }
#endif
    }
    else
    {
        // Another code is in charge of finalizing MPI and this is not the
        // responsiblity of Nektar++
        m_controls_mpi = false;
    }

    m_comm = MPI_COMM_WORLD;
    MPI_Comm_size(m_comm, &m_size);
    MPI_Comm_rank(m_comm, &m_rank);

#ifdef NEKTAR_USING_PETSC
    PetscInitializeNoArguments();
#endif

    m_type = "Parallel MPI";

    m_gpu_aware = IsMpiGPUAware();

#if defined(NEKTAR_USE_MPIX_STREAM)
    InitStreamComm();
#endif
}

/**
 *
 */
CommMpi::CommMpi(MPI_Comm pComm) : Comm()
{
    m_comm = pComm;
    MPI_Comm_size(m_comm, &m_size);
    MPI_Comm_rank(m_comm, &m_rank);

    m_type = "Parallel MPI";

    m_gpu_aware = IsMpiGPUAware();

#if defined(NEKTAR_USE_MPIX_STREAM)
    InitStreamComm();
#endif
}

/**
 *
 */
CommMpi::~CommMpi()
{
#if defined(NEKTAR_USE_MPIX_STREAM)
    FreeStreamComm();
#endif

    int flag;
    MPI_Finalized(&flag);
    if (!flag && m_comm != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&m_comm);
    }
}

/**
 *
 */
MPI_Comm CommMpi::GetComm()
{
    return m_comm;
}

/**
 *
 */
void CommMpi::v_Finalise()
{
#ifdef NEKTAR_USING_PETSC
    PetscFinalize();
#endif
#if defined(NEKTAR_USE_MPIX_STREAM)
    FreeStreamComm();
#endif

    int flag;
    MPI_Finalized(&flag);
    if ((!flag) && m_controls_mpi)
    {
        MPI_Finalize();
    }
}

/**
 *
 */
int CommMpi::v_GetRank()
{
    return m_rank;
}

/**
 *
 */
bool CommMpi::v_TreatAsRankZero()
{
    return m_rank == 0;
}

/**
 *
 */
bool CommMpi::v_IsSerial()
{
    return m_size == 1;
}

/**
 *
 */
std::tuple<int, int, int> CommMpi::v_GetVersion()
{
    int version, subversion;
    int retval = MPI_Get_version(&version, &subversion);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing GetVersion.");

    return std::make_tuple(version, subversion, 0);
}

/**
 *
 */
void CommMpi::v_Block()
{
    MPI_Barrier(m_comm);
}

/**
 *
 */
double CommMpi::v_Wtime()
{
    return MPI_Wtime();
}

#if defined(NEKTAR_USE_MPIX_STREAM)
/**
 * Create the MPIX stream communicator over #m_comm, when a device is
 * visible. Collective, so it is called on construction: point-to-point
 * calls could not create it.
 */
void CommMpi::InitStreamComm()
{
    // Device communication is enqueued on a stream whenever a device is
    // visible. An MPI enqueuing on device streams is built with device
    // support, so is GPU-aware.
    if (nekGetNumDevice() > 0)
    {
        m_gpu_aware = true;
        m_mpix      = std::make_unique<MpixState>();
        m_mpix->Init(m_comm);
    }
}

/**
 * Free the MPIX stream communicator.
 */
void CommMpi::FreeStreamComm()
{
    if (m_mpix)
    {
        m_mpix->Free();
        m_mpix.reset();
    }
}
#endif

/**
 * Device data is ordered on @p streamID: with MPIX streams the send is
 * enqueued on MpixState::commStream, without blocking the host; otherwise
 * @p streamID is synchronised first.
 */
void CommMpi::v_Send(const void *buf, int count, CommDataType dt, int dest,
                     std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->EnqueueOnStream(*streamID, [buf, count, dt,
                                            dest](MPI_Comm streamComm) {
            int retval = MPIX_Send_enqueue(buf, count, dt, dest, 0, streamComm);

            ASSERTL0(retval == MPI_SUCCESS,
                     "MPI error performing stream-ordered send.");
        });
        return;
#elif defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    if (MPISYNC)
    {
        MPI_Ssend(buf, count, dt, dest, 0, m_comm);
    }
    else
    {
        MPI_Send(buf, count, dt, dest, 0, m_comm);
    }
}

/**
 * Device data is ordered on @p streamID: with MPIX streams the receive is
 * enqueued on MpixState::commStream, without blocking the host; otherwise
 * @p streamID is synchronised first.
 */
void CommMpi::v_Recv(void *buf, int count, CommDataType dt, int source,
                     std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->EnqueueOnStream(
            *streamID, [buf, count, dt, source](MPI_Comm streamComm) {
                int retval = MPIX_Recv_enqueue(buf, count, dt, source, 0,
                                               streamComm, MPI_STATUS_IGNORE);

                ASSERTL0(retval == MPI_SUCCESS,
                         "MPI error performing stream-ordered receive.");
            });
        return;
#elif defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    MPI_Recv(buf, count, dt, source, 0, m_comm, MPI_STATUS_IGNORE);
}

/**
 * Device data is ordered on @p streamID: with MPIX streams the send and
 * receive are enqueued on MpixState::commStream and completed by one wait,
 * without blocking the host; otherwise @p streamID is synchronised first.
 */
void CommMpi::v_SendRecv(const void *sendbuf, int sendcount,
                         CommDataType sendtype, int dest, void *recvbuf,
                         int recvcount, CommDataType recvtype, int source,
                         std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->EnqueueOnStream(*streamID, [sendbuf, sendcount, sendtype, dest,
                                            recvbuf, recvcount, recvtype,
                                            source](MPI_Comm streamComm) {
            MPI_Request request[2];
            int retval = MPIX_Irecv_enqueue(recvbuf, recvcount, recvtype,
                                            source, 0, streamComm, &request[0]);
            ASSERTL0(retval == MPI_SUCCESS, "MPI error performing "
                                            "stream-ordered send-receive of "
                                            "data.");

            retval = MPIX_Isend_enqueue(sendbuf, sendcount, sendtype, dest, 0,
                                        streamComm, &request[1]);
            ASSERTL0(retval == MPI_SUCCESS, "MPI error performing "
                                            "stream-ordered send-receive of "
                                            "data.");

            retval = MPIX_Waitall_enqueue(2, request, MPI_STATUSES_IGNORE);
            ASSERTL0(retval == MPI_SUCCESS, "MPI error performing "
                                            "stream-ordered send-receive of "
                                            "data.");
        });
        return;
#elif defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    MPI_Status status;
    int retval = MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, 0, recvbuf,
                              recvcount, recvtype, source, 0, m_comm, &status);

    ASSERTL0(retval == MPI_SUCCESS,
             "MPI error performing send-receive of data.");
}

/**
 * Device data is ordered on @p streamID: with MPIX streams the all-reduce is
 * enqueued on MpixState::commStream, without blocking the host; otherwise
 * @p streamID is synchronised first.
 */
void CommMpi::v_AllReduce(void *buf, int count, CommDataType dt,
                          enum ReduceOperator pOp,
                          std::optional<unsigned int> streamID)
{
    if (GetSize() == 1)
    {
        return;
    }

    MPI_Op vOp = GetMpiOp(pOp);

    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->EnqueueOnStream(
            *streamID, [buf, count, dt, vOp](MPI_Comm streamComm) {
                int retval = MPIX_Allreduce_enqueue(MPI_IN_PLACE, buf, count,
                                                    dt, vOp, streamComm);

                ASSERTL0(retval == MPI_SUCCESS,
                         "MPI error performing stream-ordered All-reduce.");
            });
        return;
#elif defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Allreduce(MPI_IN_PLACE, buf, count, dt, vOp, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-reduce.");
}

/**
 * Device data is ordered on @p streamID: with MPIX streams the all-reduce is
 * enqueued on MpixState::commStream and neither the host nor @p streamID
 * blocks until v_AllReduceEnd(); otherwise @p streamID is synchronised first.
 */
void CommMpi::v_AllReduceBegin(void *buf, int count, CommDataType dt,
                               enum ReduceOperator pOp,
                               CommRequestSharedPtr request,
                               std::optional<unsigned int> streamID)
{
    if (GetSize() == 1)
    {
        return;
    }

    MPI_Op vOp = GetMpiOp(pOp);

    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->EnqueueOnStream(
            *streamID,
            [buf, count, dt, vOp](MPI_Comm streamComm) {
                int retval = MPIX_Allreduce_enqueue(MPI_IN_PLACE, buf, count,
                                                    dt, vOp, streamComm);

                ASSERTL0(retval == MPI_SUCCESS,
                         "MPI error performing stream-ordered All-reduce.");
            },
            false);
        return;
#elif defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    int retval = MPI_Iallreduce(MPI_IN_PLACE, buf, count, dt, vOp, m_comm,
                                req->GetRequest(0));

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-reduce.");
}

/**
 * With MPIX streams, the work queued next on @p streamID is ordered after
 * the all-reduce, without blocking the host.
 */
void CommMpi::v_AllReduceEnd(CommRequestSharedPtr request,
                             std::optional<unsigned int> streamID)
{
    if (GetSize() == 1)
    {
        return;
    }

    if (streamID)
    {
#if defined(NEKTAR_USE_MPIX_STREAM)
        m_mpix->JoinCommDone(*streamID);
        return;
#endif
    }

    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    if (req->GetNumRequest() != 0)
    {
        MPI_Wait(req->GetRequest(0), MPI_STATUS_IGNORE);
    }
}

/**
 *
 */
void CommMpi::v_AlltoAll(const void *sendbuf, int sendcount,
                         CommDataType sendtype, void *recvbuf, int recvcount,
                         CommDataType recvtype,
                         std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                              recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-to-All.");
}

/**
 *
 */
void CommMpi::v_AlltoAllv(const void *sendbuf, const int *sendcounts,
                          const int *senddispls, CommDataType sendtype,
                          void *recvbuf, const int *recvcounts,
                          const int *recvdispls, CommDataType recvtype,
                          std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval =
        MPI_Alltoallv(sendbuf, sendcounts, senddispls, sendtype, recvbuf,
                      recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-to-All-v.");
}

/**
 *
 */
void CommMpi::v_AllGather(const void *sendbuf, int sendcount,
                          CommDataType sendtype, void *recvbuf, int recvcount,
                          CommDataType recvtype,
                          std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                               recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgather.");
}

/**
 *
 */
void CommMpi::v_AllGatherv(const void *sendbuf, int sendcount,
                           CommDataType sendtype, void *recvbuf,
                           const int *recvcounts, const int *recvdispls,
                           CommDataType recvtype,
                           std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgatherv.");
}

/**
 *
 */
void CommMpi::v_AllGatherv(void *recvbuf, const int *recvcounts,
                           const int *recvdispls, CommDataType recvtype,
                           std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf,
                                recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgatherv.");
}

/**
 *
 */
void CommMpi::v_Bcast(void *buffer, int count, CommDataType dt, int root,
                      std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Bcast(buffer, count, dt, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Bcast-v.");
}

/**
 *
 */
void CommMpi::v_Gather(const void *sendbuf, int sendcount,
                       CommDataType sendtype, void *recvbuf, int recvcount,
                       CommDataType recvtype, int root,
                       std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Gather.");
}

/**
 *
 */
void CommMpi::v_Scatter(const void *sendbuf, int sendcount,
                        CommDataType sendtype, void *recvbuf, int recvcount,
                        CommDataType recvtype, int root,
                        std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

    int retval = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             recvtype, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Scatter.");
}

void CommMpi::v_Gatherv(const void *sendbuf, int sendcount,
                        CommDataType sendtype, void *recvbuf,
                        const int *recvcounts, const int *recvdispls,
                        CommDataType recvtype, int root)
{
    MPI_Gatherv(const_cast<void *>(sendbuf), sendcount, sendtype, recvbuf,
                recvcounts, recvdispls, recvtype, root, m_comm);
}

/**
 *
 */
CommSharedPtr CommMpi::v_DistGraphCreateAdjacent(
    [[maybe_unused]] int indegree, [[maybe_unused]] const int *sources,
    [[maybe_unused]] const int *sourceweights, [[maybe_unused]] int reorder)
{
#if MPI_VERSION < 3
    ASSERTL0(false, "MPI_Dist_graph_create_adjacent is not supported in your "
                    "installed MPI version.");

    return CommSharedPtr();
#else
    if (sourceweights == nullptr)
    {
        sourceweights = MPI_UNWEIGHTED;
    }

    MPI_Comm neighComm;
    int retval = MPI_Dist_graph_create_adjacent(
        m_comm, indegree, sources, sourceweights, indegree, sources,
        sourceweights, MPI_INFO_NULL, reorder, &neighComm);

    ASSERTL0(retval == MPI_SUCCESS,
             "MPI error performing Dist_graph_create_adjacent.");

    return std::shared_ptr<Comm>(new CommMpi(neighComm));
#endif
}

/**
 *
 */
void CommMpi::v_NeighborAlltoAllv([[maybe_unused]] const void *sendbuf,
                                  [[maybe_unused]] const int *sendcounts,
                                  [[maybe_unused]] const int *senddispls,
                                  [[maybe_unused]] CommDataType sendtype,
                                  [[maybe_unused]] void *recvbuf,
                                  [[maybe_unused]] const int *recvcounts,
                                  [[maybe_unused]] const int *recvdispls,
                                  [[maybe_unused]] CommDataType recvtype,
                                  std::optional<unsigned int> streamID)
{
    if (streamID)
    {
#if defined(NEKTAR_ENABLE_DEVICE_SUPPORT)
        nekStreamSynchronize(*streamID);
#endif
    }

#if MPI_VERSION < 3
    ASSERTL0(false, "MPI_Neighbor_alltoallv is not supported in your "
                    "installed MPI version.");
#else
    int retval = MPI_Neighbor_alltoallv(sendbuf, sendcounts, senddispls,
                                        sendtype, recvbuf, recvcounts,
                                        recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing NeighborAllToAllV.");
#endif
}

/**
 *
 */
void CommMpi::v_Irsend(const void *buf, int count, CommDataType dt, int dest,
                       CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Irsend(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_Isend(const void *buf, int count, CommDataType dt, int dest,
                      CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Isend(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_SendInit(const void *buf, int count, CommDataType dt, int dest,
                         CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Send_init(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_Irecv(void *buf, int count, CommDataType dt, int source,
                      CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Irecv(buf, count, dt, source, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_RecvInit(void *buf, int count, CommDataType dt, int source,
                         CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Recv_init(buf, count, dt, source, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_StartAll(CommRequestSharedPtr request)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    if (req->GetNumRequest() != 0)
    {
        MPI_Startall(req->GetNumRequest(), req->GetRequest(0));
    }
}

/**
 *
 */
void CommMpi::v_WaitAll(CommRequestSharedPtr request)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    if (req->GetNumRequest() != 0)
    {
        MPI_Waitall(req->GetNumRequest(), req->GetRequest(0),
                    MPI_STATUSES_IGNORE);
    }
}

/**
 *
 */
CommRequestSharedPtr CommMpi::v_CreateRequest(int num)
{
    return std::shared_ptr<CommRequest>(new CommRequestMpi(num));
}

/**
 * Processes are considered as a grid of size pRows*pColumns. Comm
 * objects are created corresponding to the rows and columns of this
 * grid. The row and column to which this process belongs is stored in
 * #m_commRow and #m_commColumn.
 */
void CommMpi::v_SplitComm(int pRows, int pColumns, int pTime)
{
    if (pTime == 0)
    {
        ASSERTL0(pRows * pColumns == m_size,
                 "Rows/Columns do not match comm size.");

        MPI_Comm newComm;

        // Compute row and column in grid.
        int myCol = m_rank % pColumns;
        int myRow = (m_rank - myCol) / pColumns;

        // Split Comm into rows - all processes with same myRow are put in
        // the same communicator. The rank within this communicator is the
        // column index.
        MPI_Comm_split(m_comm, myRow, myCol, &newComm);
        m_commRow = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into columns - all processes with same myCol are put
        // in the same communicator. The rank within this communicator is
        // the row index.
        MPI_Comm_split(m_comm, myCol, myRow, &newComm);
        m_commColumn = std::shared_ptr<Comm>(new CommMpi(newComm));
    }
    else
    {
        ASSERTL0(pRows * pColumns * pTime == m_size,
                 "Rows/Columns/Time do not match comm size.");

        MPI_Comm newComm;

        // Compute row and column in grid.
        int mySpace = m_rank % (pRows * pColumns);
        int myTime  = (m_rank - mySpace) / (pRows * pColumns);
        int myCol   = mySpace % pColumns;
        int myRow   = (mySpace - myCol) / pColumns;

        // Split Comm - all processes with same mySpace are put in
        // the same communicator. The rank within this communicator is the
        // time index.
        MPI_Comm_split(m_comm, mySpace, myTime, &newComm);
        m_commTime = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm - all processes with same myTime are put in
        // the same communicator. The rank within this communicator is the
        // spatial index.
        MPI_Comm_split(m_comm, myTime, mySpace, &newComm);
        m_commSpace = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into rows - all processes with same myRow are put in
        // the same communicator. The rank within this communicator is the
        // column index.
        MPI_Comm_split(m_comm, myRow + pRows * pColumns * myTime, myCol,
                       &newComm);
        m_commRow = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into columns - all processes with same myCol are put
        // in the same communicator. The rank within this communicator is
        // the row index.
        MPI_Comm_split(m_comm, myCol + pRows * pColumns * myTime, myRow,
                       &newComm);
        m_commColumn = std::shared_ptr<Comm>(new CommMpi(newComm));
    }
}

/**
 * Create a new communicator if the flag is non-zero.
 */
CommSharedPtr CommMpi::v_CommCreateIf(int flag)
{
    MPI_Comm newComm;
    // color == MPI_UNDEF => not in the new communicator
    // key == 0 on all => use rank to order them. OpenMPI, at least,
    // implies this is faster than ordering them ourselves.
    MPI_Comm_split(m_comm, flag ? flag : MPI_UNDEFINED, 0, &newComm);

    if (flag == 0)
    {
        // flag == 0 => get back MPI_COMM_NULL, return a null ptr instead.
        return std::shared_ptr<Comm>();
    }
    else
    {
        // Return a real communicator
        return std::shared_ptr<Comm>(new CommMpi(newComm));
    }
}

/**
 *
 */
std::pair<CommSharedPtr, CommSharedPtr> CommMpi::v_SplitCommNode()
{
    std::pair<CommSharedPtr, CommSharedPtr> ret;

#if MPI_VERSION < 3
    ASSERTL0(false, "Not implemented for non-MPI-3 versions.");
#else
    // Create an intra-node communicator.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, m_rank,
                        MPI_INFO_NULL, &nodeComm);

    // For rank 0 of the intra-node communicator, split the main
    // communicator. Everyone else will get a null communicator.
    ret.first  = std::shared_ptr<Comm>(new CommMpi(nodeComm));
    ret.second = CommMpi::v_CommCreateIf(ret.first->GetRank() == 0);
    if (ret.first->GetRank() == 0)
    {
        ret.second->SplitComm(1, ret.second->GetSize());
    }
#endif

    return ret;
}

} // namespace Nektar::LibUtilities
