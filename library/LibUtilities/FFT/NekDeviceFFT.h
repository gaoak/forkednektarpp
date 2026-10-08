///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFT.h
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
// Description: Device FFT wrapper for Nektar++. Implemented over cuFFT in
// NekDeviceFFT.cu and hipFFT in NekDeviceFFT.hip; this header is backend
// agnostic and may be included from a plain .cpp.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <string>

#include <LibUtilities/FFT/NektarFFT.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>

namespace Nektar::LibUtilities
{

/**
 * \brief Device-FFT-backed NektarFFT implementation supporting batched real-
 * to-complex and complex-to-real transforms with optional graph capture.
 *
 * TData must be double or float. Owns its plans and buffers; all work is
 * submitted to the stream that the backend's stream registry (CUDAStream,
 * HIPStream or SYCLQueue) holds for the stream ID given at construction.
 * Non-copyable and non-movable.
 *
 * Graph capture is a CUDA and HIP facility, and needs a non-zero stream ID:
 * ID 0 is the legacy default stream, which cannot be captured. The SYCL
 * backend has no equivalent, so HasGraph() stays false there and the capture
 * entry points throw rather than record.
 *
 * Internal template name is NekDeviceFFTImpl. The public aliases NekDeviceFFT
 * and NekDeviceFFTFloat refer to the double and float specialisations
 * respectively.
 */
template <typename TData> class NekDeviceFFTImpl : public NektarFFT<TData>
{
public:
    static std::shared_ptr<NektarFFT<TData>> create(int N)
    {
        return MemoryManager<NekDeviceFFTImpl<TData>>::AllocateSharedPtr(N, 1);
    }

    static std::string className;

    /// \brief Construct a NekDeviceFFTImpl for transforms of size \p N with
    /// batch count \p M.
    ///
    /// \param N        Transform size (number of real points).
    /// \param M        Batch size (default 1).
    /// \param streamID Id of the registry stream all work is submitted to.
    NekDeviceFFTImpl(int N, int M = 1, unsigned int streamID = 0);

    ~NekDeviceFFTImpl() override;

    // Non-copyable and non-movable: owns raw device resources.
    NekDeviceFFTImpl(const NekDeviceFFTImpl &)            = delete;
    NekDeviceFFTImpl &operator=(const NekDeviceFFTImpl &) = delete;
    NekDeviceFFTImpl(NekDeviceFFTImpl &&)                 = delete;
    NekDeviceFFTImpl &operator=(NekDeviceFFTImpl &&)      = delete;

    /// \brief Copy host physical-space data to the internal device buffer.
    void UploadPhys(const TData *phys);

    /// \brief Copy internal device physical-space buffer back to the host.
    void DownloadPhys(TData *phys);

    /// \brief Copy host coefficient data to the internal device buffer.
    void UploadCoef(const TData *coef);

    /// \brief Copy internal device coefficient buffer back to the host.
    void DownloadCoef(TData *coef);

    /// \brief Execute forward transform + 1/N normalisation on the device.
    void FFTFwdTransDevice();

    /// \brief Execute backward transform on the device (unnormalised).
    void FFTBwdTransDevice();

    /// \brief Execute forward + backward without normalisation (benchmarking).
    void FFTExecOnlyDevice();

    /// \brief Multiply the complex spectrum by i*k*beta in-place.
    void WavenumberMultiply(TData beta);

    /// \brief Begin recording a device graph for the current stream.
    void BeginGraphCapture();

    /// \brief End graph recording and instantiate the captured graph.
    void EndGraphCapture();

    /// \brief Launch the previously captured device graph.
    void LaunchGraph();

    /// \brief Return true if a device graph has been captured and is ready.
    bool HasGraph() const
    {
        return m_hasGraph;
    }

    /// \brief Return true if the plan cache has been warmed up for the given
    /// device, transform size, and batch count.
    static bool IsCacheWarmed(int deviceId, int N, int M);

    /// \brief Return the id of the registry stream used by this object.
    unsigned int GetStreamID() const
    {
        return m_streamID;
    }

    /// \brief Make this object's stream wait on the last event recorded on
    ///        another registry stream, before any subsequent call touches
    ///        device memory that stream produced.
    ///
    /// The event is the one the backend registry holds for \p
    /// producerStreamID (recorded through CUDAStream::RecordEvent,
    /// HIPStream::RecordEvent or SYCLQueue::SetEvent). Callers that only
    /// go through UploadPhys/UploadCoef (host pointers) do not need this,
    /// since those copies read host memory, not another stream's device
    /// writes. No-op if no event has been recorded on \p producerStreamID,
    /// or if it is this object's own stream.
    ///
    /// \param producerStreamID Id of the stream that produced the data.
    void WaitOnStream(unsigned int producerStreamID);

    /// \brief Return the batch size (M).
    int BatchSize() const
    {
        return m_batch;
    }

    /// \brief Return N/2 (number of independent complex Fourier modes).
    int HalfN() const
    {
        return m_halfN;
    }

protected:
    void v_FFTFwdTrans(TData *inarray, TData *outarray) override;
    void v_FFTBwdTrans(TData *inarray, TData *outarray) override;

private:
    int m_halfN;
    int m_batch;

    // Handles are held opaquely so this header needs no device or FFT
    // headers; the backend source casts them back, guarded by static_asserts.
    // uintptr_t because a plan handle is an int under cuFFT but a pointer
    // under hipFFT.
    // Under oneMath a single descriptor drives both directions, so the
    // SYCL backend uses m_planForward alone and leaves m_planBackward zero.
    std::uintptr_t m_planForward  = 0; // cufftHandle / hipfftHandle
    std::uintptr_t m_planBackward = 0; // cufftHandle / hipfftHandle
    unsigned int m_streamID;

    TData *m_d_phys     = nullptr;
    void *m_d_cmplx     = nullptr; // device half-spectrum buffer
    void *m_d_cbParams  = nullptr; // StoreScaledParams<TData> on device
    void *m_d_workspace = nullptr;
    TData *m_h_staging  = nullptr;

    bool m_useCallback = false;

    int m_blockSizeWave      = 0;
    int m_blockSizeC2C       = 0;
    int m_blockSizeScaledC2C = 0;
    int m_blockSizeCtC       = 0;
    int m_blockSizeScale     = 0;

    void *m_graph     = nullptr; // cudaGraph_t / hipGraph_t
    void *m_graphExec = nullptr; // cudaGraphExec_t / hipGraphExec_t
    bool m_hasGraph   = false;

    void ComputeKernelParams();
    void LaunchWavenumberMultiplyImpl(TData beta, TData normScale);
    void LaunchComplexToCoefImpl();
    void LaunchScaledComplexToCoefImpl();
    void LaunchCoefToComplexImpl();
    void LaunchScaleRealsImpl(TData alpha);
    void DestroyGraph();
    void DestroyRawGraph();
    void WarmUpPlans();
};

// Backward-compatible aliases so existing call sites (including unit tests)
// that use the unparameterised name NekDeviceFFT continue to compile unchanged.
using NekDeviceFFT      = NekDeviceFFTImpl<double>;
using NekDeviceFFTFloat = NekDeviceFFTImpl<float>;

// className is specialised in the backend source, and must be declared before
// the instantiation declarations below or the specialisation comes too late.
// With no initializer these are declarations, not definitions.
template <> std::string NekDeviceFFTImpl<double>::className;
template <> std::string NekDeviceFFTImpl<float>::className;

// Suppress redundant per-TU instantiation; definitions live in
// NekDeviceFFT.cu (CUDA), NekDeviceFFT.hip (HIP) and NekDeviceFFT.cpp (SYCL).
extern template class NekDeviceFFTImpl<double>;
extern template class NekDeviceFFTImpl<float>;

} // namespace Nektar::LibUtilities
