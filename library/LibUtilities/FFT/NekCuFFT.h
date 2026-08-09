///////////////////////////////////////////////////////////////////////////////
//
// File: NekCuFFT.h
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
// Description: CUDA (cuFFT) backend FFT wrapper for Nektar++.
//
// Internal backend header — include NekDeviceFFT.h instead for
// portable code that should work across CUDA/HIP/SYCL backends.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_FFT_NEKCUFFT_H
#define NEKTAR_LIB_UTILITIES_FFT_NEKCUFFT_H

#ifndef NEKTAR_ENABLE_CUDA
#error "NekCuFFT.h requires CUDA support (NEKTAR_ENABLE_CUDA). \
Configure with NEKTAR_ENABLE_DEVICE=CUDA."
#endif

#include <type_traits>

#include <cufft.h>

#include <LibUtilities/FFT/NektarFFT.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>

namespace Nektar::LibUtilities
{

/**
 * \brief cuFFT-backed NektarFFT implementation supporting batched real-to-
 * complex and complex-to-real transforms with optional CUDA graph capture.
 *
 * TData must be double or float. Owns all CUDA resources (stream, plans,
 * device buffers). Non-copyable and non-movable.
 *
 * Internal template name is NekCuFFTImpl. The public aliases NekCuFFT and
 * NekCuFFTFloat refer to the double and float specialisations respectively.
 */
template <typename TData> class NekCuFFTImpl : public NektarFFT<TData>
{
public:
    // Complex element type for the chosen real precision.
    using CmplxType = std::conditional_t<std::is_same_v<TData, double>,
                                         cufftDoubleComplex, cufftComplex>;

    static std::shared_ptr<NektarFFT<TData>> create(int N)
    {
        return MemoryManager<NekCuFFTImpl<TData>>::AllocateSharedPtr(N, 1);
    }

    static std::string className;

    /// \brief Construct a NekCuFFTImpl for transforms of size \p N with batch
    /// count \p M.
    ///
    /// \param N                  Transform size (number of real points).
    /// \param M                  Batch size (default 1).
    /// \param highPriorityStream Use a high-priority CUDA stream.
    NekCuFFTImpl(int N, int M = 1, bool highPriorityStream = false);

    ~NekCuFFTImpl() override;

    // Non-copyable and non-movable: owns raw CUDA resources.
    NekCuFFTImpl(const NekCuFFTImpl &)            = delete;
    NekCuFFTImpl &operator=(const NekCuFFTImpl &) = delete;
    NekCuFFTImpl(NekCuFFTImpl &&)                 = delete;
    NekCuFFTImpl &operator=(NekCuFFTImpl &&)      = delete;

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

    /// \brief Begin recording a CUDA graph for the current stream.
    void BeginGraphCapture();

    /// \brief End graph recording and instantiate the captured graph.
    void EndGraphCapture();

    /// \brief Launch the previously captured CUDA graph.
    void LaunchGraph();

    /// \brief Return true if a CUDA graph has been captured and is ready.
    bool HasGraph() const
    {
        return m_hasGraph;
    }

    /// \brief Return true if the plan cache has been warmed up for the given
    /// device, transform size, and batch count.
    static bool IsCacheWarmed(int deviceId, int N, int M);

    /// \brief Return the CUDA stream used by this object.
    cudaStream_t GetStream() const
    {
        return m_stream;
    }

    /// \brief Make this object's internal stream wait on an externally
    ///        recorded event before any subsequent call touches device
    ///        memory that event guards.
    ///
    /// m_stream is a private stream owned by this object; it does not
    /// otherwise participate in any caller-side stream/event bookkeeping
    /// (e.g. the Operators library's per-block stream discipline). A
    /// caller that is about to feed this object device-resident data
    /// produced on a different stream should record an event on that
    /// producer stream and pass it here first -- callers that only ever
    /// go through UploadPhys/UploadCoef (host pointers) do not need this,
    /// since those copies read host memory, not another stream's device
    /// writes. No-op if \p event is null.
    void WaitOnEvent(cudaEvent_t event);

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

    cufftHandle m_planForward  = 0;
    cufftHandle m_planBackward = 0;
    cudaStream_t m_stream      = nullptr;

    TData *m_d_phys      = nullptr;
    CmplxType *m_d_cmplx = nullptr;
    void *m_d_cbParams   = nullptr; // StoreScaledParams<TData> on device
    void *m_d_workspace  = nullptr;
    TData *m_h_staging   = nullptr;

    bool m_useCallback = false;

    int m_blockSizeWave      = 0;
    int m_blockSizeC2C       = 0;
    int m_blockSizeScaledC2C = 0;
    int m_blockSizeCtC       = 0;
    int m_blockSizeScale     = 0;

    cudaGraph_t m_graph         = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_hasGraph             = false;

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
// that use the unparameterised name NekCuFFT continue to compile unchanged.
using NekCuFFT      = NekCuFFTImpl<double>;
using NekCuFFTFloat = NekCuFFTImpl<float>;

// Suppress redundant per-TU instantiation; definitions live in NekCuFFT.cu.
extern template class NekCuFFTImpl<double>;
extern template class NekCuFFTImpl<float>;

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIB_UTILITIES_FFT_NEKCUFFT_H
