///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFT.hip
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
// Description: hipFFT implementation of NekDeviceFFT.h, mirroring
// NekDeviceFFT.cu. hipFFT has no equivalent of the cufftXt store callbacks, so
// the forward 1/N normalisation is always applied by an explicit kernel.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <LibUtilities/FFT/NekDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTHIPCUDAHelper.h>

namespace Nektar::LibUtilities
{

namespace
{

// NekDeviceFFT.h holds the device handles opaquely; these cast them back, and
// the static_asserts fail the build if that storage stops being wide enough.
static_assert(sizeof(hipfftHandle) <= sizeof(std::uintptr_t),
              "NekDeviceFFT.h stores FFT plan handles as uintptr_t");
static_assert(std::is_pointer_v<hipStream_t> && std::is_pointer_v<hipEvent_t> &&
                  std::is_pointer_v<hipGraph_t> &&
                  std::is_pointer_v<hipGraphExec_t>,
              "NekDeviceFFT.h stores HIP handles as void *");

inline hipStream_t AsStream(void *p)
{
    return static_cast<hipStream_t>(p);
}

inline hipEvent_t AsEvent(void *p)
{
    return static_cast<hipEvent_t>(p);
}

inline hipGraph_t AsGraph(void *p)
{
    return static_cast<hipGraph_t>(p);
}

inline hipGraphExec_t AsGraphExec(void *p)
{
    return static_cast<hipGraphExec_t>(p);
}

inline hipfftHandle AsPlan(std::uintptr_t h)
{
    return reinterpret_cast<hipfftHandle>(h);
}

template <typename TData> inline DeviceFFTCmplx<TData> *AsCmplx(void *p)
{
    return static_cast<DeviceFFTCmplx<TData> *>(p);
}

class PlanCache
{
public:
    struct Entry
    {
        std::size_t wsFwd = 0;
        std::size_t wsBwd = 0;
        bool warmedUp     = false;
    };

    static PlanCache &Instance()
    {
        static PlanCache instance;
        return instance;
    }

    bool Lookup(int deviceId, int N, int M, std::size_t elemSize,
                Entry &entry) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_map.find({deviceId, N, M, elemSize});
        if (it == m_map.end())
        {
            return false;
        }
        entry = it->second;
        return true;
    }

    void Register(int deviceId, int N, int M, std::size_t elemSize,
                  const Entry &entry)
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        m_map[{deviceId, N, M, elemSize}] = entry;
    }

private:
    PlanCache() = default;

    struct Key
    {
        int deviceId, N, M;
        std::size_t elemSize; // sizeof(TData): distinguishes double from float
        bool operator==(const Key &other) const noexcept
        {
            return deviceId == other.deviceId && N == other.N && M == other.M &&
                   elemSize == other.elemSize;
        }
    };

    struct KeyHash
    {
        std::size_t operator()(const Key &k) const noexcept
        {
            constexpr std::size_t basis = 14695981039346656037ULL;
            constexpr std::size_t prime = 1099511628211ULL;
            auto mix = [&](std::size_t hash, unsigned v) noexcept {
                hash ^= static_cast<std::size_t>(v);
                return hash * prime;
            };
            std::size_t hash = basis;
            hash             = mix(hash, static_cast<unsigned>(k.deviceId));
            hash             = mix(hash, static_cast<unsigned>(k.N));
            hash             = mix(hash, static_cast<unsigned>(k.M));
            hash             = mix(hash, static_cast<unsigned>(k.elemSize));
            return hash;
        }
    };

    mutable std::shared_mutex m_mutex;
    std::unordered_map<Key, Entry, KeyHash> m_map;
};

} // anonymous namespace

// Factory registration: double uses GetNektarFFTFactory(), float uses
// GetNektarFFTFloatFactory().
template <>
std::string NekDeviceFFTImpl<double>::className =
    GetNektarFFTFactory().RegisterCreatorFunction(
        "NekDeviceFFT", NekDeviceFFTImpl<double>::create);

template <>
std::string NekDeviceFFTImpl<float>::className =
    GetNektarFFTFloatFactory().RegisterCreatorFunction(
        "NekDeviceFFT", NekDeviceFFTImpl<float>::create);

template <typename TData>
NekDeviceFFTImpl<TData>::NekDeviceFFTImpl(int N, int M, bool highPriorityStream)
    : NektarFFT<TData>(N), m_halfN(N / 2), m_batch(M)
{
    const std::size_t nPhys  = static_cast<std::size_t>(M) * N;
    const std::size_t nCmplx = static_cast<std::size_t>(M) * (m_halfN + 1);

    CHECK_HIPCUDA_ERROR(hipMalloc(&m_d_phys, nPhys * sizeof(TData)));
    CHECK_HIPCUDA_ERROR(
        hipMalloc(&m_d_cmplx, nCmplx * sizeof(DeviceFFTCmplx<TData>)));
    CHECK_HIPCUDA_ERROR(hipHostMalloc(&m_h_staging, nPhys * sizeof(TData)));

    {
        int leastPriority, greatestPriority;
        CHECK_HIPCUDA_ERROR(
            hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        const int priority =
            highPriorityStream ? greatestPriority : leastPriority;
        hipStream_t stream = nullptr;
        CHECK_HIPCUDA_ERROR(
            hipStreamCreateWithPriority(&stream, hipStreamDefault, priority));
        m_stream = stream;
    }

    {
        hipfftHandle planFwd = nullptr, planBwd = nullptr;
        CHECK_HIPCUDA_FFT_ERROR(hipfftCreate(&planFwd));
        CHECK_HIPCUDA_FFT_ERROR(hipfftCreate(&planBwd));
        m_planForward  = reinterpret_cast<std::uintptr_t>(planFwd);
        m_planBackward = reinterpret_cast<std::uintptr_t>(planBwd);
    }

    CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(AsPlan(m_planForward), 0));
    CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(AsPlan(m_planBackward), 0));

    CHECK_HIPCUDA_FFT_ERROR(
        hipfftSetStream(AsPlan(m_planForward), AsStream(m_stream)));
    CHECK_HIPCUDA_FFT_ERROR(
        hipfftSetStream(AsPlan(m_planBackward), AsStream(m_stream)));

    std::size_t wsFwd = 0, wsBwd = 0;
    {
        int dims[]    = {N};
        int inembed[] = {N};
        int onembed[] = {m_halfN + 1};

        if constexpr (std::is_same_v<TData, double>)
        {
            CHECK_HIPCUDA_FFT_ERROR(hipfftMakePlanMany(
                AsPlan(m_planForward), 1, dims, inembed, 1, N, onembed, 1,
                m_halfN + 1, HIPFFT_D2Z, M, &wsFwd));
            CHECK_HIPCUDA_FFT_ERROR(hipfftMakePlanMany(
                AsPlan(m_planBackward), 1, dims, onembed, 1, m_halfN + 1,
                inembed, 1, N, HIPFFT_Z2D, M, &wsBwd));
        }
        else
        {
            CHECK_HIPCUDA_FFT_ERROR(hipfftMakePlanMany(
                AsPlan(m_planForward), 1, dims, inembed, 1, N, onembed, 1,
                m_halfN + 1, HIPFFT_R2C, M, &wsFwd));
            CHECK_HIPCUDA_FFT_ERROR(hipfftMakePlanMany(
                AsPlan(m_planBackward), 1, dims, onembed, 1, m_halfN + 1,
                inembed, 1, N, HIPFFT_C2R, M, &wsBwd));
        }
    }

    // No callback path: m_useCallback stays false and m_d_cbParams unused;
    // LaunchScaleRealsImpl applies the 1/N normalisation instead.

    int deviceId = 0;
    CHECK_HIPCUDA_ERROR(hipGetDevice(&deviceId));

    PlanCache::Entry cacheEntry;
    const bool cached =
        PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), cacheEntry);
    if (!cached)
        cacheEntry = {wsFwd, wsBwd, false};

    const std::size_t workspaceBytes =
        std::max(cacheEntry.wsFwd, cacheEntry.wsBwd);

    if (workspaceBytes > 0)
    {
        CHECK_HIPCUDA_ERROR(hipMalloc(&m_d_workspace, workspaceBytes));
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftSetWorkArea(AsPlan(m_planForward), m_d_workspace));
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftSetWorkArea(AsPlan(m_planBackward), m_d_workspace));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftSetAutoAllocation(AsPlan(m_planForward), 1));
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftSetAutoAllocation(AsPlan(m_planBackward), 1));
    }

    ComputeKernelParams();

    if (!cached || !cacheEntry.warmedUp)
    {
        WarmUpPlans();
        cacheEntry.warmedUp = true;
        PlanCache::Instance().Register(deviceId, N, M, sizeof(TData),
                                       cacheEntry);
    }
}

template <typename TData> NekDeviceFFTImpl<TData>::~NekDeviceFFTImpl()
{
    DestroyGraph();

    // Best-effort teardown: a destructor must not throw, so these status
    // codes are discarded deliberately.
    (void)hipfftDestroy(AsPlan(m_planForward));
    (void)hipfftDestroy(AsPlan(m_planBackward));
    (void)hipStreamDestroy(AsStream(m_stream));

    (void)hipFree(m_d_workspace);
    (void)hipFree(m_d_phys);
    (void)hipFree(m_d_cmplx);
    (void)hipHostFree(m_h_staging);
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyGraph()
{
    // Best-effort, as below: nothing useful to do on failure, and this runs
    // from the destructor.
    if (m_graphExec)
    {
        (void)hipGraphExecDestroy(AsGraphExec(m_graphExec));
        m_graphExec = nullptr;
    }
    if (m_graph)
    {
        (void)hipGraphDestroy(AsGraph(m_graph));
        m_graph = nullptr;
    }
    m_hasGraph = false;
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyRawGraph()
{
    if (m_graph)
    {
        (void)hipGraphDestroy(AsGraph(m_graph));
        m_graph = nullptr;
    }
}

template <typename TData> void NekDeviceFFTImpl<TData>::ComputeKernelParams()
{
    int minGridSize;
    CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeWave,
        WavenumberMultiplyKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeC2C,
        ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, false>, 0, 0));
    CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeScaledC2C,
        ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, true>, 0, 0));
    CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeCtC,
        CoefToComplexKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeScale,
        ScaleComplexKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    (void)minGridSize;
}

template <typename TData> void NekDeviceFFTImpl<TData>::WarmUpPlans()
{
    CHECK_HIPCUDA_ERROR(hipMemset(m_d_phys, 0,
                                  static_cast<std::size_t>(m_batch) *
                                      this->m_N * sizeof(TData)));
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecC2R(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(AsStream(m_stream)));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchWavenumberMultiplyImpl(TData beta,
                                                           TData normScale)
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeWave - 1) / m_blockSizeWave);
    WavenumberMultiplyKernel<TData, DeviceFFTCmplx<TData>>
        <<<grid, m_blockSizeWave, 0, AsStream(m_stream)>>>(
            AsCmplx<TData>(m_d_cmplx), m_halfN, beta, normScale);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchComplexToCoefImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeC2C - 1) / m_blockSizeC2C);
    ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, false>
        <<<grid, m_blockSizeC2C, 0, AsStream(m_stream)>>>(
            AsCmplx<TData>(m_d_cmplx), m_d_phys, this->m_N, m_halfN, TData(0));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchScaledComplexToCoefImpl()
{
    const TData invN = TData(1) / static_cast<TData>(this->m_N);
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeScaledC2C - 1) /
                        m_blockSizeScaledC2C);
    ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, true>
        <<<grid, m_blockSizeScaledC2C, 0, AsStream(m_stream)>>>(
            AsCmplx<TData>(m_d_cmplx), m_d_phys, this->m_N, m_halfN, invN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchCoefToComplexImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeCtC - 1) / m_blockSizeCtC);
    CoefToComplexKernel<TData, DeviceFFTCmplx<TData>>
        <<<grid, m_blockSizeCtC, 0, AsStream(m_stream)>>>(
            m_d_phys, AsCmplx<TData>(m_d_cmplx), this->m_N, m_halfN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchScaleRealsImpl(TData alpha)
{
    const int nComplex = m_batch * (m_halfN + 1);
    const int blocks   = (nComplex + m_blockSizeScale - 1) / m_blockSizeScale;
    ScaleComplexKernel<TData, DeviceFFTCmplx<TData>>
        <<<blocks, m_blockSizeScale, 0, AsStream(m_stream)>>>(
            AsCmplx<TData>(m_d_cmplx), nComplex, alpha);
}

template <typename TData> void NekDeviceFFTImpl<TData>::WaitOnEvent(void *event)
{
    if (event != nullptr)
    {
        CHECK_HIPCUDA_ERROR(
            hipStreamWaitEvent(AsStream(m_stream), AsEvent(event), 0));
    }
}

template <typename TData>
void NekDeviceFFTImpl<TData>::UploadPhys(const TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, phys, nBytes);
    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                                       hipMemcpyHostToDevice,
                                       AsStream(m_stream)));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadPhys(TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                       hipMemcpyDeviceToHost,
                                       AsStream(m_stream)));
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(AsStream(m_stream)));
    std::memcpy(phys, m_h_staging, nBytes);
}

// Stages coef data through m_d_phys as a scratch buffer, which overwrites
// any physical-space data currently held there.
template <typename TData>
void NekDeviceFFTImpl<TData>::UploadCoef(const TData *coef)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, coef, nBytes);

    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                                       hipMemcpyHostToDevice,
                                       AsStream(m_stream)));

    LaunchCoefToComplexImpl();
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadCoef(TData *coef)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    LaunchComplexToCoefImpl();

    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                       hipMemcpyDeviceToHost,
                                       AsStream(m_stream)));

    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(AsStream(m_stream)));
    std::memcpy(coef, m_h_staging, nBytes);
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTFwdTransDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
    }

    LaunchScaleRealsImpl(TData(1) / static_cast<TData>(this->m_N));
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTBwdTransDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecC2R(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTExecOnlyDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecC2R(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
}

template <typename TData>
void NekDeviceFFTImpl<TData>::WavenumberMultiply(TData beta)
{
    LaunchWavenumberMultiplyImpl(beta, TData(1));
}

template <typename TData> void NekDeviceFFTImpl<TData>::BeginGraphCapture()
{
    m_hasGraph = false;
    DestroyRawGraph();
    CHECK_HIPCUDA_ERROR(
        hipStreamBeginCapture(AsStream(m_stream), hipStreamCaptureModeGlobal));
}

template <typename TData> void NekDeviceFFTImpl<TData>::EndGraphCapture()
{
    hipGraph_t graph = nullptr;
    CHECK_HIPCUDA_ERROR(hipStreamEndCapture(AsStream(m_stream), &graph));
    m_graph = graph;

    if (m_graphExec)
    {
        hipGraphNode_t errorNode;
        hipGraphExecUpdateResult updateResult;
        const hipError_t updateErr =
            hipGraphExecUpdate(AsGraphExec(m_graphExec), AsGraph(m_graph),
                               &errorNode, &updateResult);
        const bool updated = (updateErr == hipSuccess &&
                              updateResult == hipGraphExecUpdateSuccess);
        if (updated)
        {
            m_hasGraph = true;
            return;
        }
        (void)hipGraphExecDestroy(AsGraphExec(m_graphExec));
        m_graphExec = nullptr;
    }

    hipGraphExec_t graphExec = nullptr;
    CHECK_HIPCUDA_ERROR(
        hipGraphInstantiate(&graphExec, AsGraph(m_graph), nullptr, nullptr, 0));
    m_graphExec = graphExec;
    m_hasGraph  = true;
}

template <typename TData> void NekDeviceFFTImpl<TData>::LaunchGraph()
{
    if (!m_hasGraph)
        throw std::runtime_error(
            "NekDeviceFFT::LaunchGraph: no graph captured. "
            "Call BeginGraphCapture / EndGraphCapture first.");
    CHECK_HIPCUDA_ERROR(
        hipGraphLaunch(AsGraphExec(m_graphExec), AsStream(m_stream)));
}

template <typename TData>
bool NekDeviceFFTImpl<TData>::IsCacheWarmed(int deviceId, int N, int M)
{
    PlanCache::Entry entry;
    return PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), entry) &&
           entry.warmedUp;
}

template <typename TData>
void NekDeviceFFTImpl<TData>::v_FFTFwdTrans(TData *inarray, TData *outarray)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    UploadPhys(inarray);

    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(AsPlan(m_planForward), m_d_phys,
                                              AsCmplx<TData>(m_d_cmplx)));
    }

    LaunchScaledComplexToCoefImpl();

    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                       hipMemcpyDeviceToHost,
                                       AsStream(m_stream)));
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(AsStream(m_stream)));
    std::memcpy(outarray, m_h_staging, nBytes);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::v_FFTBwdTrans(TData *inarray, TData *outarray)
{
    UploadCoef(inarray);
    FFTBwdTransDevice();
    DownloadPhys(outarray);
}

// Explicit instantiations for double and float.
template class NekDeviceFFTImpl<double>;
template class NekDeviceFFTImpl<float>;

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
