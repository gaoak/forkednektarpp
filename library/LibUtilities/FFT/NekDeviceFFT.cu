///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFT.cu
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
// Description: cuFFT implementation of NekDeviceFFT.h. See NekDeviceFFT.hip
// for the HIP counterpart.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <cstring>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cufftXt.h>

#include <LibUtilities/Backends/CUDAStream.hpp>
#include <LibUtilities/FFT/NekDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTHIPCUDAHelper.h>

namespace Nektar::LibUtilities
{

namespace
{

// NekDeviceFFT.h holds the device handles opaquely; these cast them back, and
// the static_asserts fail the build if that storage stops being wide enough.
static_assert(sizeof(cufftHandle) <= sizeof(std::uintptr_t),
              "NekDeviceFFT.h stores FFT plan handles as uintptr_t");
static_assert(std::is_pointer_v<cudaGraph_t> &&
                  std::is_pointer_v<cudaGraphExec_t>,
              "NekDeviceFFT.h stores CUDA graph handles as void *");

inline cudaStream_t AsStream(unsigned int streamID)
{
    return CUDAStream::GetInstance(streamID);
}

inline cudaGraph_t AsGraph(void *p)
{
    return static_cast<cudaGraph_t>(p);
}

inline cudaGraphExec_t AsGraphExec(void *p)
{
    return static_cast<cudaGraphExec_t>(p);
}

inline cufftHandle AsPlan(std::uintptr_t h)
{
    return static_cast<cufftHandle>(h);
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

// Callback parameter block, templated so invN/beta match the plan precision.
template <typename TReal> struct StoreScaledParams
{
    TReal invN;
    TReal beta;
    int halfN;
};

// Double-precision store callback.
__device__ static void StoreScaledCBDouble(void *dataOut, size_t offset,
                                           cufftDoubleComplex element,
                                           void *callerInfo, void *)
{
    const StoreScaledParams<double> *p =
        reinterpret_cast<const StoreScaledParams<double> *>(callerInfo);
    if (p->beta == 0.0)
    {
        reinterpret_cast<cufftDoubleComplex *>(dataOut)[offset] = {
            element.x * p->invN, element.y * p->invN};
        return;
    }
    const int k = static_cast<int>(offset % static_cast<size_t>(p->halfN + 1));
    if (k == 0 || k == p->halfN)
    {
        reinterpret_cast<cufftDoubleComplex *>(dataOut)[offset] = {0.0, 0.0};
        return;
    }
    const double scale = static_cast<double>(k) * p->beta * p->invN;
    reinterpret_cast<cufftDoubleComplex *>(dataOut)[offset] = {
        -element.y * scale, element.x * scale};
}

__device__ cufftCallbackStoreZ d_StoreScaledCBDouble = StoreScaledCBDouble;

// Single-precision store callback.
__device__ static void StoreScaledCBFloat(void *dataOut, size_t offset,
                                          cufftComplex element,
                                          void *callerInfo, void *)
{
    const StoreScaledParams<float> *p =
        reinterpret_cast<const StoreScaledParams<float> *>(callerInfo);
    if (p->beta == 0.0f)
    {
        reinterpret_cast<cufftComplex *>(dataOut)[offset] = {
            element.x * p->invN, element.y * p->invN};
        return;
    }
    const int k = static_cast<int>(offset % static_cast<size_t>(p->halfN + 1));
    if (k == 0 || k == p->halfN)
    {
        reinterpret_cast<cufftComplex *>(dataOut)[offset] = {0.0f, 0.0f};
        return;
    }
    const float scale = static_cast<float>(k) * p->beta * p->invN;
    reinterpret_cast<cufftComplex *>(dataOut)[offset] = {-element.y * scale,
                                                         element.x * scale};
}

__device__ cufftCallbackStoreC d_StoreScaledCBFloat = StoreScaledCBFloat;

template <typename TData>
static cufftResult RegisterFwdScaleCallback(cufftHandle plan,
                                            StoreScaledParams<TData> *d_params)
{
    if constexpr (std::is_same_v<TData, double>)
    {
        cufftCallbackStoreZ hostCbPtr;
        const cudaError_t copyErr = cudaMemcpyFromSymbol(
            &hostCbPtr, d_StoreScaledCBDouble, sizeof(hostCbPtr));
        if (copyErr != cudaSuccess)
            return CUFFT_INTERNAL_ERROR;
        return cufftXtSetCallback(plan, reinterpret_cast<void **>(&hostCbPtr),
                                  CUFFT_CB_ST_COMPLEX_DOUBLE,
                                  reinterpret_cast<void **>(&d_params));
    }
    else
    {
        cufftCallbackStoreC hostCbPtr;
        const cudaError_t copyErr = cudaMemcpyFromSymbol(
            &hostCbPtr, d_StoreScaledCBFloat, sizeof(hostCbPtr));
        if (copyErr != cudaSuccess)
            return CUFFT_INTERNAL_ERROR;
        return cufftXtSetCallback(plan, reinterpret_cast<void **>(&hostCbPtr),
                                  CUFFT_CB_ST_COMPLEX,
                                  reinterpret_cast<void **>(&d_params));
    }
}

template <typename TData> static bool ProbeCallbackSupport()
{
    int dims[]        = {8};
    cufftHandle probe = 0;
    constexpr cufftType planType =
        std::is_same_v<TData, double> ? CUFFT_D2Z : CUFFT_R2C;
    // Probe output stride: D2Z gives halfN+1=5 complex, R2C same.
    if (cufftPlanMany(&probe, 1, dims, nullptr, 1, 8, nullptr, 1, 5, planType,
                      1) != CUFFT_SUCCESS)
    {
        return false;
    }

    cufftSetAutoAllocation(probe, 1);

    StoreScaledParams<TData> *d_probe = nullptr;
    if (cudaMalloc(&d_probe, sizeof(StoreScaledParams<TData>)) != cudaSuccess)
    {
        cufftDestroy(probe);
        return false;
    }

    const cufftResult cbResult =
        RegisterFwdScaleCallback<TData>(probe, d_probe);

    cudaFree(d_probe);
    cufftDestroy(probe);

    return cbResult == CUFFT_SUCCESS;
}

template <typename TData> static bool CallbackSupportedOnce()
{
    static const bool s_supported = ProbeCallbackSupport<TData>();
    return s_supported;
}

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
NekDeviceFFTImpl<TData>::NekDeviceFFTImpl(int N, int M, unsigned int streamID)
    : NektarFFT<TData>(N), m_halfN(N / 2), m_batch(M), m_streamID(streamID)
{
    const std::size_t nPhys  = static_cast<std::size_t>(M) * N;
    const std::size_t nCmplx = static_cast<std::size_t>(M) * (m_halfN + 1);

    CHECK_HIPCUDA_ERROR(cudaMalloc(&m_d_phys, nPhys * sizeof(TData)));
    CHECK_HIPCUDA_ERROR(
        cudaMalloc(&m_d_cmplx, nCmplx * sizeof(DeviceFFTCmplx<TData>)));
    CHECK_HIPCUDA_ERROR(cudaMallocHost(&m_h_staging, nPhys * sizeof(TData)));

    {
        cufftHandle planFwd = 0, planBwd = 0;
        CHECK_HIPCUDA_FFT_ERROR(cufftCreate(&planFwd));
        CHECK_HIPCUDA_FFT_ERROR(cufftCreate(&planBwd));
        m_planForward  = static_cast<std::uintptr_t>(planFwd);
        m_planBackward = static_cast<std::uintptr_t>(planBwd);
    }

    CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(AsPlan(m_planForward), 0));
    CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(AsPlan(m_planBackward), 0));

    CHECK_HIPCUDA_FFT_ERROR(
        cufftSetStream(AsPlan(m_planForward), AsStream(m_streamID)));
    CHECK_HIPCUDA_FFT_ERROR(
        cufftSetStream(AsPlan(m_planBackward), AsStream(m_streamID)));

    std::size_t wsFwd = 0, wsBwd = 0;
    {
        int dims[]    = {N};
        int inembed[] = {N};
        int onembed[] = {m_halfN + 1};

        if constexpr (std::is_same_v<TData, double>)
        {
            CHECK_HIPCUDA_FFT_ERROR(cufftMakePlanMany(
                AsPlan(m_planForward), 1, dims, inembed, 1, N, onembed, 1,
                m_halfN + 1, CUFFT_D2Z, M, &wsFwd));
            CHECK_HIPCUDA_FFT_ERROR(cufftMakePlanMany(
                AsPlan(m_planBackward), 1, dims, onembed, 1, m_halfN + 1,
                inembed, 1, N, CUFFT_Z2D, M, &wsBwd));
        }
        else
        {
            CHECK_HIPCUDA_FFT_ERROR(cufftMakePlanMany(
                AsPlan(m_planForward), 1, dims, inembed, 1, N, onembed, 1,
                m_halfN + 1, CUFFT_R2C, M, &wsFwd));
            CHECK_HIPCUDA_FFT_ERROR(cufftMakePlanMany(
                AsPlan(m_planBackward), 1, dims, onembed, 1, m_halfN + 1,
                inembed, 1, N, CUFFT_C2R, M, &wsBwd));
        }
    }

    if (CallbackSupportedOnce<TData>())
    {
        const StoreScaledParams<TData> h_params{
            TData(1.0) / static_cast<TData>(N), TData(0.0), m_halfN};
        StoreScaledParams<TData> *d_params = nullptr;
        CHECK_HIPCUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_params),
                                       sizeof(StoreScaledParams<TData>)));
        CHECK_HIPCUDA_ERROR(cudaMemcpy(d_params, &h_params, sizeof(h_params),
                                       cudaMemcpyHostToDevice));
        m_d_cbParams = d_params;
        CHECK_HIPCUDA_FFT_ERROR(
            RegisterFwdScaleCallback<TData>(AsPlan(m_planForward), d_params));
        m_useCallback = true;

        std::size_t wsFwdPostCb = 0;
        if (cufftGetSize(AsPlan(m_planForward), &wsFwdPostCb) == CUFFT_SUCCESS)
            wsFwd = std::max(wsFwd, wsFwdPostCb);
    }

    int deviceId = 0;
    CHECK_HIPCUDA_ERROR(cudaGetDevice(&deviceId));

    PlanCache::Entry cacheEntry;
    const bool cached =
        PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), cacheEntry);
    if (!cached)
        cacheEntry = {wsFwd, wsBwd, false};

    const std::size_t workspaceBytes =
        std::max(cacheEntry.wsFwd, cacheEntry.wsBwd);

    if (workspaceBytes > 0)
    {
        CHECK_HIPCUDA_ERROR(cudaMalloc(&m_d_workspace, workspaceBytes));
        CHECK_HIPCUDA_FFT_ERROR(
            cufftSetWorkArea(AsPlan(m_planForward), m_d_workspace));
        CHECK_HIPCUDA_FFT_ERROR(
            cufftSetWorkArea(AsPlan(m_planBackward), m_d_workspace));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(
            cufftSetAutoAllocation(AsPlan(m_planForward), 1));
        CHECK_HIPCUDA_FFT_ERROR(
            cufftSetAutoAllocation(AsPlan(m_planBackward), 1));
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

    cufftDestroy(AsPlan(m_planForward));
    cufftDestroy(AsPlan(m_planBackward));

    cudaFree(m_d_workspace);
    cudaFree(m_d_cbParams);
    cudaFree(m_d_phys);
    cudaFree(m_d_cmplx);
    cudaFreeHost(m_h_staging);
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyGraph()
{
    if (m_graphExec)
    {
        cudaGraphExecDestroy(AsGraphExec(m_graphExec));
        m_graphExec = nullptr;
    }
    if (m_graph)
    {
        cudaGraphDestroy(AsGraph(m_graph));
        m_graph = nullptr;
    }
    m_hasGraph = false;
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyRawGraph()
{
    if (m_graph)
    {
        cudaGraphDestroy(AsGraph(m_graph));
        m_graph = nullptr;
    }
}

template <typename TData> void NekDeviceFFTImpl<TData>::ComputeKernelParams()
{
    int minGridSize;
    CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeWave,
        WavenumberMultiplyKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeC2C,
        ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, false>, 0, 0));
    CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeScaledC2C,
        ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, true>, 0, 0));
    CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeCtC,
        CoefToComplexKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &m_blockSizeScale,
        ScaleComplexKernel<TData, DeviceFFTCmplx<TData>>, 0, 0));
    (void)minGridSize;
}

template <typename TData> void NekDeviceFFTImpl<TData>::WarmUpPlans()
{
    CHECK_HIPCUDA_ERROR(cudaMemset(m_d_phys, 0,
                                   static_cast<std::size_t>(m_batch) *
                                       this->m_N * sizeof(TData)));
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecC2R(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(AsStream(m_streamID)));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchWavenumberMultiplyImpl(TData beta,
                                                           TData normScale)
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeWave - 1) / m_blockSizeWave);
    WavenumberMultiplyKernel<TData, DeviceFFTCmplx<TData>>
        <<<grid, m_blockSizeWave, 0, AsStream(m_streamID)>>>(
            AsCmplx<TData>(m_d_cmplx), m_halfN, beta, normScale);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchComplexToCoefImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeC2C - 1) / m_blockSizeC2C);
    ComplexToCoefKernel<TData, DeviceFFTCmplx<TData>, false>
        <<<grid, m_blockSizeC2C, 0, AsStream(m_streamID)>>>(
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
        <<<grid, m_blockSizeScaledC2C, 0, AsStream(m_streamID)>>>(
            AsCmplx<TData>(m_d_cmplx), m_d_phys, this->m_N, m_halfN, invN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchCoefToComplexImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeCtC - 1) / m_blockSizeCtC);
    CoefToComplexKernel<TData, DeviceFFTCmplx<TData>>
        <<<grid, m_blockSizeCtC, 0, AsStream(m_streamID)>>>(
            m_d_phys, AsCmplx<TData>(m_d_cmplx), this->m_N, m_halfN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchScaleRealsImpl(TData alpha)
{
    const int nComplex = m_batch * (m_halfN + 1);
    const int blocks   = (nComplex + m_blockSizeScale - 1) / m_blockSizeScale;
    ScaleComplexKernel<TData, DeviceFFTCmplx<TData>>
        <<<blocks, m_blockSizeScale, 0, AsStream(m_streamID)>>>(
            AsCmplx<TData>(m_d_cmplx), nComplex, alpha);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::WaitOnStream(unsigned int producerStreamID)
{
    const cudaEvent_t event = CUDAStream::GetEvent(producerStreamID);
    if (producerStreamID != m_streamID && event != nullptr)
    {
        CHECK_HIPCUDA_ERROR(cudaStreamWaitEvent(AsStream(m_streamID), event));
    }
}

template <typename TData>
void NekDeviceFFTImpl<TData>::UploadPhys(const TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, phys, nBytes);
    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                                        cudaMemcpyHostToDevice,
                                        AsStream(m_streamID)));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadPhys(TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                        cudaMemcpyDeviceToHost,
                                        AsStream(m_streamID)));
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(AsStream(m_streamID)));
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

    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                                        cudaMemcpyHostToDevice,
                                        AsStream(m_streamID)));

    LaunchCoefToComplexImpl();
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadCoef(TData *coef)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    LaunchComplexToCoefImpl();

    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                        cudaMemcpyDeviceToHost,
                                        AsStream(m_streamID)));

    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(AsStream(m_streamID)));
    std::memcpy(coef, m_h_staging, nBytes);
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTFwdTransDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
    }

    if (!m_useCallback)
    {
        LaunchScaleRealsImpl(TData(1) / static_cast<TData>(this->m_N));
    }
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTBwdTransDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecC2R(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTExecOnlyDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecZ2D(
            AsPlan(m_planBackward), AsCmplx<TData>(m_d_cmplx), m_d_phys));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecC2R(
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
    if (m_streamID == 0)
    {
        throw std::runtime_error(
            "NekDeviceFFT::BeginGraphCapture: stream 0 is the legacy default "
            "stream, which cannot be captured; use a non-zero stream ID.");
    }

    m_hasGraph = false;
    DestroyRawGraph();
    CHECK_HIPCUDA_ERROR(cudaStreamBeginCapture(AsStream(m_streamID),
                                               cudaStreamCaptureModeGlobal));
}

template <typename TData> void NekDeviceFFTImpl<TData>::EndGraphCapture()
{
    cudaGraph_t graph = nullptr;
    CHECK_HIPCUDA_ERROR(cudaStreamEndCapture(AsStream(m_streamID), &graph));
    m_graph = graph;

    if (m_graphExec)
    {
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo info{};
        const cudaError_t updateErr = cudaGraphExecUpdate(
            AsGraphExec(m_graphExec), AsGraph(m_graph), &info);
        (void)cudaGetLastError();
        const bool updated = (updateErr == cudaSuccess &&
                              info.result == cudaGraphExecUpdateSuccess);
#else
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdateResult updateResult;
        const cudaError_t updateErr =
            cudaGraphExecUpdate(AsGraphExec(m_graphExec), AsGraph(m_graph),
                                &errorNode, &updateResult);
        const bool updated = (updateErr == cudaSuccess &&
                              updateResult == cudaGraphExecUpdateSuccess);
#endif
        if (updated)
        {
            m_hasGraph = true;
            return;
        }
        cudaGraphExecDestroy(AsGraphExec(m_graphExec));
        m_graphExec = nullptr;
    }

    cudaGraphExec_t graphExec = nullptr;
#if CUDART_VERSION >= 12000
    CHECK_HIPCUDA_ERROR(cudaGraphInstantiate(&graphExec, AsGraph(m_graph), 0));
#else
    CHECK_HIPCUDA_ERROR(cudaGraphInstantiate(&graphExec, AsGraph(m_graph),
                                             nullptr, nullptr, 0));
#endif
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
        cudaGraphLaunch(AsGraphExec(m_graphExec), AsStream(m_streamID)));
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
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(AsPlan(m_planForward), m_d_phys,
                                             AsCmplx<TData>(m_d_cmplx)));
    }

    if (m_useCallback)
        LaunchComplexToCoefImpl();
    else
        LaunchScaledComplexToCoefImpl();

    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                                        cudaMemcpyDeviceToHost,
                                        AsStream(m_streamID)));
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(AsStream(m_streamID)));
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
