///////////////////////////////////////////////////////////////////////////////
//
// File: NekCuFFT.cu
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
// Description: cuFFT-based FFT wrapper for Nektar++.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <cstring>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cufft.h>
#include <cufftXt.h>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/FFT/NekCuFFT.h>

#ifdef NEKTAR_ENABLE_NVTX
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
struct NvtxRange
{
    NvtxRange(const char *label)
    {
        nvtxRangePushA(label);
    }
    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#define NVTX_RANGE(label) NvtxRange _nvtx_##__LINE__(label)
#define NVTX_PUSH(label) nvtxRangePushA(label)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_RANGE(label) ((void)0)
#define NVTX_PUSH(label) ((void)0)
#define NVTX_POP() ((void)0)
#endif

namespace Nektar::LibUtilities
{

namespace
{

inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " +
                                 cudaGetErrorString(err));
}

inline void checkCufft(cufftResult err, const char *msg)
{
    if (err != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(msg) +
                                 " (cufftResult=" + std::to_string(err) + ")");
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

// Helper alias: maps a real scalar type to its cuFFT complex element type.
template <typename TReal>
using CufftCmplx = std::conditional_t<std::is_same_v<TReal, double>,
                                      cufftDoubleComplex, cufftComplex>;

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

// Scales all complex elements in-place by alpha.
template <typename TReal, typename TComplex>
__global__ static void ScaleReals2(TComplex *__restrict__ d, int nComplex,
                                   TReal alpha)
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nComplex)
    {
        TComplex val = d[i];
        val.x *= alpha;
        val.y *= alpha;
        d[i] = val;
    }
}

// Converts cuFFT half-complex output to Nektar++ coefficient layout.
// With Scaled=true, folds 1/N into the output (used when no store-callback).
template <typename TReal, typename TComplex, bool Scaled>
__global__ static void ComplexToCoefKernel(const TComplex *__restrict__ d_cmplx,
                                           TReal *__restrict__ d_coef, int N,
                                           int halfN, TReal invN)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    const TComplex cx = d_cmplx[b * (halfN + 1) + k];
    TReal *coef       = d_coef + b * N;

    if (k == 0)
    {
        coef[0] = Scaled ? cx.x * invN : cx.x;
        coef[1] = TReal(0);
    }
    else if (k < halfN)
    {
        const TReal factor = Scaled ? TReal(2) * invN : TReal(2);
        coef[2 * k]        = cx.x * factor;
        coef[2 * k + 1]    = cx.y * factor;
    }
    // k == halfN: Nyquist bin has no Nektar++ slot, left unwritten.
}

// Converts Nektar++ coefficient layout to cuFFT half-complex input.
template <typename TReal, typename TComplex>
__global__ static void CoefToComplexKernel(const TReal *__restrict__ d_coef,
                                           TComplex *__restrict__ d_cmplx,
                                           int N, int halfN)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    const TReal *coef = d_coef + b * N;
    TComplex *cx      = d_cmplx + b * (halfN + 1);

    if (k == 0)
    {
        cx[0] = {__ldg(&coef[0]), TReal(0)};
    }
    else if (k == halfN)
    {
        cx[k] = {TReal(0), TReal(0)};
    }
    else
    {
        cx[k] = {__ldg(&coef[2 * k]) * TReal(0.5),
                 __ldg(&coef[2 * k + 1]) * TReal(0.5)};
    }
}

// Multiplies by i*k*beta*normScale in-place. DC and Nyquist are zeroed.
template <typename TReal, typename TComplex>
__global__ static void WavenumberMultiplyKernel(TComplex *__restrict__ d_cmplx,
                                                int halfN, TReal beta,
                                                TReal normScale)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    if (k == 0 || k == halfN)
    {
        d_cmplx[b * (halfN + 1) + k] = {TReal(0), TReal(0)};
        return;
    }

    const TComplex cx            = d_cmplx[b * (halfN + 1) + k];
    const TReal scale            = static_cast<TReal>(k) * beta * normScale;
    d_cmplx[b * (halfN + 1) + k] = {-cx.y * scale, cx.x * scale};
}

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
std::string NekCuFFTImpl<double>::className =
    GetNektarFFTFactory().RegisterCreatorFunction("NekDeviceFFT",
                                                  NekCuFFTImpl<double>::create);

template <>
std::string NekCuFFTImpl<float>::className =
    GetNektarFFTFloatFactory().RegisterCreatorFunction(
        "NekDeviceFFT", NekCuFFTImpl<float>::create);

template <typename TData>
NekCuFFTImpl<TData>::NekCuFFTImpl(int N, int M, bool highPriorityStream)
    : NektarFFT<TData>(N), m_halfN(N / 2), m_batch(M)
{
    const std::size_t nPhys  = static_cast<std::size_t>(M) * N;
    const std::size_t nCmplx = static_cast<std::size_t>(M) * (m_halfN + 1);

    cudaMalloc(&m_d_phys, nPhys * sizeof(TData));
    cudaMalloc(&m_d_cmplx, nCmplx * sizeof(CufftCmplx<TData>));
    cudaMallocHost(&m_h_staging, nPhys * sizeof(TData));

    {
        int leastPriority, greatestPriority;
        checkCuda(
            cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
            "NekCuFFT: cudaDeviceGetStreamPriorityRange");
        const int priority =
            highPriorityStream ? greatestPriority : leastPriority;
        checkCuda(cudaStreamCreateWithPriority(&m_stream, cudaStreamDefault,
                                               priority),
                  "NekCuFFT: cudaStreamCreateWithPriority");
    }

    checkCufft(cufftCreate(&m_planForward), "NekCuFFT: cufftCreate forward");
    checkCufft(cufftCreate(&m_planBackward), "NekCuFFT: cufftCreate backward");

    checkCufft(cufftSetAutoAllocation(m_planForward, 0),
               "NekCuFFT: cufftSetAutoAllocation forward");
    checkCufft(cufftSetAutoAllocation(m_planBackward, 0),
               "NekCuFFT: cufftSetAutoAllocation backward");

    checkCufft(cufftSetStream(m_planForward, m_stream),
               "NekCuFFT: cufftSetStream forward");
    checkCufft(cufftSetStream(m_planBackward, m_stream),
               "NekCuFFT: cufftSetStream backward");

    std::size_t wsFwd = 0, wsBwd = 0;
    {
        int dims[]    = {N};
        int inembed[] = {N};
        int onembed[] = {m_halfN + 1};

        if constexpr (std::is_same_v<TData, double>)
        {
            checkCufft(cufftMakePlanMany(m_planForward, 1, dims, inembed, 1, N,
                                         onembed, 1, m_halfN + 1, CUFFT_D2Z, M,
                                         &wsFwd),
                       "NekCuFFT: cufftMakePlanMany forward");
            checkCufft(cufftMakePlanMany(m_planBackward, 1, dims, onembed, 1,
                                         m_halfN + 1, inembed, 1, N, CUFFT_Z2D,
                                         M, &wsBwd),
                       "NekCuFFT: cufftMakePlanMany backward");
        }
        else
        {
            checkCufft(cufftMakePlanMany(m_planForward, 1, dims, inembed, 1, N,
                                         onembed, 1, m_halfN + 1, CUFFT_R2C, M,
                                         &wsFwd),
                       "NekCuFFT: cufftMakePlanMany forward");
            checkCufft(cufftMakePlanMany(m_planBackward, 1, dims, onembed, 1,
                                         m_halfN + 1, inembed, 1, N, CUFFT_C2R,
                                         M, &wsBwd),
                       "NekCuFFT: cufftMakePlanMany backward");
        }
    }

    if (CallbackSupportedOnce<TData>())
    {
        const StoreScaledParams<TData> h_params{
            TData(1.0) / static_cast<TData>(N), TData(0.0), m_halfN};
        StoreScaledParams<TData> *d_params = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_params),
                             sizeof(StoreScaledParams<TData>)),
                  "NekCuFFT: cudaMalloc m_d_cbParams");
        checkCuda(cudaMemcpy(d_params, &h_params, sizeof(h_params),
                             cudaMemcpyHostToDevice),
                  "NekCuFFT: cudaMemcpy m_d_cbParams");
        m_d_cbParams = d_params;
        checkCufft(RegisterFwdScaleCallback<TData>(m_planForward, d_params),
                   "NekCuFFT: RegisterFwdScaleCallback");
        m_useCallback = true;

        std::size_t wsFwdPostCb = 0;
        if (cufftGetSize(m_planForward, &wsFwdPostCb) == CUFFT_SUCCESS)
            wsFwd = std::max(wsFwd, wsFwdPostCb);
    }

    int deviceId = 0;
    checkCuda(cudaGetDevice(&deviceId), "NekCuFFT: cudaGetDevice");

    PlanCache::Entry cacheEntry;
    const bool cached =
        PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), cacheEntry);
    if (!cached)
        cacheEntry = {wsFwd, wsBwd, false};

    const std::size_t workspaceBytes =
        std::max(cacheEntry.wsFwd, cacheEntry.wsBwd);

    if (workspaceBytes > 0)
    {
        checkCuda(cudaMalloc(&m_d_workspace, workspaceBytes),
                  "NekCuFFT: cudaMalloc workspace");
        checkCufft(cufftSetWorkArea(m_planForward, m_d_workspace),
                   "NekCuFFT: cufftSetWorkArea forward");
        checkCufft(cufftSetWorkArea(m_planBackward, m_d_workspace),
                   "NekCuFFT: cufftSetWorkArea backward");
    }
    else
    {
        checkCufft(cufftSetAutoAllocation(m_planForward, 1),
                   "NekCuFFT: re-enable auto-alloc forward (zero workspace)");
        checkCufft(cufftSetAutoAllocation(m_planBackward, 1),
                   "NekCuFFT: re-enable auto-alloc backward (zero workspace)");
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

template <typename TData> NekCuFFTImpl<TData>::~NekCuFFTImpl()
{
    DestroyGraph();

    cufftDestroy(m_planForward);
    cufftDestroy(m_planBackward);
    cudaStreamDestroy(m_stream);

    cudaFree(m_d_workspace);
    cudaFree(m_d_cbParams);
    cudaFree(m_d_phys);
    cudaFree(m_d_cmplx);
    cudaFreeHost(m_h_staging);
}

template <typename TData> void NekCuFFTImpl<TData>::DestroyGraph()
{
    if (m_graphExec)
    {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    if (m_graph)
    {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
    m_hasGraph = false;
}

template <typename TData> void NekCuFFTImpl<TData>::DestroyRawGraph()
{
    if (m_graph)
    {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
}

template <typename TData> void NekCuFFTImpl<TData>::ComputeKernelParams()
{
    int minGridSize;
    checkCuda(cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize, &m_blockSizeWave,
                  WavenumberMultiplyKernel<TData, CufftCmplx<TData>>, 0, 0),
              "NekCuFFT: ComputeKernelParams (WavenumberMultiplyKernel)");
    checkCuda(cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize, &m_blockSizeC2C,
                  ComplexToCoefKernel<TData, CufftCmplx<TData>, false>, 0, 0),
              "NekCuFFT: ComputeKernelParams (ComplexToCoefKernel<false>)");
    checkCuda(cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize, &m_blockSizeScaledC2C,
                  ComplexToCoefKernel<TData, CufftCmplx<TData>, true>, 0, 0),
              "NekCuFFT: ComputeKernelParams (ComplexToCoefKernel<true>)");
    checkCuda(cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize, &m_blockSizeCtC,
                  CoefToComplexKernel<TData, CufftCmplx<TData>>, 0, 0),
              "NekCuFFT: ComputeKernelParams (CoefToComplexKernel)");
    checkCuda(cudaOccupancyMaxPotentialBlockSize(
                  &minGridSize, &m_blockSizeScale,
                  ScaleReals2<TData, CufftCmplx<TData>>, 0, 0),
              "NekCuFFT: ComputeKernelParams (ScaleReals2)");
    (void)minGridSize;
}

template <typename TData> void NekCuFFTImpl<TData>::WarmUpPlans()
{
    NVTX_RANGE("NekCuFFT::WarmUpPlans");
    checkCuda(cudaMemset(m_d_phys, 0,
                         static_cast<std::size_t>(m_batch) * this->m_N *
                             sizeof(TData)),
              "NekCuFFT: WarmUpPlans cudaMemset");
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: WarmUpPlans D2Z");
        checkCufft(cufftExecZ2D(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: WarmUpPlans Z2D");
    }
    else
    {
        checkCufft(cufftExecR2C(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: WarmUpPlans R2C");
        checkCufft(cufftExecC2R(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: WarmUpPlans C2R");
    }
    checkCuda(cudaStreamSynchronize(m_stream), "NekCuFFT: stream sync");
}

template <typename TData>
void NekCuFFTImpl<TData>::LaunchWavenumberMultiplyImpl(TData beta,
                                                       TData normScale)
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeWave - 1) / m_blockSizeWave);
    WavenumberMultiplyKernel<TData, CufftCmplx<TData>>
        <<<grid, m_blockSizeWave, 0, m_stream>>>(m_d_cmplx, m_halfN, beta,
                                                 normScale);
}

template <typename TData> void NekCuFFTImpl<TData>::LaunchComplexToCoefImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeC2C - 1) / m_blockSizeC2C);
    ComplexToCoefKernel<TData, CufftCmplx<TData>, false>
        <<<grid, m_blockSizeC2C, 0, m_stream>>>(m_d_cmplx, m_d_phys, this->m_N,
                                                m_halfN, TData(0));
}

template <typename TData>
void NekCuFFTImpl<TData>::LaunchScaledComplexToCoefImpl()
{
    const TData invN = TData(1) / static_cast<TData>(this->m_N);
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeScaledC2C - 1) /
                        m_blockSizeScaledC2C);
    ComplexToCoefKernel<TData, CufftCmplx<TData>, true>
        <<<grid, m_blockSizeScaledC2C, 0, m_stream>>>(m_d_cmplx, m_d_phys,
                                                      this->m_N, m_halfN, invN);
}

template <typename TData> void NekCuFFTImpl<TData>::LaunchCoefToComplexImpl()
{
    const dim3 grid(static_cast<unsigned>(m_batch),
                    (m_halfN + 1 + m_blockSizeCtC - 1) / m_blockSizeCtC);
    CoefToComplexKernel<TData, CufftCmplx<TData>>
        <<<grid, m_blockSizeCtC, 0, m_stream>>>(m_d_phys, m_d_cmplx, this->m_N,
                                                m_halfN);
}

template <typename TData>
void NekCuFFTImpl<TData>::LaunchScaleRealsImpl(TData alpha)
{
    const int nComplex = m_batch * (m_halfN + 1);
    const int blocks   = (nComplex + m_blockSizeScale - 1) / m_blockSizeScale;
    ScaleReals2<TData, CufftCmplx<TData>>
        <<<blocks, m_blockSizeScale, 0, m_stream>>>(m_d_cmplx, nComplex, alpha);
}

template <typename TData>
void NekCuFFTImpl<TData>::WaitOnEvent(cudaEvent_t event)
{
    if (event != nullptr)
    {
        checkCuda(cudaStreamWaitEvent(m_stream, event),
                  "NekCuFFT: WaitOnEvent");
    }
}

template <typename TData>
void NekCuFFTImpl<TData>::UploadPhys(const TData *phys)
{
    NVTX_RANGE("NekCuFFT::UploadPhys");
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, phys, nBytes);
    checkCuda(cudaMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                              cudaMemcpyHostToDevice, m_stream),
              "NekCuFFT: UploadPhys");
}

template <typename TData> void NekCuFFTImpl<TData>::DownloadPhys(TData *phys)
{
    NVTX_RANGE("NekCuFFT::DownloadPhys");
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    checkCuda(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                              cudaMemcpyDeviceToHost, m_stream),
              "NekCuFFT: DownloadPhys");
    checkCuda(cudaStreamSynchronize(m_stream), "NekCuFFT: stream sync");
    std::memcpy(phys, m_h_staging, nBytes);
}

// Stages coef data through m_d_phys as a scratch buffer, which overwrites
// any physical-space data currently held there.
template <typename TData>
void NekCuFFTImpl<TData>::UploadCoef(const TData *coef)
{
    NVTX_RANGE("NekCuFFT::UploadCoef");
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, coef, nBytes);

    NVTX_PUSH("NekCuFFT::UploadCoef::H2D");
    checkCuda(cudaMemcpyAsync(m_d_phys, m_h_staging, nBytes,
                              cudaMemcpyHostToDevice, m_stream),
              "NekCuFFT: UploadCoef H2D");
    NVTX_POP();

    NVTX_PUSH("NekCuFFT::UploadCoef::CoefToComplex");
    LaunchCoefToComplexImpl();
    NVTX_POP();
}

template <typename TData> void NekCuFFTImpl<TData>::DownloadCoef(TData *coef)
{
    NVTX_RANGE("NekCuFFT::DownloadCoef");
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    NVTX_PUSH("NekCuFFT::DownloadCoef::ComplexToCoef");
    LaunchComplexToCoefImpl();
    NVTX_POP();

    NVTX_PUSH("NekCuFFT::DownloadCoef::D2H");
    checkCuda(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                              cudaMemcpyDeviceToHost, m_stream),
              "NekCuFFT: DownloadCoef D2H");
    NVTX_POP();

    checkCuda(cudaStreamSynchronize(m_stream), "NekCuFFT: stream sync");
    std::memcpy(coef, m_h_staging, nBytes);
}

template <typename TData> void NekCuFFTImpl<TData>::FFTFwdTransDevice()
{
    NVTX_RANGE("NekCuFFT::FFTFwdTransDevice");

    NVTX_PUSH("NekCuFFT::FFTFwdTransDevice::cufftExec");
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: FFTFwdTransDevice D2Z");
    }
    else
    {
        checkCufft(cufftExecR2C(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: FFTFwdTransDevice R2C");
    }
    NVTX_POP();

    if (!m_useCallback)
    {
        NVTX_PUSH("NekCuFFT::FFTFwdTransDevice::ScaleReals");
        LaunchScaleRealsImpl(TData(1) / static_cast<TData>(this->m_N));
        NVTX_POP();
    }
}

template <typename TData> void NekCuFFTImpl<TData>::FFTBwdTransDevice()
{
    NVTX_RANGE("NekCuFFT::FFTBwdTransDevice");
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecZ2D(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: FFTBwdTransDevice Z2D");
    }
    else
    {
        checkCufft(cufftExecC2R(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: FFTBwdTransDevice C2R");
    }
}

template <typename TData> void NekCuFFTImpl<TData>::FFTExecOnlyDevice()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: FFTExecOnlyDevice D2Z");
        checkCufft(cufftExecZ2D(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: FFTExecOnlyDevice Z2D");
    }
    else
    {
        checkCufft(cufftExecR2C(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: FFTExecOnlyDevice R2C");
        checkCufft(cufftExecC2R(m_planBackward, m_d_cmplx, m_d_phys),
                   "NekCuFFT: FFTExecOnlyDevice C2R");
    }
}

template <typename TData>
void NekCuFFTImpl<TData>::WavenumberMultiply(TData beta)
{
    NVTX_RANGE("NekCuFFT::WavenumberMultiply");
    LaunchWavenumberMultiplyImpl(beta, TData(1));
}

template <typename TData> void NekCuFFTImpl<TData>::BeginGraphCapture()
{
    NVTX_RANGE("NekCuFFT::BeginGraphCapture");
    m_hasGraph = false;
    DestroyRawGraph();
    checkCuda(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal),
              "NekCuFFT: BeginGraphCapture");
}

template <typename TData> void NekCuFFTImpl<TData>::EndGraphCapture()
{
    NVTX_RANGE("NekCuFFT::EndGraphCapture");
    checkCuda(cudaStreamEndCapture(m_stream, &m_graph),
              "NekCuFFT: EndGraphCapture (cudaStreamEndCapture)");

    if (m_graphExec)
    {
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo info{};
        const cudaError_t updateErr =
            cudaGraphExecUpdate(m_graphExec, m_graph, &info);
        (void)cudaGetLastError();
        const bool updated = (updateErr == cudaSuccess &&
                              info.result == cudaGraphExecUpdateSuccess);
#else
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdateResult updateResult;
        const cudaError_t updateErr = cudaGraphExecUpdate(
            m_graphExec, m_graph, &errorNode, &updateResult);
        const bool updated = (updateErr == cudaSuccess &&
                              updateResult == cudaGraphExecUpdateSuccess);
#endif
        if (updated)
        {
            m_hasGraph = true;
            return;
        }
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

#if CUDART_VERSION >= 12000
    checkCuda(cudaGraphInstantiate(&m_graphExec, m_graph, 0),
              "NekCuFFT: EndGraphCapture (cudaGraphInstantiate)");
#else
    checkCuda(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0),
              "NekCuFFT: EndGraphCapture (cudaGraphInstantiate)");
#endif
    m_hasGraph = true;
}

template <typename TData> void NekCuFFTImpl<TData>::LaunchGraph()
{
    NVTX_RANGE("NekCuFFT::LaunchGraph");
    if (!m_hasGraph)
        throw std::runtime_error(
            "NekCuFFT::LaunchGraph: no graph captured. "
            "Call BeginGraphCapture / EndGraphCapture first.");
    checkCuda(cudaGraphLaunch(m_graphExec, m_stream), "NekCuFFT: LaunchGraph");
}

template <typename TData>
bool NekCuFFTImpl<TData>::IsCacheWarmed(int deviceId, int N, int M)
{
    PlanCache::Entry entry;
    return PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), entry) &&
           entry.warmedUp;
}

template <typename TData>
void NekCuFFTImpl<TData>::v_FFTFwdTrans(TData *inarray, TData *outarray)
{
    NVTX_RANGE("NekCuFFT::v_FFTFwdTrans");
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    UploadPhys(inarray);

    NVTX_PUSH("NekCuFFT::v_FFTFwdTrans::cufftExec");
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: v_FFTFwdTrans D2Z");
    }
    else
    {
        checkCufft(cufftExecR2C(m_planForward, m_d_phys, m_d_cmplx),
                   "NekCuFFT: v_FFTFwdTrans R2C");
    }
    NVTX_POP();

    NVTX_PUSH("NekCuFFT::v_FFTFwdTrans::CoefReshape");
    if (m_useCallback)
        LaunchComplexToCoefImpl();
    else
        LaunchScaledComplexToCoefImpl();
    NVTX_POP();

    checkCuda(cudaMemcpyAsync(m_h_staging, m_d_phys, nBytes,
                              cudaMemcpyDeviceToHost, m_stream),
              "NekCuFFT: v_FFTFwdTrans D2H");
    checkCuda(cudaStreamSynchronize(m_stream), "NekCuFFT: stream sync");
    std::memcpy(outarray, m_h_staging, nBytes);
}

template <typename TData>
void NekCuFFTImpl<TData>::v_FFTBwdTrans(TData *inarray, TData *outarray)
{
    NVTX_RANGE("NekCuFFT::v_FFTBwdTrans");
    UploadCoef(inarray);
    FFTBwdTransDevice();
    DownloadPhys(outarray);
}

// Explicit instantiations for double and float.
template class NekCuFFTImpl<double>;
template class NekCuFFTImpl<float>;

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
