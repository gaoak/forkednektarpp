///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFT.cpp
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
// Description: SYCL implementation of NekDeviceFFT.h, mirroring
// NekDeviceFFT.hip. One plan serves both directions, there being no separate
// backward plan to make, and SYCL offers no graph capture, so the capture
// entry points report that instead of recording.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_SYCL)

#include <cstring>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/FFT/NekDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTSYCLHelper.h>

namespace Nektar::LibUtilities
{

namespace
{

// NekDeviceFFT.h holds the device handles opaquely; these cast them back, and
// the static_assert fails the build if that storage stops being wide enough.
static_assert(sizeof(DFTPlan<double> *) <= sizeof(std::uintptr_t),
              "NekDeviceFFT.h stores FFT plan handles as uintptr_t");

inline sycl::queue &AsQueue(void *p)
{
    return *static_cast<sycl::queue *>(p);
}

inline sycl::event &AsEvent(void *p)
{
    return *static_cast<sycl::event *>(p);
}

template <typename TData> inline DFTPlan<TData> *AsPlan(std::uintptr_t h)
{
    return reinterpret_cast<DFTPlan<TData> *>(h);
}

template <typename TData> inline DFTCmplx<TData> *AsCmplx(void *p)
{
    return static_cast<DFTCmplx<TData> *>(p);
}

class PlanCache
{
public:
    // No workspace sizes: a committed plan owns whatever scratch it needs,
    // so warm-up is all this cache carries.
    struct Entry
    {
        bool warmedUp = false;
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

/// \brief Message the graph entry points report.
constexpr const char *kNoGraphSupport =
    "NekDeviceFFT: graph capture is not available on the SYCL backend; "
    "resubmit the pipeline instead.";

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

    {
        // SYCLQueue owns the registry queues and hands out no private ones,
        // and SYCL has no queue priority, so highPriorityStream has nothing
        // to select here: this object gets a queue of its own either way.
        (void)highPriorityStream;
        sycl::queue *queue =
            new sycl::queue(SYCLQueue::GetInstance(0).get_context(),
                            SYCLQueue::GetInstance(0).get_device(),
                            sycl::property::queue::in_order());
        m_stream = queue;
    }

    sycl::queue &Q = AsQueue(m_stream);

    m_d_phys    = sycl::malloc_device<TData>(nPhys, Q);
    m_d_cmplx   = sycl::malloc_device<DFTCmplx<TData>>(nCmplx, Q);
    m_h_staging = sycl::malloc_host<TData>(nPhys, Q);

    {
        // One plan drives both directions, so m_planBackward stays unused;
        // the real field is contiguous in N points per batch entry and the
        // half spectrum in halfN + 1 elements.
        DFTPlan<TData> *plan = new DFTPlan<TData>(MakeDFTPlan<TData>(
            Q, static_cast<size_t>(M), static_cast<std::int64_t>(N), 1,
            static_cast<std::int64_t>(N)));
        m_planForward        = reinterpret_cast<std::uintptr_t>(plan);
    }

    // No callback path: m_useCallback stays false and m_d_cbParams unused;
    // LaunchScaleRealsImpl applies the 1/N normalisation instead.

    const int deviceId = static_cast<int>(nekGetDevice());

    PlanCache::Entry cacheEntry;
    const bool cached =
        PlanCache::Instance().Lookup(deviceId, N, M, sizeof(TData), cacheEntry);

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

    sycl::queue &Q = AsQueue(m_stream);

    DestroyDFTPlan(*AsPlan<TData>(m_planForward));
    delete AsPlan<TData>(m_planForward);
    m_planForward = 0;

    sycl::free(m_d_phys, Q);
    sycl::free(m_d_cmplx, Q);
    sycl::free(m_h_staging, Q);

    delete static_cast<sycl::queue *>(m_stream);
    m_stream = nullptr;
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyGraph()
{
    // Nothing is ever captured, so there is nothing to release; the flags are
    // reset for the same reason the sibling backends reset theirs.
    m_graphExec = nullptr;
    m_graph     = nullptr;
    m_hasGraph  = false;
}

template <typename TData> void NekDeviceFFTImpl<TData>::DestroyRawGraph()
{
    m_graph = nullptr;
}

template <typename TData> void NekDeviceFFTImpl<TData>::ComputeKernelParams()
{
    // The kernels run over exact ranges rather than a launch grid, so the
    // block sizes the sibling backends pick by occupancy stay unused.
}

template <typename TData> void NekDeviceFFTImpl<TData>::WarmUpPlans()
{
    sycl::queue &Q = AsQueue(m_stream);

    Q.memset(m_d_phys, 0,
             static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData));
    ComputeForward(Q, *AsPlan<TData>(m_planForward), m_d_phys,
                   AsCmplx<TData>(m_d_cmplx));
    ComputeBackward(Q, *AsPlan<TData>(m_planForward), AsCmplx<TData>(m_d_cmplx),
                    m_d_phys);
    Q.wait();
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchWavenumberMultiplyImpl(TData beta,
                                                           TData normScale)
{
    WavenumberMultiplyKernel(AsQueue(m_stream), AsCmplx<TData>(m_d_cmplx),
                             static_cast<size_t>(m_batch), m_halfN, beta,
                             normScale);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchComplexToCoefImpl()
{
    ComplexToCoefKernel<TData, false>(
        AsQueue(m_stream), AsCmplx<TData>(m_d_cmplx), m_d_phys,
        static_cast<size_t>(m_batch), this->m_N, m_halfN, TData(0));
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchScaledComplexToCoefImpl()
{
    const TData invN = TData(1) / static_cast<TData>(this->m_N);
    ComplexToCoefKernel<TData, true>(
        AsQueue(m_stream), AsCmplx<TData>(m_d_cmplx), m_d_phys,
        static_cast<size_t>(m_batch), this->m_N, m_halfN, invN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchCoefToComplexImpl()
{
    CoefToComplexKernel(AsQueue(m_stream), m_d_phys, AsCmplx<TData>(m_d_cmplx),
                        static_cast<size_t>(m_batch), this->m_N, m_halfN);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::LaunchScaleRealsImpl(TData alpha)
{
    const std::size_t nComplex =
        static_cast<std::size_t>(m_batch) * (m_halfN + 1);
    ScaleComplexKernel(AsQueue(m_stream), AsCmplx<TData>(m_d_cmplx), nComplex,
                       alpha);
}

template <typename TData> void NekDeviceFFTImpl<TData>::WaitOnEvent(void *event)
{
    if (event != nullptr)
    {
#if defined(__ADAPTIVECPP__)
        AsQueue(m_stream).submit([&](sycl::handler &cgh) {
            cgh.depends_on(AsEvent(event));
            cgh.AdaptiveCpp_enqueue_custom_operation(
                [=]([[maybe_unused]] sycl::interop_handle ih) {});
        });
#elif defined(__DPCPP_COMPILER)
        AsQueue(m_stream).submit([&](sycl::handler &cgh) {
            cgh.ext_oneapi_barrier({AsEvent(event)});
        });
#endif
    }
}

template <typename TData>
void NekDeviceFFTImpl<TData>::UploadPhys(const TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    std::memcpy(m_h_staging, phys, nBytes);
    AsQueue(m_stream).memcpy(m_d_phys, m_h_staging, nBytes);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadPhys(TData *phys)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);
    AsQueue(m_stream).memcpy(m_h_staging, m_d_phys, nBytes);
    AsQueue(m_stream).wait();
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

    AsQueue(m_stream).memcpy(m_d_phys, m_h_staging, nBytes);

    LaunchCoefToComplexImpl();
}

template <typename TData>
void NekDeviceFFTImpl<TData>::DownloadCoef(TData *coef)
{
    const std::size_t nBytes =
        static_cast<std::size_t>(m_batch) * this->m_N * sizeof(TData);

    LaunchComplexToCoefImpl();

    AsQueue(m_stream).memcpy(m_h_staging, m_d_phys, nBytes);

    AsQueue(m_stream).wait();
    std::memcpy(coef, m_h_staging, nBytes);
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTFwdTransDevice()
{
    ComputeForward(AsQueue(m_stream), *AsPlan<TData>(m_planForward), m_d_phys,
                   AsCmplx<TData>(m_d_cmplx));

    LaunchScaleRealsImpl(TData(1) / static_cast<TData>(this->m_N));
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTBwdTransDevice()
{
    ComputeBackward(AsQueue(m_stream), *AsPlan<TData>(m_planForward),
                    AsCmplx<TData>(m_d_cmplx), m_d_phys);
}

template <typename TData> void NekDeviceFFTImpl<TData>::FFTExecOnlyDevice()
{
    ComputeForward(AsQueue(m_stream), *AsPlan<TData>(m_planForward), m_d_phys,
                   AsCmplx<TData>(m_d_cmplx));
    ComputeBackward(AsQueue(m_stream), *AsPlan<TData>(m_planForward),
                    AsCmplx<TData>(m_d_cmplx), m_d_phys);
}

template <typename TData>
void NekDeviceFFTImpl<TData>::WavenumberMultiply(TData beta)
{
    LaunchWavenumberMultiplyImpl(beta, TData(1));
}

template <typename TData> void NekDeviceFFTImpl<TData>::BeginGraphCapture()
{
    throw std::runtime_error(kNoGraphSupport);
}

template <typename TData> void NekDeviceFFTImpl<TData>::EndGraphCapture()
{
    throw std::runtime_error(kNoGraphSupport);
}

template <typename TData> void NekDeviceFFTImpl<TData>::LaunchGraph()
{
    // Same error the sibling backends raise for a launch without a capture:
    // m_hasGraph can never become true here.
    throw std::runtime_error("NekDeviceFFT::LaunchGraph: no graph captured. "
                             "Call BeginGraphCapture / EndGraphCapture first.");
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

    ComputeForward(AsQueue(m_stream), *AsPlan<TData>(m_planForward), m_d_phys,
                   AsCmplx<TData>(m_d_cmplx));

    LaunchScaledComplexToCoefImpl();

    AsQueue(m_stream).memcpy(m_h_staging, m_d_phys, nBytes);
    AsQueue(m_stream).wait();
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

#endif // NEKTAR_ENABLE_SYCL
