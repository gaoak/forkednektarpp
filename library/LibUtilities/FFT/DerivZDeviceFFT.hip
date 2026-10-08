///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZDeviceFFT.hip
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
// Description: hipFFT z-derivative pipeline for the Nektar++ DerivZ
// operators: cached hipFFT plans driving D2Z, the wavenumber multiply and
// Z2D as three launches. ROCm offers no fused single-kernel transform, so
// that is the only pipeline here.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <unordered_map>

#include <LibUtilities/FFT/DerivZDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTHIPCUDAHelper.h>

namespace Nektar::LibUtilities
{

namespace
{

// ---------------------------------------------------------------------
// hipFFT pipeline: cached plans driving D2Z, the wavenumber multiply and
// Z2D as three separate launches.
// ---------------------------------------------------------------------

struct DirectPlanKey
{
    int deviceId; // hipGetDevice() out-parameter
    unsigned int nhomo;
    size_t NXY;
    size_t compStride;
    hipStream_t stream;

    bool operator==(const DirectPlanKey &o) const noexcept
    {
        return deviceId == o.deviceId && nhomo == o.nhomo && NXY == o.NXY &&
               compStride == o.compStride && stream == o.stream;
    }
};

struct DirectPlanKeyHash
{
    std::size_t operator()(const DirectPlanKey &k) const noexcept
    {
        constexpr std::size_t basis = 14695981039346656037ULL;
        constexpr std::size_t prime = 1099511628211ULL;
        auto mix                    = [&](std::size_t hash,
                       std::size_t v) noexcept -> std::size_t {
            hash ^= v;
            return hash * prime;
        };
        std::size_t hash = basis;
        hash             = mix(hash, static_cast<std::size_t>(k.deviceId));
        hash             = mix(hash, static_cast<std::size_t>(k.nhomo));
        hash             = mix(hash, static_cast<std::size_t>(k.NXY));
        hash             = mix(hash, static_cast<std::size_t>(k.compStride));
        hash             = mix(hash, reinterpret_cast<std::size_t>(k.stream));
        return hash;
    }
};

template <typename TData> struct DirectPlanEntry
{
    hipfftHandle planFwd           = nullptr;
    hipfftHandle planBwd           = nullptr;
    DeviceFFTCmplx<TData> *d_cmplx = nullptr;
    void *d_workspace              = nullptr;
    /// Where the inverse transform lands when the caller is accumulating;
    /// hipFFT owns its output store, so it cannot add into the caller's
    /// buffer directly. Allocated on the first APPEND prepare and null
    /// otherwise, so a caller that overwrites pays nothing for it.
    TData *d_accum    = nullptr;
    int halfN         = 0;
    int blockSizeWave = 0;
};

template <typename TData> class DirectPlanCache
{
public:
    static DirectPlanCache &Instance()
    {
        static DirectPlanCache instance;
        return instance;
    }

    /// Null when the key is not cached. The entry is handed back by
    /// reference so that a later prepare can attach the accumulate buffer to
    /// the cached copy rather than to a temporary.
    DirectPlanEntry<TData> *Find(const DirectPlanKey &key)
    {
        auto it = m_map.find(key);
        return it == m_map.end() ? nullptr : &it->second;
    }

    DirectPlanEntry<TData> &Register(const DirectPlanKey &key,
                                     const DirectPlanEntry<TData> &entry)
    {
        return m_map[key] = entry;
    }

private:
    DirectPlanCache() = default;

    std::unordered_map<DirectPlanKey, DirectPlanEntry<TData>, DirectPlanKeyHash>
        m_map;
};

template <typename TData>
DirectPlanEntry<TData> CreateEntry(unsigned int nhomo, size_t NXY,
                                   size_t compStride, hipStream_t stream)
{
    DirectPlanEntry<TData> e;
    // halfN and blockSizeWave stay int: both only ever feed hipFFT's int-typed
    // plan descriptors and hipOccupancyMaxPotentialBlockSize.
    e.halfN = static_cast<int>(nhomo / 2);

    const std::size_t nCmplx = NXY * static_cast<std::size_t>(e.halfN + 1);
    CHECK_HIPCUDA_ERROR(hipMalloc(reinterpret_cast<void **>(&e.d_cmplx),
                                  nCmplx * sizeof(DeviceFFTCmplx<TData>)));

    CHECK_HIPCUDA_FFT_ERROR(hipfftCreate(&e.planFwd));
    CHECK_HIPCUDA_FFT_ERROR(hipfftCreate(&e.planBwd));

    CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(e.planFwd, 0));
    CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(e.planBwd, 0));

    CHECK_HIPCUDA_FFT_ERROR(hipfftSetStream(e.planFwd, stream));
    CHECK_HIPCUDA_FFT_ERROR(hipfftSetStream(e.planBwd, stream));

    const int nhomoI      = static_cast<int>(nhomo);
    const int compStrideI = static_cast<int>(compStride);
    const int batchI      = static_cast<int>(NXY);

    int dims[]    = {nhomoI};
    int inembed[] = {nhomoI};
    int onembed[] = {e.halfN + 1};

    // The plan type has to track TData: the single-precision plans are
    // R2C/C2R, and executing them with the double-precision entry points (or
    // the reverse) is rejected by hipFFT at exec time.
    constexpr hipfftType fwdType =
        std::is_same_v<TData, double> ? HIPFFT_D2Z : HIPFFT_R2C;
    constexpr hipfftType bwdType =
        std::is_same_v<TData, double> ? HIPFFT_Z2D : HIPFFT_C2R;

    std::size_t wsFwd = 0, wsBwd = 0;

    CHECK_HIPCUDA_FFT_ERROR(
        hipfftMakePlanMany(e.planFwd, 1, dims, inembed, compStrideI, 1, onembed,
                           1, e.halfN + 1, fwdType, batchI, &wsFwd));

    CHECK_HIPCUDA_FFT_ERROR(
        hipfftMakePlanMany(e.planBwd, 1, dims, onembed, 1, e.halfN + 1, inembed,
                           compStrideI, 1, bwdType, batchI, &wsBwd));

    const std::size_t wsBytes = std::max(wsFwd, wsBwd);
    if (wsBytes > 0)
    {
        CHECK_HIPCUDA_ERROR(hipMalloc(&e.d_workspace, wsBytes));
        CHECK_HIPCUDA_FFT_ERROR(hipfftSetWorkArea(e.planFwd, e.d_workspace));
        CHECK_HIPCUDA_FFT_ERROR(hipfftSetWorkArea(e.planBwd, e.d_workspace));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(e.planFwd, 1));
        CHECK_HIPCUDA_FFT_ERROR(hipfftSetAutoAllocation(e.planBwd, 1));
    }

    {
        int dummy;
        CHECK_HIPCUDA_ERROR(hipOccupancyMaxPotentialBlockSize(
            &dummy, &e.blockSizeWave, WavenumberMultiplyKernel<TData>, 0, 0));
        (void)dummy;
    }

    TData *d_tmp = nullptr;
    const std::size_t nPhys =
        static_cast<std::size_t>(nhomo) * static_cast<std::size_t>(compStride);
    CHECK_HIPCUDA_ERROR(
        hipMalloc(reinterpret_cast<void **>(&d_tmp), nPhys * sizeof(TData)));
    CHECK_HIPCUDA_ERROR(hipMemset(d_tmp, 0, nPhys * sizeof(TData)));
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(e.planFwd, d_tmp, e.d_cmplx));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecZ2D(e.planBwd, e.d_cmplx, d_tmp));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(e.planFwd, d_tmp, e.d_cmplx));
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecC2R(e.planBwd, e.d_cmplx, d_tmp));
    }
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(stream));
    CHECK_HIPCUDA_ERROR(hipFree(d_tmp));

    return e;
}

// Cache lookup shared by DerivZDirect() and DerivZPrepare(), so that
// preparing a stream and running on it cannot disagree about the key.
template <typename TData>
DirectPlanEntry<TData> &GetOrCreateEntry(unsigned int nhomo, size_t NXY,
                                         size_t compStride, hipStream_t stream)
{
    int deviceId = 0;
    CHECK_HIPCUDA_ERROR(hipGetDevice(&deviceId));

    const DirectPlanKey key{deviceId, nhomo, NXY, compStride, stream};

    if (DirectPlanEntry<TData> *hit =
            DirectPlanCache<TData>::Instance().Find(key))
    {
        return *hit;
    }

    return DirectPlanCache<TData>::Instance().Register(
        key, CreateEntry<TData>(nhomo, NXY, compStride, stream));
}

/// Attach the accumulate buffer to @p entry if it does not have one. It is
/// laid out exactly as the inverse transform writes it -- plane p at
/// p * compStride -- and zeroed once here: the transform never writes the
/// padding between NXY and compStride, so it stays zero for the life of the
/// buffer and the sum below can run flat over the whole span.
/// This allocates, which a graph capture will not tolerate, so it is reached
/// only from DerivZPrepare().
template <typename TData>
void EnsureAccumBuffer(DirectPlanEntry<TData> &entry, unsigned int nhomo,
                       size_t compStride)
{
    if (entry.d_accum == nullptr)
    {
        const size_t nbytes = nhomo * compStride * sizeof(TData);
        CHECK_HIPCUDA_ERROR(
            hipMalloc(reinterpret_cast<void **>(&entry.d_accum), nbytes));
        CHECK_HIPCUDA_ERROR(hipMemset(entry.d_accum, 0, nbytes));
    }
}

/// Add the transform in @p src into @p dst over the whole plane-major span.
/// The padding is zero in @p src, so a padded slot only ever adds zero to
/// itself.
template <typename TData>
__global__ void AccumulateKernel(const TData *src, TData *dst, size_t nsize)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < nsize)
    {
        dst[i] += src[i];
    }
}

} // anonymous namespace

template <typename TData, DerivZOrder DERIVORDER, bool APPEND>
void DerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                  size_t NXY, size_t compStride, TData beta,
                  unsigned int streamID)
{
    // The declaration hands the stream over as an id to keep the header
    // independent of the HIP runtime; the registry resolves it to the real
    // handle here.
    hipStream_t stream = HIPStream::GetInstance(streamID);

    // The plans depend only on the problem size, so both orders share one
    // entry; only the wavenumber multiply differs.
    auto &entry = GetOrCreateEntry<TData>(nhomo, NXY, compStride, stream);

    // hipFFT owns its output store, so an accumulating call transforms into
    // the buffer the plan holds and is summed into d_out below.
    TData *d_target = APPEND ? entry.d_accum : d_out;

    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecD2Z(
            entry.planFwd, const_cast<TData *>(d_in), entry.d_cmplx));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(hipfftExecR2C(
            entry.planFwd, const_cast<TData *>(d_in), entry.d_cmplx));
    }

    const TData invN = 1.0 / static_cast<TData>(nhomo);
    const dim3 grid(
        static_cast<unsigned int>(NXY),
        static_cast<unsigned>((entry.halfN + 1 + entry.blockSizeWave - 1) /
                              entry.blockSizeWave));
    if constexpr (DERIVORDER == DerivZOrder::First)
    {
        WavenumberMultiplyKernel<<<grid, entry.blockSizeWave, 0, stream>>>(
            entry.d_cmplx, entry.halfN, beta, invN);
    }
    else
    {
        WavenumberMultiply2Kernel<<<grid, entry.blockSizeWave, 0, stream>>>(
            entry.d_cmplx, entry.halfN, beta, invN);
    }

    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftExecZ2D(entry.planBwd, entry.d_cmplx, d_target));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftExecC2R(entry.planBwd, entry.d_cmplx, d_target));
    }

    if constexpr (APPEND)
    {
        const size_t nsize         = nhomo * compStride;
        const unsigned int addGrid = static_cast<unsigned int>(
            (nsize + entry.blockSizeWave - 1) / entry.blockSizeWave);
        AccumulateKernel<<<addGrid, entry.blockSizeWave, 0, stream>>>(
            entry.d_accum, d_out, nsize);
    }
}

template <typename TData, bool APPEND>
void DerivZPrepare(unsigned int nhomo, size_t NXY, size_t compStride,
                   unsigned int streamID)
{
    auto &entry = GetOrCreateEntry<TData>(nhomo, NXY, compStride,
                                          HIPStream::GetInstance(streamID));

    // Allocating is illegal inside a graph capture, so the buffer an
    // accumulating call transforms through is created here.
    if constexpr (APPEND)
    {
        EnsureAccumBuffer<TData>(entry, nhomo, compStride);
    }
    else
    {
        (void)entry;
    }
}

template void DerivZDirect<double, DerivZOrder::First, false>(
    const double *d_in, double *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, double beta, unsigned int streamID);
template void DerivZDirect<double, DerivZOrder::First, true>(
    const double *d_in, double *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, double beta, unsigned int streamID);
template void DerivZDirect<double, DerivZOrder::Second, false>(
    const double *d_in, double *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, double beta, unsigned int streamID);
template void DerivZDirect<double, DerivZOrder::Second, true>(
    const double *d_in, double *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, double beta, unsigned int streamID);
template void DerivZDirect<float, DerivZOrder::First, false>(
    const float *d_in, float *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, float beta, unsigned int streamID);
template void DerivZDirect<float, DerivZOrder::First, true>(
    const float *d_in, float *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, float beta, unsigned int streamID);
template void DerivZDirect<float, DerivZOrder::Second, false>(
    const float *d_in, float *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, float beta, unsigned int streamID);
template void DerivZDirect<float, DerivZOrder::Second, true>(
    const float *d_in, float *d_out, unsigned int nhomo, size_t NXY,
    size_t compStride, float beta, unsigned int streamID);

template void DerivZPrepare<double, false>(unsigned int nhomo, size_t NXY,
                                           size_t compStride,
                                           unsigned int streamID);
template void DerivZPrepare<double, true>(unsigned int nhomo, size_t NXY,
                                          size_t compStride,
                                          unsigned int streamID);
template void DerivZPrepare<float, false>(unsigned int nhomo, size_t NXY,
                                          size_t compStride,
                                          unsigned int streamID);
template void DerivZPrepare<float, true>(unsigned int nhomo, size_t NXY,
                                         size_t compStride,
                                         unsigned int streamID);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
