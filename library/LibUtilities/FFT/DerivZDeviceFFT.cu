///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZDeviceFFT.cu
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
// Description: cuFFT z-derivative pipeline for the Nektar++ DerivZ
// operators: cached cuFFT plans driving D2Z, the wavenumber multiply and Z2D
// as three launches, or a single fused cuFFTDx kernel when the build defines
// NEKTAR_USE_CUFFTDX.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

// The two implementations share no machinery, so each pulls in only what it
// needs; see the matching split in the anonymous namespace below.
#if !defined(NEKTAR_USE_CUFFTDX)

#include <algorithm>
#include <unordered_map>

#include <LibUtilities/FFT/DerivZDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTHIPCUDAHelper.h>

#else // NEKTAR_USE_CUFFTDX

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cufftdx/cufftdx.hpp>

#include <LibUtilities/Backends/CUDAStream.hpp>
#include <LibUtilities/FFT/DerivZDeviceFFT.h>

#ifndef CUFFTDX_TARGET_SM
#define CUFFTDX_TARGET_SM 700
#endif

#endif // !NEKTAR_USE_CUFFTDX

namespace Nektar::LibUtilities
{

namespace
{

#if !defined(NEKTAR_USE_CUFFTDX)

// ---------------------------------------------------------------------
// cuFFT pipeline: cached plans driving D2Z, the wavenumber multiply and
// Z2D as three separate launches. Not compiled when the fused cuFFTDx
// kernel below replaces it.
// ---------------------------------------------------------------------

struct DirectPlanKey
{
    int deviceId; // cudaGetDevice() out-parameter
    unsigned int nhomo;
    size_t NXY;
    size_t compStride;
    cudaStream_t stream;

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
    cufftHandle planFwd            = 0;
    cufftHandle planBwd            = 0;
    DeviceFFTCmplx<TData> *d_cmplx = nullptr;
    void *d_workspace              = nullptr;
    /// Where the inverse transform lands when the caller is accumulating;
    /// cuFFT owns its output store, so it cannot add into the caller's
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
                                   size_t compStride, cudaStream_t stream)
{
    DirectPlanEntry<TData> e;
    // halfN and blockSizeWave stay int: both only ever feed cuFFT's int-typed
    // plan descriptors and cudaOccupancyMaxPotentialBlockSize.
    e.halfN = static_cast<int>(nhomo / 2);

    const std::size_t nCmplx = NXY * static_cast<std::size_t>(e.halfN + 1);
    CHECK_HIPCUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&e.d_cmplx),
                                   nCmplx * sizeof(DeviceFFTCmplx<TData>)));

    CHECK_HIPCUDA_FFT_ERROR(cufftCreate(&e.planFwd));
    CHECK_HIPCUDA_FFT_ERROR(cufftCreate(&e.planBwd));

    CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(e.planFwd, 0));
    CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(e.planBwd, 0));

    CHECK_HIPCUDA_FFT_ERROR(cufftSetStream(e.planFwd, stream));
    CHECK_HIPCUDA_FFT_ERROR(cufftSetStream(e.planBwd, stream));

    const int nhomoI      = static_cast<int>(nhomo);
    const int compStrideI = static_cast<int>(compStride);
    const int batchI      = static_cast<int>(NXY);

    int dims[]    = {nhomoI};
    int inembed[] = {nhomoI};
    int onembed[] = {e.halfN + 1};

    // The plan type has to track TData: the single-precision plans are
    // R2C/C2R, and executing them with the double-precision entry points (or
    // the reverse) is rejected by cuFFT at exec time.
    constexpr cufftType fwdType =
        std::is_same_v<TData, double> ? CUFFT_D2Z : CUFFT_R2C;
    constexpr cufftType bwdType =
        std::is_same_v<TData, double> ? CUFFT_Z2D : CUFFT_C2R;

    std::size_t wsFwd = 0, wsBwd = 0;

    CHECK_HIPCUDA_FFT_ERROR(
        cufftMakePlanMany(e.planFwd, 1, dims, inembed, compStrideI, 1, onembed,
                          1, e.halfN + 1, fwdType, batchI, &wsFwd));

    CHECK_HIPCUDA_FFT_ERROR(cufftMakePlanMany(e.planBwd, 1, dims, onembed, 1,
                                              e.halfN + 1, inembed, compStrideI,
                                              1, bwdType, batchI, &wsBwd));

    const std::size_t wsBytes = std::max(wsFwd, wsBwd);
    if (wsBytes > 0)
    {
        CHECK_HIPCUDA_ERROR(cudaMalloc(&e.d_workspace, wsBytes));
        CHECK_HIPCUDA_FFT_ERROR(cufftSetWorkArea(e.planFwd, e.d_workspace));
        CHECK_HIPCUDA_FFT_ERROR(cufftSetWorkArea(e.planBwd, e.d_workspace));
    }
    else
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(e.planFwd, 1));
        CHECK_HIPCUDA_FFT_ERROR(cufftSetAutoAllocation(e.planBwd, 1));
    }

    {
        int dummy;
        CHECK_HIPCUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(
            &dummy, &e.blockSizeWave, WavenumberMultiplyKernel<TData>, 0, 0));
        (void)dummy;
    }

    TData *d_tmp = nullptr;
    const std::size_t nPhys =
        static_cast<std::size_t>(nhomo) * static_cast<std::size_t>(compStride);
    CHECK_HIPCUDA_ERROR(
        cudaMalloc(reinterpret_cast<void **>(&d_tmp), nPhys * sizeof(TData)));
    CHECK_HIPCUDA_ERROR(cudaMemset(d_tmp, 0, nPhys * sizeof(TData)));
    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(e.planFwd, d_tmp, e.d_cmplx));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecZ2D(e.planBwd, e.d_cmplx, d_tmp));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(e.planFwd, d_tmp, e.d_cmplx));
        CHECK_HIPCUDA_FFT_ERROR(cufftExecC2R(e.planBwd, e.d_cmplx, d_tmp));
    }
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_HIPCUDA_ERROR(cudaFree(d_tmp));

    return e;
}

// Cache lookup shared by DerivZDirect() and DerivZPrepare(), so that
// preparing a stream and running on it cannot disagree about the key.
template <typename TData>
DirectPlanEntry<TData> &GetOrCreateEntry(unsigned int nhomo, size_t NXY,
                                         size_t compStride, cudaStream_t stream)
{
    int deviceId = 0;
    CHECK_HIPCUDA_ERROR(cudaGetDevice(&deviceId));

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
            cudaMalloc(reinterpret_cast<void **>(&entry.d_accum), nbytes));
        CHECK_HIPCUDA_ERROR(cudaMemset(entry.d_accum, 0, nbytes));
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

#else // NEKTAR_USE_CUFFTDX

// ---------------------------------------------------------------------
// Fused cuFFTDx pipeline: D2Z + wavenumber multiply + Z2D in a single
// kernel, one thread block per xy pencil, reading and writing the block
// device memory in place.
// ---------------------------------------------------------------------

constexpr int kTarget = CUFFTDX_TARGET_SM;

template <typename TReal, int N_PLANES> struct DxFFT
{
private:
    // Probe descriptor: no ElementsPerThread, so cuFFTDx reports the value it
    // suggests for this size and precision rather than one we impose.
    using Probe =
        decltype(cufftdx::Block() + cufftdx::Size<N_PLANES>() +
                 cufftdx::Type<cufftdx::fft_type::c2c>() +
                 cufftdx::Direction<cufftdx::fft_direction::forward>() +
                 cufftdx::Precision<TReal>() + cufftdx::FFTsPerBlock<1>() +
                 cufftdx::SM<kTarget>());

public:
    // Pinned on both directions so the forward and inverse transforms share
    // one register array in the kernel below.
    static constexpr int kEPT = static_cast<int>(Probe::elements_per_thread);

    using FFT = decltype(cufftdx::Block() + cufftdx::Size<N_PLANES>() +
                         cufftdx::Type<cufftdx::fft_type::c2c>() +
                         cufftdx::Direction<cufftdx::fft_direction::forward>() +
                         cufftdx::Precision<TReal>() +
                         cufftdx::ElementsPerThread<kEPT>() +
                         cufftdx::FFTsPerBlock<1>() + cufftdx::SM<kTarget>());

    using IFFT =
        decltype(cufftdx::Block() + cufftdx::Size<N_PLANES>() +
                 cufftdx::Type<cufftdx::fft_type::c2c>() +
                 cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                 cufftdx::Precision<TReal>() +
                 cufftdx::ElementsPerThread<kEPT>() +
                 cufftdx::FFTsPerBlock<1>() + cufftdx::SM<kTarget>());
};

// Reads from block device memory with stride compStride instead of NXY,
// writing directly to the z-slot in the output block. One block per xy
// pencil (blockIdx.x == j). Avoids separate gather/scatter copies.
template <typename TReal, int N_PLANES, DerivZOrder DERIVORDER, bool APPEND>
__global__ void DerivZDxDirectKernel(const TReal *__restrict__ d_in,
                                     TReal *__restrict__ d_out, int NXY,
                                     int compStride, TReal beta)
{
    using Traits       = DxFFT<TReal, N_PLANES>;
    using FFT_t        = typename Traits::FFT;
    using IFFT_t       = typename Traits::IFFT;
    using complex_type = typename FFT_t::value_type;

    constexpr int kEPT   = Traits::kEPT;
    constexpr int half_n = N_PLANES / 2;

    const int j = static_cast<int>(blockIdx.x);

    extern __shared__ char shmem_raw[];
    complex_type *smem = reinterpret_cast<complex_type *>(shmem_raw);

    complex_type thread_data[kEPT];

    for (int i = 0; i < kEPT; ++i)
    {
        const int k    = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        thread_data[i] = complex_type{d_in[k * compStride + j], TReal(0)};
    }

    FFT_t().execute(thread_data, smem);

    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;

        // Nektar's eFourier basis spans k = 0 to N/2 - 1 only, its second
        // slot being a structural zero rather than a Nyquist mode, so k = N/2
        // is not representable at either derivative order.
        if (k == 0 || k == half_n)
        {
            thread_data[i] = complex_type{TReal(0), TReal(0)};
        }
        else
        {
            const int wn      = (k <= half_n) ? k : (k - N_PLANES);
            const TReal betaK = static_cast<TReal>(wn) * beta;

            if constexpr (DERIVORDER == DerivZOrder::First)
            {
                thread_data[i] = complex_type{-thread_data[i].y * betaK,
                                              thread_data[i].x * betaK};
            }
            else
            {
                const TReal scale = -betaK * betaK;
                thread_data[i]    = complex_type{thread_data[i].x * scale,
                                              thread_data[i].y * scale};
            }
        }
    }

    IFFT_t().execute(thread_data, smem);

    // Nothing else writes the result, so an accumulating call costs only
    // the read-modify-write here: no staging buffer and no second pass.
    const TReal inv_n = TReal(1) / static_cast<TReal>(N_PLANES);
    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        if constexpr (APPEND)
        {
            d_out[k * compStride + j] += thread_data[i].x * inv_n;
        }
        else
        {
            d_out[k * compStride + j] = thread_data[i].x * inv_n;
        }
    }
}

// TReal is a template parameter of the launcher, not just the kernel: the
// attr_set latch below and the kernel whose attribute it raises must be the
// same instantiation, or one precision would never get its shared-memory
// limit lifted.
template <typename TReal, int N_PLANES, DerivZOrder DERIVORDER, bool APPEND>
void LaunchDerivZDxDirect(const TReal *d_in, TReal *d_out, int NXY,
                          int compStride, TReal beta, cudaStream_t stream)
{
    using Traits = DxFFT<TReal, N_PLANES>;
    using FFT_t  = typename Traits::FFT;

    constexpr unsigned int shmem_size = FFT_t::shared_memory_size;

    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(
            DerivZDxDirectKernel<TReal, N_PLANES, DERIVORDER, APPEND>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        attr_set = true;
    }

    // Always launch directly (instead of via graphs):
    // device pointers vary per call so a static graph cache
    // keyed on {NXY, compStride} would replay stale pointers for
    // differently-allocated fields of the same shape. The outer
    // DerivZOpDevice already handles graph capture/replay for the full
    // z-pipeline. When the stream is being captured by that outer graph the
    // kernel launch is recorded into it automatically.
    DerivZDxDirectKernel<TReal, N_PLANES, DERIVORDER, APPEND>
        <<<NXY, FFT_t::block_dim, shmem_size, stream>>>(d_in, d_out, NXY,
                                                        compStride, beta);
}

// Switches on the plane count to pick the compile-time FFT size. Takes the
// same argument order as DerivZDirect, so the dispatch below forwards
// its arguments unchanged.
template <typename TReal, DerivZOrder DERIVORDER, bool APPEND>
void DerivZDxDispatch(const TReal *d_in, TReal *d_out, unsigned int nhomo,
                      size_t NXY, size_t compStride, TReal beta,
                      cudaStream_t stream)
{
    // The launcher below drives a kernel launch, so NXY and compStride narrow
    // to the int that the grid dimension and in-kernel indexing use.
    const int NXYI        = static_cast<int>(NXY);
    const int compStrideI = static_cast<int>(compStride);

    switch (nhomo)
    {
        case 16:
            LaunchDerivZDxDirect<TReal, 16, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        case 32:
            LaunchDerivZDxDirect<TReal, 32, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        case 64:
            LaunchDerivZDxDirect<TReal, 64, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        case 128:
            LaunchDerivZDxDirect<TReal, 128, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        case 256:
            LaunchDerivZDxDirect<TReal, 256, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        case 512:
            LaunchDerivZDxDirect<TReal, 512, DERIVORDER, APPEND>(
                d_in, d_out, NXYI, compStrideI, beta, stream);
            break;
        default:
            throw std::runtime_error(
                "DerivZDirect: unsupported number of homogeneous planes " +
                std::to_string(nhomo) +
                " for the fused cuFFTDx kernel; supported values: 16, 32, 64, "
                "128, 256, 512");
    }
}

#endif // !NEKTAR_USE_CUFFTDX

} // anonymous namespace

template <typename TData, DerivZOrder DERIVORDER, bool APPEND>
void DerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                  size_t NXY, size_t compStride, TData beta,
                  unsigned int streamID)
{
    // The declaration hands the stream over as an id to keep the header
    // independent of the CUDA runtime; the registry resolves it to the real
    // handle here.
    cudaStream_t stream = CUDAStream::GetInstance(streamID);

#if defined(NEKTAR_USE_CUFFTDX)
    DerivZDxDispatch<TData, DERIVORDER, APPEND>(d_in, d_out, nhomo, NXY,
                                                compStride, beta, stream);
#else
    // The plans depend only on the problem size, so both orders share one
    // entry; only the wavenumber multiply differs.
    auto &entry = GetOrCreateEntry<TData>(nhomo, NXY, compStride, stream);

    // cuFFT owns its output store, so an accumulating call transforms into
    // the buffer the plan holds and is summed into d_out below.
    TData *d_target = APPEND ? entry.d_accum : d_out;

    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecD2Z(
            entry.planFwd, const_cast<TData *>(d_in), entry.d_cmplx));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(cufftExecR2C(
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
            cufftExecZ2D(entry.planBwd, entry.d_cmplx, d_target));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(
            cufftExecC2R(entry.planBwd, entry.d_cmplx, d_target));
    }

    if constexpr (APPEND)
    {
        const size_t nsize         = nhomo * compStride;
        const unsigned int addGrid = static_cast<unsigned int>(
            (nsize + entry.blockSizeWave - 1) / entry.blockSizeWave);
        AccumulateKernel<<<addGrid, entry.blockSizeWave, 0, stream>>>(
            entry.d_accum, d_out, nsize);
    }
#endif
}

template <typename TData, bool APPEND>
void DerivZPrepare([[maybe_unused]] unsigned int nhomo,
                   [[maybe_unused]] size_t NXY,
                   [[maybe_unused]] size_t compStride,
                   [[maybe_unused]] unsigned int streamID)
{
    // The fused cuFFTDx kernel keeps no per-size state, so there is nothing
    // to prepare in that build.
#if !defined(NEKTAR_USE_CUFFTDX)
    auto &entry = GetOrCreateEntry<TData>(nhomo, NXY, compStride,
                                          CUDAStream::GetInstance(streamID));

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
#endif
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
