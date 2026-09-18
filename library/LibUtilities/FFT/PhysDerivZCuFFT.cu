///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZCuFFT.cu
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
// Description: cuFFT z-derivative pipeline for Nektar++ PhysDeriv, with a
// fused cuFFTDx kernel used instead under NEKTAR_USE_CUFFTDX.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <cuda_runtime.h>

// The two implementations share no machinery, so each pulls in only what it
// needs; see the matching split in the anonymous namespace below.
#if !defined(NEKTAR_USE_CUFFTDX)

#include <algorithm>
#include <type_traits>
#include <unordered_map>

#include <cufft.h>

#include <LibUtilities/FFT/NekCuFFTHelper.h>

#else // NEKTAR_USE_CUFFTDX

#include <stdexcept>
#include <string>

#include <cufftdx/cufftdx.hpp>

#ifndef CUFFTDX_TARGET_SM
#define CUFFTDX_TARGET_SM 700
#endif

#endif // !NEKTAR_USE_CUFFTDX

#include <LibUtilities/FFT/PhysDerivZCuFFT.h>

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
    cufftHandle planFwd        = 0;
    cufftHandle planBwd        = 0;
    CufftCmplx<TData> *d_cmplx = nullptr;
    void *d_workspace          = nullptr;
    int halfN                  = 0;
    int blockSizeWave          = 0;
};

template <typename TData> class DirectPlanCache
{
public:
    static DirectPlanCache &Instance()
    {
        static DirectPlanCache instance;
        return instance;
    }

    bool Lookup(const DirectPlanKey &key, DirectPlanEntry<TData> &entry) const
    {
        auto it = m_map.find(key);
        if (it == m_map.end())
        {
            return false;
        }
        entry = it->second;
        return true;
    }

    void Register(const DirectPlanKey &key, const DirectPlanEntry<TData> &entry)
    {
        m_map[key] = entry;
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
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&e.d_cmplx),
                         nCmplx * sizeof(CufftCmplx<TData>)),
              "NekCuFFTDirect: cudaMalloc d_cmplx");

    checkCufft(cufftCreate(&e.planFwd), "NekCuFFTDirect: cufftCreate forward");
    checkCufft(cufftCreate(&e.planBwd), "NekCuFFTDirect: cufftCreate backward");

    checkCufft(cufftSetAutoAllocation(e.planFwd, 0),
               "NekCuFFTDirect: cufftSetAutoAllocation forward");
    checkCufft(cufftSetAutoAllocation(e.planBwd, 0),
               "NekCuFFTDirect: cufftSetAutoAllocation backward");

    checkCufft(cufftSetStream(e.planFwd, stream),
               "NekCuFFTDirect: cufftSetStream forward");
    checkCufft(cufftSetStream(e.planBwd, stream),
               "NekCuFFTDirect: cufftSetStream backward");

    const int nhomoI      = static_cast<int>(nhomo);
    const int compStrideI = static_cast<int>(compStride);
    const int batchI      = static_cast<int>(NXY);

    int dims[]    = {nhomoI};
    int inembed[] = {nhomoI};
    int onembed[] = {e.halfN + 1};

    std::size_t wsFwd = 0, wsBwd = 0;

    checkCufft(cufftMakePlanMany(e.planFwd, 1, dims, inembed, compStrideI, 1,
                                 onembed, 1, e.halfN + 1, CUFFT_D2Z, batchI,
                                 &wsFwd),
               "NekCuFFTDirect: cufftMakePlanMany forward");

    checkCufft(cufftMakePlanMany(e.planBwd, 1, dims, onembed, 1, e.halfN + 1,
                                 inembed, compStrideI, 1, CUFFT_Z2D, batchI,
                                 &wsBwd),
               "NekCuFFTDirect: cufftMakePlanMany backward");

    const std::size_t wsBytes = std::max(wsFwd, wsBwd);
    if (wsBytes > 0)
    {
        checkCuda(cudaMalloc(&e.d_workspace, wsBytes),
                  "NekCuFFTDirect: cudaMalloc workspace");
        checkCufft(cufftSetWorkArea(e.planFwd, e.d_workspace),
                   "NekCuFFTDirect: cufftSetWorkArea forward");
        checkCufft(cufftSetWorkArea(e.planBwd, e.d_workspace),
                   "NekCuFFTDirect: cufftSetWorkArea backward");
    }
    else
    {
        checkCufft(cufftSetAutoAllocation(e.planFwd, 1),
                   "NekCuFFTDirect: re-enable auto-alloc forward");
        checkCufft(cufftSetAutoAllocation(e.planBwd, 1),
                   "NekCuFFTDirect: re-enable auto-alloc backward");
    }

    {
        int dummy;
        checkCuda(cudaOccupancyMaxPotentialBlockSize(
                      &dummy, &e.blockSizeWave, WavenumberMultiplyKernel<TData>,
                      0, 0),
                  "NekCuFFTDirect: cudaOccupancyMaxPotentialBlockSize");
        (void)dummy;
    }

    TData *d_tmp = nullptr;
    const std::size_t nPhys =
        static_cast<std::size_t>(nhomo) * static_cast<std::size_t>(compStride);
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&d_tmp), nPhys * sizeof(TData)),
        "NekCuFFTDirect: cudaMalloc warm-up buffer");
    checkCuda(cudaMemset(d_tmp, 0, nPhys * sizeof(TData)),
              "NekCuFFTDirect: cudaMemset warm-up buffer");
    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(e.planFwd, d_tmp, e.d_cmplx),
                   "NekCuFFTDirect: warm-up D2Z");
        checkCufft(cufftExecZ2D(e.planBwd, e.d_cmplx, d_tmp),
                   "NekCuFFTDirect: warm-up Z2D");
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        checkCufft(cufftExecR2C(e.planFwd, d_tmp, e.d_cmplx),
                   "NekCuFFTDirect: warm-up R2C");
        checkCufft(cufftExecC2R(e.planBwd, e.d_cmplx, d_tmp),
                   "NekCuFFTDirect: warm-up C2R");
    }
    checkCuda(cudaStreamSynchronize(stream), "NekCuFFTDirect: warm-up sync");
    checkCuda(cudaFree(d_tmp), "NekCuFFTDirect: warm-up buffer free");

    return e;
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
template <typename TReal, int N_PLANES>
__global__ void PhysDerivZDxDirectKernel(const TReal *__restrict__ d_in,
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
        if (k == 0 || k == half_n)
        {
            thread_data[i] = complex_type{TReal(0), TReal(0)};
        }
        else
        {
            const int wn      = (k <= half_n) ? k : (k - N_PLANES);
            const TReal scale = static_cast<TReal>(wn) * beta;
            thread_data[i]    = complex_type{-thread_data[i].y * scale,
                                          thread_data[i].x * scale};
        }
    }

    IFFT_t().execute(thread_data, smem);

    const TReal inv_n = TReal(1) / static_cast<TReal>(N_PLANES);
    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        d_out[k * compStride + j] = thread_data[i].x * inv_n;
    }
}

// TReal is a template parameter of the launcher, not just the kernel: the
// attr_set latch below and the kernel whose attribute it raises must be the
// same instantiation, or one precision would never get its shared-memory
// limit lifted.
template <typename TReal, int N_PLANES>
void LaunchPhysDerivZDxDirect(const TReal *d_in, TReal *d_out, int NXY,
                              int compStride, TReal beta, cudaStream_t stream)
{
    using Traits = DxFFT<TReal, N_PLANES>;
    using FFT_t  = typename Traits::FFT;

    constexpr unsigned int shmem_size = FFT_t::shared_memory_size;

    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(PhysDerivZDxDirectKernel<TReal, N_PLANES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem_size);
        attr_set = true;
    }

    // Always launch directly (instead of via graphs):
    // device pointers vary per call so a static graph cache
    // keyed on {NXY, compStride} would replay stale pointers for
    // differently-allocated fields of the same shape. The outer
    // PhysDerivZOpDevice already handles graph capture/replay for the full
    // z-pipeline. When the stream is being captured by that outer graph the
    // kernel launch is recorded into it automatically.
    PhysDerivZDxDirectKernel<TReal, N_PLANES>
        <<<NXY, FFT_t::block_dim, shmem_size, stream>>>(d_in, d_out, NXY,
                                                        compStride, beta);
}

// Switches on the plane count to pick the compile-time FFT size. Takes the
// same argument order as PhysDerivZDirect, so the dispatch below forwards
// its arguments unchanged.
template <typename TReal>
void PhysDerivZDxDispatch(const TReal *d_in, TReal *d_out, unsigned int nhomo,
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
            LaunchPhysDerivZDxDirect<TReal, 16>(d_in, d_out, NXYI, compStrideI,
                                                beta, stream);
            break;
        case 32:
            LaunchPhysDerivZDxDirect<TReal, 32>(d_in, d_out, NXYI, compStrideI,
                                                beta, stream);
            break;
        case 64:
            LaunchPhysDerivZDxDirect<TReal, 64>(d_in, d_out, NXYI, compStrideI,
                                                beta, stream);
            break;
        case 128:
            LaunchPhysDerivZDxDirect<TReal, 128>(d_in, d_out, NXYI, compStrideI,
                                                 beta, stream);
            break;
        case 256:
            LaunchPhysDerivZDxDirect<TReal, 256>(d_in, d_out, NXYI, compStrideI,
                                                 beta, stream);
            break;
        case 512:
            LaunchPhysDerivZDxDirect<TReal, 512>(d_in, d_out, NXYI, compStrideI,
                                                 beta, stream);
            break;
        default:
            throw std::runtime_error(
                "PhysDerivZDirect: unsupported number of homogeneous planes " +
                std::to_string(nhomo) +
                " for the fused cuFFTDx kernel; supported values: 16, 32, 64, "
                "128, 256, 512");
    }
}

#endif // !NEKTAR_USE_CUFFTDX

} // anonymous namespace

template <typename TData>
void PhysDerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                      size_t NXY, size_t compStride, TData beta,
                      cudaStream_t stream)
{
#if defined(NEKTAR_USE_CUFFTDX)
    PhysDerivZDxDispatch(d_in, d_out, nhomo, NXY, compStride, beta, stream);
#else
    int deviceId = 0;
    checkCuda(cudaGetDevice(&deviceId), "NekCuFFTDirect: cudaGetDevice");

    const DirectPlanKey key{deviceId, nhomo, NXY, compStride, stream};

    DirectPlanEntry<TData> entry;
    if (!DirectPlanCache<TData>::Instance().Lookup(key, entry))
    {
        entry = CreateEntry<TData>(nhomo, NXY, compStride, stream);
        DirectPlanCache<TData>::Instance().Register(key, entry);
    }

    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecD2Z(entry.planFwd, const_cast<TData *>(d_in),
                                entry.d_cmplx),
                   "NekCuFFTDirect: cufftExecD2Z");
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        checkCufft(cufftExecR2C(entry.planFwd, const_cast<TData *>(d_in),
                                entry.d_cmplx),
                   "NekCuFFTDirect: cufftExecR2C");
    }

    const TData invN = 1.0 / static_cast<TData>(nhomo);
    const dim3 grid(
        static_cast<unsigned int>(NXY),
        static_cast<unsigned>((entry.halfN + 1 + entry.blockSizeWave - 1) /
                              entry.blockSizeWave));
    WavenumberMultiplyKernel<<<grid, entry.blockSizeWave, 0, stream>>>(
        entry.d_cmplx, entry.halfN, beta, invN);

    if constexpr (std::is_same_v<TData, double>)
    {
        checkCufft(cufftExecZ2D(entry.planBwd, entry.d_cmplx, d_out),
                   "NekCuFFTDirect: cufftExecZ2D");
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        checkCufft(cufftExecC2R(entry.planBwd, entry.d_cmplx, d_out),
                   "NekCuFFTDirect: cufftExecZ2D");
    }
#endif
}

template void PhysDerivZDirect<double>(const double *d_in, double *d_out,
                                       unsigned int nhomo, size_t NXY,
                                       size_t compStride, double beta,
                                       cudaStream_t stream);
template void PhysDerivZDirect<float>(const float *d_in, float *d_out,
                                      unsigned int nhomo, size_t NXY,
                                      size_t compStride, float beta,
                                      cudaStream_t stream);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
