///////////////////////////////////////////////////////////////////////////////
//
// File: NekCuFFTDx.cu
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
// Description: cuFFTDx-based z-derivative for the Nektar++ 3DH1 pipeline.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_CUDA) && defined(NEKTAR_USE_CUFFTDX)

#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cufftdx/cufftdx.hpp>

#include <LibUtilities/FFT/NekCuFFTDx.h>

#ifndef CUFFTDX_TARGET_SM
#define CUFFTDX_TARGET_SM 700
#endif

namespace Nektar::LibUtilities
{

namespace
{

constexpr int kEPT    = 4;
constexpr int kTarget = CUFFTDX_TARGET_SM;

struct DxGraphKey
{
    int nxy;
    int comp_stride;
    bool operator==(const DxGraphKey &o) const
    {
        return nxy == o.nxy && comp_stride == o.comp_stride;
    }
};

struct DxGraphKeyHash
{
    std::size_t operator()(const DxGraphKey &k) const
    {
        return std::hash<int>()(k.nxy) ^
               (std::hash<int>()(k.comp_stride) * 2654435761u);
    }
};

struct DxGraphEntry
{
    cudaGraph_t raw      = nullptr;
    cudaGraphExec_t exec = nullptr;
};

template <int N_PLANES> struct DxFFT
{
    using FFT = decltype(cufftdx::Block() + cufftdx::Size<N_PLANES>() +
                         cufftdx::Type<cufftdx::fft_type::c2c>() +
                         cufftdx::Direction<cufftdx::fft_direction::forward>() +
                         cufftdx::Precision<double>() +
                         cufftdx::ElementsPerThread<kEPT>() +
                         cufftdx::FFTsPerBlock<1>() + cufftdx::SM<kTarget>());

    using IFFT =
        decltype(cufftdx::Block() + cufftdx::Size<N_PLANES>() +
                 cufftdx::Type<cufftdx::fft_type::c2c>() +
                 cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                 cufftdx::Precision<double>() +
                 cufftdx::ElementsPerThread<kEPT>() +
                 cufftdx::FFTsPerBlock<1>() + cufftdx::SM<kTarget>());
};

template <int N_PLANES>
__global__ void PhysDerivZDxKernel(const double *__restrict__ d_in,
                                   double *__restrict__ d_out, int NXY,
                                   double beta)
{
    using Traits       = DxFFT<N_PLANES>;
    using FFT_t        = typename Traits::FFT;
    using IFFT_t       = typename Traits::IFFT;
    using complex_type = typename FFT_t::value_type;

    constexpr int half_n = N_PLANES / 2;

    const int j = static_cast<int>(blockIdx.x);

    extern __shared__ char shmem_raw[];
    complex_type *smem = reinterpret_cast<complex_type *>(shmem_raw);

    complex_type thread_data[kEPT];

    for (int i = 0; i < kEPT; ++i)
    {
        const int k    = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        thread_data[i] = complex_type{d_in[k * NXY + j], 0.0};
    }

    FFT_t().execute(thread_data, smem);

    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        if (k == 0 || k == half_n)
        {
            thread_data[i] = complex_type{0.0, 0.0};
        }
        else
        {
            const int wn       = (k <= half_n) ? k : (k - N_PLANES);
            const double scale = static_cast<double>(wn) * beta;
            thread_data[i]     = complex_type{-thread_data[i].y * scale,
                                          thread_data[i].x * scale};
        }
    }

    IFFT_t().execute(thread_data, smem);

    const double inv_n = 1.0 / static_cast<double>(N_PLANES);
    for (int i = 0; i < kEPT; ++i)
    {
        const int k        = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        d_out[k * NXY + j] = thread_data[i].x * inv_n;
    }
}

template <int N_PLANES>
void LaunchPhysDerivZDx(const double *d_in, double *d_out, int NXY, double beta,
                        cudaStream_t stream)
{
    using Traits = DxFFT<N_PLANES>;
    using FFT_t  = typename Traits::FFT;

    constexpr unsigned int shmem_size = FFT_t::shared_memory_size;

    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(PhysDerivZDxKernel<N_PLANES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem_size);
        attr_set = true;
    }

    cudaStreamCaptureStatus captureStatus;
    cudaStreamIsCapturing(stream, &captureStatus);
    if (captureStatus == cudaStreamCaptureStatusActive)
    {
        // Stream is already being captured by the caller; the kernel launch
        // below is recorded into that outer graph automatically.
        PhysDerivZDxKernel<N_PLANES>
            <<<NXY, FFT_t::block_dim, shmem_size, stream>>>(d_in, d_out, NXY,
                                                            beta);
        return;
    }

    static std::unordered_map<DxGraphKey, DxGraphEntry, DxGraphKeyHash> s_cache;
    const DxGraphKey key{NXY, 0};
    auto it = s_cache.find(key);
    if (it == s_cache.end())
    {
        DxGraphEntry entry;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        PhysDerivZDxKernel<N_PLANES>
            <<<NXY, FFT_t::block_dim, shmem_size, stream>>>(d_in, d_out, NXY,
                                                            beta);
        cudaStreamEndCapture(stream, &entry.raw);
        cudaGraphInstantiate(&entry.exec, entry.raw, nullptr, nullptr, 0);
        it = s_cache.emplace(key, entry).first;
    }
    // NXY is part of the key, so grid dimensions never change for a given
    // cache entry -- cudaGraphExecUpdate is not needed.
    cudaGraphLaunch(it->second.exec, stream);
}

// Reads from block device memory with stride compStride instead of NXY,
// writing directly to the z-slot in the output block. One block per xy
// pencil (blockIdx.x == j). Avoids separate gather/scatter copies.
template <int N_PLANES>
__global__ void PhysDerivZDxDirectKernel(const double *__restrict__ d_in,
                                         double *__restrict__ d_out, int NXY,
                                         int compStride, double beta)
{
    using Traits       = DxFFT<N_PLANES>;
    using FFT_t        = typename Traits::FFT;
    using IFFT_t       = typename Traits::IFFT;
    using complex_type = typename FFT_t::value_type;

    constexpr int half_n = N_PLANES / 2;

    const int j = static_cast<int>(blockIdx.x);

    extern __shared__ char shmem_raw[];
    complex_type *smem = reinterpret_cast<complex_type *>(shmem_raw);

    complex_type thread_data[kEPT];

    for (int i = 0; i < kEPT; ++i)
    {
        const int k    = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        thread_data[i] = complex_type{d_in[k * compStride + j], 0.0};
    }

    FFT_t().execute(thread_data, smem);

    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        if (k == 0 || k == half_n)
        {
            thread_data[i] = complex_type{0.0, 0.0};
        }
        else
        {
            const int wn       = (k <= half_n) ? k : (k - N_PLANES);
            const double scale = static_cast<double>(wn) * beta;
            thread_data[i]     = complex_type{-thread_data[i].y * scale,
                                          thread_data[i].x * scale};
        }
    }

    IFFT_t().execute(thread_data, smem);

    const double inv_n = 1.0 / static_cast<double>(N_PLANES);
    for (int i = 0; i < kEPT; ++i)
    {
        const int k = static_cast<int>(threadIdx.x) + i * FFT_t::stride;
        d_out[k * compStride + j] = thread_data[i].x * inv_n;
    }
}

template <int N_PLANES>
void LaunchPhysDerivZDxDirect(const double *d_in, double *d_out, int NXY,
                              int compStride, double beta, cudaStream_t stream)
{
    using Traits = DxFFT<N_PLANES>;
    using FFT_t  = typename Traits::FFT;

    constexpr unsigned int shmem_size = FFT_t::shared_memory_size;

    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(PhysDerivZDxDirectKernel<N_PLANES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem_size);
        attr_set = true;
    }

    // Always launch directly (instead of via graphs):
    // device pointers vary per call so a static graph cache
    // keyed on {NXY, compStride} would replay stale pointers for
    // differently-allocated fields of the same shape. The outer
    // PhysDerivZOpDeviceDx already handles graph capture/replay for the full
    // z-pipeline. When the stream is being captured by that outer graph the
    // kernel launch is recorded into it automatically.
    PhysDerivZDxDirectKernel<N_PLANES>
        <<<NXY, FFT_t::block_dim, shmem_size, stream>>>(d_in, d_out, NXY,
                                                        compStride, beta);
}

} // anonymous namespace

void PhysDerivZDx(const double *d_in, double *d_out, int NXY, int NPlanes,
                  double beta, cudaStream_t stream)
{
    switch (NPlanes)
    {
        case 16:
            LaunchPhysDerivZDx<16>(d_in, d_out, NXY, beta, stream);
            break;
        case 32:
            LaunchPhysDerivZDx<32>(d_in, d_out, NXY, beta, stream);
            break;
        case 64:
            LaunchPhysDerivZDx<64>(d_in, d_out, NXY, beta, stream);
            break;
        case 128:
            LaunchPhysDerivZDx<128>(d_in, d_out, NXY, beta, stream);
            break;
        case 256:
            LaunchPhysDerivZDx<256>(d_in, d_out, NXY, beta, stream);
            break;
        case 512:
            LaunchPhysDerivZDx<512>(d_in, d_out, NXY, beta, stream);
            break;
        default:
            throw std::runtime_error(
                "NekCuFFTDx::PhysDerivZDx: unsupported NPlanes=" +
                std::to_string(NPlanes) +
                "; supported values: 16, 32, 64, 128, 256, 512");
    }
}

void PhysDerivZDxDirect(const double *d_in, double *d_out, int NXY,
                        int compStride, int NPlanes, double beta,
                        cudaStream_t stream)
{
    switch (NPlanes)
    {
        case 16:
            LaunchPhysDerivZDxDirect<16>(d_in, d_out, NXY, compStride, beta,
                                         stream);
            break;
        case 32:
            LaunchPhysDerivZDxDirect<32>(d_in, d_out, NXY, compStride, beta,
                                         stream);
            break;
        case 64:
            LaunchPhysDerivZDxDirect<64>(d_in, d_out, NXY, compStride, beta,
                                         stream);
            break;
        case 128:
            LaunchPhysDerivZDxDirect<128>(d_in, d_out, NXY, compStride, beta,
                                          stream);
            break;
        case 256:
            LaunchPhysDerivZDxDirect<256>(d_in, d_out, NXY, compStride, beta,
                                          stream);
            break;
        case 512:
            LaunchPhysDerivZDxDirect<512>(d_in, d_out, NXY, compStride, beta,
                                          stream);
            break;
        default:
            throw std::runtime_error(
                "NekCuFFTDx::PhysDerivZDxDirect: unsupported NPlanes=" +
                std::to_string(NPlanes) +
                "; supported values: 16, 32, 64, 128, 256, 512");
    }
}

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_CUDA && NEKTAR_USE_CUFFTDX
