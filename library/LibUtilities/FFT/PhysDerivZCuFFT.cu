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
// Description: cuFFT-based z-derivative pipeline for Nektar++ PhysDeriv.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cufft.h>

#include <LibUtilities/FFT/PhysDerivZCuFFT.h>

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

template <typename T> struct cufftComplexData;

template <> struct cufftComplexData<double>
{
    using type = cufftDoubleComplex;
};

template <> struct cufftComplexData<float>
{
    using type = cufftComplex;
};

template <typename T>
using cufftComplexData_t = typename cufftComplexData<T>::type;

// Multiplies by i*k*beta*normScale in-place. DC and Nyquist are zeroed.
// Batch goes in x, wavenumber range in y.
template <typename TData>
__global__ static void WavenumberMultiplyKernel(
    cufftComplexData_t<TData> *__restrict__ d_cmplx, int halfN, TData beta,
    TData normScale)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    if (k == 0 || k == halfN)
    {
        d_cmplx[b * (halfN + 1) + k] = {0.0, 0.0};
        return;
    }

    const cufftComplexData_t<TData> cx = d_cmplx[b * (halfN + 1) + k];
    const TData scale            = static_cast<TData>(k) * beta * normScale;
    d_cmplx[b * (halfN + 1) + k] = {-cx.y * scale, cx.x * scale};
}

struct DirectPlanKey
{
    int deviceId;
    int nhomo;
    int NXY;
    int compStride;
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
    cufftHandle planFwd                = 0;
    cufftHandle planBwd                = 0;
    cufftComplexData_t<TData> *d_cmplx = nullptr;
    void *d_workspace                  = nullptr;
    int halfN                          = 0;
    int blockSizeWave                  = 0;
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
DirectPlanEntry<TData> CreateEntry(int nhomo, int NXY, int compStride,
                                   cudaStream_t stream)
{
    DirectPlanEntry<TData> e;
    e.halfN = nhomo / 2;

    const std::size_t nCmplx = static_cast<std::size_t>(NXY) * (e.halfN + 1);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&e.d_cmplx),
                         nCmplx * sizeof(cufftComplexData_t<TData>)),
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

    int dims[]    = {nhomo};
    int inembed[] = {nhomo};
    int onembed[] = {e.halfN + 1};

    std::size_t wsFwd = 0, wsBwd = 0;

    checkCufft(cufftMakePlanMany(e.planFwd, 1, dims, inembed, compStride, 1,
                                 onembed, 1, e.halfN + 1, CUFFT_D2Z, NXY,
                                 &wsFwd),
               "NekCuFFTDirect: cufftMakePlanMany forward");

    checkCufft(cufftMakePlanMany(e.planBwd, 1, dims, onembed, 1, e.halfN + 1,
                                 inembed, compStride, 1, CUFFT_Z2D, NXY,
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

} // anonymous namespace

template <typename TData>
void PhysDerivZDirect(const TData *d_in, TData *d_out, int nhomo, int NXY,
                      int compStride, TData beta, cudaStream_t stream)
{
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
        static_cast<unsigned>(NXY),
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
}

template void PhysDerivZDirect<double>(const double *d_in, double *d_out,
                                       int nhomo, int NXY, int compStride,
                                       double beta, cudaStream_t stream);
template void PhysDerivZDirect<float>(const float *d_in, float *d_out,
                                      int nhomo, int NXY, int compStride,
                                      float beta, cudaStream_t stream);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
