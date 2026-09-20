///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZDeviceFFT.hip
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
// Description: hipFFT z-derivative pipeline for Nektar++ PhysDeriv. The HIP
// counterpart of PhysDerivZDeviceFFT.cu; there is no ROCm equivalent of
// cuFFTDx, so only the three-launch hipFFT pipeline is provided here.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_DEVICE)

#include <algorithm>
#include <unordered_map>

#include <LibUtilities/FFT/NekDeviceFFTHIPCUDAHelper.h>
#include <LibUtilities/FFT/PhysDerivZDeviceFFT.h>

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
    int halfN                      = 0;
    int blockSizeWave              = 0;
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

} // anonymous namespace

template <typename TData>
void PhysDerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                      size_t NXY, size_t compStride, TData beta, void *vstream)
{
    // The declaration hands the stream over as void * to keep the header
    // independent of the HIP runtime; everything below works on the real
    // handle.
    hipStream_t stream = static_cast<hipStream_t>(vstream);

    int deviceId = 0;
    CHECK_HIPCUDA_ERROR(hipGetDevice(&deviceId));

    const DirectPlanKey key{deviceId, nhomo, NXY, compStride, stream};

    DirectPlanEntry<TData> entry;
    if (!DirectPlanCache<TData>::Instance().Lookup(key, entry))
    {
        entry = CreateEntry<TData>(nhomo, NXY, compStride, stream);
        DirectPlanCache<TData>::Instance().Register(key, entry);
    }

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
    WavenumberMultiplyKernel<<<grid, entry.blockSizeWave, 0, stream>>>(
        entry.d_cmplx, entry.halfN, beta, invN);

    if constexpr (std::is_same_v<TData, double>)
    {
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftExecZ2D(entry.planBwd, entry.d_cmplx, d_out));
    }
    else if constexpr (std::is_same_v<TData, float>)
    {
        CHECK_HIPCUDA_FFT_ERROR(
            hipfftExecC2R(entry.planBwd, entry.d_cmplx, d_out));
    }
}

template void PhysDerivZDirect<double>(const double *d_in, double *d_out,
                                       unsigned int nhomo, size_t NXY,
                                       size_t compStride, double beta,
                                       void *vstream);
template void PhysDerivZDirect<float>(const float *d_in, float *d_out,
                                      unsigned int nhomo, size_t NXY,
                                      size_t compStride, float beta,
                                      void *vstream);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_DEVICE
