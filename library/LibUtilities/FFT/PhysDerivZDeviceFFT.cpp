///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZDeviceFFT.cpp
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
// Description: SYCL z-derivative pipeline for Nektar++ PhysDeriv. The
// counterpart of PhysDerivZDeviceFFT.cu, with the cuFFT plans replaced by the
// DFT plan NekDeviceFFTSYCLHelper.h provides and the wavenumber multiply by a
// SYCL kernel.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_SYCL)

#include <unordered_map>

#include <LibUtilities/FFT/NekDeviceFFTSYCLHelper.h>
#include <LibUtilities/FFT/PhysDerivZDeviceFFT.h>

namespace Nektar::LibUtilities
{

namespace
{

// ---------------------------------------------------------------------
// SYCL pipeline: cached plans driving the real-to-complex transform, the
// wavenumber multiply and the complex-to-real transform as three separate
// submissions on the caller's queue.
// ---------------------------------------------------------------------

struct DirectPlanKey
{
    // No device id: the queue already names the device it was built on.
    sycl::queue *queue;
    unsigned int nhomo;
    size_t NXY;
    size_t compStride;

    bool operator==(const DirectPlanKey &o) const noexcept
    {
        return queue == o.queue && nhomo == o.nhomo && NXY == o.NXY &&
               compStride == o.compStride;
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
        hash             = mix(hash, reinterpret_cast<std::size_t>(k.queue));
        hash             = mix(hash, static_cast<std::size_t>(k.nhomo));
        hash             = mix(hash, static_cast<std::size_t>(k.NXY));
        hash             = mix(hash, static_cast<std::size_t>(k.compStride));
        return hash;
    }
};

template <typename TData> struct DirectPlanEntry
{
    DFTPlan<TData> plan;
    DFTCmplx<TData> *d_cmplx = nullptr;
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
                                   size_t compStride, sycl::queue &Q)
{
    DirectPlanEntry<TData> e;

    // The real field is plane-major, with compStride between successive
    // planes of one xy pencil and 1 between pencils; the half spectrum is
    // contiguous in halfN + 1 elements per pencil. The transform pair is
    // unnormalised, so the 1/nhomo normalisation is folded into the
    // wavenumber multiply below.
    e.plan = MakeDFTPlan<TData>(Q, NXY, static_cast<std::int64_t>(nhomo),
                                static_cast<std::int64_t>(compStride), 1);

    const std::size_t nCmplx = NXY * static_cast<std::size_t>(e.plan.halfN + 1);
    e.d_cmplx                = sycl::malloc_device<DFTCmplx<TData>>(nCmplx, Q);

    return e;
}

// Cache lookup shared by PhysDerivZDirect() and PhysDerivZPrepare(), so that
// preparing a queue and running on it cannot disagree about the key.
template <typename TData>
DirectPlanEntry<TData> GetOrCreateEntry(unsigned int nhomo, size_t NXY,
                                        size_t compStride, sycl::queue &Q)
{
    const DirectPlanKey key{&Q, nhomo, NXY, compStride};

    DirectPlanEntry<TData> entry;
    if (!DirectPlanCache<TData>::Instance().Lookup(key, entry))
    {
        entry = CreateEntry<TData>(nhomo, NXY, compStride, Q);
        DirectPlanCache<TData>::Instance().Register(key, entry);
    }

    return entry;
}

} // anonymous namespace

template <typename TData>
void PhysDerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                      size_t NXY, size_t compStride, TData beta,
                      unsigned int streamID)
{
    // The declaration hands the queue over as an id to keep the header
    // independent of SYCL; the registry resolves it to the real handle here.
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    DirectPlanEntry<TData> entry =
        GetOrCreateEntry<TData>(nhomo, NXY, compStride, Q);

    // The three stages are chained on their events rather than left to the
    // queue, so the pipeline holds whether or not the caller's queue is
    // in-order.
    sycl::event eFwd = ComputeForward(Q, entry.plan, d_in, entry.d_cmplx);

    const TData invN = 1.0 / static_cast<TData>(nhomo);
    sycl::event eWav = WavenumberMultiplyKernel(
        Q, entry.d_cmplx, NXY, entry.plan.halfN, beta, invN, {eFwd});

    sycl::event eBwd =
        ComputeBackward(Q, entry.plan, entry.d_cmplx, d_out, {eWav});

    // Publish the tail of the pipeline so that a later
    // SetStreamDependencies() on this stream id waits for it.
    SYCLQueue::SetEvent(streamID, eBwd);
}

template <typename TData>
void PhysDerivZ2Direct(const TData *d_in, TData *d_out, unsigned int nhomo,
                       size_t NXY, size_t compStride, TData beta,
                       unsigned int streamID)
{
    // The declaration hands the queue over as an id to keep the header
    // independent of SYCL; the registry resolves it to the real handle here.
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    // The plans depend only on the problem size, so the first derivative's
    // entry serves here too; only the wavenumber multiply differs.
    DirectPlanEntry<TData> entry =
        GetOrCreateEntry<TData>(nhomo, NXY, compStride, Q);

    // The three stages are chained on their events rather than left to the
    // queue, so the pipeline holds whether or not the caller's queue is
    // in-order.
    sycl::event eFwd = ComputeForward(Q, entry.plan, d_in, entry.d_cmplx);

    const TData invN = 1.0 / static_cast<TData>(nhomo);
    sycl::event eWav = WavenumberMultiply2Kernel(
        Q, entry.d_cmplx, NXY, entry.plan.halfN, beta, invN, {eFwd});

    sycl::event eBwd =
        ComputeBackward(Q, entry.plan, entry.d_cmplx, d_out, {eWav});

    // Publish the tail of the pipeline so that a later
    // SetStreamDependencies() on this stream id waits for it.
    SYCLQueue::SetEvent(streamID, eBwd);
}

template <typename TData>
void PhysDerivZPrepare(unsigned int nhomo, size_t NXY, size_t compStride,
                       unsigned int streamID)
{
    GetOrCreateEntry<TData>(nhomo, NXY, compStride,
                            SYCLQueue::GetInstance(streamID));
}

template void PhysDerivZDirect<double>(const double *d_in, double *d_out,
                                       unsigned int nhomo, size_t NXY,
                                       size_t compStride, double beta,
                                       unsigned int streamID);
template void PhysDerivZDirect<float>(const float *d_in, float *d_out,
                                      unsigned int nhomo, size_t NXY,
                                      size_t compStride, float beta,
                                      unsigned int streamID);

template void PhysDerivZ2Direct<double>(const double *d_in, double *d_out,
                                        unsigned int nhomo, size_t NXY,
                                        size_t compStride, double beta,
                                        unsigned int streamID);
template void PhysDerivZ2Direct<float>(const float *d_in, float *d_out,
                                       unsigned int nhomo, size_t NXY,
                                       size_t compStride, float beta,
                                       unsigned int streamID);

template void PhysDerivZPrepare<double>(unsigned int nhomo, size_t NXY,
                                        size_t compStride,
                                        unsigned int streamID);
template void PhysDerivZPrepare<float>(unsigned int nhomo, size_t NXY,
                                       size_t compStride,
                                       unsigned int streamID);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_SYCL
