///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZDeviceFFT.cpp
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
// Description: SYCL z-derivative pipeline for the Nektar++ DerivZ operators:
// cached DFT descriptors from NekDeviceFFTSYCLHelper.h driving the forward
// transform, the wavenumber multiply and the backward transform, each
// submitted as its own kernel and chained on its predecessor's event.
//
///////////////////////////////////////////////////////////////////////////////

#if defined(NEKTAR_ENABLE_SYCL)

#include <unordered_map>

#include <LibUtilities/FFT/DerivZDeviceFFT.h>
#include <LibUtilities/FFT/NekDeviceFFTSYCLHelper.h>

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
    /// Where the inverse transform lands when the caller is accumulating;
    /// oneMath owns its output store, so it cannot add into the caller's
    /// buffer directly. Allocated on the first APPEND prepare and null
    /// otherwise, so a caller that overwrites pays nothing for it.
    TData *d_accum = nullptr;
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

// Cache lookup shared by DerivZDirect() and DerivZPrepare(), so that
// preparing a queue and running on it cannot disagree about the key.
template <typename TData>
DirectPlanEntry<TData> &GetOrCreateEntry(unsigned int nhomo, size_t NXY,
                                         size_t compStride, sycl::queue &Q)
{
    const DirectPlanKey key{&Q, nhomo, NXY, compStride};

    if (DirectPlanEntry<TData> *hit =
            DirectPlanCache<TData>::Instance().Find(key))
    {
        return *hit;
    }

    return DirectPlanCache<TData>::Instance().Register(
        key, CreateEntry<TData>(nhomo, NXY, compStride, Q));
}

/// Attach the accumulate buffer to @p entry if it does not have one. It is
/// laid out exactly as the inverse transform writes it -- plane p at
/// p * compStride -- and zeroed once here: the transform never writes the
/// padding between NXY and compStride, so it stays zero for the life of the
/// buffer and the sum below can run flat over the whole span.
/// This allocates, which the callers resolve before any capture, so it is
/// reached only from DerivZPrepare().
template <typename TData>
void EnsureAccumBuffer(DirectPlanEntry<TData> &entry, unsigned int nhomo,
                       size_t compStride, sycl::queue &Q)
{
    if (entry.d_accum == nullptr)
    {
        const size_t nsize = nhomo * compStride;
        entry.d_accum      = sycl::malloc_device<TData>(nsize, Q);
        Q.memset(entry.d_accum, 0, nsize * sizeof(TData)).wait();
    }
}

/// Add the transform in @p src into @p dst over the whole plane-major span.
/// The padding is zero in @p src, so a padded slot only ever adds zero to
/// itself.
template <typename TData>
sycl::event AccumulateKernel(sycl::queue &Q, const TData *src, TData *dst,
                             size_t nsize, const std::vector<sycl::event> &deps)
{
    return Q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(nsize),
                       [=](sycl::id<1> idx) { dst[idx[0]] += src[idx[0]]; });
    });
}

} // anonymous namespace

template <typename TData, DerivZOrder DERIVORDER, bool APPEND>
void DerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                  size_t NXY, size_t compStride, TData beta,
                  unsigned int streamID)
{
    // The declaration hands the queue over as an id to keep the header
    // independent of SYCL; the registry resolves it to the real handle here.
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    // The plans depend only on the problem size, so both orders share one
    // entry; only the wavenumber multiply differs.
    auto &entry = GetOrCreateEntry<TData>(nhomo, NXY, compStride, Q);

    // oneMath owns its output store, so an accumulating call transforms into
    // the buffer the plan holds and is summed into d_out below.
    TData *d_target = APPEND ? entry.d_accum : d_out;

    // The three stages are chained on their events rather than left to the
    // queue, so the pipeline holds whether or not the caller's queue is
    // in-order.
    sycl::event eFwd = ComputeForward(Q, entry.plan, d_in, entry.d_cmplx);

    const TData invN = 1.0 / static_cast<TData>(nhomo);
    sycl::event eWav;
    if constexpr (DERIVORDER == DerivZOrder::First)
    {
        eWav = WavenumberMultiplyKernel(Q, entry.d_cmplx, NXY, entry.plan.halfN,
                                        beta, invN, {eFwd});
    }
    else
    {
        eWav = WavenumberMultiply2Kernel(Q, entry.d_cmplx, NXY,
                                         entry.plan.halfN, beta, invN, {eFwd});
    }

    sycl::event eBwd =
        ComputeBackward(Q, entry.plan, entry.d_cmplx, d_target, {eWav});

    if constexpr (APPEND)
    {
        eBwd = AccumulateKernel(Q, entry.d_accum, d_out, nhomo * compStride,
                                {eBwd});
    }

    // Publish the tail of the pipeline so that a later
    // SetStreamDependencies() on this stream id waits for it.
    SYCLQueue::SetEvent(streamID, eBwd);
}

template <typename TData, bool APPEND>
void DerivZPrepare(unsigned int nhomo, size_t NXY, size_t compStride,
                   unsigned int streamID)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    auto &entry    = GetOrCreateEntry<TData>(nhomo, NXY, compStride, Q);

    // The callers resolve every allocation before opening a graph capture on
    // the backends that have one, so the buffer an accumulating call
    // transforms through is created here for symmetry with those.
    if constexpr (APPEND)
    {
        EnsureAccumBuffer<TData>(entry, nhomo, compStride, Q);
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

#endif // NEKTAR_ENABLE_SYCL
