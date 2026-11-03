///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceRandomGenerator.cpp
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
// Description: DeviceRandomGenerator over rocRAND on HIP targets, oneMath, or
// oneMKL on Intel targets. Where none is configured, numbers are generated on
// the host and copied to the device.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Math/DeviceRandomGenerator.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

#if defined(SYCL_ENABLE_HIP)
#include <LibUtilities/BasicUtils/Math/MathKernels.hpp>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#elif defined(NEKTAR_ENABLE_ONEMATH)
#include "oneapi/math.hpp"
#elif defined(NEKTAR_ENABLE_ONEMKL)
#include "oneapi/mkl.hpp"
#endif

#include <exception>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Nektar::Math
{

#if !defined(SYCL_ENABLE_HIP) &&                                               \
    (defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL))
#define NEKTAR_SYCL_HAVE_RNG_LIBRARY

// The two libraries differ only in the namespace below.
#if defined(NEKTAR_ENABLE_ONEMATH)
namespace rng = oneapi::math::rng;
#else
namespace rng = oneapi::mkl::rng;
#endif
#endif

#if defined(SYCL_ENABLE_HIP)
static void CheckRandStatus(const rocrand_status status, const char *call)
{
    if (status != ROCRAND_STATUS_SUCCESS)
    {
        NEKERROR(ErrorUtil::efatal,
                 std::string("DeviceRandomGenerator: ") + call +
                     " returned status " +
                     std::to_string(static_cast<int>(status)));
    }
}

/**
 * Run call on generator in a native command on the queue of streamID, so that
 * rocRAND launches on the stream behind that queue. rocRAND reports any HIP
 * error pending on the thread as a launch failure of its own kernel, and the
 * SYCL runtime leaves such errors on the threads that run native commands, so
 * the pending error is cleared first. oneMath's rocRAND backend does not.
 */
template <typename Call>
static void RunOnStream(const rocrand_generator generator,
                        const unsigned int streamID, const char *name,
                        const Call &call)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([&](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        auto native = [=](sycl::interop_handle ih) {
            (void)hipGetLastError();
            CheckRandStatus(
                rocrand_set_stream(generator,
                                    static_cast<hipStream_t>(
                                       ih.get_native_queue<sycl_backend>())),
                "rocrand_set_stream");
            CheckRandStatus(call(generator), name);
        };
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(native);
#elif defined(__DPCPP_COMPILER)
        cgh.ext_codeplay_enqueue_native_command(native);
#endif
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <typename TData>
static void GenerateUniform(const rocrand_generator generator, const size_t n,
                            TData *x, const unsigned int streamID)
{
    if constexpr (std::is_same_v<TData, double>)
    {
        RunOnStream(generator, streamID, "rocrand_generate_uniform_double",
                    [=](rocrand_generator g) {
                        return rocrand_generate_uniform_double(g, x, n);
                    });
    }
    else
    {
        RunOnStream(generator, streamID, "rocrand_generate_uniform",
                    [=](rocrand_generator g) {
                        return rocrand_generate_uniform(g, x, n);
                    });
    }
}

template <typename TData>
static void GenerateNormal(const rocrand_generator generator, const TData mean,
                           const TData sigma, const size_t n, TData *x,
                           const unsigned int streamID)
{
    if constexpr (std::is_same_v<TData, double>)
    {
        RunOnStream(generator, streamID, "rocrand_generate_normal_double",
                    [=](rocrand_generator g) {
                        return rocrand_generate_normal_double(g, x, n, mean,
                                                              sigma);
                    });
    }
    else
    {
        RunOnStream(generator, streamID, "rocrand_generate_normal",
                    [=](rocrand_generator g) {
                        return rocrand_generate_normal(g, x, n, mean, sigma);
                    });
    }
}
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
template <typename Distribution, typename TData>
static void Generate(const Distribution &distribution,
                     rng::philox4x32x10 &engine, const size_t n, TData *x,
                     const unsigned int streamID)
{
    try
    {
        sycl::event event =
            rng::generate(distribution, engine, static_cast<std::int64_t>(n), x,
                          getSYCLDefaultExecutionDependencies(streamID));
        SYCLQueue::SetEvent(streamID, event);
    }
    catch (const std::exception &e)
    {
        NEKERROR(ErrorUtil::efatal,
                 std::string("DeviceRandomGenerator: generate failed: ") +
                     e.what());
    }
}
#else
static void WarnHostGeneration()
{
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        NEKERROR(ErrorUtil::ewarning,
                 "Device random numbers are generated on the host and copied, "
                 "because this build configures no oneMath or oneMKL RNG "
                 "backend.");
    }
}

// Copy the n entries of host to x on stream streamID, and wait for the copy
// before host goes out of scope.
template <typename TData>
static void CopyToDevice(const std::vector<TData> &host, TData *x,
                         const unsigned int streamID)
{
    deviceMemcpy<HostToDevice>(x, host.data(), host.size() * sizeof(TData),
                               streamID);
    nekStreamSynchronize(streamID);
}
#endif

struct DeviceRandomGenerator::Engines
{
#if defined(SYCL_ENABLE_HIP)
    // The generator of a stream, and two entries of scratch for the odd
    // entry of a normal fill, which is generated as a pair.
    struct StreamEngine
    {
        rocrand_generator generator;
        double *scratch;
    };
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
    // The engine of a stream, bound to its queue, and two entries of scratch
    // for the odd entry of a normal fill, which the cuRAND backend generates
    // only in pairs.
    struct StreamEngine
    {
        std::unique_ptr<rng::philox4x32x10> generator;
        double *scratch;
    };
#else
    struct StreamEngine
    {
        std::mt19937_64 generator;
    };
#endif

    std::unordered_map<unsigned int, StreamEngine> engines;

    ~Engines()
    {
#if defined(SYCL_ENABLE_HIP)
        for (auto &[streamID, engine] : engines)
        {
            rocrand_destroy_generator(engine.generator);
            deviceFree(engine.scratch, 2 * sizeof(double), streamID);
        }
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
        for (auto &[streamID, engine] : engines)
        {
            deviceFree(engine.scratch, 2 * sizeof(double), streamID);
        }
#endif
    }

    StreamEngine &Get(const std::uint64_t seed, const unsigned int streamID)
    {
        auto it = engines.find(streamID);
        if (it != engines.end())
        {
            return it->second;
        }

        StreamEngine engine;
#if defined(SYCL_ENABLE_HIP)
        CheckRandStatus(
            rocrand_create_generator(&engine.generator,
                                     ROCRAND_RNG_PSEUDO_PHILOX4_32_10),
            "rocrand_create_generator");
        CheckRandStatus(
            rocrand_set_seed(engine.generator, StreamSeed(seed, streamID)),
            "rocrand_set_seed");
        deviceMalloc(&engine.scratch, 2 * sizeof(double), streamID);
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
        try
        {
            engine.generator = std::make_unique<rng::philox4x32x10>(
                SYCLQueue::GetInstance(streamID), StreamSeed(seed, streamID));
        }
        catch (const std::exception &e)
        {
            NEKERROR(ErrorUtil::efatal,
                     std::string("DeviceRandomGenerator: no RNG engine for "
                                 "this device: ") +
                         e.what());
        }
        deviceMalloc(&engine.scratch, 2 * sizeof(double), streamID);
#else
        WarnHostGeneration();
        engine.generator.seed(StreamSeed(seed, streamID));
#endif

        return engines.emplace(streamID, std::move(engine)).first->second;
    }
};

DeviceRandomGenerator::DeviceRandomGenerator(const std::uint64_t seed)
    : m_seed(seed), m_engines(std::make_unique<Engines>())
{
}

DeviceRandomGenerator::~DeviceRandomGenerator() = default;

/**
 * rocRAND generates on (0, 1], which is then mapped onto (a, b] by the Math
 * kernels, on the same stream.
 */
template <typename TData>
void DeviceRandomGenerator::Uniform(const TData a, const TData b,
                                    const size_t n, TData *x,
                                    const unsigned int streamID)
{
    if (n == 0)
    {
        return;
    }

    auto &engine = m_engines->Get(m_seed, streamID);
#if defined(SYCL_ENABLE_HIP)
    GenerateUniform(engine.generator, n, x, streamID);
    mulKernel<NektarSpaces::Device>(n, b - a, x, x, streamID);
    addScalarKernel<NektarSpaces::Device>(n, a, x, x, streamID);
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
    Generate(rng::uniform<TData>(a, b), *engine.generator, n, x, streamID);
#else
    std::uniform_real_distribution<TData> distribution(a, b);
    std::vector<TData> host(n);
    for (auto &value : host)
    {
        value = distribution(engine.generator);
    }
    CopyToDevice(host, x, streamID);
#endif
}

/**
 * Normal numbers are generated in pairs, as the cuRAND backend of oneMath
 * requires, so the last entry of an odd n is taken from a pair generated into
 * scratch.
 */
template <typename TData>
void DeviceRandomGenerator::Normal(const TData mean, const TData sigma,
                                   const size_t n, TData *x,
                                   const unsigned int streamID)
{
    if (n == 0)
    {
        return;
    }

    auto &engine = m_engines->Get(m_seed, streamID);
#if defined(SYCL_ENABLE_HIP)
    auto scratch = reinterpret_cast<TData *>(engine.scratch);

    if (n % 2 == 0)
    {
        GenerateNormal(engine.generator, mean, sigma, n, x, streamID);
    }
    else
    {
        GenerateNormal(engine.generator, mean, sigma, n - 1, x, streamID);
        GenerateNormal(engine.generator, mean, sigma, size_t(2), scratch,
                       streamID);
        deviceMemcpy<DeviceToDevice>(x + n - 1, scratch, sizeof(TData),
                                     streamID);
    }
#elif defined(NEKTAR_SYCL_HAVE_RNG_LIBRARY)
    auto scratch = reinterpret_cast<TData *>(engine.scratch);

    if (n % 2 == 0)
    {
        Generate(rng::gaussian<TData>(mean, sigma), *engine.generator, n, x,
                 streamID);
    }
    else
    {
        Generate(rng::gaussian<TData>(mean, sigma), *engine.generator, n - 1, x,
                 streamID);
        Generate(rng::gaussian<TData>(mean, sigma), *engine.generator, 2,
                 scratch, streamID);
        deviceMemcpy<DeviceToDevice>(x + n - 1, scratch, sizeof(TData),
                                     streamID);
    }
#else
    std::normal_distribution<TData> distribution(mean, sigma);
    std::vector<TData> host(n);
    for (auto &value : host)
    {
        value = distribution(engine.generator);
    }
    CopyToDevice(host, x, streamID);
#endif
}

template void DeviceRandomGenerator::Uniform<double>(const double, const double,
                                                     const size_t, double *,
                                                     const unsigned int);
template void DeviceRandomGenerator::Uniform<float>(const float, const float,
                                                    const size_t, float *,
                                                    const unsigned int);
template void DeviceRandomGenerator::Normal<double>(const double, const double,
                                                    const size_t, double *,
                                                    const unsigned int);
template void DeviceRandomGenerator::Normal<float>(const float, const float,
                                                   const size_t, float *,
                                                   const unsigned int);

} // namespace Nektar::Math
