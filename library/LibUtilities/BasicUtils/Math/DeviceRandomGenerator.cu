///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceRandomGenerator.cu
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
// Description: DeviceRandomGenerator over cuRAND, or hipRAND when compiled
// as DeviceRandomGenerator.hip.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Math/DeviceRandomGenerator.hpp>
#include <LibUtilities/BasicUtils/Math/MathKernels.hpp>
#include <LibUtilities/Memory/MemoryAlloc.hpp>

#if defined(NEKTAR_ENABLE_CUDA)
#include <curand.h>
#elif defined(NEKTAR_ENABLE_HIP)
#include <hiprand/hiprand.h>
#endif

#include <string>
#include <type_traits>
#include <unordered_map>

namespace Nektar::Math
{

// The backends differ only in the names below.
#if defined(NEKTAR_ENABLE_CUDA)
using RandGenerator = curandGenerator_t;
using RandStatus    = curandStatus_t;

static constexpr RandStatus randSuccess = CURAND_STATUS_SUCCESS;
static constexpr auto randPhilox        = CURAND_RNG_PSEUDO_PHILOX4_32_10;

#define NEK_RAND(name) curand##name
#elif defined(NEKTAR_ENABLE_HIP)
using RandGenerator = hiprandGenerator_t;
using RandStatus    = hiprandStatus_t;

static constexpr RandStatus randSuccess = HIPRAND_STATUS_SUCCESS;
static constexpr auto randPhilox        = HIPRAND_RNG_PSEUDO_PHILOX4_32_10;

#define NEK_RAND(name) hiprand##name
#endif

static void CheckRandStatus(const RandStatus status, const std::string &call)
{
    if (status != randSuccess)
    {
        NEKERROR(ErrorUtil::efatal,
                 "DeviceRandomGenerator: " + call + " returned status " +
                     std::to_string(static_cast<int>(status)));
    }
}

static void GenerateNormal(RandGenerator generator, const double mean,
                           const double sigma, const size_t n, double *x)
{
    CheckRandStatus(
        NEK_RAND(GenerateNormalDouble)(generator, x, n, mean, sigma),
        "GenerateNormalDouble");
}

static void GenerateNormal(RandGenerator generator, const float mean,
                           const float sigma, const size_t n, float *x)
{
    CheckRandStatus(NEK_RAND(GenerateNormal)(generator, x, n, mean, sigma),
                    "GenerateNormal");
}

struct DeviceRandomGenerator::Engines
{
    // The generator of a stream, and two entries of scratch for the odd
    // entry of a normal fill, which the libraries generate only in pairs.
    struct StreamEngine
    {
        RandGenerator generator;
        double *scratch;
    };

    std::unordered_map<unsigned int, StreamEngine> engines;

    ~Engines()
    {
        for (auto &[streamID, engine] : engines)
        {
            NEK_RAND(DestroyGenerator)(engine.generator);
            deviceFree(engine.scratch, 2 * sizeof(double), streamID);
        }
    }

    StreamEngine &Get(const std::uint64_t seed, const unsigned int streamID)
    {
        auto it = engines.find(streamID);
        if (it != engines.end())
        {
            return it->second;
        }

        StreamEngine engine;
        CheckRandStatus(
            NEK_RAND(CreateGenerator)(&engine.generator, randPhilox),
            "CreateGenerator");
        CheckRandStatus(NEK_RAND(SetPseudoRandomGeneratorSeed)(
                            engine.generator, StreamSeed(seed, streamID)),
                        "SetPseudoRandomGeneratorSeed");
#if defined(NEKTAR_ENABLE_CUDA)
        CheckRandStatus(NEK_RAND(SetStream)(engine.generator,
                                            CUDAStream::GetInstance(streamID)),
                        "SetStream");
#else
        CheckRandStatus(NEK_RAND(SetStream)(engine.generator,
                                            HIPStream::GetInstance(streamID)),
                        "SetStream");
#endif
        deviceMalloc(&engine.scratch, 2 * sizeof(double), streamID);

        return engines.emplace(streamID, engine).first->second;
    }
};

DeviceRandomGenerator::DeviceRandomGenerator(const std::uint64_t seed)
    : m_seed(seed), m_engines(std::make_unique<Engines>())
{
}

DeviceRandomGenerator::~DeviceRandomGenerator() = default;

/**
 * The library generates on (0, 1], which is then mapped onto (a, b] by the
 * Math kernels, on the same stream.
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
    if constexpr (std::is_same_v<TData, double>)
    {
        CheckRandStatus(NEK_RAND(GenerateUniformDouble)(engine.generator, x, n),
                        "GenerateUniformDouble");
    }
    else
    {
        CheckRandStatus(NEK_RAND(GenerateUniform)(engine.generator, x, n),
                        "GenerateUniform");
    }

    mulKernel<NektarSpaces::Device>(n, b - a, x, x, streamID);
    addScalarKernel<NektarSpaces::Device>(n, a, x, x, streamID);
}

/**
 * The library generates normal numbers in pairs, so the last entry of an odd
 * n is taken from a pair generated into scratch.
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
    auto scratch = reinterpret_cast<TData *>(engine.scratch);

    if (n % 2 == 0)
    {
        GenerateNormal(engine.generator, mean, sigma, n, x);
    }
    else
    {
        GenerateNormal(engine.generator, mean, sigma, n - 1, x);
        GenerateNormal(engine.generator, mean, sigma, 2, scratch);
        deviceMemcpy<DeviceToDevice>(x + n - 1, scratch, sizeof(TData),
                                     streamID);
    }
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
