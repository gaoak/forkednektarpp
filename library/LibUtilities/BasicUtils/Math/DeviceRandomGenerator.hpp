///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceRandomGenerator.hpp
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
// Description: Device random number generator, over cuRAND in
// DeviceRandomGenerator.cu, hipRAND in DeviceRandomGenerator.hip and rocRAND,
// oneMath or oneMKL in DeviceRandomGenerator.cpp. This header is backend
// agnostic, so the vendor libraries stay private to LibUtilities.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LibUtilitiesDeclspec.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace Nektar::Math
{

/**
 * \brief Fills device memory with pseudo-random numbers.
 *
 * Each stream has a Philox generator of its own, created on first use with a
 * seed derived from the seed given here and the stream ID, and only ever
 * launched on that stream. Generation on different streams therefore shares
 * no state, and successive calls on one stream continue its sequence. The
 * numbers a buffer receives depend on the stream it is filled on.
 */
class DeviceRandomGenerator
{
public:
    LIB_UTILITIES_EXPORT explicit DeviceRandomGenerator(
        const std::uint64_t seed);

    LIB_UTILITIES_EXPORT ~DeviceRandomGenerator();

    DeviceRandomGenerator(const DeviceRandomGenerator &)            = delete;
    DeviceRandomGenerator &operator=(const DeviceRandomGenerator &) = delete;

    /// Fill the n entries of x with numbers uniform between a and b.
    template <typename TData>
    LIB_UTILITIES_EXPORT void Uniform(const TData a, const TData b,
                                      const size_t n, TData *x,
                                      const unsigned int streamID = 0);

    /// Fill the n entries of x with normal numbers of the given mean and
    /// standard deviation.
    template <typename TData>
    LIB_UTILITIES_EXPORT void Normal(const TData mean, const TData sigma,
                                     const size_t n, TData *x,
                                     const unsigned int streamID = 0);

private:
    struct Engines;

    std::uint64_t m_seed;
    std::unique_ptr<Engines> m_engines;

    /// Seed of the generator of stream streamID.
    static std::uint64_t StreamSeed(const std::uint64_t seed,
                                    const unsigned int streamID)
    {
        // splitmix64 of the pair, so nearby seeds and stream IDs give
        // unrelated Philox keys.
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ull * (streamID + 1ull);
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

} // namespace Nektar::Math
