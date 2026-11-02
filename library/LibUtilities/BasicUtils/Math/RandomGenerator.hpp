///////////////////////////////////////////////////////////////////////////////
//
// File: RandomGenerator.hpp
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
// Description: Seeded pseudo-random number generator for the Math fill
// functions, on the host or on the device.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends.hpp>

#if defined(NEKTAR_ENABLE_DEVICE)
#include <LibUtilities/BasicUtils/Math/DeviceRandomGenerator.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>

namespace Nektar::Math
{

/**
 * \brief Pseudo-random number generator behind Math::fillUniform and
 * Math::fillNormal.
 *
 * Host memory is filled from one Mersenne Twister engine, in call order.
 * Device memory is filled by a DeviceRandomGenerator, over cuRAND, hipRAND or
 * oneMath, with a generator per stream. Two objects built with the same seed
 * produce the same numbers for the same sequence of calls; the host and the
 * device do not produce the same numbers as each other.
 */
class RandomGenerator
{
public:
    explicit RandomGenerator(const std::uint64_t seed = 0)
        : m_hostEngine(seed)
#if defined(NEKTAR_ENABLE_DEVICE)
          ,
          m_device(std::make_unique<DeviceRandomGenerator>(seed))
#endif
    {
    }

    /// Fill the n entries of x, in MemSpace, with numbers uniform between a
    /// and b.
    template <typename MemSpace, typename TData>
    void Uniform(const TData a, const TData b, const size_t n, TData *x,
                 [[maybe_unused]] const unsigned int streamID = 0)
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            std::uniform_real_distribution<TData> distribution(a, b);
            for (size_t i = 0; i < n; ++i)
            {
                x[i] = distribution(m_hostEngine);
            }
        }
#if defined(NEKTAR_ENABLE_DEVICE)
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            m_device->Uniform(a, b, n, x, streamID);
        }
#endif
    }

    /// Fill the n entries of x, in MemSpace, with normal numbers of the given
    /// mean and standard deviation.
    template <typename MemSpace, typename TData>
    void Normal(const TData mean, const TData sigma, const size_t n, TData *x,
                [[maybe_unused]] const unsigned int streamID = 0)
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            std::normal_distribution<TData> distribution(mean, sigma);
            for (size_t i = 0; i < n; ++i)
            {
                x[i] = distribution(m_hostEngine);
            }
        }
#if defined(NEKTAR_ENABLE_DEVICE)
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            m_device->Normal(mean, sigma, n, x, streamID);
        }
#endif
    }

private:
    std::mt19937_64 m_hostEngine;
#if defined(NEKTAR_ENABLE_DEVICE)
    std::unique_ptr<DeviceRandomGenerator> m_device;
#endif
};

} // namespace Nektar::Math
