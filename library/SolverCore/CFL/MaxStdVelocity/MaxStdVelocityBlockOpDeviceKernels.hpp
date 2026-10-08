///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocityBlockOpDeviceKernels.hpp
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
// Description: MaxStdVelocity device kernel.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>

namespace Nektar::SolverCore::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief Largest standard element velocity over a block, one element per
 * thread stride.
 *
 * The arithmetic is the same as the serial implementation, and deliberately
 * kept in the same order so the two can be compared term by term: contract
 * the velocity with the derivative factors into each reference direction, add
 * the wave speed carried into that direction, and take the root of the sum of
 * squares. Only the element loop and the final reduction differ.
 *
 * Padded elements are excluded by the loop bound rather than masked
 * afterwards - their derivative factors are zero but a wave speed formed from
 * a zero density is not finite, and multiplying that by zero does not give
 * zero.
 */
template <
    typename Implementation, bool DEFORMED, typename TthreadBlock,
    typename TData,
    std::enable_if_t<std::is_same_v<Implementation, MultiRegions::SumFac>, bool>
        Enable = true>
NEK_DEVICE_KERNEL void MaxStdVelocityKernelLauncher(
    const size_t nelmt, const size_t compSize, const unsigned int dimension,
    const unsigned int coordDim, const unsigned int nqTot,
    const TData soundSpeedFactor, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    // Both the field data and the derivative factors are interleaved at the
    // warp width - anything else would break coalesced access - so the one
    // constant serves both.
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const bool haveSoundSpeed = (soundSpeedFactor != TData(0));
    const unsigned int ndf    = coordDim * dimension;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        // Derivative factors for this element's warp: packed per warp by
        // point (deformed only), then component, then lane.
        const TData *inptr = in + iwarp * warpsize * nqTot;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;

        for (unsigned int q = 0; q < nqTot; ++q)
        {
            const unsigned int index = warpsize * q + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * q + ilane : ilane;

            TData pntSq = 0.0;
            for (unsigned int i = 0; i < dimension; ++i)
            {
                // Contract the physical components into reference direction
                // i: the velocity and, if present, the wave speed carried
                // with it.
                TData stdVel   = 0.0;
                TData stdSound = 0.0;
                for (unsigned int j = 0; j < coordDim; ++j)
                {
                    const TData g =
                        dfptr[(dimension * j + i) * warpsize + dfindex];
                    stdVel += g * inptr[j * compSize + index];
                    if (haveSoundSpeed)
                    {
                        stdSound += g;
                    }
                }

                TData vel = abs(stdVel);
                if (haveSoundSpeed)
                {
                    const TData c = inptr[coordDim * compSize + index];
                    vel += soundSpeedFactor * abs(stdSound * c);
                }
                pntSq += vel * vel;
            }

            const TData value = sqrt(pntSq);
            acc               = acc > value ? acc : value;
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceMax(acc, threadBlock, out);
}

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData,
          std::enable_if_t<
              std::is_same_v<Implementation, MultiRegions::SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void MaxStdVelocityKernelLauncher(
    const size_t nelmt, const size_t compSize, const unsigned int dimension,
    const unsigned int coordDim, const unsigned int nqTot,
    const TData soundSpeedFactor, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const bool haveSoundSpeed = (soundSpeedFactor != TData(0));
    const unsigned int ndf    = coordDim * dimension;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    TData acc                 = 0.0;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
    size_t e                  = getBlockIdx<0>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nqTot * e;
        const TData *dfptr = df + ndf * dfsize * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int dfindex = DEFORMED ? idx : 0;

            TData pntSq = 0.0;
            for (unsigned int i = 0; i < dimension; ++i)
            {
                // Contract the physical components into reference direction
                // i: the velocity and, if present, the wave speed carried
                // with it.
                TData stdVel   = 0.0;
                TData stdSound = 0.0;
                for (unsigned int j = 0; j < coordDim; ++j)
                {
                    const TData g =
                        dfptr[(dimension * j + i) * dfsize + dfindex];
                    stdVel += g * inptr[j * compSize + idx];
                    if (haveSoundSpeed)
                    {
                        stdSound += g;
                    }
                }

                TData vel = abs(stdVel);
                if (haveSoundSpeed)
                {
                    const TData c = inptr[coordDim * compSize + idx];
                    vel += soundSpeedFactor * abs(stdSound * c);
                }
                pntSq += vel * vel;
            }

            const TData value = sqrt(pntSq);
            acc               = acc > value ? acc : value;
        }

        e += getBlockRange<0>(threadBlock);
    }

    blockReduceMax(acc, threadBlock, out);
}
#endif

} // namespace Nektar::SolverCore::detail
