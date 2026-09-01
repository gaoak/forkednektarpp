///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocitySerialAVXKernels.hpp
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
// Description: MaxStdVelocity, Serial and AVX kernel.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

namespace Nektar::Operators::detail
{

/**
 * @brief Largest standard element velocity over one element group, in the
 * vector type.
 *
 * One simd lane per element: @p in and @p df hold one group of
 * simd_type::width elements interleaved at that width, and the return value
 * carries each lane's maximum, to be max-combined and horizontally reduced
 * by the caller. Padded lanes are expected to have been masked to zero -
 * with zero derivative factors they then contribute exactly zero to a
 * maximum of non-negative values.
 *
 * The arithmetic is the same as the device kernel, term by term: contract
 * the velocity with the derivative factors into each reference direction,
 * add the wave speed carried into that direction, and take the root of the
 * sum of squares.
 */
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static simd_type MaxStdVelocityKernel(
    const unsigned int nqTot, const unsigned int dimension,
    const unsigned int coordDim, const size_t compVecStride,
    const typename simd_type::scalarType soundSpeedFactor, const simd_type *df,
    const simd_type *in)
{
    const bool haveSoundSpeed =
        (soundSpeedFactor != typename simd_type::scalarType(0));
    const unsigned int ndf = coordDim * dimension;

    simd_type acc = 0.0;
    for (unsigned int q = 0; q < nqTot; ++q)
    {
        const simd_type *dfptr = DEFORMED ? df + q * ndf : df;

        simd_type pntSq = 0.0;
        for (unsigned int i = 0; i < dimension; ++i)
        {
            // Contract the physical components into reference direction i:
            // the velocity and, if present, the wave speed carried with it.
            simd_type stdVel   = 0.0;
            simd_type stdSound = 0.0;
            for (unsigned int j = 0; j < coordDim; ++j)
            {
                const simd_type g = dfptr[dimension * j + i];
                stdVel.fma(g, in[j * compVecStride + q]);
                if (haveSoundSpeed)
                {
                    stdSound += g;
                }
            }

            simd_type vel = abs(stdVel);
            if (haveSoundSpeed)
            {
                vel.fma(simd_type(soundSpeedFactor),
                        abs(stdSound * in[coordDim * compVecStride + q]));
            }
            pntSq.fma(vel, vel);
        }

        acc = max(acc, sqrt(pntSq));
    }

    return acc;
}

} // namespace Nektar::Operators::detail
