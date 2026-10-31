///////////////////////////////////////////////////////////////////////////////
//
// File: ReynoldsStressKernels.hpp
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
// Description: Device-portable velocity-product kernel.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Add scale * u_a * u_b to product component (a,b), a <= b, at every
 * entry of one block.
 *
 * The velocity and the products share the block's storage layout, so one
 * storage index addresses the same point in both, whatever the interleave
 * width: the kernel walks the storage as it is laid out, neighbouring threads
 * reading and writing neighbouring entries, with no index arithmetic. The
 * padding of the velocity is zero, so the padded entries of the products
 * stay zero. The products are ordered (0,0), (0,1), ..., (0,d-1), (1,1), ...,
 * (d-1,d-1).
 */
template <unsigned int DIMENSION, typename ExecSpace, typename TData>
void ReynoldsStressKernel(const size_t compSize,
                          const unsigned int velocityComponent0,
                          const unsigned int velocityComponent1,
                          const unsigned int velocityComponent2,
                          const TData scale, const TData *velocity,
                          TData *products, const unsigned int streamID)
{
    static_assert(DIMENSION == 2 || DIMENSION == 3,
                  "ReynoldsStress supports only dimensions 2 and 3.");

    // Offsets of the velocity components. The third is unused in 2D.
    const size_t uOffset = velocityComponent0 * compSize;
    const size_t vOffset = velocityComponent1 * compSize;
    const size_t wOffset = velocityComponent2 * compSize;

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, compSize, NEKTAR_LAMBDA(const size_t idx) {
            const TData u[3] = {
                velocity[uOffset + idx], velocity[vOffset + idx],
                DIMENSION == 3 ? velocity[wOffset + idx] : TData{0}};

            unsigned int product = 0;
            for (unsigned int a = 0; a < DIMENSION; ++a)
            {
                for (unsigned int b = a; b < DIMENSION; ++b, ++product)
                {
                    products[product * compSize + idx] += scale * u[a] * u[b];
                }
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
