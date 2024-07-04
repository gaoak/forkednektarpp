///////////////////////////////////////////////////////////////////////////////
//
// File: MatrixKernels.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

// Generic kernel launchers except for CUDA.
template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
MatrixKernel([[maybe_unused]] const size_t gridSize,
             [[maybe_unused]] const size_t blockSize, const size_t nElmts,
             const size_t numPts, const size_t size, const TData *mat,
             const TData *in, TData *out)
{
    Nektar::parallel_for<ExecSpace>(
        0, nElmts, KOKKOS_LAMBDA(int e) {
            const TData *matrix = mat + e * numPts;
            const TData *inPtr  = in + e * numPts;
            TData *outPtr       = out + e * numPts;

            for (size_t j = 0; j < size * size; j += size)
            {
                for (size_t i = 0; i < numPts; ++i)
                {
                    outPtr[i] += inPtr[i] * matrix[j + i];
                }
            }
        });
}

} // namespace Nektar::Operators::detail
