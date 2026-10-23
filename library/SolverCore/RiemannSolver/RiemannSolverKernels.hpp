///////////////////////////////////////////////////////////////////////////////
//
// File: RiemannSolverKernels.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::SolverCore::detail
{

template <template <typename> typename RiemannKernel, typename ExecSpace,
          typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const size_t blksize, const unsigned int velComps,
    const unsigned int fluxComps, const TData *velbase, const TData *normbase,
    const TData *fwdbase, const TData *bwdbase, TData *fluxbase,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // Explicit vectorisation for AVX backend,
    // vec_t = tinysimd::simd<TData> for AVX,
    // vec_t = TData otherwise.
    // using vec_t = typename data_type_if<
    //    std::is_same_v<ExecSpace, NektarSpaces::AVX>, TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = blksize / vec_width;

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            RiemannKernel<ExecSpace>()(
                blksize, velComps, fluxComps, velbase + i * vec_width,
                normbase + i * vec_width, fwdbase + i * vec_width,
                bwdbase + i * vec_width, fluxbase + i * vec_width);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail