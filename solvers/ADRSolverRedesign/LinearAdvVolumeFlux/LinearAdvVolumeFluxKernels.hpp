///////////////////////////////////////////////////////////////////////////////
//
// File: LinearAdvVolumeFluxKernels.hpp
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
// Description: Volume flux kernels for linear advection
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::detail
{
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void LinearAdvVolumeFluxKernel(
    const unsigned int npts, const unsigned int velComps,
    const unsigned int nvarComps, const unsigned int velStride,
    const unsigned int inStride, const unsigned int outStride,
    const TData *velbase, const TData *inbase, TData *outbase,
    const unsigned int streamID)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize    = npts / vec_width;
    const size_t velVecStride = velStride / vec_width;
    const size_t inVecStride  = inStride / vec_width;
    const size_t outVecStride = outStride / vec_width;
    const auto velvec         = reinterpret_cast<const vec_t *>(velbase);
    const auto invec          = reinterpret_cast<const vec_t *>(inbase);
    auto outvec               = reinterpret_cast<vec_t *>(outbase);

    Nektar::LoopExecutionSetStreamID(streamID);

    // Parallelize over point groups; each i is independent.
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned int nvar = 0; nvar < nvarComps; ++nvar)
            {
                const size_t in_off = nvar * inVecStride + i;
                const vec_t in_val  = invec[in_off];

                for (unsigned int ndim = 0; ndim < velComps; ++ndim)
                {
                    const size_t vel_off = ndim * velVecStride + i;
                    const size_t out_off =
                        (nvar * velComps + ndim) * outVecStride + i;

                    outvec[out_off] = in_val * velvec[vel_off];
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}
} // namespace Nektar::detail
