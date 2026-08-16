///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarIPVolFluxKernels.hpp
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
// Description: Scalar IP diffusion volume flux kernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::detail
{

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void DiffusionScalarIPVolFluxKernel(
    const size_t npts, const unsigned int ndim, const unsigned int nvarComps,
    const size_t derivStride, const size_t outStride, const TData *diffCoeff,
    const TData *derivbase, TData *outbase, const unsigned int streamID)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = npts / vec_width;
    const auto derivvec    = reinterpret_cast<const vec_t *>(derivbase);
    auto outvec            = reinterpret_cast<vec_t *>(outbase);

    const size_t derivVecStride = derivStride / vec_width;
    const size_t outVecStride   = outStride / vec_width;

    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                for (unsigned int outDir = 0; outDir < ndim; ++outDir)
                {
                    vec_t flux = vec_t(0.0);
                    for (unsigned int derivDir = 0; derivDir < ndim; ++derivDir)
                    {
                        const auto diffIdx = LibUtilities::GetDiffCoeffMap(
                            ndim, outDir * ndim + derivDir);
                        const size_t derivIdx =
                            (f * ndim + derivDir) * derivVecStride + i;
                        flux += vec_t(diffCoeff[diffIdx]) * derivvec[derivIdx];
                    }

                    outvec[(f * ndim + outDir) * outVecStride + i] = flux;
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
