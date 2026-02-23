///////////////////////////////////////////////////////////////////////////////
//
// File: UpwindSolverKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TScalar>
NEK_FORCE_INLINE static void UpwindSolverKernel(
    const size_t blksize, const unsigned int velComps,
    const unsigned int fluxComps, const TScalar *velbase,
    const TScalar *normbase, const TScalar *fwdbase, const TScalar *bwdbase,
    TScalar *fluxbase)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // Parallelize over points; each i is independent
    const size_t groupsize = blksize / vec_width;
    const auto velvec      = reinterpret_cast<const vec_t *>(velbase);
    const auto normvec     = reinterpret_cast<const vec_t *>(normbase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            using std::abs;

            // Build nv = v·n on the fly
            vec_t nv = 0.0;
            for (unsigned int d = 0; d < velComps; ++d)
            {
                const vec_t v_i = velvec[d * groupsize + i];
                const vec_t n_i = normvec[d * groupsize + i];
                nv += v_i * n_i;
            }

            const vec_t nv_abs = abs(nv);

            // Branchless split: nv_pos=max(nv,0), nv_neg=min(nv,0)
            // Use fabs to stay device-friendly.
            const vec_t nv_pos = 0.5 * (nv + nv_abs);
            const vec_t nv_neg = 0.5 * (nv - nv_abs);

            // Blend without branches:
            // flux = nv_pos * Fwd + nv_neg * Bwd
            for (unsigned int nc = 0; nc < fluxComps; ++nc)
            {
                fluxvec[nc * groupsize + i] =
                    nv_pos * fwdvec[nc * groupsize + i] +
                    nv_neg * bwdvec[nc * groupsize + i];
            }
        });
}

} // namespace Nektar::Operators::detail
