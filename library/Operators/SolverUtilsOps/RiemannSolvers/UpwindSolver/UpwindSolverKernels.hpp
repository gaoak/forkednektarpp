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

// TODO: move to Common/Spaces.hpp and tidy.
template <bool B, typename TData> struct data_type_if
{
    typedef TData type;
};

template <typename TData> struct data_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

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
    Nektar::parallel_for<ExecSpace>(
        0u, blksize / vec_width, NEKTAR_LAMBDA(const size_t i) {
            // Build nv = v·n on the fly
            vec_t nv = 0.0;
            for (unsigned int d = 0; d < velComps; ++d)
            {
                const vec_t v_i =
                    reinterpret_cast<const vec_t *>(velbase + d * blksize)[i];
                const vec_t n_i =
                    reinterpret_cast<const vec_t *>(normbase + d * blksize)[i];
                nv += v_i * n_i;
            }

#if defined(_MSC_VER)
            const vec_t nv_abs = std::abs(nv);
#else
            const vec_t nv_abs = abs(nv);
#endif
            // Branchless split: nv_pos=max(nv,0), nv_neg=min(nv,0)
            // Use fabs to stay device-friendly.
            const vec_t nv_pos = 0.5 * (nv + nv_abs);
            const vec_t nv_neg = 0.5 * (nv - nv_abs);

            // Blend without branches:
            // flux = nv_pos * Fwd + nv_neg * Bwd
            for (unsigned int nc = 0; nc < fluxComps; ++nc)
            {
                reinterpret_cast<vec_t *>(fluxbase + nc * blksize)[i] =
                    nv_pos * reinterpret_cast<const vec_t *>(fwdbase +
                                                             nc * blksize)[i] +
                    nv_neg * reinterpret_cast<const vec_t *>(bwdbase +
                                                             nc * blksize)[i];
            }
        });
}

} // namespace Nektar::Operators::detail
