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
#include <LibUtilities/BasicUtils/NekInline.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpwindSolverKernel(
    const size_t npts, const unsigned int velComps,
    const unsigned int fluxComps, const TData *velbase, const TData *normbase,
    const TData *fwdbase, const TData *bwdbase, TData *fluxbase)
{
    // Parallelize over points; each i is independent
    Nektar::parallel_for<ExecSpace>(
        0u, npts, NEKTAR_LAMBDA(const size_t i) {
            // Build nv = v·n on the fly
            TData nv = 0.0;
            for (unsigned int d = 0; d < velComps; ++d)
            {
                const TData v_i = velbase[d * npts + i];
                const TData n_i = normbase[d * npts + i];
                nv += v_i * n_i;
            }

            // Branchless split: nv_pos=max(nv,0), nv_neg=min(nv,0)
            // Use fabs to stay device-friendly.
            const TData nv_abs = std::fabs(nv);
            const TData nv_pos = 0.5 * (nv + nv_abs);
            const TData nv_neg = 0.5 * (nv - nv_abs);

            // Blend without branches:
            // flux = nv_pos * Fwd + nv_neg * Bwd
            for (unsigned int nc = 0; nc < fluxComps; ++nc)
            {
                const size_t off = nc * npts + i;
                fluxbase[off] = nv_pos * fwdbase[off] + nv_neg * bwdbase[off];
            }
        });
}

} // namespace Nektar::Operators::detail
