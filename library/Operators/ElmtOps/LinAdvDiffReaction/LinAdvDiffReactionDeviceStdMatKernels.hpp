///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceStdMatKernels.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceStdMatKernels.hpp"

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AddAdvectionKernels(
    const size_t nelmt, const unsigned int nhomo, const unsigned int nqTot,
    const unsigned int ncoord, const size_t adveloffset,
    const size_t derivoffset, const TData *advVel, const TData *deriv,
    TData *out, const TData scale)
{
    const auto nsize = nelmt * nqTot * nhomo;
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            size_t idx0 = idx % (nelmt * nqTot);
            TData tmp   = 0.0;
            for (unsigned int d = 0; d < ncoord; d++)
            {
                tmp += advVel[d * adveloffset + idx0] *
                       deriv[d * derivoffset + idx];
            }
            out[idx] = scale * out[idx] + tmp;
        });
}
