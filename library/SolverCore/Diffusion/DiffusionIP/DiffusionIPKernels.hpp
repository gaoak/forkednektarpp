///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionIPKernels.hpp
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

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyBwdDerivTraceFromFwdOnBndKernel(
    const size_t nBndPts, const unsigned int nComps,
    const unsigned int compStride, const size_t *bndTraceOffset,
    const TData *fwdbase, TData *bwdbase, const unsigned streamID)
{
    // Derivative traces do not have physical boundary data of their own. On
    // physical boundaries legacy DiffusionIP uses dq^- = dq^+ before the IP
    Nektar::LoopExecutionSetStreamID(streamID);

    // penalty term is added, while periodic traces are handled separately.
    Nektar::parallel_for<ExecSpace>(
        0u, nBndPts, NEKTAR_LAMBDA(const size_t i) {
            const size_t offset = bndTraceOffset[i];
            for (unsigned int c = 0; c < nComps; ++c)
            {
                bwdbase[c * compStride + offset] =
                    fwdbase[c * compStride + offset];
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::SolverCore::detail
