///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
void SetModeBlkKernel(const unsigned mode, const size_t nelmtgrps,
                      const unsigned width, const unsigned numdata,
                      const TData val, TData *blkptr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmtgrps * width, NEKTAR_LAMBDA(size_t idx) {
            unsigned e = idx / width;
            unsigned i = idx % width;

            blkptr[(e * numdata + mode) * width + i] = val;
        });
}

template <typename ExecSpace, typename TData>
void CopyModeBlkKernel(const unsigned mode, const size_t nelmtgrps,
                       const unsigned width, const unsigned numdata,
                       const TData *fromblkptr, TData *toblkptr)
{
    Nektar::parallel_for<ExecSpace>(
        0, nelmtgrps * width, NEKTAR_LAMBDA(size_t idx) {
            unsigned e = idx / width;
            unsigned i = idx % width;

            toblkptr[(e * numdata + mode) * width + i] =
                fromblkptr[(e * numdata + mode) * width + i];
        });
}

} // namespace Nektar::Operators::detail
