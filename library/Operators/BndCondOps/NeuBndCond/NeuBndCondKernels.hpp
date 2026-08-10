///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondKernels.hpp
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

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
void NeuBndCondKernel(const size_t bndExpSize, const size_t *mapPtr,
                      const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, bndExpSize, NEKTAR_LAMBDA(const size_t i) {
            Nektar::atomic_add<ExecSpace, NektarSpaces::GlobalScope>(
                outptr + mapPtr[i], inptr[i]);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void NeuBndCondKernel(const size_t bndExpSize, const TData *signPtr,
                      const size_t *mapPtr, const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, bndExpSize, NEKTAR_LAMBDA(const size_t i) {
            Nektar::atomic_add<ExecSpace, NektarSpaces::GlobalScope>(
                outptr + mapPtr[i], signPtr[i] * inptr[i]);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::Operators::detail
