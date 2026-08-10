///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
void DirBndCondKernel(const size_t nsize, const size_t *mapPtr,
                      const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize,
        NEKTAR_LAMBDA(const size_t i) { outptr[mapPtr[i]] = inptr[i]; });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void DirBndCondKernel(const size_t nsize, const TData *signPtr,
                      const size_t *mapPtr, const TData *inptr, TData *outptr,
                      const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            outptr[mapPtr[i]] = signPtr[i] * inptr[i];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void ParallelDirBndSignKernel(const size_t nsize, const int *signPtr,
                              TData *outptr, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) { outptr[signPtr[i]] *= -1; });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
void LocalDirBndCondKernel(const size_t nsize, const size_t *id0Ptr,
                           const size_t *id1Ptr, const TData *signPtr,
                           const TData *inptr, TData *outptr,
                           const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    // Note: inptr and outptr might alias each other.
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            outptr[id0Ptr[i]] = inptr[id1Ptr[i]] * signPtr[i];
        });

    Nektar::LoopExecutionSetStreamID(0);

    // Record event for synchronization.
#if defined(NEKTAR_ENABLE_CUDA)
    CUDAStream::RecordEvent(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    HIPStream::RecordEvent(streamID);
#endif
}

} // namespace Nektar::Operators::detail
