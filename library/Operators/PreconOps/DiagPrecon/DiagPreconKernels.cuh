///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconKernels.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include "Operators/Common/Spaces.hpp"
#include "Operators/Field/MemoryRegion.hpp"

#include <cstddef>
#include <type_traits>

namespace Nektar::Operators::detail
{

// CUDA Kernels
template <typename TData>
__global__ void SetDiagonalKernel(const size_t nm, const size_t nelmt,
                                  const size_t mode, const size_t offset,
                                  const TData val, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        out[offset + e * nm + mode] = val;

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void CopyDiagonalKernel(const size_t nm, const size_t nelmt,
                                   const size_t mode, const size_t inoffset,
                                   const size_t outoffset, const TData *in,
                                   TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        out[outoffset + e * nm + mode] = in[inoffset + e * nm + mode];

        e += blockDim.x * gridDim.x;
    }
}

// CUDA Kernel launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    SetDiagonalKernel(const size_t nmTot, const size_t nElmts,
                      const size_t mode, const size_t offset, const TData val,
                      MemoryRegion<TData> &out)
{
    using MemSpace = typename ExecSpace::memory_space;

    unsigned int blockSize = 256u;
    unsigned int gridSize  = (nElmts + blockSize - 1u) / blockSize;

    TData *outptr = out.template GetPtr<MemSpace, WriteOnly>();
    SetDiagonalKernel<<<gridSize, blockSize>>>(nmTot, nElmts, mode, offset, val,
                                               outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    CopyDiagonalKernel(const size_t nmTot, const size_t nElmts,
                       const size_t mode, const size_t inoffset,
                       const size_t outoffset, MemoryRegion<TData> &in,
                       MemoryRegion<TData> &out)
{
    using MemSpace = typename ExecSpace::memory_space;

    unsigned int blockSize = 256u;
    unsigned int gridSize  = (nElmts + blockSize - 1u) / blockSize;

    const TData *inptr = in.template GetPtr<MemSpace, ReadOnly>();
    TData *outptr      = out.template GetPtr<MemSpace, WriteOnly>();
    CopyDiagonalKernel<<<gridSize, blockSize>>>(nmTot, nElmts, mode, inoffset,
                                                outoffset, inptr, outptr);
}

} // namespace Nektar::Operators::detail

#endif
