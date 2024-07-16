///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondCUDASumFacKernels.cuh
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

#include "Operators/Spaces.hpp"

#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict__ offsetPtr,
    const BoundaryConditionType *__restrict__ bctypePtr,
    const int *__restrict__ ncoeffPtr, const int *__restrict__ mapPtr,
    const TData *__restrict__ inPtr, TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypePtr[i] == eNeumann || bctypePtr[i] == eRobin)
        {
            unsigned int offset = offsetPtr[i];
            unsigned int ncoeff = ncoeffPtr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outPtr[mapPtr[offset + j]] += inPtr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict__ offsetPtr,
    const BoundaryConditionType *__restrict__ bctypePtr,
    const int *__restrict__ ncoeffPtr, const TData *__restrict__ signPtr,
    const int *__restrict__ mapPtr, const TData *__restrict__ inPtr,
    TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypePtr[i] == eNeumann || bctypePtr[i] == eRobin)
        {
            unsigned int offset = offsetPtr[i];
            unsigned int ncoeff = ncoeffPtr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outPtr[mapPtr[offset + j]] +=
                    signPtr[offset + j] * inPtr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    NeuBndCondKernel(const size_t nsize, const int *offsetPtr,
                     const BoundaryConditionType *bctypePtr,
                     const int *ncoeffPtr, const int *mapPtr,
                     const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    NeuBndCondKernel<TData><<<gridSize, blockSize>>>(
        nsize, offsetPtr, bctypePtr, ncoeffPtr, mapPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    NeuBndCondKernel(const size_t nsize, const int *offsetPtr,
                     const BoundaryConditionType *bctypePtr,
                     const int *ncoeffPtr, const TData *signPtr,
                     const int *mapPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    NeuBndCondKernel<TData><<<gridSize, blockSize>>>(
        nsize, offsetPtr, bctypePtr, ncoeffPtr, signPtr, mapPtr, inPtr, outPtr);
}

#endif

} // namespace Nektar::Operators::detail
