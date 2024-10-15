///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondCUDAKernels.cuh
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <bool negflag, typename TData>
__global__ void RobBndCond1DKernel(const unsigned int nsize,
                                   const unsigned int *__restrict__ offsetPtr,
                                   const TData *__restrict__ matPtr,
                                   const unsigned int *__restrict__ mapPtr,
                                   const TData *__restrict__ incoeffPtr,
                                   TData *__restrict__ coeffPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        const unsigned int offset = offsetPtr[i];
        const unsigned int map    = mapPtr[i];

        if constexpr (negflag)
        {
            atomic_sub<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
                coeffPtr + offset + map, matPtr[i] * incoeffPtr[offset + map]);
        }
        else
        {
            atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
                coeffPtr + offset + map, matPtr[i] * incoeffPtr[offset + map]);
        }

        i += blockDim.x * gridDim.x;
    }
}

template <bool negflag, typename TData>
__global__ void RobBndCond2DKernel(
    const unsigned int nsize, const unsigned int *__restrict__ ncoeffPtr,
    const unsigned int *__restrict__ offsetPtr,
    const unsigned int *__restrict__ matOffsetPtr,
    const unsigned int *__restrict__ mapOffsetPtr,
    const TData *__restrict__ matPtr, const unsigned int *__restrict__ mapPtr,
    const int *__restrict__ signPtr, const TData *__restrict__ incoeffPtr,
    TData *__restrict__ coeffPtr)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];
    TData *vEdgeCoeffs = (TData *)shared;
    unsigned int j     = blockIdx.x;

    while (j < nsize)
    {
        const unsigned int ncoeff    = ncoeffPtr[j];
        const unsigned int offset    = offsetPtr[j];
        const unsigned int matOffset = matOffsetPtr[j];
        const unsigned int mapOffset = mapOffsetPtr[j];

        for (unsigned int i = threadIdx.x; i < ncoeff; i += blockDim.x)
        {
            const unsigned int index = mapOffset + i;
            vEdgeCoeffs[i] =
                incoeffPtr[offset + mapPtr[index]] * signPtr[index];
        }

        __syncthreads();

        for (unsigned int i = threadIdx.x; i < ncoeff; i += blockDim.x)
        {
            TData tmp = 0.0;
            for (unsigned int k = 0; k < ncoeff; k++)
            {
                tmp += matPtr[matOffset + ncoeff * k + i] * vEdgeCoeffs[k];
            }

            const unsigned int index = mapOffset + i;
            if constexpr (negflag)
            {
                atomic_sub<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
                    coeffPtr + offset + mapPtr[index], tmp * signPtr[index]);
            }
            else
            {
                atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
                    coeffPtr + offset + mapPtr[index], tmp * signPtr[index]);
            }
        }

        __syncthreads();

        j += gridDim.x;
    }
}

template <typename ExecSpace, bool negflag, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    RobBndCond1DKernel(const unsigned int nsize, const unsigned int *offsetPtr,
                       const TData *matPtr, const unsigned int *mapPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    RobBndCond1DKernel<negflag><<<gridSize, blockSize>>>(
        nsize, offsetPtr, matPtr, mapPtr, incoeffPtr, coeffPtr);
}

template <typename ExecSpace, bool negflag, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    RobBndCond2DKernel(const unsigned int nmaxcoeff, const unsigned int nsize,
                       const unsigned int *ncoeffPtr,
                       const unsigned int *offsetPtr,
                       const unsigned int *matOffsetPtr,
                       const unsigned int *mapOffsetPtr, const TData *matPtr,
                       const unsigned int *mapPtr, const int *signPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;
    const unsigned int gridSize  = nsize;

    RobBndCond2DKernel<negflag>
        <<<gridSize, blockSize, sizeof(TData) * nmaxcoeff>>>(
            nsize, ncoeffPtr, offsetPtr, matOffsetPtr, mapOffsetPtr, matPtr,
            mapPtr, signPtr, incoeffPtr, coeffPtr);
}

} // namespace Nektar::Operators::detail

#endif
