///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondCUDASumFacKernels.cuh
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

#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData>
__global__ void DirBndCondKernel(
    const unsigned int nsize, const int *__restrict__ offsetptr,
    const BoundaryConditionType *__restrict__ bctypeptr,
    const int *__restrict__ ncoeffptr, const int *__restrict__ mapptr,
    const TData *__restrict__ inptr, TData *__restrict__ outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eDirichlet)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] = inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void DirBndCondKernel(
    const unsigned int nsize, const int *__restrict__ offsetptr,
    const BoundaryConditionType *__restrict__ bctypeptr,
    const int *__restrict__ ncoeffptr, const TData *__restrict__ signptr,
    const int *__restrict__ mapptr, const TData *__restrict__ inptr,
    TData *__restrict__ outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eDirichlet)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] =
                    signptr[offset + j] * inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void LocalDirBndCondKernel(const unsigned int nsize,
                                      const int *__restrict__ id0ptr,
                                      const int *__restrict__ id1ptr,
                                      const TData *__restrict__ signptr,
                                      TData *__restrict__ outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outptr[id0ptr[i]] = outptr[id1ptr[i]] * signptr[i];
        i += blockDim.x * gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DirBndCondKernel([[maybe_unused]] const size_t gridSize,
                             [[maybe_unused]] const size_t blockSize,
                             const unsigned int nsize, const int *offsetptr,
                             const BoundaryConditionType *bctypeptr,
                             const int *ncoeffptr, const int *mapptr,
                             const TData *inptr, TData *outptr)
{
    DirBndCondKernel<TData><<<gridSize, blockSize>>>(
        nsize, offsetptr, bctypeptr, ncoeffptr, mapptr, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DirBndCondKernel([[maybe_unused]] const size_t gridSize,
                             [[maybe_unused]] const size_t blockSize,
                             const unsigned int nsize, const int *offsetptr,
                             const BoundaryConditionType *bctypeptr,
                             const int *ncoeffptr, const TData *signptr,
                             const int *mapptr, const TData *inptr,
                             TData *outptr)
{
    DirBndCondKernel<TData><<<gridSize, blockSize>>>(
        nsize, offsetptr, bctypeptr, ncoeffptr, signptr, mapptr, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    LocalDirBndCondKernel([[maybe_unused]] const size_t gridSize,
                                  [[maybe_unused]] const size_t blockSize,
                                  const unsigned int nsize, const int *id0ptr,
                                  const int *id1ptr, const TData *signptr,
                                  TData *outptr)
{
    LocalDirBndCondKernel<TData>
        <<<gridSize, blockSize>>>(nsize, id0ptr, id1ptr, signptr, outptr);
}

#endif

} // namespace Nektar::Operators::detail
