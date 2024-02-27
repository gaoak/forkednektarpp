///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondCUDAKernels.cuh
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

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict offsetptr,
    const BoundaryConditionType *__restrict bctypeptr,
    const int *__restrict ncoeffptr, const int *__restrict mapptr,
    const TData *__restrict inptr, TData *__restrict outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] += inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict offsetptr,
    const BoundaryConditionType *__restrict bctypeptr,
    const int *__restrict ncoeffptr, const TData *__restrict signptr,
    const int *__restrict mapptr, const TData *__restrict inptr,
    TData *__restrict outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] +=
                    signptr[offset + j] * inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

} // namespace Nektar::Operators::detail
