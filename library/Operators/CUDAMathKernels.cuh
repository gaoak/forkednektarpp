///////////////////////////////////////////////////////////////////////////////
//
// File: CUDAMathKernels.cuh
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

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void negKernel(const size_t nsize, const TData *x, TData *y)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        y[i] = -x[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void addKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void subKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] - y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void daxpyKernel(const size_t nsize, const TData alpha,
                            const TData *x, const TData *y, TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = alpha * x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void vdivKernel(const size_t nsize, const TData *x, const TData *y,
                           TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] / y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void dotKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *out)
{
    extern __shared__ TData s[];

    size_t i      = blockDim.x * blockIdx.x + threadIdx.x;
    size_t sIndex = threadIdx.x;

    TData tmp = 0.0;
    while (i < nsize)
    {
        tmp += x[i] * y[i];
        i += blockDim.x * gridDim.x;
    }

    s[sIndex] = tmp;

    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0)
    {
        if (sIndex < i)
        {
            s[sIndex] += s[sIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, s[0]);
    }
}

} // namespace Nektar::Operators::detail
