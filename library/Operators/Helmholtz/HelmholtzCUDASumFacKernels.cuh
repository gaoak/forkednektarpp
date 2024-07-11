///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzCUDASumFacKernels.cuh
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

namespace Nektar::Operators::detail
{

// CUDA Kernels
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData>
__global__ void DiffusionCoeff1DKernel(const unsigned int nsize,
                                       const TData *diffCoeff, TData *deriv0)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        deriv0[i] *= diffCoeff[0];

        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void DiffusionCoeff2DKernel(const unsigned int nsize,
                                       const TData *diffCoeff, TData *deriv0,
                                       TData *deriv1)
{
    __shared__ TData s_diffCoeff[4];

    // Copy to shared memory.
    unsigned int ind = threadIdx.x;
    if (ind < 4)
    {
        s_diffCoeff[ind] = diffCoeff[ind];
    }

    __syncthreads();

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        TData deriv[2] = {deriv0[i], deriv1[i]};

        deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1];
        deriv1[i] = s_diffCoeff[2] * deriv[0] + s_diffCoeff[3] * deriv[1];

        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void DiffusionCoeff3DKernel(const unsigned int nsize,
                                       const TData *diffCoeff, TData *deriv0,
                                       TData *deriv1, TData *deriv2)
{
    __shared__ TData s_diffCoeff[9];

    // Copy to shared memory.
    unsigned int ind = threadIdx.x;
    if (ind < 9)
    {
        s_diffCoeff[ind] = diffCoeff[ind];
    }

    __syncthreads();

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        TData deriv[3] = {deriv0[i], deriv1[i], deriv2[i]};

        deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1] +
                    s_diffCoeff[2] * deriv[2];
        deriv1[i] = s_diffCoeff[3] * deriv[0] + s_diffCoeff[4] * deriv[1] +
                    s_diffCoeff[5] * deriv[2];
        deriv2[i] = s_diffCoeff[6] * deriv[0] + s_diffCoeff[7] * deriv[1] +
                    s_diffCoeff[8] * deriv[2];

        i += blockDim.x * gridDim.x;
    }
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DiffusionCoeff1DKernel(const unsigned int nsize, const TData *diffCoeff,
                           TData *deriv0)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DiffusionCoeff1DKernel<<<gridSize, blockSize>>>(nsize, diffCoeff, deriv0);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DiffusionCoeff2DKernel(const unsigned int nsize, const TData *diffCoeff,
                           TData *deriv0, TData *deriv1)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DiffusionCoeff2DKernel<<<gridSize, blockSize>>>(nsize, diffCoeff, deriv0,
                                                    deriv1);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DiffusionCoeff3DKernel(const unsigned int nsize, const TData *diffCoeff,
                           TData *deriv0, TData *deriv1, TData *deriv2)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DiffusionCoeff3DKernel<<<gridSize, blockSize>>>(nsize, diffCoeff, deriv0,
                                                    deriv1, deriv2);
}

#endif

} // namespace Nektar::Operators::detail
