///////////////////////////////////////////////////////////////////////////////
//
// File: CUDAMathKernelsLauncher.cu
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

#include <cuda_runtime.h>

#include "Operators/CUDAMathKernels.cuh"

#include "CUDAMathKernelsLauncher.hpp"

using namespace Nektar::Operators;

void dotKernelLauncher(const size_t n, double *x, double *y, double *h_out)
{
    constexpr size_t gridSize  = 1024;
    constexpr size_t blockSize = 256;
    double *buffer, *d_out;
    cudaMalloc((void **)&buffer, sizeof(double) * gridSize);
    cudaMemset(buffer, 0, sizeof(double) * gridSize);
    cudaMalloc((void **)&d_out, sizeof(double));
    cudaMemset(d_out, 0, sizeof(double));
    dotKernel<blockSize><<<gridSize, blockSize>>>(n, x, y, buffer);
    reduceKernel<gridSize><<<1, gridSize>>>(gridSize, buffer, d_out);
    cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
}

void addKernelLauncher(const size_t n, double *x, double *y, double *z)
{
    constexpr size_t gridSize  = 1024;
    constexpr size_t blockSize = 256;
    addKernel<<<gridSize, blockSize>>>(n, x, y, z);
}

void subKernelLauncher(const size_t n, double *x, double *y, double *z)
{
    constexpr size_t gridSize  = 1024;
    constexpr size_t blockSize = 256;
    subKernel<<<gridSize, blockSize>>>(n, x, y, z);
}

void daxpyKernelLauncher(const size_t n, double alpha, double *x, double *y,
                         double *z)
{
    constexpr size_t gridSize  = 1024;
    constexpr size_t blockSize = 256;
    daxpyKernel<<<gridSize, blockSize>>>(n, alpha, x, y, z);
}

void vdivKernelLauncher(const size_t n, double *x, double *y, double *z)
{
    constexpr size_t gridSize  = 1024;
    constexpr size_t blockSize = 256;
    vdivKernel<<<gridSize, blockSize>>>(n, x, y, z);
}
