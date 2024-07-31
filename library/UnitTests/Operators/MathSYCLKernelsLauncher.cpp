///////////////////////////////////////////////////////////////////////////////
//
// File: MathSYCLKernelsLauncher.cpp
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

#include "Operators/MathKernels/MathSYCLKernels.hpp"

#include "MathKernelsLauncher.hpp"

using namespace Nektar;
using ExecSpace = NektarSpaces::SYCL;

void negKernelLauncher(const size_t n, const double *x, double *y)
{
    negKernel<ExecSpace, double>(n, x, y);
}

void addKernelLauncher(const size_t n, const double *x, const double *y,
                       double *z)
{
    addKernel<ExecSpace, double>(n, x, y, z);
}

void subKernelLauncher(const size_t n, const double *x, const double *y,
                       double *z)
{
    subKernel<ExecSpace, double>(n, x, y, z);
}

void daxpyKernelLauncher(const size_t n, const double alpha, const double *x,
                         const double *y, double *z)
{
    daxpyKernel<ExecSpace, double>(n, alpha, x, y, z);
}

void divKernelLauncher(const size_t n, const double *x, const double *y,
                       double *z)
{
    divKernel<ExecSpace, double>(n, x, y, z);
}

void sumKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    reduceSumKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void maxKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    reduceMaxKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void minKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    reduceMinKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void innerproductKernelLauncher(const size_t n, const double *x,
                                const double *y, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    ddotKernel<ExecSpace, double>(n, x, y, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void l1normKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    l1normKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void l2normKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    l2normKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void lpnormKernelLauncher(const size_t n, const int p, const double *x,
                          double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    lpnormKernel<ExecSpace, double>(n, p, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}

void linfnormKernelLauncher(const size_t n, const double *x, double *h_out)
{
    double *d_out = sycl::malloc_device<double>(1, SYCLQueue::GetInstance());
    linfnormKernel<ExecSpace, double>(n, x, d_out);
    SYCLQueue::GetInstance().memcpy(h_out, d_out, sizeof(double)).wait();
    sycl::free(d_out, SYCLQueue::GetInstance());
}
