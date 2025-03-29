///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSYCLSumFacKernelLaunchers.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

namespace Nektar::Operators::detail
{

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *dbasis0, const TData *w0,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr = &shmem[0];
#pragma forceinline
                IProductWRTDerivBase1DKernel<Implementation, DEFORMED>(
                    ncoord, nm0, nq0, nelmt, dbasis0, w0, df, jac, in, out, wsp,
                    shmemptr, item_ct1);
            });
    });
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *dbasis0,
    const TData *w0, const TData *df, const TData *jac, const TData *in,
    TData *out, TData *wsp)
{
    IProductWRTDerivBase1DKernel<ExecSpace, Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, dbasis0, w0, df, jac, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *D0, const TData *D1, const TData *w0,
    const TData *w1, const TData *f0, const TData *f1, const TData *df,
    const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr = &shmem[0];
#pragma forceinline
                IProductWRTDerivBase2DKernel<SHAPE_TYPE, Implementation,
                                             DEFORMED>(
                    ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified,
                    index0, basis0, basis1, D0, D1, w0, w1, f0, f1, df, jac, in,
                    out, wsp, shmemptr, item_ct1);
            });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *D0, const TData *D1, const TData *w0, const TData *w1,
    const TData *f0, const TData *f1, const TData *df, const TData *jac,
    const TData *in, TData *out, TData *wsp)
{
    IProductWRTDerivBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                 DEFORMED>(
        ncoord, nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        D0, D1, w0, w1, f0, f1, df, jac, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *D0, const TData *D1,
    const TData *D2, const TData *w0, const TData *w1, const TData *w2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr = &shmem[0];
#pragma forceinline
                IProductWRTDerivBase3DKernel<SHAPE_TYPE, Implementation,
                                             DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified,
                    index0, index1, index2, basis0, basis1, basis2, D0, D1, D2,
                    w0, w1, w2, f0, f1, f1m, f2, df, jac, in, out, wsp,
                    shmemptr, item_ct1);
            });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *D0, const TData *D1,
    const TData *D2, const TData *w0, const TData *w1, const TData *w2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    IProductWRTDerivBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                 DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, df,
        jac, in, out, wsp);
}

} // namespace Nektar::Operators::detail

#endif
