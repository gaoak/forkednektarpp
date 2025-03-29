///////////////////////////////////////////////////////////////////////////////
//
// File: MassSYCLSumFacKernelLaunchers.hpp
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
NEK_FORCE_INLINE static void Mass1DKernel(const unsigned int nm0,
                                          const unsigned int nq0,
                                          const unsigned int nelmt,
                                          const TData *basis0, const TData *w0,
                                          const TData *jac, TData *wsp,
                                          const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        MassSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
                             TData *shmemptr = &shmem[0];
#pragma forceinline
                             Mass1DKernel<Implementation, DEFORMED>(
                                 nm0, nq0, nelmt, basis0, w0, jac, in, out, wsp,
                                 shmemptr, item_ct1);
                         });
    });
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void Mass1DKernel(const unsigned int nelmt,
                                          const TData *basis0, const TData *w0,
                                          const TData *jac, TData *wsp,
                                          const TData *in, TData *out)
{
    Mass1DKernel<ExecSpace, Implementation, DEFORMED>(nm0, nq0, nelmt, basis0,
                                                      w0, jac, wsp, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *w0, const TData *w1, const TData *jac, TData *wsp,
    const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
                             TData *shmemptr = &shmem[0];
#pragma forceinline
                             Mass2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                                 nm0, nm1, nmTot, nq0, nq1, nelmt, isModified,
                                 index0, basis0, basis1, w0, w1, jac, in, out,
                                 wsp, shmemptr, item_ct1);
                         });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void Mass2DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const TData *basis0, const TData *basis1, const TData *w0, const TData *w1,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    Mass2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1, w0, w1,
        jac, wsp, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2, nm0,
                                                         nm1, nm2);
    const unsigned int blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
                             TData *shmemptr = &shmem[0];
#pragma forceinline
                             Mass3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                                 nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                 isModified, index0, index1, index2, index3,
                                 basis0, basis1, basis2, w0, w1, w2, jac, in,
                                 out, wsp, shmemptr, item_ct1);
                         });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void Mass3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    Mass3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        index3, basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out);
}

} // namespace Nektar::Operators::detail

#endif
