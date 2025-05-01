///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSYCLSumFacKernelLaunchers.hpp
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

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, typename TData>
NEK_DEVICE_INLINE void PhysDeriv1DKernel(const unsigned int nelmt,
                                         const TData *__restrict__ D0,
                                         const TData *__restrict__ df,
                                         const TData *__restrict__ in,
                                         TData *__restrict__ out,
                                         const sycl::nd_item<1> &item_ct1)
{
    PhysDeriv1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt, D0, df, in,
                                                out, item_ct1);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TData>
NEK_DEVICE_INLINE void PhysDeriv2DKernel(
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ shmemptr, const sycl::nd_item<1> &item_ct1)
{
    PhysDeriv2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out, shmemptr,
        item_ct1);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TData>
NEK_DEVICE_INLINE void PhysDeriv3DKernel(
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ shmemptr,
    const sycl::nd_item<1> &item_ct1)
{
    PhysDeriv3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out,
        shmemptr, item_ct1);
}

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int ncoord,
                                               const unsigned int nq0,
                                               const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                             PhysDeriv1DKernel<Implementation, DEFORMED>(
                                 ncoord, nq0, nelmt, D0, df, in, out, item_ct1);
                         });
    });
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int ncoord, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
#if defined(NEKTAR_DEBUG)
    PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED>(ncoord, nq0, nelmt,
                                                           D0, df, in, out);
#else
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                PhysDeriv1DKernel<Implementation, DEFORMED, ncoord, nq0>(
                    nelmt, D0, df, in, out, item_ct1);
            });
    });
#endif
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *f0,
    const TData *f1, const TData *df, const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
    const unsigned int blocksize =
        GetDeviceBlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(sizeof(TData) * shmemsize);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr = &shmem[0];
#pragma forceinline
                PhysDeriv2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                    ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out,
                    shmemptr, item_ct1);
            });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *D1,
                                               const TData *f0, const TData *f1,
                                               const TData *df, const TData *in,
                                               TData *out)
{
#if defined(NEKTAR_DEBUG)
    PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out);
#else
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
    const unsigned int blocksize =
        GetDeviceBlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(sizeof(TData) * shmemsize);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
                             TData *shmemptr = &shmem[0];
#pragma forceinline
                             PhysDeriv2DKernel<SHAPE_TYPE, Implementation,
                                               DEFORMED, ncoord, nq0, nq1>(
                                 nelmt, D0, D1, f0, f1, df, in, out, shmemptr,
                                 item_ct1);
                         });
    });
#endif
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2);
    const unsigned int blocksize =
        GetDeviceBlockSize<Implementation>(nq0 * nq1 * nq2);
    const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(sizeof(TData) * shmemsize);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridsize * blocksize, blocksize),
            [=](sycl::nd_item<1> item_ct1) {
                TData *shmemptr = &shmem[0];
#pragma forceinline
                PhysDeriv3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                    nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in,
                    out, shmemptr, item_ct1);
            });
    });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
#if defined(NEKTAR_DEBUG)
    PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out);
#else
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2);
    const unsigned int blocksize =
        GetDeviceBlockSize<Implementation>(nq0 * nq1 * nq2);
    const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(sizeof(TData) * shmemsize);

    Q.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gridsize * blocksize, blocksize),
                         [=](sycl::nd_item<1> item_ct1) {
                             TData *shmemptr = &shmem[0];
#pragma forceinline
                             PhysDeriv3DKernel<SHAPE_TYPE, Implementation,
                                               DEFORMED, nq0, nq1, nq2>(
                                 nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in,
                                 out, shmemptr, item_ct1);
                         });
    });
#endif
}

} // namespace Nektar::Operators::detail

#endif
