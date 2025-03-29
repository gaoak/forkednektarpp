///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseCUDASumFacKernelLaunchers.hpp
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

namespace Nektar::Operators::detail
{

// Non-size based version.
template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
__global__ void IProductWRTBase1DKernelLauncher(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale, const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase1DKernel<Implementation, SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, (TData *)shmemptr,
        threadBlock);
}

// Size based template version.
template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
__global__ void IProductWRTBase1DKernelLauncher(
    const unsigned int nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ w0, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase1DKernel<Implementation, SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, (TData *)shmemptr,
        threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TData>
__global__ void IProductWRTBase2DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData scale,
    const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase2DKernel<SHAPE_TYPE, Implementation, SCALE, APPEND,
                            DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, jac, in, out, wsp, scale, (TData *)shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nmTot, unsigned int nq0,
          unsigned int nq1, typename TData>
__global__ void IProductWRTBase2DKernelLauncher(
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale, const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase2DKernel<SHAPE_TYPE, Implementation, SCALE, APPEND,
                            DEFORMED>(
        nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        w0, w1, jac, in, out, wsp, scale, (TData *)shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TData>
__global__ void IProductWRTBase3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale, const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase3DKernel<SHAPE_TYPE, Implementation, SCALE, APPEND,
                            DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, basis0, basis1, basis2, w0, w1, w2, jac, in, out, wsp, scale,
        (TData *)shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, signed int nm0,
          unsigned int nm1, unsigned int nm2, unsigned nmTot, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
__global__ void IProductWRTBase3DKernelLauncher(
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData scale, const cudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    IProductWRTBase3DKernel<SHAPE_TYPE, Implementation, SCALE, APPEND,
                            DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, basis0, basis1, basis2, w0, w1, w2, jac, in, out, wsp, scale,
        (TData *)shmemptr, threadBlock);
}

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *basis0, const TData *w0, const TData *jac, const TData *in,
    TData *out, const TData scale = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase1DKernelLauncher<Implementation, SCALE, APPEND, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(nm0, nq0, nelmt, basis0, w0, jac,
                                             in, out, scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nelmt, const TData *basis0, const TData *w0,
    const TData *jac, const TData *in, TData *out, const TData scale = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase1DKernelLauncher<Implementation, SCALE, APPEND, DEFORMED,
                                    nm0, nq0>
        <<<gridsize, blocksize, shmemsize>>>(nelmt, basis0, w0, jac, in, out,
                                             scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *w0, const TData *w1, const TData *jac, const TData *in,
    TData *out, TData *wsp, const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                    nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation, SCALE, APPEND,
                                    DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
            basis1, w0, w1, jac, in, out, wsp, scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nq0,
          unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const TData *basis0, const TData *basis1, const TData *w0, const TData *w1,
    const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                    nm0, nm1);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase2DKernelLauncher<SHAPE_TYPE, Implementation, SCALE, APPEND,
                                    DEFORMED, nm0, nm1, nmTot, nq0, nq1>
        <<<gridsize, blocksize, shmemsize>>>(nelmt, isModified, index0, basis0,
                                             basis1, w0, w1, jac, in, out, wsp,
                                             scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation, SCALE, APPEND,
                                    DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
            index1, index2, basis0, basis1, basis2, w0, w1, w2, jac, in, out,
            wsp, scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nq0, unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out, TData *wsp,
    const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) *
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetCUDABlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetCUDAGridSize<Implementation>(nelmt);

    IProductWRTBase3DKernelLauncher<SHAPE_TYPE, Implementation, SCALE, APPEND,
                                    DEFORMED, nm0, nm1, nm2, nmTot, nq0, nq1,
                                    nq2><<<gridsize, blocksize, shmemsize>>>(
        nelmt, isModified, index0, index1, index2, basis0, basis1, basis2, w0,
        w1, w2, jac, in, out, wsp, scale, cudaBlock1D());
    CHECK_LAST_CUDA_ERROR();
}

} // namespace Nektar::Operators::detail

#endif
