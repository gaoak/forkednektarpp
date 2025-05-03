///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionHIPCUDASumFacKernelLaunchers.hpp
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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__))

namespace Nektar::Operators::detail
{

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TData>
__global__ void LinAdvDiffReaction1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction1DKernel<Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, basis0, D0, w0, df, jac, coeff, advVel0, in,
        out, wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nq0, typename TData>
__global__ void LinAdvDiffReaction1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ D0,
    const TData *__restrict__ w0, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ advVel0, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction1DKernel<Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, basis0, D0, w0, df, jac, coeff, advVel0, in,
        out, wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void LinAdvDiffReaction2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ advVel1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, advVel0,
        advVel1, in, out, wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TData>
__global__ void LinAdvDiffReaction2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ advVel1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, advVel0,
        advVel1, in, out, wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
__global__ void LinAdvDiffReaction3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ advVel2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, advVel0, advVel1, advVel2, in, out,
        wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TData>
__global__ void LinAdvDiffReaction3DKernelLauncher(
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ advVel2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    LinAdvDiffReaction3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, advVel0, advVel1, advVel2, in, out,
        wsp, lambda, (TData *)shmemptr, threadBlock);
}

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *basis0, const TData *D0,
    const TData *w0, const TData *df, const TData *jac, const TData *coeff,
    const TData *advVel0, const TData *in, TData *out, TData *wsp,
    const TData lambda = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction1DKernelLauncher<Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            ncoord, nm0, nq0, nelmt, basis0, D0, w0, df, jac, coeff, advVel0,
            in, out, wsp, lambda, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction1DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *basis0,
    const TData *D0, const TData *w0, const TData *df, const TData *jac,
    const TData *coeff, const TData *advVel0, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction1DKernelLauncher<Implementation, DEFORMED, nm0, nq0>
        <<<gridsize, blocksize, shmemsize>>>(ncoord, nelmt, basis0, D0, w0, df,
                                             jac, coeff, advVel0, in, out, wsp,
                                             lambda, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *D0, const TData *D1, const TData *w0,
    const TData *w1, const TData *f0, const TData *f1, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *advVel0,
    const TData *advVel1, const TData *in, TData *out, TData *wsp,
    const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0,
            basis0, basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff,
            advVel0, advVel1, in, out, wsp, lambda, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction2DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *D0, const TData *D1, const TData *w0, const TData *w1,
    const TData *f0, const TData *f1, const TData *nodToMod, const TData *df,
    const TData *jac, const TData *coeff, const TData *advVel0,
    const TData *advVel1, const TData *in, TData *out, TData *wsp,
    const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nm0, nm1);
    const unsigned int blocksize =
        GetDeviceBlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction2DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                       nm0, nm1, nmTot, nq0, nq1>
        <<<gridsize, blocksize, shmemsize>>>(
            ncoord, nelmt, isModified, index0, basis0, basis1, D0, D1, w0, w1,
            f0, f1, nodToMod, df, jac, coeff, advVel0, advVel1, in, out, wsp,
            lambda, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *advVel0,
    const TData *advVel1, const TData *advVel2, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>
        <<<gridsize, blocksize, shmemsize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
            index1, index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1,
            w2, f0, f1, f1m, f2, nodToMod, df, jac, coeff, advVel0, advVel1,
            advVel2, in, out, wsp, lambda, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void LinAdvDiffReaction3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *D0, const TData *D1, const TData *D2,
    const TData *w0, const TData *w1, const TData *w2, const TData *f0,
    const TData *f1, const TData *f1m, const TData *f2, const TData *nodToMod,
    const TData *df, const TData *jac, const TData *coeff, const TData *advVel0,
    const TData *advVel1, const TData *advVel2, const TData *in, TData *out,
    TData *wsp, const TData lambda = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        sizeof(TData) * HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(
                            nq0, nq1, nq2, nm0, nm1, nm2);
    const unsigned int blocksize = GetDeviceBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetDeviceGridSize<Implementation>(nelmt);

    GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

    LinAdvDiffReaction3DKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED,
                                       nm0, nm1, nm2, nmTot, nq0, nq1, nq2>
        <<<gridsize, blocksize, shmemsize>>>(
            nelmt, isModified, index0, index1, index2, index3, basis0, basis1,
            basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, nodToMod, df, jac,
            coeff, advVel0, advVel1, advVel2, in, out, wsp, lambda,
            hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar::Operators::detail

#endif
