///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondHIPCUDAKernelLaunchers.hpp
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

template <bool negflag, typename TData>
__global__ void RobBndCond1DKernelLauncher(const size_t nsize,
                                           const size_t *__restrict__ offsetPtr,
                                           const TData *__restrict__ matPtr,
                                           const size_t *__restrict__ mapPtr,
                                           const TData *__restrict__ incoeffPtr,
                                           TData *__restrict__ coeffPtr,
                                           const hipcudaBlock1D &threadBlock)
{
    RobBndCond1DKernel<negflag>(nsize, offsetPtr, matPtr, mapPtr, incoeffPtr,
                                coeffPtr, threadBlock);
}

template <bool negflag, typename TData>
__global__ void RobBndCond2DKernelLauncher(
    const size_t nsize, const unsigned int *__restrict__ ncoeffPtr,
    const size_t *__restrict__ offsetPtr,
    const size_t *__restrict__ matOffsetPtr,
    const size_t *__restrict__ mapOffsetPtr, const TData *__restrict__ matPtr,
    const size_t *__restrict__ mapPtr, const int *__restrict__ signPtr,
    const TData *__restrict__ incoeffPtr, TData *__restrict__ coeffPtr,
    const hipcudaBlock1D &threadBlock)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shmemptr[];

    RobBndCond2DKernel<negflag>(
        nsize, ncoeffPtr, offsetPtr, matOffsetPtr, mapOffsetPtr, matPtr, mapPtr,
        signPtr, incoeffPtr, coeffPtr, (TData *)shmemptr, threadBlock);
}

// Kernel Launchers.
template <typename ExecSpace, bool negflag, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    RobBndCond1DKernel(const size_t nsize, const size_t *offsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::Device::warpSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    RobBndCond1DKernelLauncher<negflag>
        <<<gridSize, blockSize>>>(nsize, offsetPtr, matPtr, mapPtr, incoeffPtr,
                                  coeffPtr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool negflag, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    RobBndCond2DKernel(const unsigned int nmaxcoeff, const size_t nsize,
                       const unsigned int *ncoeffPtr, const size_t *offsetPtr,
                       const size_t *matOffsetPtr, const size_t *mapOffsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const int *signPtr, const TData *incoeffPtr,
                       TData *coeffPtr)
{
    const unsigned int shmemsize = sizeof(TData) * nmaxcoeff;
    const unsigned int blockSize = NektarSpaces::Device::warpSize;
    const unsigned int gridSize  = nsize;

    RobBndCond2DKernelLauncher<negflag><<<gridSize, blockSize, shmemsize>>>(
        nsize, ncoeffPtr, offsetPtr, matOffsetPtr, mapOffsetPtr, matPtr, mapPtr,
        signPtr, incoeffPtr, coeffPtr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar::Operators::detail

#endif
