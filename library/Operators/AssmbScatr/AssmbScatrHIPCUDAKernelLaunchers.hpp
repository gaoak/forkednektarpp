///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrHIPCUDAKernelLaunchers.hpp
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

template <typename TData>
__global__ void AssembleScatrKernelLauncher(
    const unsigned int nvals, const unsigned *__restrict__ blkGSInfo,
    const int *__restrict__ sign, TData *__restrict__ inoutptr,
    const hipcudaBlock1D &threadBlock)

{
    AssembleScatrKernel<>(nvals, blkGSInfo, sign, inoutptr, threadBlock);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    AssembleScatrKernel(const unsigned int nvals, const unsigned *GSInfo,
                        const int *sign, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    AssembleScatrKernelLauncher<><<<gridSize, blockSize>>>(
        nvals, GSInfo, sign, inoutptr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename TData>
__global__ void AssembleScatrKernelLauncher(
    const unsigned nvals, const unsigned *__restrict__ nassemble,
    const unsigned *__restrict__ index, const unsigned *__restrict__ offset,
    const int *__restrict__ sign, TData *__restrict__ inoutptr,
    const hipcudaBlock1D &threadBlock)

{
    AssembleScatrKernel(nvals, nassemble, index, offset, sign, inoutptr,
                        threadBlock);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *nassemble,
                        const unsigned *index, const unsigned *offset,
                        const int *sign, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    AssembleScatrKernelLauncher<<<gridSize, blockSize>>>(
        nvals, nassemble, index, offset, sign, inoutptr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename TData>
__global__ void AssembleScatrBndKernelLauncher(
    const unsigned nvals, const unsigned *__restrict__ GSInfo,
    const int *__restrict__ sign, TData *__restrict__ inoutptr,
    TData *__restrict__ bndptr, const hipcudaBlock1D &threadBlock)
{
    AssembleScatrBndKernel<>(nvals, GSInfo, sign, inoutptr, bndptr,
                             threadBlock);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    AssembleScatrBndKernel(const unsigned nvals, const unsigned *GSInfo,
                           const int *sign, TData *inoutptr, TData *bndptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    AssembleScatrBndKernelLauncher<><<<gridSize, blockSize>>>(
        nvals, GSInfo, sign, inoutptr, bndptr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename TData>
__global__ void AssembleFromBndKernelLauncher(
    const unsigned nvals, const unsigned *__restrict__ GSInfo,
    const int *__restrict__ sign, const TData *__restrict__ bndptr,
    TData *__restrict__ inoutptr, const hipcudaBlock1D &threadBlock)
{
    AssembleFromBndKernel<>(nvals, GSInfo, sign, bndptr, inoutptr, threadBlock);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    AssembleFromBndKernel(const unsigned nvals, const unsigned *GSInfo,
                          const int *sign, const TData *bndptr, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    AssembleFromBndKernelLauncher<><<<gridSize, blockSize>>>(
        nvals, GSInfo, sign, bndptr, inoutptr, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar::Operators::detail

#endif
