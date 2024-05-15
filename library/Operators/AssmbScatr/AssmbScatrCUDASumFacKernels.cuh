///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrCUDASumFacKernels.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

template <typename TData>
__global__ void AssembleKernel(const unsigned int ncoeff,
                               const unsigned int nelmt,
                               const unsigned int offset,
                               const int *__restrict__ assmbptr,
                               const TData *__restrict__ signptr,
                               const TData *__restrict__ inptr,
                               TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            atomicAdd(outptr + assmbptr[index + i],
                      signptr[index + i] * inptr[index + i]);
        }
        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int ncoeff,
                               const unsigned int nelmt,
                               const unsigned int offset,
                               const int *__restrict__ assmbptr,
                               const TData sign,
                               const TData *__restrict__ inptr,
                               TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            atomicAdd(outptr + assmbptr[index + i], sign * inptr[index + i]);
        }
        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int ncoeff,
                               const unsigned int nelmt,
                               const unsigned int offset,
                               const int *__restrict__ assmbptr,
                               const TData *__restrict__ inptr,
                               TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            atomicAdd(outptr + assmbptr[index + i], inptr[index + i]);
        }
        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int ncoeff,
                                    const unsigned int nelmt,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbptr,
                                    const TData *__restrict__ signptr,
                                    const TData *__restrict__ inptr,
                                    TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            outptr[index + i] = signptr[index + i] * inptr[assmbptr[index + i]];
        }
        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int ncoeff,
                                    const unsigned int nelmt,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbptr,
                                    const TData sign,
                                    const TData *__restrict__ inptr,
                                    TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            outptr[index + i] = sign * inptr[assmbptr[index + i]];
        }
        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int ncoeff,
                                    const unsigned int nelmt,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbptr,
                                    const TData *__restrict__ inptr,
                                    TData *__restrict__ outptr)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int index = offset + e * ncoeff;

        for (unsigned int i = 0; i < ncoeff; i++)
        {
            outptr[index + i] = inptr[assmbptr[index + i]];
        }
        e += blockDim.x * gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel([[maybe_unused]] const size_t gridSize,
                           [[maybe_unused]] const size_t blockSize,
                           const unsigned int ncoeff, const unsigned int nelmt,
                           const unsigned int offset, const int *assmbptr,
                           const TData *signptr, const TData *inptr,
                           TData *outptr)
{
    AssembleKernel<TData><<<gridSize, blockSize>>>(
        ncoeff, nelmt, offset, assmbptr, signptr, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel([[maybe_unused]] const size_t gridSize,
                           [[maybe_unused]] const size_t blockSize,
                           const unsigned int ncoeff, const unsigned int nelmt,
                           const unsigned int offset, const int *assmbptr,
                           const TData sign, const TData *inptr, TData *outptr)
{
    AssembleKernel<TData><<<gridSize, blockSize>>>(
        ncoeff, nelmt, offset, assmbptr, sign, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel([[maybe_unused]] const size_t gridSize,
                           [[maybe_unused]] const size_t blockSize,
                           const unsigned int ncoeff, const unsigned int nelmt,
                           const unsigned int offset, const int *assmbptr,
                           const TData *inptr, TData *outptr)
{
    AssembleKernel<TData><<<gridSize, blockSize>>>(ncoeff, nelmt, offset,
                                                   assmbptr, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                                [[maybe_unused]] const size_t blockSize,
                                const unsigned int ncoeff,
                                const unsigned int nelmt,
                                const unsigned int offset, const int *assmbptr,
                                const TData *signptr, const TData *inptr,
                                TData *outptr)
{
    GlobalToLocalKernel<TData><<<gridSize, blockSize>>>(
        ncoeff, nelmt, offset, assmbptr, signptr, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::CUDA>::value, void>::type
GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                            [[maybe_unused]] const size_t blockSize,
                            const unsigned int ncoeff, const unsigned int nelmt,
                            const unsigned int offset, const int *assmbptr,
                            const TData sign, const TData *inptr, TData *outptr)
{
    GlobalToLocalKernel<TData><<<gridSize, blockSize>>>(
        ncoeff, nelmt, offset, assmbptr, sign, inptr, outptr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel([[maybe_unused]] const size_t gridSize,
                                [[maybe_unused]] const size_t blockSize,
                                const unsigned int ncoeff,
                                const unsigned int nelmt,
                                const unsigned int offset, const int *assmbptr,
                                const TData *inptr, TData *outptr)
{
    GlobalToLocalKernel<TData><<<gridSize, blockSize>>>(
        ncoeff, nelmt, offset, assmbptr, inptr, outptr);
}

#endif

} // namespace Nektar::Operators::detail
