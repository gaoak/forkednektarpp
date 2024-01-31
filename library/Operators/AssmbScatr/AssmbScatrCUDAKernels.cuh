#pragma once

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void AssembleKernel(const unsigned int ncoeff,
                               const unsigned int nelmt,
                               const unsigned int offset,
                               const int *__restrict assmbptr,
                               const TData *__restrict signptr,
                               const TData *__restrict inptr,
                               TData *__restrict outptr)
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
                               const int *__restrict assmbptr, const TData sign,
                               const TData *__restrict inptr,
                               TData *__restrict outptr)
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
                               const int *__restrict assmbptr,
                               const TData *__restrict inptr,
                               TData *__restrict outptr)
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
                                    const int *__restrict assmbptr,
                                    const TData *__restrict signptr,
                                    const TData *__restrict inptr,
                                    TData *__restrict outptr)
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
__global__ void GlobalToLocalKernel(
    const unsigned int ncoeff, const unsigned int nelmt,
    const unsigned int offset, const int *__restrict assmbptr, const TData sign,
    const TData *__restrict inptr, TData *__restrict outptr)
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
                                    const int *__restrict assmbptr,
                                    const TData *__restrict inptr,
                                    TData *__restrict outptr)
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

} // namespace Nektar::Operators::detail
