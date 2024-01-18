namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void AssembleKernel(const size_t ncoeff, const size_t nelmt,
                               const size_t offset, const int *assmbptr,
                               const TData *signptr, const TData *inptr,
                               TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        atomicAdd(outptr + assmbptr[index + i],
                  signptr[index + i] * inptr[index + i]);
    }
}

template <typename TData>
__global__ void AssembleKernel(const size_t ncoeff, const size_t nelmt,
                               const size_t offset, const int *assmbptr,
                               const TData sign, const TData *inptr,
                               TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        atomicAdd(outptr + assmbptr[index + i], sign * inptr[index + i]);
    }
}

template <typename TData>
__global__ void AssembleKernel(const size_t ncoeff, const size_t nelmt,
                               const size_t offset, const int *assmbptr,
                               const TData *inptr, TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        atomicAdd(outptr + assmbptr[index + i], inptr[index + i]);
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const size_t ncoeff, const size_t nelmt,
                                    const size_t offset, const int *assmbptr,
                                    const TData *signptr, const TData *inptr,
                                    TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        outptr[index + i] = signptr[index + i] * inptr[assmbptr[index + i]];
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const size_t ncoeff, const size_t nelmt,
                                    const size_t offset, const int *assmbptr,
                                    const TData sign, const TData *inptr,
                                    TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        outptr[index + i] = sign * inptr[assmbptr[index + i]];
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const size_t ncoeff, const size_t nelmt,
                                    const size_t offset, const int *assmbptr,
                                    const TData *inptr, TData *outptr)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t index = offset + e * ncoeff;

    for (size_t i = 0; i < ncoeff; i++)
    {
        outptr[index + i] = inptr[assmbptr[index + i]];
    }
}

} // namespace Nektar::Operators::detail
