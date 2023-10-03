namespace Nektar::Operators::detail
{
template <typename TData>
__global__ void IdentityKernel(const size_t numPts, const size_t nelmt,
                               const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const TData *inptr = in + e * numPts;
    TData *outptr      = out + e * numPts;

    for (size_t i = 0; i < numPts; ++i)
    {
        outptr[i] = inptr[i];
    }
}

} // namespace Nektar::Operators::detail
