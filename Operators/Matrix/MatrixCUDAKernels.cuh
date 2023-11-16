namespace Nektar::Operators::detail
{
template <typename TData>
__global__ void MatrixKernel(const size_t numPts, const size_t nelmt,
                             const size_t size, const TData *mat,
                             const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const TData *matrix = mat + e * numPts;
    const TData *inptr  = in + e * numPts;
    TData *outptr       = out + e * numPts;

    for (size_t j = 0; j < size * size; j += size)
    {
        for (size_t i = 0; i < numPts; ++i)
        {
            outptr[i] += inptr[i] * matrix[j + i];
        }
    }
}

} // namespace Nektar::Operators::detail
