namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void SetDiagonalKernel(const size_t nm, const size_t nelmt,
                                  const size_t mode, const TData val,
                                  TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        out[e * nm + mode] = val;

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void CopyDiagonalKernel(const size_t nm, const size_t nelmt,
                                   const size_t mode, const TData *in,
                                   TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        out[e * nm + mode] = in[e * nm + mode];

        e += blockDim.x * gridDim.x;
    }
}

} // namespace Nektar::Operators::detail
