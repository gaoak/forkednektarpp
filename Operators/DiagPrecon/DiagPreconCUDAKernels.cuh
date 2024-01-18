namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void DiagPreconKernel(const size_t nGlobal, const size_t nDir,
                                 const TData *diagptr, TData *inoutptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nDir || i >= nGlobal)
    {
        return;
    }

    inoutptr[i] /= diagptr[i];
}

} // namespace Nektar::Operators::detail
