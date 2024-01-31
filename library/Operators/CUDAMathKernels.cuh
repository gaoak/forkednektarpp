namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void negKernel(const size_t nsize, const TData *x, TData *y)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        y[i] = -x[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void addKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void subKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] - y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void daxpyKernel(const size_t nsize, const TData alpha,
                            const TData *x, const TData *y, TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = alpha * x[i] + y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void vdivKernel(const size_t nsize, const TData *x, const TData *y,
                           TData *z)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        z[i] = x[i] / y[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void dotKernel(const size_t nsize, const TData *x, const TData *y,
                          TData *out)
{
    extern __shared__ TData s[];

    size_t i      = blockDim.x * blockIdx.x + threadIdx.x;
    size_t sIndex = threadIdx.x;

    TData tmp = 0.0;
    while (i < nsize)
    {
        tmp += x[i] * y[i];
        i += blockDim.x * gridDim.x;
    }

    s[sIndex] = tmp;

    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0)
    {
        if (sIndex < i)
        {
            s[sIndex] += s[sIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, s[0]);
    }
}

} // namespace Nektar::Operators::detail
