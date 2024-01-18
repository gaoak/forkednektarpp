namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void DiffusionCoeff1DKernel(const size_t nsize,
                                       const TData *diffCoeff, TData *deriv0)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    deriv0[i] *= diffCoeff[0];
}

template <typename TData>
__global__ void DiffusionCoeff2DKernel(const size_t nsize,
                                       const TData *diffCoeff, TData *deriv0,
                                       TData *deriv1)
{
    __shared__ TData s_diffCoeff[4];

    size_t ind = threadIdx.x;
    if (ind < 4)
    {
        s_diffCoeff[ind] = diffCoeff[ind];
    }

    __syncthreads();

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    TData deriv[2] = {deriv0[i], deriv1[i]};

    deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1];
    deriv1[i] = s_diffCoeff[2] * deriv[0] + s_diffCoeff[3] * deriv[1];
}

template <typename TData>
__global__ void DiffusionCoeff3DKernel(const size_t nsize, TData *diffCoeff,
                                       TData *deriv0, TData *deriv1,
                                       TData *deriv2)
{
    __shared__ TData s_diffCoeff[9];

    size_t ind = threadIdx.x;
    if (ind < 9)
    {
        s_diffCoeff[ind] = diffCoeff[ind];
    }

    __syncthreads();

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    TData deriv[3] = {deriv0[i], deriv1[i], deriv2[i]};

    deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1] +
                s_diffCoeff[2] * deriv[2];
    deriv1[i] = s_diffCoeff[3] * deriv[0] + s_diffCoeff[4] * deriv[1] +
                s_diffCoeff[5] * deriv[2];
    deriv2[i] = s_diffCoeff[6] * deriv[0] + s_diffCoeff[7] * deriv[1] +
                s_diffCoeff[8] * deriv[2];
}

} // namespace Nektar::Operators::detail
