namespace Nektar::Operators::detail
{
template <typename TData>
__global__ void DiffusionCoeff1DKernel(const size_t nq0, const size_t nelmt,
                                       const TData *diffCoeff, TData *deriv0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Assign pointers.
    TData *derivptr = deriv0 + nq0 * e;

    for (size_t i = 0; i < nq0; i++)
    {
        derivptr[i] *= diffCoeff[0];
    }
}

template <typename TData>
__global__ void DiffusionCoeff2DKernel(const size_t nq0, const size_t nq1,
                                       const size_t nelmt,
                                       const TData *diffCoeff, TData *deriv0,
                                       TData *deriv1)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    constexpr size_t ncoord = 2;

    // Assign pointers.
    TData **derivptr = new TData *[ncoord];
    derivptr[0]      = deriv0 + nq0 * nq1 * e;
    derivptr[1]      = deriv1 + nq0 * nq1 * e;

    for (size_t j = 0, cnt = 0; j < nq1; j++)
    {
        for (size_t i = 0; i < nq0; i++)
        {
            TData deriv[2] = {derivptr[0][cnt], derivptr[1][cnt]};
            for (size_t d = 0; d < ncoord; d++)
            {
                derivptr[d][cnt] = diffCoeff[d * ncoord + 0] * deriv[0] +
                                   diffCoeff[d * ncoord + 1] * deriv[1];
            }
            cnt++;
        }
    }
}

template <typename TData>
__global__ void DiffusionCoeff3DKernel(const size_t nq0, const size_t nq1,
                                       const size_t nq2, const size_t nelmt,
                                       TData *diffCoeff, TData *deriv0,
                                       TData *deriv1, TData *deriv2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    constexpr size_t ncoord = 3;

    // Assign pointers.
    TData **derivptr = new TData *[ncoord];
    derivptr[0]      = deriv0 + nq0 * nq1 * nq2 * e;
    derivptr[1]      = deriv1 + nq0 * nq1 * nq2 * e;
    derivptr[2]      = deriv2 + nq0 * nq1 * nq2 * e;

    for (size_t k = 0, cnt = 0; k < nq2; k++)
    {
        for (size_t j = 0; j < nq1; j++)
        {
            for (size_t i = 0; i < nq0; i++)
            {
                TData deriv[3] = {derivptr[0][cnt], derivptr[1][cnt],
                                  derivptr[2][cnt]};
                for (size_t d = 0; d < ncoord; d++)
                {
                    derivptr[d][cnt] = diffCoeff[d * ncoord + 0] * deriv[0] +
                                       diffCoeff[d * ncoord + 1] * deriv[1] +
                                       diffCoeff[d * ncoord + 2] * deriv[2];
                }
                cnt++;
            }
        }
    }
}
} // namespace Nektar::Operators::detail
