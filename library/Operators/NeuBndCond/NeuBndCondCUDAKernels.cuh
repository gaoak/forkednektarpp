#pragma once

#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict offsetptr,
    const BoundaryConditionType *__restrict bctypeptr,
    const int *__restrict ncoeffptr, const int *__restrict mapptr,
    const TData *__restrict inptr, TData *__restrict outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] += inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void NeuBndCondKernel(
    const unsigned int nsize, const int *__restrict offsetptr,
    const BoundaryConditionType *__restrict bctypeptr,
    const int *__restrict ncoeffptr, const TData *__restrict signptr,
    const int *__restrict mapptr, const TData *__restrict inptr,
    TData *__restrict outptr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
        {
            unsigned int offset = offsetptr[i];
            unsigned int ncoeff = ncoeffptr[i];
            for (unsigned int j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] +=
                    signptr[offset + j] * inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

} // namespace Nektar::Operators::detail
