#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void DirBndCondKernel(const size_t nsize, const int *offsetptr,
                                 const BoundaryConditionType *bctypeptr,
                                 const int *ncoeffptr, const int *mapptr,
                                 const TData *inptr, TData *outptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    if (bctypeptr[i] == eDirichlet)
    {
        int offset = offsetptr[i];
        for (size_t j = 0; j < ncoeffptr[i]; j++)
        {
            outptr[mapptr[offset + j]] = inptr[offset + j];
        }
    }
}

template <typename TData>
__global__ void DirBndCondKernel(const size_t nsize, const int *offsetptr,
                                 const BoundaryConditionType *bctypeptr,
                                 const int *ncoeffptr, const TData *signptr,
                                 const int *mapptr, const TData *inptr,
                                 TData *outptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    if (bctypeptr[i] == eDirichlet)
    {
        int offset = offsetptr[i];
        for (size_t j = 0; j < ncoeffptr[i]; j++)
        {
            outptr[mapptr[offset + j]] =
                signptr[offset + j] * inptr[offset + j];
        }
    }
}

template <typename TData>
__global__ void LocalDirBndCondKernel(const size_t nsize, const int *id0ptr,
                                      const int *id1ptr, const TData *signptr,
                                      TData *outptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    outptr[id0ptr[i]] = outptr[id1ptr[i]] * signptr[i];
}

} // namespace Nektar::Operators::detail
