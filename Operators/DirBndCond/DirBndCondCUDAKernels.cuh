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

    while (i < nsize)
    {
        if (bctypeptr[i] == eDirichlet)
        {
            size_t offset = offsetptr[i];
            size_t ncoeff = ncoeffptr[i];
            for (size_t j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] = inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
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

    while (i < nsize)
    {
        if (bctypeptr[i] == eDirichlet)
        {
            size_t offset = offsetptr[i];
            size_t ncoeff = ncoeffptr[i];
            for (size_t j = 0; j < ncoeff; j++)
            {
                outptr[mapptr[offset + j]] =
                    signptr[offset + j] * inptr[offset + j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void LocalDirBndCondKernel(const size_t nsize, const int *id0ptr,
                                      const int *id1ptr, const TData *signptr,
                                      TData *outptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outptr[id0ptr[i]] = outptr[id1ptr[i]] * signptr[i];
        i += blockDim.x * gridDim.x;
    }
}

} // namespace Nektar::Operators::detail
