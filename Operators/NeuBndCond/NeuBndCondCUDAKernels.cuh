#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void NeuBndCondKernel(const size_t nsize, const int *offsetptr,
                                 const BoundaryConditionType *bctypeptr,
                                 const int *ncoeffptr, const int *mapptr,
                                 const TData *inptr, TData *outptr)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nsize)
    {
        return;
    }

    if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
    {
        size_t offset = offsetptr[i];
        size_t ncoeff = ncoeffptr[i];
        for (size_t j = 0; j < ncoeff; j++)
        {
            // atomicAdd(outptr + mapptr[offset + j], inptr[offset + j]);
            outptr[mapptr[offset + j]] += inptr[offset + j];
        }
    }
}

template <typename TData>
__global__ void NeuBndCondKernel(const size_t nsize, const int *offsetptr,
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

    if (bctypeptr[i] == eNeumann || bctypeptr[i] == eRobin)
    {
        size_t offset = offsetptr[i];
        size_t ncoeff = ncoeffptr[i];
        for (size_t j = 0; j < ncoeff; j++)
        {
            // atomicAdd(outptr + mapptr[offset + j], signptr[offset + j] *
            // inptr[offset + j]);
            outptr[mapptr[offset + j]] +=
                signptr[offset + j] * inptr[offset + j];
        }
    }
}

} // namespace Nektar::Operators::detail
