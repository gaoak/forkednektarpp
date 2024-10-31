///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondKokkosKernels.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(NEKTAR_ENABLE_KOKKOS)

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::KOKKOS>,
                               void>::type
RobBndCond1DKernel(const unsigned int nsize, const unsigned int *offsetPtr,
                   const TData *matPtr, const unsigned int *mapPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0u, nsize, Kokkos::ChunkSize(blockSize)),
        KOKKOS_LAMBDA(const unsigned int i) {
            const unsigned int offset = offsetPtr[i];
            const unsigned int map    = mapPtr[i];
            if (negflag)
            {
                Kokkos::atomic_sub(coeffPtr + offset + map,
                                   matPtr[i] * incoeffPtr[offset + map]);
            }
            else
            {
                Kokkos::atomic_add(coeffPtr + offset + map,
                                   matPtr[i] * incoeffPtr[offset + map]);
            }
        });
}

template <typename ExecSpace, bool negflag, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::KOKKOS>,
                               void>::type
RobBndCond2DKernel(const unsigned int nmaxcoeff, const unsigned int nsize,
                   const unsigned int *ncoeffPtr, const unsigned int *offsetPtr,
                   const unsigned int *matOffsetPtr,
                   const unsigned int *mapOffsetPtr, const TData *matPtr,
                   const unsigned int *mapPtr, const int *signPtr,
                   const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(nmaxcoeff);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nsize, blockSize)
            .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                vEdgeCoeffs(team.team_scratch(0), nmaxcoeff);
            const unsigned int j         = team.league_rank();
            const unsigned int ncoeff    = ncoeffPtr[j];
            const unsigned int offset    = offsetPtr[j];
            const unsigned int matOffset = matOffsetPtr[j];
            const unsigned int mapOffset = mapOffsetPtr[j];

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ncoeff),
                                 [&](const unsigned int i) {
                                     const unsigned int index = mapOffset + i;
                                     vEdgeCoeffs(i) =
                                         incoeffPtr[offset + mapPtr[index]] *
                                         signPtr[index];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, ncoeff),
                [&](const unsigned int i) {
                    TData tmp = 0.0;
                    for (unsigned int k = 0; k < ncoeff; k++)
                    {
                        tmp +=
                            matPtr[matOffset + ncoeff * k + i] * vEdgeCoeffs(k);
                    }

                    const unsigned int index = mapOffset + i;
                    if (negflag)
                    {
                        Kokkos::atomic_sub(coeffPtr + offset + mapPtr[index],
                                           tmp * signPtr[index]);
                    }
                    else
                    {
                        Kokkos::atomic_add(coeffPtr + offset + mapPtr[index],
                                           tmp * signPtr[index]);
                    }
                });

            team.team_barrier();
        });
}

} // namespace Nektar::Operators::detail

#endif
