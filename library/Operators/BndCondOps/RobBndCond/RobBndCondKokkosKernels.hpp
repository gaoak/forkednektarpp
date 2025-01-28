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

#include "Operators/LoopExecution/LoopExecution.hpp"

using team_handle = Kokkos::TeamPolicy<>::member_type;
template <typename TData>
using ScratchMemoryView =
    Kokkos::View<TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

namespace Nektar::Operators::detail
{

template <bool negflag, typename TData>
KOKKOS_INLINE_FUNCTION static void RobBndCond1DKernel(
    const unsigned int *__restrict__ offsetPtr,
    const TData *__restrict__ matPtr, const unsigned int *__restrict__ mapPtr,
    const TData *__restrict__ incoeffPtr, TData *__restrict__ coeffPtr,
    const unsigned int i)
{

    const unsigned int offset = offsetPtr[i];
    const unsigned int map    = mapPtr[i];

    TData *const ptr = coeffPtr + offset + map;
    const TData val  = matPtr[i] * incoeffPtr[offset + map];
    if constexpr (negflag)
    {
        Nektar::atomic_sub<NektarSpaces::KOKKOS, NektarSpaces::GlobalScope>(
            ptr, val);
    }
    else
    {
        Nektar::atomic_add<NektarSpaces::KOKKOS, NektarSpaces::GlobalScope>(
            ptr, val);
    }
}

template <bool negflag, typename TData>
KOKKOS_INLINE_FUNCTION static void RobBndCond2DKernel(
    const unsigned int *__restrict__ ncoeffPtr,
    const unsigned int *__restrict__ offsetPtr,
    const unsigned int *__restrict__ matOffsetPtr,
    const unsigned int *__restrict__ mapOffsetPtr,
    const TData *__restrict__ matPtr, const unsigned int *__restrict__ mapPtr,
    const int *__restrict__ signPtr, const TData *__restrict__ incoeffPtr,
    TData *__restrict__ coeffPtr, TData *__restrict__ shmemptr,
    const team_handle &team)
{
    TData *vEdgeCoeffs = shmemptr;

    const unsigned int j         = team.league_rank();
    const unsigned int ncoeff    = ncoeffPtr[j];
    const unsigned int offset    = offsetPtr[j];
    const unsigned int matOffset = matOffsetPtr[j];
    const unsigned int mapOffset = mapOffsetPtr[j];

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, ncoeff), [&](const unsigned int i) {
            const unsigned int index = mapOffset + i;
            vEdgeCoeffs[i] =
                incoeffPtr[offset + mapPtr[index]] * signPtr[index];
        });

    team.team_barrier();

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, ncoeff), [&](const unsigned int i) {
            TData tmp = 0.0;
            for (unsigned int k = 0; k < ncoeff; k++)
            {
                tmp += matPtr[matOffset + ncoeff * k + i] * vEdgeCoeffs[k];
            }

            const unsigned int index = mapOffset + i;
            TData *const ptr         = coeffPtr + offset + mapPtr[index];
            const TData val          = tmp * signPtr[index];
            if constexpr (negflag)
            {
                Nektar::atomic_sub<NektarSpaces::KOKKOS,
                                   NektarSpaces::GlobalScope>(ptr, val);
            }
            else
            {
                Nektar::atomic_add<NektarSpaces::KOKKOS,
                                   NektarSpaces::GlobalScope>(ptr, val);
            }
        });

    team.team_barrier();
}

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
            RobBndCond1DKernel<negflag>(offsetPtr, matPtr, mapPtr, incoeffPtr,
                                        coeffPtr, i);
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
    constexpr unsigned int slevel = 0u;
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nmaxcoeff);

    const unsigned int blockSize = NektarSpaces::vector_width<TData>::value;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nsize, blockSize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel),
                                           nmaxcoeff);
            RobBndCond2DKernel<negflag>(
                ncoeffPtr, offsetPtr, matOffsetPtr, mapOffsetPtr, matPtr,
                mapPtr, signPtr, incoeffPtr, coeffPtr, shmem.data(), team);
        });
}

} // namespace Nektar::Operators::detail

#endif
