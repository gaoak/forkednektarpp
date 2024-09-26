///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsKokkos.hpp
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

namespace Nektar
{

template <size_t VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int dataLen,
           TData *inout)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * dataLen;

    TData *buffer = (TData *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock = team.league_rank();
            const unsigned int offset    = dataLen * VectorWidth * metaBlock;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, dataLen),
                [&](const unsigned int &idx) {
                    for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
                    {
                        buffer[offset + vecElem * dataLen + idx] =
                            inout[offset + vecElem * dataLen + idx];
                    }
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, dataLen),
                [&](const unsigned int &idx) {
                    for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
                    {
                        inout[offset + idx * VectorWidth + vecElem] =
                            buffer[offset + vecElem * dataLen + idx];
                    }
                });

            team.team_barrier();
        });

    Kokkos::kokkos_free(buffer);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int dataLen, TData *inout)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * dataLen;

    TData *buffer = (TData *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock = team.league_rank();
            const unsigned int offset    = dataLen * VectorWidth * metaBlock;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, dataLen),
                [&](const unsigned int &idx) {
                    for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
                    {
                        buffer[offset + vecElem * dataLen + idx] =
                            inout[offset + vecElem * dataLen + idx];
                    }
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, dataLen),
                [&](const unsigned int &idx) {
                    for (size_t vecElem = 0; vecElem < VectorWidth; ++vecElem)
                    {
                        inout[offset + vecElem * dataLen + idx] =
                            buffer[offset + idx * VectorWidth + vecElem];
                    }
                });

            team.team_barrier();
        });

    Kokkos::kokkos_free(buffer);
}

template <typename ExecSpace>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
BuildInterleaveMapKernel(const unsigned int numMetaBlocks,
                         const unsigned int ncoeff,
                         const unsigned int newVecWidth,
                         const unsigned int offset, int *deInterleaveMapPtr,
                         int *interleaveMapPtr)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * ncoeff;
    // allocate buffer for all teams
    int *buffer = (int *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    // rewrite UtilsAVX.hpp BuildInterleaveMapKernel for Kokkos:
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock  = team.league_rank();
            const unsigned int teamOffset = newVecWidth * ncoeff * metaBlock;
            // assign count+0, count+1, count+2, count+3, count+4, ....
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, ncoeff * newVecWidth),
                [&](const unsigned int &i) {
                    buffer[teamOffset + i] = offset + teamOffset + i;
                });
            team.team_barrier();
            // get the deinterleave map
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, ncoeff),
                [&](const unsigned int &n) {
                    for (unsigned int vecElem = 0; vecElem < newVecWidth;
                         ++vecElem)
                    {
                        deInterleaveMapPtr[teamOffset + n * newVecWidth +
                                           vecElem] =
                            buffer[teamOffset + vecElem * ncoeff + n];
                    }
                });
            team.team_barrier();
            // get the interleave map
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, ncoeff * newVecWidth),
                [&](const unsigned int &i) {
                    interleaveMapPtr[deInterleaveMapPtr[teamOffset + i]] =
                        offset + teamOffset + i;
                });
            team.team_barrier();
        });

    Kokkos::kokkos_free(buffer);
}

} // namespace Nektar

#endif
