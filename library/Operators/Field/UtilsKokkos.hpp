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
interleave(const unsigned int numMetaBlocks, const unsigned int npts,
           TData *inout)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer = (TData *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock = team.league_rank();
            const unsigned int offset    = npts * VectorWidth * metaBlock;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * VectorWidth),
                [&](const unsigned int &idx) {
                    buffer[offset + idx] = inout[offset + idx];
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * VectorWidth),
                [&](const unsigned int &idx) {
                    unsigned int vecElem = idx % VectorWidth;
                    unsigned int iElem   = idx / VectorWidth;
                    inout[offset + idx] =
                        buffer[offset + vecElem * npts + iElem];
                });
        });

    Kokkos::kokkos_free(buffer);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int npts, TData *inout)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer = (TData *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock = team.league_rank();
            const unsigned int offset    = npts * VectorWidth * metaBlock;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * VectorWidth),
                [&](const unsigned int &idx) {
                    buffer[offset + idx] = inout[offset + idx];
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * VectorWidth),
                [&](const unsigned int &idx) {
                    unsigned int vecElem = idx / npts;
                    unsigned int iElem   = idx % npts;
                    inout[offset + idx] =
                        buffer[offset + iElem * VectorWidth + vecElem];
                });
        });

    Kokkos::kokkos_free(buffer);
}

template <typename ExecSpace>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
BuildInterleaveMapKernel(const unsigned int numMetaBlocks,
                         const unsigned int npts,
                         const unsigned int newVecWidth,
                         const unsigned int offset, int *deInterleaveMapPtr,
                         int *interleaveMapPtr)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * npts;
    // allocate buffer for all teams
    int *buffer = (int *)
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
            bufferSize);

    // rewrite UtilsAVX.hpp BuildInterleaveMapKernel for Kokkos:
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(numMetaBlocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int metaBlock  = team.league_rank();
            const unsigned int teamOffset = newVecWidth * npts * metaBlock;
            // assign count+0, count+1, count+2, count+3, count+4, ....
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * newVecWidth),
                [&](const unsigned int &idx) {
                    buffer[teamOffset + idx] = offset + teamOffset + idx;
                });

            team.team_barrier();

            // get the deinterleave map
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts),
                [&](const unsigned int &idx) {
                    unsigned int vecElem = idx % newVecWidth;
                    unsigned int iElem   = idx / newVecWidth;
                    deInterleaveMapPtr[teamOffset + idx] =
                        buffer[teamOffset + vecElem * npts + iElem];
                });

            team.team_barrier();

            // get the interleave map
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, npts * newVecWidth),
                [&](const unsigned int &idx) {
                    interleaveMapPtr[deInterleaveMapPtr[teamOffset + idx]] =
                        offset + teamOffset + idx;
                });
        });

    Kokkos::kokkos_free(buffer);
}

} // namespace Nektar

#endif
