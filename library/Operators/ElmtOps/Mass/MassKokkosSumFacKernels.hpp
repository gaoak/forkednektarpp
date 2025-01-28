///////////////////////////////////////////////////////////////////////////////
//
// File: MassKokkosSumFacKernels.hpp
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

namespace Nektar::Operators::detail
{

using team_handle = Kokkos::TeamPolicy<>::member_type;
template <typename TData>
using ScratchMemoryView =
    Kokkos::View<TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename Implementation, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void Mass1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp,
    [[maybe_unused]] TData *__restrict__ shmemptr, const team_handle &team)
{
    unsigned int jacsize = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nq0;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nm0 * warpsize * iwarp;
            TData *wspptr      = wsp + nq0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, wspptr);
            IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nq0, basis0, w0, jacptr, wspptr, outptr,
                (TData)1.0);
            e += team.team_size() * team.league_size();
        }
    }
    else
    {
        TData *bwd = shmemptr;

        const unsigned int e = team.league_rank();

        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * e;
        TData *outptr       = out + nm0 * e;
        BwdTransSegSumFacQPKernel(nm0, nq0, basis0, inptr, bwd, team);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                             [&](const unsigned int &i) {
                                 if constexpr (DEFORMED)
                                 {
                                     bwd[i] *= jacptr[i] * w0[i];
                                 }
                                 else
                                 {
                                     bwd[i] *= jacptr[0] * w0[i];
                                 }
                             });

        team.team_barrier();

        IProductWRTBaseSegSumFacQPKernel<false, false, DEFORMED>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, team);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void Mass2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, [[maybe_unused]] const unsigned int *index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] TData *__restrict__ shmemptr, const team_handle &team)
{
    const unsigned int nqTot = nq0 * nq1;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 = wsp + nqTot * nelmt + nq1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, bwd, wsp0);
                IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr,
                    bwd, outptr, wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 =
                    wsp + nqTot * nelmt + std::max(nq1, nm0) * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, bwd, wsp0);
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0,
                    w1, jacptr, bwd, outptr, wsp0, (TData)1.0);
            }
            e += team.team_size() * team.league_size();
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset, nmode0, nmode1;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            offset = std::max(nm0 * nq1, nm1 * nq0);
            nmode0 = nm0;
            nmode1 = nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nmTot;
        }

        TData *tmp      = shmemptr;
        TData *bwd      = tmp + nmTot;
        TData *s_wsp0   = bwd + nqTot;
        TData *s_basis0 = s_wsp0 + offset;
        TData *s_basis1 = s_basis0 + nm0 * nq0;

        // Copy to shared memory.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmode0 * nq0),
            [&](const unsigned int &idx) { s_basis0[idx] = basis0[idx]; });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmode1 * nq1),
            [&](const unsigned int &idx) { s_basis1[idx] = basis1[idx]; });

        const unsigned int e = team.league_rank();

        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        // Copy to shared memory.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmTot),
            [&](const unsigned int &idx) { tmp[idx] = inptr[idx]; });

        team.team_barrier();

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                       s_basis1, tmp, bwd, s_wsp0, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            BwdTransTriSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                      s_basis0, s_basis1, tmp, bwd, s_wsp0,
                                      team);
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                             [&](const unsigned int &idx) {
                                 const unsigned int i = idx % nq0;
                                 const unsigned int j = idx / nq0;
                                 if constexpr (DEFORMED)
                                 {
                                     bwd[idx] *= jacptr[idx] * w0[i] * w1[j];
                                 }
                                 else
                                 {
                                     bwd[idx] *= jacptr[0] * w0[i] * w1[j];
                                 }
                             });

        team.team_barrier();

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                outptr, s_wsp0, (TData)1.0, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, outptr, s_wsp0, (TData)1.0, team);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void Mass3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    [[maybe_unused]] const unsigned int *index2,
    [[maybe_unused]] const unsigned int *index3,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] TData *__restrict__ shmemptr, const team_handle &team)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (nqTot + nq1 * nq2) * nelmt + nq2 * warpsize * iwarp;
                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, bwd,
                                        wsp0, wsp1);
                IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (nqTot + nq1 * nq2 + std::max(nq2, nm0)) * nelmt +
                              nm2 * warpsize * iwarp;
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                    wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 = wsp +
                              (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (nqTot + std::max(nq1 * nq2, nm0 * nm1) +
                               std::max(nq2, nm0)) *
                                  nelmt +
                              nm1 * warpsize * iwarp;
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, bwd, wsp0, wsp1);
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                    wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 = wsp +
                              (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
                IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, bwd, outptr, wsp0, wsp1,
                    (TData)1.0);
            }
            e += team.team_size() * team.league_size();
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset0, offset1, nmode0, nmode1, nmode2;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
            offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            offset0 = nm0 * nm1 * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            offset0 = nm0 * nm1 * nq2;
            offset1 = nm0 * nq1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }

        TData *tmp      = shmemptr;
        TData *bwd      = tmp + nmTot;
        TData *s_wsp0   = bwd + nqTot;
        TData *s_wsp1   = s_wsp0 + offset0;
        TData *s_basis0 = s_wsp1 + offset1;
        TData *s_basis1 = s_basis0 + nmode0 * nq0;
        TData *s_basis2 = s_basis1 + nmode1 * nq1;

        // Copy to shared memory.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmode0 * nq0),
            [&](const unsigned int &idx) { s_basis0[idx] = basis0[idx]; });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmode1 * nq1),
            [&](const unsigned int &idx) { s_basis1[idx] = basis1[idx]; });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmode2 * nq2),
            [&](const unsigned int &idx) { s_basis2[idx] = basis2[idx]; });

        const unsigned int e = team.league_rank();

        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        // Copy to shared memory.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nmTot),
            [&](const unsigned int &idx) { tmp[idx] = inptr[idx]; });

        team.team_barrier();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                      s_basis0, s_basis1, s_basis2, tmp, bwd,
                                      s_wsp0, s_wsp1, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            BwdTransTetSumFacQPKernel(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
                s_basis0, s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            BwdTransPrismSumFacQPKernel(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                      isModified, s_basis0, s_basis1, s_basis2,
                                      tmp, bwd, s_wsp0, s_wsp1, team);
        }

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nqTot), [&](const unsigned int &idx) {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);
                if constexpr (DEFORMED)
                {
                    bwd[idx] *= jacptr[idx] * w0[i] * w1[j] * w2[k];
                }
                else
                {
                    bwd[idx] *= jacptr[0] * w0[i] * w1[j] * w2[k];
                }
            });

        team.team_barrier();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, team);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacQPKernel<false, false, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, bwd, outptr, s_wsp1,
                s_wsp0, (TData)1.0, team);
        }
    }
}

// Kernel launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void Mass1DKernel(const unsigned int nm0,
                                          const unsigned int nq0,
                                          const unsigned int nelmt,
                                          const TData *basis0, const TData *w0,
                                          const TData *jac, TData *wsp,
                                          const TData *in, TData *out)
{
    constexpr unsigned int slevel = 0u;
    const unsigned int nshared = MassSharedMemorySize<Implementation>(nq0, nm0);
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nshared);
    const unsigned int blocksize = GetKokkosBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel), nshared);
            Mass1DKernel<Implementation, DEFORMED>(nm0, nq0, nelmt, basis0, w0,
                                                   jac, in, out, wsp,
                                                   shmem.data(), team);
        });
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void Mass1DKernel(const unsigned int nelmt,
                                          const TData *basis0, const TData *w0,
                                          const TData *jac, TData *wsp,
                                          const TData *in, TData *out)
{
    Mass1DKernel<ExecSpace, Implementation, DEFORMED>(nm0, nq0, nelmt, basis0,
                                                      w0, jac, wsp, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *w0, const TData *w1, const TData *jac, TData *wsp,
    const TData *in, TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    constexpr unsigned int slevel = 0u;
    const unsigned int nshared =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nm0, nm1);
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nshared);
    const unsigned int blocksize = GetKokkosBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel), nshared);
            Mass2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
                basis1, w0, w1, jac, in, out, wsp, shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void Mass2DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const TData *basis0, const TData *basis1, const TData *w0, const TData *w1,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    Mass2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1, w0, w1,
        jac, wsp, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

    constexpr unsigned int slevel = 0u;
    const unsigned int nshared =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2, nm0,
                                                         nm1, nm2);
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nshared);
    const unsigned int blocksize = GetKokkosBlockSize<Implementation>(nmTot);
    const unsigned int gridsize  = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel), nshared);
            Mass3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
                index1, index2, index3, basis0, basis1, basis2, w0, w1, w2, jac,
                in, out, wsp, shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void Mass3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2,
    const unsigned int *index3, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, TData *wsp, const TData *in, TData *out)
{
    Mass3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        index3, basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out);
}

} // namespace Nektar::Operators::detail

#endif
