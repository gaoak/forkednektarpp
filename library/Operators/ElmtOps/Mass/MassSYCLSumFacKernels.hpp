///////////////////////////////////////////////////////////////////////////////
//
// File: MassSYCLSumFacKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict basis0, const TData *__restrict w0,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out, TData *__restrict wsp,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
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

        unsigned int e = item_ct1.get_global_id(2);
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
            e += item_ct1.get_global_range(2);
        }
    }
    else
    {
        TData *s_wsp0 = (TData *)shmemptr;

        unsigned int e = item_ct1.get_group(2);
        while (e < nelmt)
        {
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * e;
            TData *outptr       = out + nm0 * e;
            BwdTransSegSumFacQPKernel(nm0, nq0, basis0, inptr, s_wsp0,
                                      item_ct1);

            for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                 i += item_ct1.get_local_range(2))
            {
                const unsigned int jacindex = DEFORMED ? i : 0;
                s_wsp0[i] *= jacptr[jacindex] * w0[i];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            IProductWRTBaseSegSumFacQPKernel<false, false, DEFORMED>(
                nm0, nq0, basis0, s_wsp0, outptr, (TData)1.0, item_ct1);
            e += item_ct1.get_group_range(2);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, [[maybe_unused]] const unsigned int *index0,
    const TData *__restrict basis0, const TData *__restrict basis1,
    const TData *__restrict w0, const TData *__restrict w1,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out, [[maybe_unused]] TData *__restrict wsp,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
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

        unsigned int e = item_ct1.get_global_id(2);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 = wsp + nqTot * nelmt + nq1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, wsp0, wsp1);
                IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr,
                    wsp0, outptr, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 =
                    wsp + nqTot * nelmt + std::max(nq1, nm0) * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, wsp0, wsp1);
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0,
                    w1, jacptr, wsp0, outptr, wsp1, (TData)1.0);
            }
            e += item_ct1.get_global_range(2);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *s_wsp0   = (TData *)shmemptr;
            TData *s_wsp1   = s_wsp0 + nmTot;
            TData *s_wsp2   = s_wsp1 + nqTot;
            TData *s_basis0 = s_wsp2 + std::max(nm0 * nq1, nm1 * nq0);
            TData *s_basis1 = s_basis0 + nm0 * nq0;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransQuadSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                           s_basis1, s_wsp0, s_wsp1, s_wsp2,
                                           item_ct1);

                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = idx / nq0;
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBaseQuadSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1,
                    s_wsp1, outptr, s_wsp2, (TData)1.0, item_ct1);
                e += item_ct1.get_group_range(2);
            }
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *s_wsp0   = (TData *)shmemptr;
            TData *s_wsp1   = s_wsp0 + nmTot;
            TData *s_wsp2   = s_wsp1 + nqTot;
            TData *s_basis0 = s_wsp2 + nm0 * nq1;
            TData *s_basis1 = s_basis0 + nm0 * nq0;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransTriSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                          s_basis0, s_basis1, s_wsp0, s_wsp1,
                                          s_wsp2, item_ct1);

                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = idx / nq0;
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBaseTriSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, s_wsp1, outptr, s_wsp2, (TData)1.0,
                    item_ct1);
                e += item_ct1.get_group_range(2);
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Mass3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    [[maybe_unused]] const unsigned int *index2,
    [[maybe_unused]] const unsigned int *index3, const TData *__restrict basis0,
    const TData *__restrict basis1, const TData *__restrict basis2,
    const TData *__restrict w0, const TData *__restrict w1,
    const TData *__restrict w2, const TData *__restrict jac,
    const TData *__restrict in, TData *__restrict out,
    [[maybe_unused]] TData *__restrict wsp,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
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

        unsigned int e = item_ct1.get_global_id(2);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 =
                    wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (nqTot + nq2 * nq1) * nelmt + nq2 * warpsize * iwarp;

                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, wsp0,
                                        wsp1, wsp2);
                IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    w0, w1, w2, jacptr, wsp0, outptr, wsp1, wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 =
                    wsp + nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (nqTot + nq2 * nq1) * nelmt + nq2 * warpsize * iwarp;
                TData *prod = wsp + (nqTot + nq1 * nq2 + nq2) * nelmt +
                              nm2 * warpsize * iwarp;

                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, wsp0, wsp1, wsp2);
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, wsp0, outptr, wsp1,
                    wsp2, prod, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 = wsp + nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (nqTot + std::max(nq2 * nq1, nm0 * nm1)) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp3 = wsp +
                              (nqTot + std::max(nq1 * nq2, nm0 * nm1) +
                               std::max(nq2, nm0)) *
                                  nelmt +
                              nm1 * warpsize * iwarp;

                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, wsp0, wsp1, wsp2);
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, wsp0, outptr, wsp1,
                    wsp2, wsp3, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + nqTot * warpsize * iwarp;
                TData *wsp1 = wsp + nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;

                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, wsp0, wsp1, wsp2);
                IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, w0, w1, w2, jacptr, wsp0, outptr, wsp1,
                    wsp2, (TData)1.0);
            }
            e += item_ct1.get_global_range(2);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *s_wsp0 = (TData *)shmemptr;
            TData *s_wsp1 = s_wsp0 + nmTot;
            TData *s_wsp2 = s_wsp1 + nqTot;
            TData *s_wsp3 = s_wsp2 + std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
            TData *s_basis0 =
                s_wsp3 + std::max(nq1 * nq0 * nm2, nm0 * nm1 * nq2);
            TData *s_basis1 = s_basis0 + nm0 * nq0;
            TData *s_basis2 = s_basis1 + nm1 * nq1;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm2 * nq2;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis2[idx] = basis2[idx];
            }

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransHexSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          s_basis0, s_basis1, s_basis2, s_wsp0,
                                          s_wsp1, s_wsp2, s_wsp3, item_ct1);

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = (idx / nq0) % nq1;
                    const unsigned int k        = idx / (nq0 * nq1);
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j] * w2[k];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBaseHexSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, s_wsp1, outptr, s_wsp2, s_wsp3,
                    (TData)1.0, item_ct1);
                e += item_ct1.get_group_range(2);
            }
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

            TData *s_wsp0   = (TData *)shmemptr;
            TData *s_wsp1   = s_wsp0 + nmTot;
            TData *s_wsp2   = s_wsp1 + nqTot;
            TData *s_wsp3   = s_wsp2 + nm01 * nq2;
            TData *s_basis0 = s_wsp3 + nm0 * nq1 * nq2;
            TData *s_basis1 = s_basis0 + nm0 * nq0;
            TData *s_basis2 = s_basis1 + nm01 * nq1;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm01 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2);
                 idx < nmode2 * nq2; idx += item_ct1.get_local_range(2))
            {
                s_basis2[idx] = basis2[idx];
            }

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransTetSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, index0, index3, s_basis0,
                                          s_basis1, s_basis2, s_wsp0, s_wsp1,
                                          s_wsp2, s_wsp3, item_ct1);

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = (idx / nq0) % nq1;
                    const unsigned int k        = idx / (nq0 * nq1);
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j] * w2[k];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBaseTetSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2,
                    s_wsp1, outptr, s_wsp3, s_wsp2, (TData)1.0, item_ct1);
                e += item_ct1.get_group_range(2);
            }
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

            TData *s_wsp0   = (TData *)shmemptr;
            TData *s_wsp1   = s_wsp0 + nmTot;
            TData *s_wsp2   = s_wsp1 + nqTot;
            TData *s_wsp3   = s_wsp2 + nm0 * nm1 * nq2;
            TData *s_basis0 = s_wsp3 + nm0 * nq1 * nq2;
            TData *s_basis1 = s_basis0 + nm0 * nq0;
            TData *s_basis2 = s_basis1 + nm1 * nq1;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm02 * nq2;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis2[idx] = basis2[idx];
            }

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransPrismSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                            isModified, s_basis0, s_basis1,
                                            s_basis2, s_wsp0, s_wsp1, s_wsp2,
                                            s_wsp3, item_ct1);

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = (idx / nq0) % nq1;
                    const unsigned int k        = idx / (nq0 * nq1);
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j] * w2[k];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBasePrismSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2,
                    s_wsp1, outptr, s_wsp3, s_wsp2, (TData)1.0, item_ct1);
                e += item_ct1.get_group_range(2);
            }
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

            TData *s_wsp0   = (TData *)shmemptr;
            TData *s_wsp1   = s_wsp0 + nmTot;
            TData *s_wsp2   = s_wsp1 + nqTot;
            TData *s_wsp3   = s_wsp2 + nm0 * nm1 * nq2;
            TData *s_basis0 = s_wsp3 + nm0 * nq1 * nq2;
            TData *s_basis1 = s_basis0 + nm0 * nq0;
            TData *s_basis2 = s_basis1 + nm1 * nq1;

            // Copy to shared memory.
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis0[idx] = basis0[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_basis1[idx] = basis1[idx];
            }

            for (unsigned int idx = item_ct1.get_local_id(2);
                 idx < nmode2 * nq2; idx += item_ct1.get_local_range(2))
            {
                s_basis2[idx] = basis2[idx];
            }

            unsigned int e = item_ct1.get_group(2);
            while (e < nelmt)
            {
                const TData *jacptr = jac + jacsize * e;
                const TData *inptr  = in + nmTot * e;
                TData *outptr       = out + nmTot * e;

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
                     idx += item_ct1.get_local_range(2))
                {
                    s_wsp0[idx] = inptr[idx];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                BwdTransPyrSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, s_basis0, s_basis1,
                                          s_basis2, s_wsp0, s_wsp1, s_wsp2,
                                          s_wsp3, item_ct1);

                // Copy to shared memory.
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i        = idx % nq0;
                    const unsigned int j        = (idx / nq0) % nq1;
                    const unsigned int k        = idx / (nq0 * nq1);
                    const unsigned int jacindex = DEFORMED ? idx : 0;
                    s_wsp1[idx] *= jacptr[jacindex] * w0[i] * w1[j] * w2[k];
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                IProductWRTBasePyrSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, s_wsp1,
                    outptr, s_wsp3, s_wsp2, (TData)1.0, item_ct1);
                e += item_ct1.get_group_range(2);
            }
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
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        MassSharedMemorySize<Implementation>(nq0, nm0);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 Mass1DKernel<Implementation, DEFORMED>(nm0, nq0, nelmt, basis0,
                                                        w0, jac, in, out, wsp,
                                                        shmemptr, item_ct1);
             });
     }).wait();
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
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nm0, nm1);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nq0 * nq1);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 Mass2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                     nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0,
                     basis0, basis1, w0, w1, jac, in, out, wsp, shmemptr,
                     item_ct1);
             });
     }).wait();
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
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        MassSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2, nm0,
                                                         nm1, nm2);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nm0 * nm1 * nm2);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 Mass3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                     nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified,
                     index0, index1, index2, index3, basis0, basis1, basis2, w0,
                     w1, w2, jac, in, out, wsp, shmemptr, item_ct1);
             });
     }).wait();
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
