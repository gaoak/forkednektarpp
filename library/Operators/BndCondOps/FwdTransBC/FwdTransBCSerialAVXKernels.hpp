///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCSerialAVXKernels.hpp
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
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "StdRegions/Operators/BwdTransSumFacStdKernels.hpp"
#include "StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp"
#include "Utils/UtilsSerialAVXKernels.hpp"

#include <LibUtilities/BasicUtils/NekInline.hpp>

namespace Nektar::Operators::detail
{
template <typename ExecSpace, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void FwdTransBCSegKernel(
    const unsigned int nm0, const unsigned int nq0, const simd_type *basis0,
    const simd_type *w0, const unsigned int offset_seg,
    const simd_type *invintmass, [[maybe_unused]] const simd_type *jac,
    const simd_type *in, simd_type *out, simd_type *wsp1, simd_type *wsp2)
{
    const auto nmInt = nm0 - 2;

    // Step 1: Set vertex modes and evaluate mass matrix (BwdTrans +
    // IProd) Assume outblock is initialised as zero
    // TODO vertex map changes with GLL_Lagrange
    out[0] = in[0];
    out[1] = in[nq0 - 1];

    // Step 1: Evaluate vertex contributions.
    BwdTransSegKernel(nm0, nq0, basis0, out, wsp2);

    // Note negative scale to substract this from Dirichlet
    // condition
    if constexpr (DEFORMED)
    {
        IProductSegKernel<true, false, DEFORMED>(nm0, nq0, wsp2, basis0, w0,
                                                 jac, wsp1, -1.0);

        // Step 2: Evaluate IProd of Dirichlet condition and
        // Step 3: Subtract vertex contribution from Dirichlet condition
        IProductSegKernel<false, true, DEFORMED>(nm0, nq0, in, basis0, w0, jac,
                                                 wsp1, 1.0);
    }
    else
    {
        simd_type tmp = 1.0;
        IProductSegKernel<true, false, DEFORMED>(nm0, nq0, wsp2, basis0, w0,
                                                 &tmp, wsp1, -1.0);

        // Step 2: Evaluate IProd of Dirichlet condition and
        // Step 3: Subtract vertex contribution from Dirichlet condition
        IProductSegKernel<false, true, DEFORMED>(nm0, nq0, in, basis0, w0, &tmp,
                                                 wsp1, 1.0);
    }

    // Step 4: Project edge interior modes onto boundary
    MatVecKernel(nmInt, invintmass, wsp1 + offset_seg, out + offset_seg);

    // Divide by Jacobian.
    if constexpr (DEFORMED)
    {
        DivideByJacobian<ExecSpace, DEFORMED, simd_type>(
            1, nmInt, jac, out + offset_seg, out + offset_seg);
    }
}

template <typename ExecSpace, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void FwdTransBCQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const simd_type *basis0, const simd_type *basis1,
    const simd_type *w0, const simd_type *w1, const unsigned int offset_seg,
    const simd_type *invintmass0, const simd_type *invintmass1,
    [[maybe_unused]] const simd_type *tJac, const unsigned int *tMap,
    const int *tSign, const unsigned int nmTotInt, const unsigned int *iMap,
    [[maybe_unused]] const simd_type *invintmass,
    [[maybe_unused]] const simd_type *dmat,
    [[maybe_unused]] const simd_type *jac, const simd_type *in, simd_type *out,
    simd_type *wsp1, simd_type *wsp2, simd_type *wsp3, simd_type *wsp4)
{
    // Get relevant sizes in phys and coeff space
    const unsigned int nqTot     = nq0 * nq1;
    const unsigned int nmEdgeTot = 2 * nm0 + 2 * nm1;

    /// Step 1: Extract edge modes
    for (unsigned int i = 0; i < nq0; i++)
    {
        wsp1[i]             = in[i];
        wsp1[i + nq0 + nq1] = in[nq0 * (nq1 - 1) + i];
    }
    for (unsigned int i = 0; i < nq1; i++)
    {
        wsp1[i + nq0]           = in[nq0 - 1 + i * nq0];
        wsp1[i + 2 * nq0 + nq1] = in[i * nq0];
    }

    // Zero wsp2
    for (unsigned int i = 0; i < nqTot; i++)
    {
        wsp2[i] = 0.0;
    }

    // Remove vertex contribution for every edge
    // TODO Do we need a separate check whether segment is deformed?
    simd_type tmp = 1.0;
    FwdTransBCSegKernel<ExecSpace, false>(nm0, nq0, basis0, w0, offset_seg,
                                          invintmass0, &tmp, wsp1, wsp2, wsp3,
                                          wsp4);
    FwdTransBCSegKernel<ExecSpace, false>(nm1, nq1, basis1, w1, offset_seg,
                                          invintmass1, &tmp, wsp1 + nq0,
                                          wsp2 + nm0, wsp3, wsp4);
    FwdTransBCSegKernel<ExecSpace, false>(nm0, nq0, basis0, w0, offset_seg,
                                          invintmass0, &tmp, wsp1 + nq0 + nq1,
                                          wsp2 + nm0 + nm1, wsp3, wsp4);
    FwdTransBCSegKernel<ExecSpace, false>(
        nm1, nq1, basis1, w1, offset_seg, invintmass1, &tmp,
        wsp1 + 2 * nq0 + nq1, wsp2 + 2 * nm0 + nm1, wsp3, wsp4);

    // Map edge modes (without vertex contribution) back into face
    for (unsigned int j = 0u; j < nmEdgeTot; j++)
    {
        out[tMap[j]] = tSign[j] * wsp2[j];
    }

    /// Step 2: Evaluate edge contributions via mass matrix
    BwdTransQuadKernel(nm0, nm1, nq0, nq1, basis0, basis1, wsp2, out, wsp1);

    // Complete mass matrix with IProduct
    if constexpr (DEFORMED)
    {
        IProductQuadKernel<true, false, DEFORMED, simd_type>(
            nm0, nm1, nq0, nq1, wsp1, basis0, basis1, w0, w1, jac, wsp3, wsp2,
            -1.0);

        /// Step 3: Evaluate IProd of Dirichlet condition and
        /// Step 4: Subtract vertex contribution from Dirichlet condition
        IProductQuadKernel<false, true, DEFORMED, simd_type>(
            nm0, nm1, nq0, nq1, in, basis0, basis1, w0, w1, jac, wsp3, wsp2,
            1.0);
    }
    else
    {
        simd_type tmp = 1.0;
        IProductQuadKernel<true, false, DEFORMED, simd_type>(
            nm0, nm1, nq0, nq1, wsp1, basis0, basis1, w0, w1, &tmp, wsp3, wsp2,
            -1.0);

        /// Step 3: Evaluate IProd of Dirichlet condition and
        /// Step 4: Subtract vertex contribution from Dirichlet condition
        IProductQuadKernel<false, true, DEFORMED, simd_type>(
            nm0, nm1, nq0, nq1, in, basis0, basis1, w0, w1, &tmp, wsp3, wsp2,
            1.0);
    }

    /// Step 4: Project face/interior modes onto boundary
    // Map to interior coeffs
    for (unsigned int j = 0u; j < nmTotInt; j++)
    {
        wsp1[j] = wsp2[iMap[j]];
    }

    // Projection ie matrix vector product with inverse interior mass
    if constexpr (DEFORMED)
    {
        MatVecKernel(nmTotInt, dmat, wsp1, wsp2);
    }
    else
    {
        MatVecKernel(nmTotInt, invintmass, wsp1, wsp2);
    }

    // Map to volume
    for (unsigned int j = 0u; j < nmTotInt; j++)
    {
        out[iMap[j]] = wsp2[j];
    }
}

template <typename ExecSpace, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void FwdTransBCTriKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const bool isModified, const simd_type *basis0,
    const simd_type *basis1, const simd_type *w0, const simd_type *w1,
    const simd_type *interp1to0, const unsigned int offset_seg,
    const simd_type *invintmass0, [[maybe_unused]] const simd_type *tJac,
    const unsigned int *tMap, const int *tSign, const unsigned int nmTotInt,
    const unsigned int *iMap, [[maybe_unused]] const simd_type *invintmass,
    [[maybe_unused]] const simd_type *dmat,
    [[maybe_unused]] const simd_type *jac, const simd_type *in, simd_type *out,
    simd_type *wsp1, simd_type *wsp2, simd_type *wsp3, simd_type *wsp4)
{
    // Get relevant sizes in phys and coeff space
    const unsigned int nqTot     = nq0 * nq1;
    const unsigned int nmEdgeTot = nm0 + 2 * nm1;

    /// Step 1: Extract edge modes
    for (unsigned int i = 0; i < nq0; i++)
    {
        // Interpolate basis1 to basis0 for edge1 and edge2
        simd_type tmp  = 0.0;
        simd_type tmp2 = 0.0;
        for (unsigned int j = 0u; j < nq1; j++)
        {
            tmp += in[nq0 - 1 + j * nq0] * interp1to0[i * nq1 + j];
            tmp2 += in[j * nq0] * interp1to0[i * nq1 + j];
        }

        wsp1[i]           = in[i];
        wsp1[i + nq0]     = tmp;
        wsp1[i + 2 * nq0] = tmp2;
    }

    // Zero wsp2
    for (unsigned int i = 0; i < nqTot; i++)
    {
        wsp2[i] = 0.0;
    }

    // Remove vertex contribution for every edge
    // TODO Do we need a separate check whether segment is deformed?
    simd_type tmp = 1.0;
    FwdTransBCSegKernel<ExecSpace, false>(nm0, nq0, basis0, w0, offset_seg,
                                          invintmass0, &tmp, wsp1, wsp2, wsp3,
                                          wsp4);
    FwdTransBCSegKernel<ExecSpace, false>(nm0, nq0, basis0, w0, offset_seg,
                                          invintmass0, &tmp, wsp1 + nq0,
                                          wsp2 + nm0, wsp3, wsp4);
    FwdTransBCSegKernel<ExecSpace, false>(nm0, nq0, basis0, w0, offset_seg,
                                          invintmass0, &tmp, wsp1 + 2 * nq0,
                                          wsp2 + nm0 + nm1, wsp3, wsp4);

    // Map edge modes (without vertex contribution) back into face
    for (unsigned int j = 0u; j < nmEdgeTot; j++)
    {
        out[tMap[j]] = tSign[j] * wsp2[j];
    }

    /// Step 2: Evaluate edge contributions via mass matrix
    BwdTransTriKernel(nm0, nm1, nq0, nq1, isModified, basis0, basis1, wsp2, out,
                      wsp1);

    // Complete mass matrix with IProduct
    if constexpr (DEFORMED)
    {
        IProductTriKernel<true, false, DEFORMED>(nm0, nm1, nq0, nq1, isModified,
                                                 wsp1, basis0, basis1, w0, w1,
                                                 jac, wsp3, wsp2, -1.0);

        /// Step 3: Evaluate IProd of Dirichlet condition and
        /// Step 4: Subtract vertex contribution from Dirichlet condition
        IProductTriKernel<false, true, DEFORMED>(nm0, nm1, nq0, nq1, isModified,
                                                 in, basis0, basis1, w0, w1,
                                                 jac, wsp3, wsp2, 1.0);
    }
    else
    {
        simd_type tmp = 1.0;
        IProductTriKernel<true, false, DEFORMED>(nm0, nm1, nq0, nq1, isModified,
                                                 wsp1, basis0, basis1, w0, w1,
                                                 &tmp, wsp3, wsp2, -1.0);

        /// Step 3: Evaluate IProd of Dirichlet condition and
        /// Step 4: Subtract vertex contribution from Dirichlet condition
        IProductTriKernel<false, true, DEFORMED>(nm0, nm1, nq0, nq1, isModified,
                                                 in, basis0, basis1, w0, w1,
                                                 &tmp, wsp3, wsp2, 1.0);
    }

    /// Step 4: Project face/interior modes onto boundary
    // Map to interior coeffs
    for (unsigned int j = 0u; j < nmTotInt; j++)
    {
        wsp1[j] = wsp2[iMap[j]];
    }

    // Projection ie matrix vector product with inverse interior mass
    if constexpr (DEFORMED)
    {
        MatVecKernel(nmTotInt, dmat, wsp1, wsp2);
    }
    else
    {
        MatVecKernel(nmTotInt, invintmass, wsp1, wsp2);
    }

    // Map to volume
    for (unsigned int j = 0u; j < nmTotInt; j++)
    {
        out[iMap[j]] = wsp2[j];
    }
}

template <typename ExecSpace, LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename simd_type>
NEK_FORCE_INLINE static void FwdTransBC1DKernel(
    const unsigned int nm0, const unsigned int nq0, const simd_type *basis0,
    const simd_type *w0, const unsigned int offset_seg,
    const simd_type *invintmass, const simd_type *jac, const simd_type *in,
    simd_type *out, simd_type *wsp1, simd_type *wsp2)
{
    FwdTransBCSegKernel<ExecSpace, DEFORMED>(
        nm0, nq0, basis0, w0, offset_seg, invintmass, jac, in, out, wsp1, wsp2);
}

template <typename ExecSpace, LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename simd_type>
NEK_FORCE_INLINE static void FwdTransBC2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, [[maybe_unused]] const bool isModified,
    const simd_type *basis0, const simd_type *basis1, const simd_type *w0,
    const simd_type *w1, [[maybe_unused]] const simd_type *interp1to0,
    const unsigned int offset_seg, const simd_type *invintmass0,
    [[maybe_unused]] const simd_type *invintmass1, const simd_type *tJac,
    const unsigned int *tMap, const int *tSign, const unsigned int nmTotInt,
    const unsigned int *iMap, const simd_type *invintmass,
    const simd_type *dmat, const simd_type *jac, const simd_type *in,
    simd_type *out, simd_type *wsp1, simd_type *wsp2, simd_type *wsp3,
    simd_type *wsp4)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        FwdTransBCQuadKernel<ExecSpace, DEFORMED>(
            nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, offset_seg, invintmass0,
            invintmass1, tJac, tMap, tSign, nmTotInt, iMap, invintmass, dmat,
            jac, in, out, wsp1, wsp2, wsp3, wsp4);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        FwdTransBCTriKernel<ExecSpace, DEFORMED>(
            nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1, interp1to0,
            offset_seg, invintmass0, tJac, tMap, tSign, nmTotInt, iMap,
            invintmass, dmat, jac, in, out, wsp1, wsp2, wsp3, wsp4);
    }
    // else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    // {
    //     const auto nmTot  = nm0 * (nm0 + 1) / 2;
    //     simd_type *outtmp = wsp0 + nq1;
    //
    //     FwdTransBCTriKernel<SCALE, APPEND, simd_type>(
    //         nm0, nm1, nq0, nq1, isModified, in, B0, B1, wsp0, outtmp, scale);
    //     MatVecKernel(nmTot, NtoMTrans, outtmp, out);
    // }
}

} // namespace Nektar::Operators::detail
