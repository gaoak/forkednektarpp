///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXSumFacKernels.hpp
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

#include "StdRegions/Operators/BwdTransSumFacStdKernels.hpp"

namespace Nektar::Operators::detail
{

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

// Workspace - used to dynamically get the workspace size needed for
// temporary memory.
NEK_FORCE_INLINE static void BwdTrans1DWorkspace(
    [[maybe_unused]] LibUtilities::ShapeType SHAPE_TYPE,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nq0)

{
}

NEK_FORCE_INLINE static void BwdTrans2DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1, unsigned int &wsp0Size)
{
    if (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        wsp0Size = std::max(wsp0Size, nm0 + nm0 * (nm0 + 1) / 2);
    }
    else if (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        wsp0Size = std::max(wsp0Size, nm0);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm1 * nq0);
    }
}

NEK_FORCE_INLINE static void BwdTrans3DWorkspace(
    LibUtilities::ShapeType SHAPE_TYPE, [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2, unsigned int &wsp0Size,
    unsigned int &wsp1Size)
{
    if (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        wsp0Size = std::max(wsp0Size, nq0 * nm1 * nm2);
        wsp1Size = std::max(wsp1Size, nq0 * nq1 * nm2);
    }
    else if (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        wsp0Size =
            std::max(wsp0Size, nm0 * nm1 + nm0 * (nm0 + 1) * (nm0 + 2) / 6);
        wsp1Size = std::max(wsp1Size, nm0);
    }
    else if (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        wsp0Size = std::max(wsp0Size, nm0 * nm1 + nm0 * nm0 * (nm0 + 1) / 2);
        wsp1Size = std::max(wsp1Size, nm0);
    }
    else
    {
        wsp0Size = std::max(wsp0Size, nm0 * nm1);
        wsp1Size = std::max(wsp1Size, nm0);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void BwdTrans1DKernel(
    const unsigned int nm0, const unsigned int nq0,
    const typename simd_type::scalarType *basis0, const simd_type *in,
    simd_type *out)
{
    BwdTransSegKernel<APPEND>(nm0, nq0, basis0, in, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void BwdTrans2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, [[maybe_unused]] const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    [[maybe_unused]] const simd_type *NtoM, simd_type *wsp0,
    const simd_type *in, simd_type *out)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eNodalTri)
    {
        const auto nmTot = nm0 * (nm0 + 1) / 2;
        simd_type *in1   = wsp0 + nm0;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransTriKernel<APPEND>(nm0, nm1, nq0, nq1, isModified, basis0,
                                  basis1, wsp0, in1, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
    {
        BwdTransTriKernel<APPEND>(nm0, nm1, nq0, nq1, isModified, basis0,
                                  basis1, wsp0, in, out);
    }
    else
    {
        BwdTransQuadKernel<APPEND>(nm0, nm1, nq0, nq1, basis0, basis1, wsp0, in,
                                   out);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void BwdTrans3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    [[maybe_unused]] const simd_type *NtoM, simd_type *wsp0, simd_type *wsp1,
    const simd_type *in, simd_type *out)
{
    if constexpr (SHAPE_TYPE == LibUtilities::eHexahedron)
    {
        BwdTransHexKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1,
                                  basis2, wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
    {
        BwdTransTetKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalTet)
    {
        const auto nmTot = nm0 * (nm0 + 1) * (nm0 + 2) / 6;
        simd_type *in1   = wsp0 + nm0 * nm1;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransTetKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in1, out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
    {
        BwdTransPrismKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                    basis0, basis1, basis2, wsp0, wsp1, in,
                                    out);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::eNodalPrism)
    {
        const auto nmTot = nm0 * nm0 * (nm0 + 1) / 2;
        simd_type *in1   = wsp0 + nm0 * nm1;

        MatVecKernel(nmTot, NtoM, in, in1);
        BwdTransPrismKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                    basis0, basis1, basis2, wsp0, wsp1, in1,
                                    out);
    }
    else
    {
        BwdTransPyrKernel<APPEND>(nm0, nm1, nm2, nq0, nq1, nq2, isModified,
                                  basis0, basis1, basis2, wsp0, wsp1, in, out);
    }
}

} // namespace Nektar::Operators::detail
