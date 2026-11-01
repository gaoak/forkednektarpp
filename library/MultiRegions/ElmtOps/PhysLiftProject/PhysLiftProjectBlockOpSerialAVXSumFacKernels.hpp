///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpSerialAVXSumFacKernels.hpp
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
// Description: Serial/AVX kernels of the lift-and-project element operator, one
// element group per call.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpSerialAVXSumFacKernels.hpp
 * @brief Serial/AVX kernels of the lift-and-project operator's own steps:
 * one SIMD vector of elements per call.
 *
 * @details
 * The lift, the basis transpose and the backward transform are the
 * launchers of IProductWRTPhysTrace, IProductWRTBase and BwdTrans, which the
 * block operator (PhysLiftProjectBlockOpSerialAVXSumFac.hpp) calls on each
 * element group; the inverse mass where it is diagonal is
 * MultiplyByElmtInvMass's StdInvMassDiagonalScaleKernel(), and the
 * Pointwise form's kernels are shared with the StdMat implementation
 * (PhysLiftProjectBlockOpSerialAVXKernels.hpp). The kernel here is the scale
 * after the dense standard inverse mass.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

namespace Nektar::MultiRegions::detail
{

/// `inout[m] *= coefScale` over the @p nmTot coefficients of one element
/// group: the scale over the Jacobian after the dense standard inverse mass.
template <typename simd_type>
NEK_FORCE_INLINE static void PhysLiftProjectCoefScaleKernel(
    const unsigned int nmTot, const simd_type coefScale, simd_type *inout)
{
    for (unsigned int m = 0; m < nmTot; ++m)
    {
        inout[m] = inout[m] * coefScale;
    }
}

} // namespace Nektar::MultiRegions::detail
