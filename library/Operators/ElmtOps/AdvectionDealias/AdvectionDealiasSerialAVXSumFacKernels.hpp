///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasSerialAVXSumFacKernels.hpp
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
// Description: Fine-grid product for 3/2-rule dealiased advection.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace Nektar::Operators::detail
{

template <typename simd_type>
NEK_FORCE_INLINE void AdvectionDealiasCombineKernel(
    const unsigned int nqTot, const unsigned int coordDim,
    const simd_type *advVelPtr, const unsigned int advVelOffset,
    const simd_type *gradPtr, const unsigned int gradOffset, simd_type *out,
    const typename simd_type::scalarType scale)
{
    for (unsigned int j = 0; j < nqTot; ++j)
    {
        simd_type tmp = advVelPtr[j] * gradPtr[j];
        for (unsigned int d = 1; d < coordDim; ++d)
        {
            tmp.fma(advVelPtr[d * advVelOffset + j],
                    gradPtr[d * gradOffset + j]);
        }

        out[j] = scale * tmp;
    }
}

} // namespace Nektar::Operators::detail
