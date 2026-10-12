///////////////////////////////////////////////////////////////////////////////
//
// File: AvgAssmbScatrKernels.hpp
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
// Description: Kernel for the AvgAssmbScatr operator: the inverse
// multiplicity of the local coefficients.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Replace every multiplicity in @p ptr by its reciprocal, and by zero
 * where it is zero.
 *
 * The multiplicity of a local coefficient is a whole number, zero exactly
 * where its orientation sign is zero, so a value below one half is zero.
 *
 * @param   nsize     Number of values.
 * @param   ptr       Multiplicities on entry, their reciprocals on return.
 * @param   streamID  Stream the kernel runs on.
 */
template <typename ExecSpace, typename TData>
void InverseMultiplicityKernel(const size_t nsize, TData *ptr,
                               const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const size_t i) {
            ptr[i] = (ptr[i] < TData(0.5)) ? TData(0) : TData(1) / ptr[i];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::MultiRegions::detail
