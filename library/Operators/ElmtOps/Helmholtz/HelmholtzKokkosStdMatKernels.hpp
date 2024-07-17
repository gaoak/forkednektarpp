///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzKokkosStdMatKernels.hpp
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

#include "Common/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

// Generic kernel launchers except for CUDA.
template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
DiffusionCoeff1DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0)
{
    Kokkos::parallel_for(
        nsize,
        KOKKOS_LAMBDA(const unsigned int i) { deriv0[i] *= diffCoeff[0]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
DiffusionCoeff2DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0, TData *deriv1)
{
    Kokkos::parallel_for(
        nsize, KOKKOS_LAMBDA(const unsigned int i) {
            TData deriv[2] = {deriv0[i], deriv1[i]};

            deriv0[i] = diffCoeff[0] * deriv[0] + diffCoeff[1] * deriv[1];
            deriv1[i] = diffCoeff[2] * deriv[0] + diffCoeff[3] * deriv[1];
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
DiffusionCoeff3DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0, TData *deriv1, TData *deriv2)
{
    Kokkos::parallel_for(
        nsize, KOKKOS_LAMBDA(const unsigned int i) {
            TData deriv[3] = {deriv0[i], deriv1[i], deriv2[i]};

            deriv0[i] = diffCoeff[0] * deriv[0] + diffCoeff[1] * deriv[1] +
                        diffCoeff[2] * deriv[2];
            deriv1[i] = diffCoeff[3] * deriv[0] + diffCoeff[4] * deriv[1] +
                        diffCoeff[5] * deriv[2];
            deriv2[i] = diffCoeff[6] * deriv[0] + diffCoeff[7] * deriv[1] +
                        diffCoeff[8] * deriv[2];
        });
}

} // namespace Nektar::Operators::detail

#endif
