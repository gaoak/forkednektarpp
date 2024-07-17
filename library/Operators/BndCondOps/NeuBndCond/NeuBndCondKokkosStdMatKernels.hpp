///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondKokkosStdMatKernels.hpp
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

#include "Operators/Common/LoopExecution.hpp"

#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
NeuBndCondKernel(const size_t bndExpSize, const int *offsetPtr,
                 const BoundaryConditionType *bctypePtr, const int *ncoeffPtr,
                 const int *mapPtr, const TData *inPtr, TData *outPtr)
{
    Kokkos::parallel_for(
        bndExpSize, KOKKOS_LAMBDA(const unsigned int i) {
            if (bctypePtr[i] == eNeumann || bctypePtr[i] == eRobin)
            {
                unsigned int offset = offsetPtr[i];
                unsigned int ncoeff = ncoeffPtr[i];
                for (unsigned int j = 0; j < ncoeff; j++)
                {
                    outPtr[mapPtr[offset + j]] += inPtr[offset + j];
                }
            }
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
NeuBndCondKernel(const size_t bndExpSize, const int *offsetPtr,
                 const BoundaryConditionType *bctypePtr, const int *ncoeffPtr,
                 const TData *signPtr, const int *mapPtr, const TData *inPtr,
                 TData *outPtr)
{
    Kokkos::parallel_for(
        bndExpSize, KOKKOS_LAMBDA(const unsigned int i) {
            if (bctypePtr[i] == eNeumann || bctypePtr[i] == eRobin)
            {
                unsigned int offset = offsetPtr[i];
                unsigned int ncoeff = ncoeffPtr[i];
                for (unsigned int j = 0; j < ncoeff; j++)
                {
                    outPtr[mapPtr[offset + j]] +=
                        signPtr[offset + j] * inPtr[offset + j];
                }
            }
        });
}

} // namespace Nektar::Operators::detail

#endif
