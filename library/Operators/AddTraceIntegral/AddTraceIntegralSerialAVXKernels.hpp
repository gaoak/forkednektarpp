///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralSerialAVXKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
AddTraceIntegralKernel(const unsigned int nsize,
                       const int *traceCoeffsToElmtMapPtr,
                       const int *traceCoeffsToElmtSignPtr,
                       const int *traceCoeffsToElmtTracePtr,
                       const TData *tracePtr, TData *outPtr)
{
    Nektar::parallel_for<ExecSpace>(0u, nsize, [&](const unsigned int i) {
        outPtr[traceCoeffsToElmtMapPtr[i]] +=
            traceCoeffsToElmtSignPtr[i] *
            tracePtr[traceCoeffsToElmtTracePtr[i]];
    });
}

template <typename ExecSpace>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value,
    void>::type
ReOrderMapKernel(const unsigned int nsize, int *traceCoeffsToElmtMapPtr,
                 int *traceCoeffsToElmtSignPtr, int *traceCoeffsToElmtTracePtr)
{
    // sort the trace map and get the permutation
    std::vector<int> permutation(nsize);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        return traceCoeffsToElmtTracePtr[i] < traceCoeffsToElmtTracePtr[j];
    });
    // apply the permutation to the map and sign
    std::vector<int> tempMap(nsize);
    std::vector<int> tempSign(nsize);
    std::vector<int> tempTrace(nsize);
    for (size_t i = 0; i < nsize; i++)
    {
        tempMap[i]   = traceCoeffsToElmtMapPtr[permutation[i]];
        tempSign[i]  = traceCoeffsToElmtSignPtr[permutation[i]];
        tempTrace[i] = traceCoeffsToElmtTracePtr[permutation[i]];
    }
    // copy back to the original map and sign
    for (size_t i = 0; i < nsize; i++)
    {
        traceCoeffsToElmtMapPtr[i]   = tempMap[i];
        traceCoeffsToElmtSignPtr[i]  = tempSign[i];
        traceCoeffsToElmtTracePtr[i] = tempTrace[i];
    }
}

} // namespace Nektar::Operators::detail
