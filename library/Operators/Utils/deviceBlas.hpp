///////////////////////////////////////////////////////////////////////////////
//
// File: deviceBlas.hpp
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

#include <string>
#if defined(NEKTAR_ENABLE_CUDA)
#include "Operators/Utils/CUBLASHandle.cuh"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Utils/SYCLQueue.hpp"
#endif

template <typename deviceHandle, typename TData>
void deviceGemm(deviceHandle handle, std::string transposeA,
                std::string transposeB, const unsigned int M,
                const unsigned int N, const unsigned int K, const TData alpha,
                const TData *a, const unsigned int lda, const TData *b,
                const unsigned int ldb, const TData beta, TData *c,
                const unsigned int ldc);

template <typename deviceHandle, typename TData>
void deviceGemmStridedBatched(
    deviceHandle handle, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const TData alpha, const TData *a, const unsigned int lda,
    const unsigned int strideA, const TData *b, const unsigned int ldb,
    const unsigned int strideB, const TData beta, TData *c,
    const unsigned int ldc, const unsigned int strideC,
    const unsigned int batchSize);
