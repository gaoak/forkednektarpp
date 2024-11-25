///////////////////////////////////////////////////////////////////////////////
//
// File: OneMKL.hpp
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

#include "SYCLQueue.hpp"

namespace OneMKL
{

void gemm(sycl::queue queue, std::string transposeA, std::string transposeB,
          const unsigned int M, const unsigned int N, const unsigned int K,
          const float alpha, const float *a, const unsigned int lda,
          const float *b, const unsigned int ldb, const float beta, float *c,
          const unsigned int ldc);

void gemm(sycl::queue queue, std::string transposeA, std::string transposeB,
          const unsigned int M, const unsigned int N, const unsigned int K,
          const double alpha, const double *a, const unsigned int lda,
          const double *b, const unsigned int ldb, const double beta, double *c,
          const unsigned int ldc);

void gemm_batch(sycl::queue queue, std::string transposeA,
                std::string transposeB, const unsigned int M,
                const unsigned int N, const unsigned int K, const float alpha,
                const float *a, const unsigned int lda,
                const unsigned int strideA, const float *b,
                const unsigned int ldb, const unsigned int strideB,
                const float beta, float *c, const unsigned int ldc,
                const unsigned int strideC, const unsigned int batchSize);

void gemm_batch(sycl::queue queue, std::string transposeA,
                std::string transposeB, const unsigned int M,
                const unsigned int N, const unsigned int K, const double alpha,
                const double *a, const unsigned int lda,
                const unsigned int strideA, const double *b,
                const unsigned int ldb, const unsigned int strideB,
                const double beta, double *c, const unsigned int ldc,
                const unsigned int strideC, const unsigned int batchSize);
} // namespace OneMKL
