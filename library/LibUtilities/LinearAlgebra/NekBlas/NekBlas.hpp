///////////////////////////////////////////////////////////////////////////////
//
// File: NekBlas.hpp
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
// Description: Backend-agnostic BLAS interface. This header must stay
// abstract: it must not include any vendor BLAS headers (e.g. libxsmm.h,
// cublas_v2.h, hipblas/hipblas.h, magma_v2.h, oneapi/math.hpp). These
// belong in the backend .cpp files, or, where a backend needs them in a
// header (e.g. libXSMMDispatchWrapper.hpp), are included directly by the
// backend-specific code that uses them.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include <LibUtilities/Backends/Backends.hpp>

namespace Nektar::NekBlas
{
// Opaque handle to the BLAS of an execution space. It carries only the stream
// ID so that no vendor handle type appears in this header; each backend looks
// up its library handle for the stream in its own compilation unit.
template <typename ExecSpace> class Handle
{
public:
    static Handle GetInstance(const unsigned int streamID)
    {
        return Handle(streamID);
    }

    unsigned int GetStreamID() const
    {
        return m_streamID;
    }

private:
    explicit Handle(const unsigned int streamID) : m_streamID(streamID)
    {
    }

    unsigned int m_streamID;
};

// Serial: blas.cpp.
template <typename TData>
void Gemm(Handle<NektarSpaces::Serial> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc);
template <typename TData>
void GemmStridedBatched(Handle<NektarSpaces::Serial> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const int K,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *b, const int ldb,
                        const int strideB, const TData beta, TData *c,
                        const int ldc, const int strideC, const int batchSize);
template <typename TData>
void Gemv(Handle<NektarSpaces::Serial> handle, std::string transpose,
          const int M, const int N, const TData alpha, const TData *a,
          const int lda, const TData *x, const int incx, const TData beta,
          TData *y, const int incy);
template <typename TData>
void GemvStridedBatched(Handle<NektarSpaces::Serial> handle,
                        std::string transpose, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *x, const int incx,
                        const int strideX, const TData beta, TData *y,
                        const int incy, const int strideY, const int batchSize);
template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::Serial> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const TData alpha,
                        const TData *a, const int lda, const int strideA,
                        const TData beta, const TData *b, const int ldb,
                        const int strideB, TData *c, const int ldc,
                        const int strideC, const int batchSize);

// AVX: xsmm.cpp.
template <typename TData>
void Gemm(Handle<NektarSpaces::AVX> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc);
template <typename TData>
void GemmStridedBatched(Handle<NektarSpaces::AVX> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const int K,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *b, const int ldb,
                        const int strideB, const TData beta, TData *c,
                        const int ldc, const int strideC, const int batchSize);
template <typename TData>
void Gemv(Handle<NektarSpaces::AVX> handle, std::string transpose, const int M,
          const int N, const TData alpha, const TData *a, const int lda,
          const TData *x, const int incx, const TData beta, TData *y,
          const int incy);
template <typename TData>
void GemvStridedBatched(Handle<NektarSpaces::AVX> handle, std::string transpose,
                        const int M, const int N, const TData alpha,
                        const TData *a, const int lda, const int strideA,
                        const TData *x, const int incx, const int strideX,
                        const TData beta, TData *y, const int incy,
                        const int strideY, const int batchSize);
template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::AVX> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const TData alpha,
                        const TData *a, const int lda, const int strideA,
                        const TData beta, const TData *b, const int ldb,
                        const int strideB, TData *c, const int ldc,
                        const int strideC, const int batchSize);

// Device: cuBlas.cpp, hipBlas.cpp, magma.cpp or oneMath.cpp.
template <typename TData>
void Gemm(Handle<NektarSpaces::Device> handle, std::string transposeA,
          std::string transposeB, const int M, const int N, const int K,
          const TData alpha, const TData *a, const int lda, const TData *b,
          const int ldb, const TData beta, TData *c, const int ldc);
template <typename TData>
void GemmStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const int K,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *b, const int ldb,
                        const int strideB, const TData beta, TData *c,
                        const int ldc, const int strideC, const int batchSize);
template <typename TData>
void Gemv(Handle<NektarSpaces::Device> handle, std::string transpose,
          const int M, const int N, const TData alpha, const TData *a,
          const int lda, const TData *x, const int incx, const TData beta,
          TData *y, const int incy);
template <typename TData>
void GemvStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transpose, const int M, const int N,
                        const TData alpha, const TData *a, const int lda,
                        const int strideA, const TData *x, const int incx,
                        const int strideX, const TData beta, TData *y,
                        const int incy, const int strideY, const int batchSize);
template <typename TData>
void GeamStridedBatched(Handle<NektarSpaces::Device> handle,
                        std::string transposeA, std::string transposeB,
                        const int M, const int N, const TData alpha,
                        const TData *a, const int lda, const int strideA,
                        const TData beta, const TData *b, const int ldb,
                        const int strideB, TData *c, const int ldc,
                        const int strideC, const int batchSize);
} // namespace Nektar::NekBlas
