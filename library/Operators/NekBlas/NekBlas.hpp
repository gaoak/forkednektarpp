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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#if defined(NEKTAR_ENABLE_CUDA)
#include "Operators/NekBlas/cuBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "Operators/NekBlas/hipBlasHandle.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Common/SYCLQueue.hpp"
#endif

struct blasHandle
{
};

template <typename ExecSpace> class NekHandle
{
public:
    static blasHandle GetInstance(void)
    {
        return blasHandle();
    }
};

#if defined(NEKTAR_ENABLE_CUDA)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static cublasHandle_t GetInstance(void)
    {
        return cuBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_HIP)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static hipblasHandle_t GetInstance(void)
    {
        return hipBlasHandle::GetInstance();
    }
};
#elif defined(NEKTAR_ENABLE_SYCL)
template <> class NekHandle<NektarSpaces::Device>
{
public:
    static sycl::queue GetInstance(void)
    {
        return SYCLQueue::GetInstance();
    }
};
#endif

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const TData alpha, const TData *a, const unsigned int lda, const TData *b,
    const unsigned int ldb, const TData beta, TData *c, const unsigned int ldc);
#if defined(NEKTAR_ENABLE_CUDA)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const TData alpha, const TData *a, const unsigned int lda, const TData *b,
    const unsigned int ldb, const TData beta, TData *c, const unsigned int ldc);
#elif defined(NEKTAR_ENABLE_HIP)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemm(THandle handle, std::string transposeA, std::string transposeB,
        const unsigned int M, const unsigned int N, const unsigned int K,
        const TData alpha, const TData *a, const unsigned int lda,
        const TData *b, const unsigned int ldb, const TData beta, TData *c,
        const unsigned int ldc);
#elif defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type NekGemm(
    THandle handle, std::string transposeA, std::string transposeB,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const TData alpha, const TData *a, const unsigned int lda, const TData *b,
    const unsigned int ldb, const TData beta, TData *c, const unsigned int ldc);
#endif

template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, blasHandle>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const TData alpha, const TData *a, const unsigned int lda,
                      const unsigned int strideA, const TData *b,
                      const unsigned int ldb, const unsigned int strideB,
                      const TData beta, TData *c, const unsigned int ldc,
                      const unsigned int strideC, const unsigned int batchSize);
#if defined(NEKTAR_ENABLE_CUDA)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, cublasHandle_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const TData alpha, const TData *a, const unsigned int lda,
                      const unsigned int strideA, const TData *b,
                      const unsigned int ldb, const unsigned int strideB,
                      const TData beta, TData *c, const unsigned int ldc,
                      const unsigned int strideC, const unsigned int batchSize);
#elif defined(NEKTAR_ENABLE_HIP)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, hipblasHandle_t>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const TData alpha, const TData *a, const unsigned int lda,
                      const unsigned int strideA, const TData *b,
                      const unsigned int ldb, const unsigned int strideB,
                      const TData beta, TData *c, const unsigned int ldc,
                      const unsigned int strideC, const unsigned int batchSize);
#elif defined(NEKTAR_ENABLE_SYCL)
template <typename THandle, typename TData>
typename std::enable_if<std::is_same_v<THandle, sycl::queue>, void>::type
NekGemmStridedBatched(THandle handle, std::string transposeA,
                      std::string transposeB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const TData alpha, const TData *a, const unsigned int lda,
                      const unsigned int strideA, const TData *b,
                      const unsigned int ldb, const unsigned int strideB,
                      const TData beta, TData *c, const unsigned int ldc,
                      const unsigned int strideC, const unsigned int batchSize);
#endif
