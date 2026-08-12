///////////////////////////////////////////////////////////////////////////////
//
// File: Math.hpp
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

#include "LibUtilities/BasicUtils/MemoryRegion.hpp"

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"

namespace Nektar::Math
{

// In-place MemoryRegion-based Math functions. Should in general NOT be used at
// the "solver" level, but only at the "Operator" level. Those functions require
// compile-time definition of the Execution space as a template parameter.
template <typename ExecSpace, typename TData>
void zero(LibUtilities::MemoryRegion<TData> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    zeroKernel<ExecSpace>(nsize, xptr);
}

template <typename ExecSpace, typename TData>
void fill(const TData &val, LibUtilities::MemoryRegion<TData> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    fillKernel<ExecSpace>(nsize, val, xptr);
}

template <typename ExecSpace, typename TData>
void copy(LibUtilities::MemoryRegion<TData> &x,
          LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::copy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    copyKernel<ExecSpace>(nsize, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void abs(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::abs - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    absKernel<ExecSpace>(nsize, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void neg(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::neg - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    negKernel<ExecSpace>(nsize, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void sqrt(LibUtilities::MemoryRegion<TData> &x,
          LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sqrt - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    sqrtKernel<ExecSpace>(nsize, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void add(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y,
         LibUtilities::MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::add - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto zptr  = z.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    addKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData>
void sub(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y,
         LibUtilities::MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sub - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto zptr  = z.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    subKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData>
void mul(const TData alpha, LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    mulKernel<ExecSpace>(nsize, alpha, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void mul(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y,
         LibUtilities::MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto zptr  = z.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    mulKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData>
void div(const TData alpha, LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    divKernel<ExecSpace>(nsize, alpha, xptr, yptr);
}

template <typename ExecSpace, typename TData>
void div(LibUtilities::MemoryRegion<TData> &x,
         LibUtilities::MemoryRegion<TData> &y,
         LibUtilities::MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto zptr  = z.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    divKernel<ExecSpace>(nsize, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData>
void daxpy(const TData alpha, LibUtilities::MemoryRegion<TData> &x,
           LibUtilities::MemoryRegion<TData> &y,
           LibUtilities::MemoryRegion<TData> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::daxpy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto zptr  = z.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    daxpyKernel<ExecSpace>(nsize, alpha, xptr, yptr, zptr);
}

template <typename ExecSpace, typename TData>
void reduceSum(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceSumKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void reduceSum(LibUtilities::MemoryRegion<uint8_t> &mask,
               LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceSumKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void reduceMax(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMaxKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void reduceMax(LibUtilities::MemoryRegion<uint8_t> &mask,
               LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceMaxKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void reduceMin(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMinKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void reduceMin(LibUtilities::MemoryRegion<uint8_t> &mask,
               LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceMinKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void ddot(LibUtilities::MemoryRegion<TData> &x,
          LibUtilities::MemoryRegion<TData> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr  = y.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    ddotKernel<ExecSpace, true>(nsize, xptr, yptr, out);
}

template <typename ExecSpace, typename TData>
void ddot(LibUtilities::MemoryRegion<uint8_t> &mask,
          LibUtilities::MemoryRegion<TData> &x,
          LibUtilities::MemoryRegion<TData> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto yptr    = y.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    ddotKernel<ExecSpace, true>(nsize, maskptr, xptr, yptr, out);
}

template <typename ExecSpace, typename TData>
void l1norm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l1normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void l1norm(LibUtilities::MemoryRegion<uint8_t> &mask,
            LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    l1normKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void l2norm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l2normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void l2norm(LibUtilities::MemoryRegion<uint8_t> &mask,
            LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    l2normKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void lpnorm(const unsigned int p, LibUtilities::MemoryRegion<TData> &x,
            TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    lpnormKernel<ExecSpace, true>(nsize, p, xptr, out);
}

template <typename ExecSpace, typename TData>
void lpnorm(const unsigned int p, LibUtilities::MemoryRegion<uint8_t> &mask,
            LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    lpnormKernel<ExecSpace, true>(nsize, p, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData>
void linfnorm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    linfnormKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData>
void linfnorm(LibUtilities::MemoryRegion<uint8_t> &mask,
              LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    linfnormKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

} // namespace Nektar::Math
