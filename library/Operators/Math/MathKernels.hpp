///////////////////////////////////////////////////////////////////////////////
//
// File: MathKernels.hpp
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

#include "Operators/Field/Field.hpp"

#include "Operators/Math/MathAVXKernels.hpp"
#include "Operators/Math/MathDeviceOnHostKernels.hpp"
#include "Operators/Math/MathHIPCUDAKernels.hpp"
#include "Operators/Math/MathSYCLKernels.hpp"
#include "Operators/Math/MathSerialKernels.hpp"

namespace Nektar::Operators
{

template <typename ExecSpace, typename TData, FieldState TFieldState>
void abs(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::abs - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        absKernel<ExecSpace>(size, xptr, yptr);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void abs(MemoryRegion<TData> &x, MemoryRegion<TData> &y)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void neg(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::neg - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        negKernel<ExecSpace>(size, xptr, yptr);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void neg(MemoryRegion<TData> &x, MemoryRegion<TData> &y)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void sqrt(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sqrt - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        sqrtKernel<ExecSpace>(size, xptr, yptr);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void sqrt(MemoryRegion<TData> &x, MemoryRegion<TData> &y)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void add(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
         Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::add - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::add - Inconsistent interleave format between "
                  "input Fields");

        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto zptr = z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        addKernel<ExecSpace>(size, xptr, yptr, zptr);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void add(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void sub(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
         Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::sub - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::sub - Inconsistent interleave format between "
                  "input Fields");

        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto zptr = z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        subKernel<ExecSpace>(size, xptr, yptr, zptr);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void sub(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void mul(const TData alpha, Field<TData, TFieldState> &x,
         Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        mulKernel<ExecSpace>(size, alpha, xptr, yptr);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void mul(const TData alpha, MemoryRegion<TData> &x, MemoryRegion<TData> &y)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void mul(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
         Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::mul - Inconsistent interleave format between "
                  "input Fields");

        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto zptr = z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        mulKernel<ExecSpace>(size, xptr, yptr, zptr);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void mul(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void div(const TData alpha, Field<TData, TFieldState> &x,
         Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        divKernel<ExecSpace>(size, alpha, xptr, yptr);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void div(const TData alpha, MemoryRegion<TData> &x, MemoryRegion<TData> &y)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void div(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
         Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::div - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernel::div - Inconsistent interleave format between "
                  "input Fields");

        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto zptr = z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        divKernel<ExecSpace>(size, xptr, yptr, zptr);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void div(MemoryRegion<TData> &x, MemoryRegion<TData> &y, MemoryRegion<TData> &z)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void daxpy(const TData alpha, Field<TData, TFieldState> &x,
           Field<TData, TFieldState> &y, Field<TData, TFieldState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size() && y.size() != z.size())
    {
        std::stringstream msg;

        msg << "MathKernel::daxpy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernel::daxpy - Inconsistent interleave format between "
                  "input Fields");

        auto xptr = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto zptr = z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
        auto size = x.GetBlocks()[blk].size() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        daxpyKernel<ExecSpace>(size, alpha, xptr, yptr, zptr);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
}

template <typename ExecSpace, typename TData>
void daxpy(const TData alpha, MemoryRegion<TData> &x, MemoryRegion<TData> &y,
           MemoryRegion<TData> &z)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceSum(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceSumKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            reduceSumKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceSum(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceSumKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceSum(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
               TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernesl::reduceSum - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceSumKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            reduceSumKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceSum(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceSumKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMax(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceMaxKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            reduceMaxKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMax(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMaxKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMax(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
               TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::reduceMax - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceMaxKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            reduceMaxKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMax(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceMaxKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMin(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceMinKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            reduceMinKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMin(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMinKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMin(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
               TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::reduceMin - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            reduceMinKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            reduceMinKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMin(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    reduceMinKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void ddot(Field<TData, TFieldState> &x, Field<TData, TFieldState> &y,
          TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::ddot - Inconsistent interleave format between "
                  "input Fields");

        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr   = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            ddotKernel<ExecSpace, true>(size, xptr, yptr, out);
        }
        else
        {
            ddotKernel<ExecSpace, false>(size, xptr, yptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void ddot(MemoryRegion<TData> &x, MemoryRegion<TData> &y, TData *out)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void ddot(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
          Field<TData, TFieldState> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1((mask.GetBlocks()[blk].GetInterleaveWidth() ==
                   x.GetBlocks()[blk].GetInterleaveWidth()) &&
                      (mask.GetBlocks()[blk].GetInterleaveWidth() ==
                       y.GetBlocks()[blk].GetInterleaveWidth()),
                  "MathKernels::ddot - Inconsistent interleave format between "
                  "input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto yptr   = y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            ddotKernel<ExecSpace, true>(size, maskptr, xptr, yptr, out);
        }
        else
        {
            ddotKernel<ExecSpace, false>(size, maskptr, xptr, yptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void ddot(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x,
          MemoryRegion<TData> &y, TData *out)
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l1norm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            l1normKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            l1normKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void l1norm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l1normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l1norm(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
            TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::l1norm - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            l1normKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            l1normKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void l1norm(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    l1normKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l2norm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            l2normKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            l2normKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void l2norm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l2normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l2norm(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
            TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::l2norm - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            l2normKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            l2normKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void l2norm(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    l2normKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void lpnorm(const unsigned int p, Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            lpnormKernel<ExecSpace, true>(size, p, xptr, out);
        }
        else
        {
            lpnormKernel<ExecSpace, false>(size, p, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void lpnorm(const unsigned int p, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    lpnormKernel<ExecSpace, true>(nsize, p, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void lpnorm(const unsigned int p, Field<uint8_t, TFieldState> &mask,
            Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::lpnorm - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            lpnormKernel<ExecSpace, true>(size, p, maskptr, xptr, out);
        }
        else
        {
            lpnormKernel<ExecSpace, false>(size, p, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void lpnorm(const unsigned int p, MemoryRegion<uint8_t> &mask,
            MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    lpnormKernel<ExecSpace, true>(nsize, p, maskptr, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void linfnorm(Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            linfnormKernel<ExecSpace, true>(size, xptr, out);
        }
        else
        {
            linfnormKernel<ExecSpace, false>(size, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void linfnorm(MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    linfnormKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void linfnorm(Field<uint8_t, TFieldState> &mask, Field<TData, TFieldState> &x,
              TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::linfnorm - Inconsistent interleave format "
                  "between input Fields");

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto xptr   = x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
        auto &block = x.GetBlocks()[blk];
        auto size   = block.GetNumElements() * block.GetNumData() *
                    x.GetNumComponents() * x.GetNumHomoModes();

        if (blk == 0)
        {
            linfnormKernel<ExecSpace, true>(size, maskptr, xptr, out);
        }
        else
        {
            linfnormKernel<ExecSpace, false>(size, maskptr, xptr, out);
        }
    }
}

template <typename ExecSpace, typename TData>
void linfnorm(MemoryRegion<uint8_t> &mask, MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto maskptr = mask.template GetPtr<MemSpace, ReadOnly>();
    auto xptr    = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize   = x.size();

    linfnormKernel<ExecSpace, true>(nsize, maskptr, xptr, out);
}

} // namespace Nektar::Operators
