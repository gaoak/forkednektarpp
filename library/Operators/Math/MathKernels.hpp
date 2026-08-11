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

#include "MultiRegions/Field/Field.hpp"

#include "Operators/Math/MathAVXKernels.hpp"
#include "Operators/Math/MathDeviceOnHostKernels.hpp"
#include "Operators/Math/MathHIPCUDAKernels.hpp"
#include "Operators/Math/MathSYCLKernels.hpp"
#include "Operators/Math/MathSerialKernels.hpp"

#include "Operators/Common/OperatorsDeclspec.hpp"

namespace Nektar::Operators
{

template <typename MemSpace> class internalMathKernelMask
{
public:
    template <typename TData, FieldState TFieldState>
    static const uint8_t *GetInstance(
        MultiRegions::BlockAccessor<TData, TFieldState> &block)
    {
        auto key =
            std::tuple<size_t, size_t, unsigned int, size_t, unsigned int>(
                block.GetNumElements(), block.GetNumElementsWithPadding(),
                block.GetNumData(), block.CompSize(),
                block.GetInterleaveWidth());
        if (m_mask.find(key) == m_mask.end())
        {
            auto mask = std::vector<uint8_t>(block.CompSize());
            auto ptr  = mask.data();

            // Loop over chunks.
            for (size_t chunk = 0, el = 0; chunk < block.GetNumElmtGroups();
                 ++chunk, el += block.GetInterleaveWidth())
            {
                // Loop over interleave width
                for (unsigned int pt = 0; pt < block.GetNumData(); ++pt)
                {
                    for (unsigned int i = 0; i < block.GetInterleaveWidth();
                         ++i)
                    {
                        // Check for padding
                        if (el + i < block.GetNumElements())
                        {
                            *(ptr++) = 1;
                        }
                        else
                        {
                            *(ptr++) = 0;
                        }
                    }
                }
            }

            if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
            {
                hostMalloc(&m_mask[key], block.CompSize() * sizeof(uint8_t),
                           NektarSpaces::host_memory_alignment);
                memcpy(m_mask[key], mask.data(),
                       block.CompSize() * sizeof(uint8_t));
            }
            else if constexpr (std::is_same_v<MemSpace,
                                              NektarSpaces::DeviceSpace>)
            {
                const unsigned int streamID = 0;
                deviceMalloc(&m_mask[key], block.CompSize() * sizeof(uint8_t),
                             streamID);
                deviceMemcpy<HostToDevice>(m_mask[key], mask.data(),
                                           block.CompSize() * sizeof(uint8_t),
                                           streamID);
                nekStreamSynchronize(streamID);
            }
        }

        return m_mask[key];
    }

private:
    inline static std::map<
        std::tuple<size_t, size_t, unsigned int, size_t, unsigned int>,
        uint8_t *>
        m_mask;
};

template <typename ExecSpace, typename TData, FieldState TFieldState>
void zero(MultiRegions::Field<TData, TFieldState> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        zeroKernel<ExecSpace>(size, xptr, streamID);
    }
}

template <typename ExecSpace, typename TData>
void zero(LibUtilities::MemoryRegion<TData> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    zeroKernel<ExecSpace>(nsize, xptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void fill(const TData &val, MultiRegions::Field<TData, TFieldState> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        fillKernel<ExecSpace>(size, val, xptr, streamID);
    }
}

template <typename ExecSpace, typename TData>
void fill(const TData &val, LibUtilities::MemoryRegion<TData> &x)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, WriteOnly>();
    auto nsize = x.size();

    fillKernel<ExecSpace>(nsize, val, xptr);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void copy(MultiRegions::Field<TData, TFieldState> &x,
          MultiRegions::Field<TData, TFieldState> &y)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::copy - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        copyKernel<ExecSpace>(size, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void abs(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y)
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
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        absKernel<ExecSpace>(size, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void neg(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y)
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
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        negKernel<ExecSpace>(size, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void sqrt(MultiRegions::Field<TData, TFieldState> &x,
          MultiRegions::Field<TData, TFieldState> &y)
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
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        sqrtKernel<ExecSpace>(size, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void add(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y,
         MultiRegions::Field<TData, TFieldState> &z)
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

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        addKernel<ExecSpace>(size, xptr, yptr, zptr, streamID);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void sub(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y,
         MultiRegions::Field<TData, TFieldState> &z)
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

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        subKernel<ExecSpace>(size, xptr, yptr, zptr, streamID);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void mul(const TData alpha, MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y)
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
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        mulKernel<ExecSpace>(size, alpha, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void mul(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y,
         MultiRegions::Field<TData, TFieldState> &z)
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

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        mulKernel<ExecSpace>(size, xptr, yptr, zptr, streamID);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void div(const TData alpha, MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y)
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
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        divKernel<ExecSpace>(size, alpha, xptr, yptr, streamID);

        y.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void div(MultiRegions::Field<TData, TFieldState> &x,
         MultiRegions::Field<TData, TFieldState> &y,
         MultiRegions::Field<TData, TFieldState> &z)
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

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        divKernel<ExecSpace>(size, xptr, yptr, zptr, streamID);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void daxpy(const TData alpha, MultiRegions::Field<TData, TFieldState> &x,
           MultiRegions::Field<TData, TFieldState> &y,
           MultiRegions::Field<TData, TFieldState> &z)
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

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto zptr =
            z.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        daxpyKernel<ExecSpace>(size, alpha, xptr, yptr, zptr, streamID);

        z.GetBlocks()[blk].template SetInterleaveWidth<TData>(
            x.GetBlocks()[blk].GetInterleaveWidth());
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceSum(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            reduceSumKernel<ExecSpace, false>(size, maskptr, xptr, out,
                                              streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceSum(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceSumKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceSum(MultiRegions::Field<uint8_t, TFieldState> &mask,
               MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernesl::reduceSum - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        reduceSumKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMax(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = std::numeric_limits<TData>::lowest();
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceFill(out, std::numeric_limits<TData>::lowest(), 1, 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            reduceMaxKernel<ExecSpace, false>(size, maskptr, xptr, out,
                                              streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMax(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMaxKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMax(MultiRegions::Field<uint8_t, TFieldState> &mask,
               MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = std::numeric_limits<TData>::lowest();
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceFill(out, std::numeric_limits<TData>::lowest(), 1, 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::reduceMax - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        reduceMaxKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMin(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = std::numeric_limits<TData>::max();
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceFill(out, std::numeric_limits<TData>::max(), 1, 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            reduceMinKernel<ExecSpace, false>(size, maskptr, xptr, out,
                                              streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void reduceMin(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    reduceMinKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void reduceMin(MultiRegions::Field<uint8_t, TFieldState> &mask,
               MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = std::numeric_limits<TData>::max();
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceFill(out, std::numeric_limits<TData>::max(), 1, 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::reduceMin - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        reduceMinKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void ddot(MultiRegions::Field<TData, TFieldState> &x,
          MultiRegions::Field<TData, TFieldState> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(x.GetBlocks()[blk].GetInterleaveWidth() ==
                      y.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::ddot - Inconsistent interleave format between "
                  "input Fields");

        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            ddotKernel<ExecSpace, false>(size, maskptr, xptr, yptr, out,
                                         streamID);

            xptr += size;
            yptr += size;
        }
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void ddot(MultiRegions::Field<uint8_t, TFieldState> &mask,
          MultiRegions::Field<TData, TFieldState> &x,
          MultiRegions::Field<TData, TFieldState> &y, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if (x.size() != y.size())
    {
        std::stringstream msg;

        msg << "MathKernel::ddot - Memory size mismatch between Field";
        NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1((mask.GetBlocks()[blk].GetInterleaveWidth() ==
                   x.GetBlocks()[blk].GetInterleaveWidth()) &&
                      (mask.GetBlocks()[blk].GetInterleaveWidth() ==
                       y.GetBlocks()[blk].GetInterleaveWidth()),
                  "MathKernels::ddot - Inconsistent interleave format between "
                  "input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto yptr =
            y.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        ddotKernel<ExecSpace, false>(size, maskptr, xptr, yptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l1norm(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            l1normKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void l1norm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l1normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l1norm(MultiRegions::Field<uint8_t, TFieldState> &mask,
            MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::l1norm - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        l1normKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l2norm(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            l2normKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void l2norm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    l2normKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void l2norm(MultiRegions::Field<uint8_t, TFieldState> &mask,
            MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::l2norm - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        l2normKernel<ExecSpace, false>(size, maskptr, xptr, out);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void lpnorm(const unsigned int p, MultiRegions::Field<TData, TFieldState> &x,
            TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            lpnormKernel<ExecSpace, false>(size, p, maskptr, xptr, out,
                                           streamID);

            xptr += size;
        }
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void lpnorm(const unsigned int p,
            MultiRegions::Field<uint8_t, TFieldState> &mask,
            MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::lpnorm - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        lpnormKernel<ExecSpace, false>(size, p, maskptr, xptr, out, streamID);
    }
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

template <typename ExecSpace, typename TData, FieldState TFieldState>
void linfnorm(MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto &block  = x.GetBlocks()[blk];
        auto maskptr = internalMathKernelMask<MemSpace>::GetInstance(block);
        auto size    = block.CompSize();
        auto ncomp   = block.GetNumComponents() * block.GetNumHomoModes();

        for (unsigned int n = 0; n < ncomp; ++n)
        {
            linfnormKernel<ExecSpace, false>(size, maskptr, xptr, out,
                                             streamID);

            xptr += size;
        }
    }
}

template <typename ExecSpace, typename TData>
void linfnorm(LibUtilities::MemoryRegion<TData> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    auto xptr  = x.template GetPtr<MemSpace, ReadOnly>();
    auto nsize = x.size();

    linfnormKernel<ExecSpace, true>(nsize, xptr, out);
}

template <typename ExecSpace, typename TData, FieldState TFieldState>
void linfnorm(MultiRegions::Field<uint8_t, TFieldState> &mask,
              MultiRegions::Field<TData, TFieldState> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        *out = 0.0;
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        deviceMemset(out, 0, sizeof(TData), 0);
    }

    for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
    {
        WARNINGL1(mask.GetBlocks()[blk].GetInterleaveWidth() ==
                      x.GetBlocks()[blk].GetInterleaveWidth(),
                  "MathKernels::linfnorm - Inconsistent interleave format "
                  "between input Fields");

        const unsigned int streamID = blk + 1;

        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto size = x.GetBlocks()[blk].CompSize() * x.GetNumComponents() *
                    x.GetNumHomoModes();

        linfnormKernel<ExecSpace, false>(size, maskptr, xptr, out, streamID);
    }
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

} // namespace Nektar::Operators
