///////////////////////////////////////////////////////////////////////////////
//
// File: Field.cpp
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

#include "Operators/Field/Field.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators
{
template <typename TData, FieldState TState>
void Field<TData, TState>::ReshapeStorage(const unsigned int &interleaveWidth,
                                          const std::string &execSpace)
{
    for (auto &block : this->GetBlocks())
    {
        if (execSpace == "Serial")
        {
            auto *ptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
            {
                ::Nektar::Operators::ReshapeStorage<NektarSpaces::Serial>(
                    interleaveWidth, block.GetInterleaveWidth(),
                    block.GetNumElementsWithPadding(), block.GetNumData(),
                    ptr + n * block.CompSize());
            }
            block.template SetInterleaveWidth<TData>(interleaveWidth);
        }
#if defined(NEKTAR_ENABLE_SIMD)
        else if (execSpace == "AVX")
        {
            auto *ptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
            {
                ::Nektar::Operators::ReshapeStorage<NektarSpaces::AVX>(
                    interleaveWidth, block.GetInterleaveWidth(),
                    block.GetNumElementsWithPadding(), block.GetNumData(),
                    ptr + n * block.CompSize());
            }
            block.template SetInterleaveWidth<TData>(interleaveWidth);
        }
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
        else if (execSpace == "Device")
        {
            auto *ptr =
                block.template GetPtr<NektarSpaces::DeviceSpace, ReadWrite>();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
            {
                ::Nektar::Operators::ReshapeStorage<NektarSpaces::Device>(
                    interleaveWidth, block.GetInterleaveWidth(),
                    block.GetNumElementsWithPadding(), block.GetNumData(),
                    ptr + n * block.CompSize());
            }
            block.template SetInterleaveWidth<TData>(interleaveWidth);
        }
#endif
        else
        {
            ASSERTL0(false, "Unknown Execution space: " + execSpace)
        }
    }
}

template void Field<float, FieldState::Phys>::ReshapeStorage(
    const unsigned int &interleaveWidth, const std::string &execSpace);
template void Field<float, FieldState::Coeff>::ReshapeStorage(
    const unsigned int &interleaveWidth, const std::string &execSpace);
template void Field<double, FieldState::Phys>::ReshapeStorage(
    const unsigned int &interleaveWidth, const std::string &execSpace);
template void Field<double, FieldState::Coeff>::ReshapeStorage(
    const unsigned int &interleaveWidth, const std::string &execSpace);
} // namespace Nektar::Operators
