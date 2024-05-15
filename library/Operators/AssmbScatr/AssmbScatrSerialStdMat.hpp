///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrSerialStdMat.hpp
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

#include "AssmbScatrImplBase.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorAssmbScatrImpl
    : public OperatorAssmbScatrImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatrImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
    }

    void Assemble(Field<TData, FieldState::Coeff> &in,
                  MemoryRegion<TData> &out) override
    {
        // Copy the data to from the input field.
        Array<OneD, TData> inArray = in.toArray();
        Array<OneD, TData> outArray(this->m_nLocal);

        if (this->m_solnType == eIterativeFull)
        {
            this->m_assmbMap->Assemble(inArray, outArray);
        }
        else
        {
            this->m_assmbMap->AssembleBnd(inArray, outArray);
        }

        out =
            MemoryRegion<TData>::template fromArray<MemSpace, TData>(outArray);
    }

    void GlobalToLocal(MemoryRegion<TData> &in,
                       Field<TData, FieldState::Coeff> &out) override
    {
        // Copy the data to from the input field.
        Array<OneD, TData> inArray = in.toArray();
        Array<OneD, TData> outArray(this->m_nLocal);

        if (this->m_solnType == eIterativeFull)
        {
            this->m_assmbMap->GlobalToLocal(inArray, outArray);
        }
        else
        {
            this->m_assmbMap->GlobalToLocalBnd(inArray, outArray);
        }

        // Copy the data to the output field.
        out.template copyArray<MemSpace>(outArray);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorAssmbScatrImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
